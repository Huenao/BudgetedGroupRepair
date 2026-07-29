from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from budgeted_group_repair_no_baran.group_gate import (
    CatBoostFeatureEncoder,
    GroupUpliftGate,
)
from budgeted_group_repair_no_baran.router_v2 import (
    CATBOOST_GATE_BACKENDS,
    FROZEN_ROUTER_V3_CATBOOST_IMPLEMENTATION_SHA256,
    ROUTER_V3_CATBOOST_REVISION,
    ROUTER_V3_VARIANTS,
    ExperimentRunner,
    _router_v3_implementation_binding_matches,
)
from budgeted_group_repair_no_baran import router_reporting, router_reporting_v3
from budgeted_group_repair_no_baran.run_state import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _runner() -> ExperimentRunner:
    runner = object.__new__(ExperimentRunner)
    runner.router_revision = ROUTER_V3_CATBOOST_REVISION
    runner.experiment_config = json.loads(
        (PROJECT_ROOT / "configs" / "experiment_router_v3_catboost.json").read_text(
            encoding="utf-8"
        )
    )
    return runner


def test_catboost_revision_freezes_20pct_full_k_matrix() -> None:
    runner = _runner()
    assert runner.is_router_v3
    assert runner.is_router_v3_catboost
    assert runner.freezes_reused_terminal_failures
    assert runner._active_gate_backends() == CATBOOST_GATE_BACKENDS
    assert tuple(runner._router_training_variants()) == ROUTER_V3_VARIANTS
    assert runner._router_budget_shares() == (0.2,)
    assert len(runner._scenario_specs()) == 5


def test_catboost_paths_isolate_five_models_and_selections(tmp_path: Path) -> None:
    runner = _runner()
    runner.paths = SimpleNamespace(
        gates_dir=tmp_path / "gates",
        selections_dir=tmp_path / "selections",
    )
    predictions = {
        runner._prediction_path("catboost", variant, "tableeg", "company")
        for variant in ROUTER_V3_VARIANTS
    }
    selections = {
        runner._selection_path(
            "catboost",
            "size_conditioned",
            variant,
            0.2,
            "tableeg",
            "company",
        )
        for variant in ROUTER_V3_VARIANTS
    }
    assert len(predictions) == len(selections) == 5
    assert all("/catboost/variant_" in str(path) for path in predictions)
    assert all("/catboost/size_conditioned/variant_" in str(path) for path in selections)


def test_catboost_encoder_preserves_native_categories_and_train_only_fill() -> None:
    encoder = CatBoostFeatureEncoder().fit(
        [
            {"kind": "alpha", "score": 1.0},
            {"kind": None, "score": None},
            {"kind": "beta", "score": 3.0},
        ]
    )
    transformed = encoder.transform([{"kind": "unseen", "score": None}])
    assert transformed == [["str:unseen", 2.0]]
    metadata = encoder.metadata() if hasattr(encoder, "metadata") else encoder.as_dict()
    assert metadata["kind"] == "catboost_native_categorical"
    assert metadata["categorical_feature_indices"] == [0]
    assert metadata["missing_category"] == "__BGR_MISSING_CATEGORY__"


def test_catboost_gate_is_deterministic_and_does_not_write_training_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    features = [
        {"kind": "alpha", "score": 1.0},
        {"kind": "beta", "score": 2.0},
        {"kind": None, "score": None},
        {"kind": "gamma", "score": 4.0},
    ]
    args = (
        features,
        [False, True, False, True],
        [True, False, False, True],
        [True, True, True, True],
        ["family"] * 4,
    )
    first = GroupUpliftGate("catboost", random_state=42).fit(*args)
    second = GroupUpliftGate("catboost", random_state=42).fit(*args)
    first_predictions = [row.as_dict() for row in first.predict(features)]
    second_predictions = [row.as_dict() for row in second.predict(features)]
    assert first_predictions == second_predictions
    metadata = first.metadata()
    assert metadata["backend"] == "catboost"
    assert metadata["full"]["encoder"]["kind"] == "catboost_native_categorical"
    parameters = metadata["full"]["helpful_head"]["parameters"]
    assert parameters["iterations"] == 200
    assert parameters["allow_writing_files"] is False
    assert parameters["thread_count"] == 1
    assert not (tmp_path / "catboost_info").exists()


def test_catboost_gate_handles_constant_binary_heads() -> None:
    features = [{"kind": "alpha", "score": float(index)} for index in range(4)]
    gate = GroupUpliftGate("catboost", random_state=42).fit(
        features,
        [False] * 4,
        [False] * 4,
        [True] * 4,
        ["family"] * 4,
    )
    metadata = gate.metadata()["full"]
    assert metadata["helpful_head"] == {"kind": "constant", "probability": 0.0}
    assert metadata["harmful_head"] == {"kind": "constant", "probability": 0.0}
    assert all(row.q_helpful == row.q_harmful == 0.0 for row in gate.predict(features))


def test_catboost_report_dispatch_and_physical_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_json(
        tmp_path / "run_manifest.json",
        {"experiment_config": {"router_revision": ROUTER_V3_CATBOOST_REVISION}},
    )
    expected = {"ok": True, "kind": "catboost-v3"}
    monkeypatch.setattr(
        router_reporting_v3,
        "build_router_v3_report",
        lambda run_dir, output_path=None: expected,
    )
    assert router_reporting.build_router_report(tmp_path) == expected
    detailed = router_reporting_v3._prepare_detailed(
        [{"suite": "source", "dataset": "beers", "physical_calls_charged": 3}]
    )
    assert detailed[0]["physical_calls"] == "3"
    assert (
        "1ac1d8e816e693bd3459ba69e57f5b633275a8ac2f2d2235ec5ddce193f4e54f"
        in FROZEN_ROUTER_V3_CATBOOST_IMPLEMENTATION_SHA256
    )


def test_frozen_catboost_hash_is_revision_scoped() -> None:
    frozen = "1ac1d8e816e693bd3459ba69e57f5b633275a8ac2f2d2235ec5ddce193f4e54f"
    assert _router_v3_implementation_binding_matches(
        ROUTER_V3_CATBOOST_REVISION, frozen, "current"
    )
    assert not _router_v3_implementation_binding_matches(
        "router_v3_exact_size_conditioned", frozen, "current"
    )
    assert not _router_v3_implementation_binding_matches(
        "router_v3_budget_sweep_exact_size_conditioned", frozen, "current"
    )
    assert not _router_v3_implementation_binding_matches(
        ROUTER_V3_CATBOOST_REVISION, "unknown", "current"
    )
    assert _router_v3_implementation_binding_matches(
        ROUTER_V3_CATBOOST_REVISION, "current", "current"
    )
