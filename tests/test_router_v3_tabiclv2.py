from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from budgeted_group_repair_no_baran.cli import parse_args
from budgeted_group_repair_no_baran.group_gate import (
    FoundationFeatureEncoder,
    GroupUpliftGate,
    TabICLv2ClassifierAdapter,
)
from budgeted_group_repair_no_baran.router_v3 import (
    ROUTER_V3_TABICLV2_REVISION,
    ROUTER_V3_TABICLV2_VARIANTS,
    TABICLV2_GATE_BACKENDS,
    ExperimentRunner,
    SafetyCapExceeded,
)
from budgeted_group_repair_no_baran import router_reporting_v3, router_v3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_NAME = "tabicl-classifier-v2-20260212.ckpt"


def _runner() -> ExperimentRunner:
    runner = object.__new__(ExperimentRunner)
    runner.router_revision = ROUTER_V3_TABICLV2_REVISION
    runner.experiment_config = json.loads(
        (
            PROJECT_ROOT / "configs" / "experiment_router_v3_tabiclv2_k14.json"
        ).read_text(encoding="utf-8")
    )
    return runner


def _checkpoint(tmp_path: Path, contents: bytes = b"tabicl-test") -> tuple[Path, str]:
    path = tmp_path / CHECKPOINT_NAME
    path.write_bytes(contents)
    return path, sha256(contents).hexdigest()


def test_tabiclv2_revision_freezes_backend_k_and_budget() -> None:
    runner = _runner()
    assert runner.is_router_v3
    assert runner.is_router_v3_tabiclv2
    assert runner.freezes_reused_terminal_failures
    assert runner._active_gate_backends() == TABICLV2_GATE_BACKENDS
    assert tuple(runner._router_training_variants()) == ROUTER_V3_TABICLV2_VARIANTS
    assert runner._router_budget_shares() == (0.2,)
    specs = runner._scenario_specs()
    assert [value["scenario"] for value in specs] == ["size_conditioned"] * 2
    assert [value["group_size_variant"] for value in specs] == ["1", "4"]


def test_foundation_encoder_preserves_categories_missing_and_unseen() -> None:
    encoder = FoundationFeatureEncoder().fit(
        [
            {"kind": "alpha", "score": 1.0},
            {"kind": None, "score": None},
            {"kind": "beta", "score": 3.0},
        ]
    )
    transformed = encoder.transform(
        [{"kind": None, "score": None}, {"kind": "unseen", "score": 4.0}]
    )
    assert list(transformed.columns) == ["kind", "score"]
    assert isinstance(transformed["kind"].dtype, pd.CategoricalDtype)
    assert transformed["kind"].astype(str).tolist() == [
        "__BGR_MISSING_CATEGORY__",
        "__BGR_UNKNOWN_CATEGORY__",
    ]
    assert transformed["score"].tolist() == [2.0, 4.0]
    metadata = encoder.as_dict()
    assert metadata["kind"] == "foundation_native_categorical"
    assert metadata["numeric_missing_strategy"] == "train_median"
    assert metadata["categorical_feature_indices"] == [0]


def test_tabiclv2_adapter_rejects_missing_wrong_name_and_hash(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint path is unset"):
        TabICLv2ClassifierAdapter({"allow_auto_download": False})
    wrong = tmp_path / "wrong.ckpt"
    wrong.write_bytes(b"x")
    with pytest.raises(ValueError, match="checkpoint basename"):
        TabICLv2ClassifierAdapter({"checkpoint_path": str(wrong)})
    checkpoint, _ = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        TabICLv2ClassifierAdapter(
            {"checkpoint_path": str(checkpoint), "checkpoint_sha256": "0" * 64}
        )


def test_tabiclv2_adapter_lazy_import_probability_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    captured: dict[str, object] = {}

    class FakeClassifier:
        def __init__(self, **kwargs: object) -> None:
            captured["parameters"] = kwargs
            self.classes_ = np.asarray([0, 1])
            self.use_amp_ = True
            self.dtype_ = "torch.float16"

        def fit(self, features: pd.DataFrame, labels: list[int]) -> "FakeClassifier":
            captured["fit_frame"] = features.copy()
            captured["labels"] = labels
            return self

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            return np.tile(np.asarray([[0.25, 0.75]]), (len(features), 1))

    monkeypatch.setitem(
        sys.modules, "tabicl", SimpleNamespace(TabICLClassifier=FakeClassifier)
    )
    monkeypatch.setattr(
        "budgeted_group_repair_no_baran.group_gate.package_version",
        lambda package: "2.1.1" if package == "tabicl" else "unknown",
    )
    adapter = TabICLv2ClassifierAdapter(
        {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest,
            "n_estimators": 1,
            "batch_size": 8,
            "kv_cache": False,
            "device": "cuda",
            "use_amp": "auto",
            "use_fa3": "auto",
            "allow_auto_download": False,
        }
    )
    frame = pd.DataFrame(
        {
            "kind": pd.Series(pd.Categorical(["a", "b"])),
            "score": pd.Series([1.0, 2.0], dtype="float64"),
        }
    )
    adapter.fit(frame, [0, 1])
    first = adapter.predict_proba(frame)
    second = adapter.predict_proba(frame)
    assert np.array_equal(first, second)
    assert np.allclose(first[:, 1], 0.75)
    assert isinstance(captured["fit_frame"], pd.DataFrame)
    assert captured["parameters"]["model_path"] == str(checkpoint)
    metadata = adapter.metadata()
    assert metadata["package"] == "tabicl"
    assert metadata["checkpoint_sha256"] == digest
    assert metadata["parameters"]["n_estimators"] == 1
    assert metadata["fit_seconds"] >= 0
    assert metadata["predict_seconds"] >= 0


def test_tabiclv2_constant_heads_do_not_construct_foundation_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "budgeted_group_repair_no_baran.group_gate.package_version",
        lambda package: "2.1.1",
    )
    features = [{"kind": "alpha", "score": float(index)} for index in range(4)]
    gate = GroupUpliftGate("tabiclv2", backend_config={}).fit(
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


def test_router_token_cap_is_bound_and_cli_plan_is_unpaid(monkeypatch: pytest.MonkeyPatch) -> None:
    args = parse_args(
        [
            "plan-router-bgr",
            "--run-id",
            "tabicl-plan",
            "--experiment-config",
            "configs/experiment_router_v3_tabiclv2_k14.json",
        ]
    )
    assert args.command == "plan-router-bgr"
    assert not hasattr(args, "token_cap")
    paid = parse_args(
        ["run-router-bgr", "--run-id", "tabicl-run", "--token-cap", "103"]
    )
    assert paid.token_cap == 103
    runner = object.__new__(ExperimentRunner)
    runner.provider_token_cap = 100
    runner.allow_uncapped_provider_usage = False
    runner.experiment_config = {"max_estimated_tokens_safety_cap": None}
    runner.llm_config = {"max_retries": 2}
    monkeypatch.setattr(runner, "_provider_safety_debit", lambda: 10)
    runner._reserve_provider_safety("within", [30])
    with pytest.raises(SafetyCapExceeded):
        runner._reserve_provider_safety("over", [31])


def test_foundation_dry_plan_freezes_exact_cap_without_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm_dir = tmp_path / "llm"
    llm_dir.mkdir()
    (llm_dir / "selected_union_plan.json").write_text(
        json.dumps(
            {
                "query_ids": ["query-1"],
                "online_query_ids": ["query-1"],
                "model_preflight_estimated_tokens": 5,
                "retry_adjusted_token_cap": 60,
            }
        ),
        encoding="utf-8",
    )
    action = SimpleNamespace(
        query_id="query-1",
        prompt_hash="prompt-1",
        estimated_total_tokens=20,
    )
    runner = object.__new__(ExperimentRunner)
    runner.router_revision = ROUTER_V3_TABICLV2_REVISION
    runner.paths = SimpleNamespace(llm_dir=llm_dir, run_dir=tmp_path)
    runner.state = SimpleNamespace(
        stage_completed=lambda stage: stage == "gate_selection"
    )
    runner.llm_config = {"max_retries": 1}
    runner.runtime_token_cap = 60
    runner._load_actions = lambda suite, dataset: (action,)
    runner._response_index = lambda: {}
    runner._provider_safety_debit = lambda: 10
    runner._execute_jobs = lambda *args, **kwargs: pytest.fail(
        "dry plan attempted a provider call"
    )
    monkeypatch.setattr(router_v3, "target_order", lambda: (("suite", "dataset"),))

    summary = runner.plan_selected_llm_stage()

    assert summary["api_called"] is False
    assert summary["retry_adjusted_token_cap"] == 60
    persisted = json.loads(
        (tmp_path / "provenance" / "selected_execution_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted == summary


def test_tabiclv2_report_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {"ok": True, "kind": "tabiclv2-v3"}
    monkeypatch.setattr(
        router_reporting_v3,
        "_build_tabiclv2_report",
        lambda root, validation, output_path=None: expected,
    )
    monkeypatch.setattr(
        router_v3,
        "validate_run",
        lambda run_dir, require_complete=True: {
            "router_revision": ROUTER_V3_TABICLV2_REVISION
        },
    )
    assert router_reporting_v3.build_router_v3_report(tmp_path) == expected
