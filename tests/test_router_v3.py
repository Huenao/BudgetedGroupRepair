from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from budgeted_group_repair_no_baran import router_v3 as router_v3_module
from budgeted_group_repair_no_baran.data import SafeCell, write_jsonl
from budgeted_group_repair_no_baran.group_context import canonical_messages
from budgeted_group_repair_no_baran.group_generator import GroupQueryAction
from budgeted_group_repair_no_baran.prompt_policy import (
    INFORMATION_POLICY,
    PROMPT_SCHEMA_VERSION,
)
from budgeted_group_repair_no_baran.router_v3 import (
    ROUTER_V3_BUDGET_SWEEP_REVISION,
    ROUTER_V3_REVISION,
    ROUTER_V3_SWEEP_BUDGETS,
    ROUTER_V3_SWEEP_VARIANTS,
    ROUTER_V3_VARIANTS,
    ExperimentRunner,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _runner() -> ExperimentRunner:
    runner = object.__new__(ExperimentRunner)
    runner.router_revision = ROUTER_V3_REVISION
    runner.experiment_config = json.loads(
        (PROJECT_ROOT / "configs" / "experiment_router_v3.json").read_text(
            encoding="utf-8"
        )
    )
    return runner


def _sweep_runner() -> ExperimentRunner:
    runner = object.__new__(ExperimentRunner)
    runner.router_revision = ROUTER_V3_BUDGET_SWEEP_REVISION
    runner.experiment_config = json.loads(
        (
            PROJECT_ROOT
            / "configs"
            / "experiment_router_v3_budget_sweep_k24_lightgbm.json"
        ).read_text(encoding="utf-8")
    )
    return runner


def _singleton_action(cell_id: str) -> GroupQueryAction:
    return GroupQueryAction(
        query_id="singleton-query",
        suite="tableeg",
        dataset="toy",
        arm="singleton",
        group_view="singleton",
        cell_ids=(cell_id,),
        group_size=1,
        prompt_schema_version=PROMPT_SCHEMA_VERSION,
        prompt_information_policy=INFORMATION_POLICY,
        messages=canonical_messages(({"role": "user", "content": "{}"},)),
        prompt_hash="prompt-hash",
        estimated_prompt_tokens=10,
        completion_token_ceiling=192,
        estimated_total_tokens=202,
        group_features={},
    )


def test_router_v3_declares_exact_independent_size_conditions() -> None:
    runner = _runner()
    expected = {
        "1": (1,),
        "2": (1, 2),
        "4": (1, 4),
        "8": (1, 8),
        "all": (1, 2, 4, 8),
    }
    assert runner._router_training_variants() == expected
    specs = runner._scenario_specs()
    assert tuple(spec["group_size_variant"] for spec in specs) == ROUTER_V3_VARIANTS
    assert all(spec["scenario"] == "size_conditioned" for spec in specs)
    assert all(spec["budget_share"] == 0.2 for spec in specs)
    assert {
        str(spec["group_size_variant"]): tuple(spec["allowed_sizes"])
        for spec in specs
    } == expected


@pytest.mark.parametrize(
    ("variant", "allowed"),
    (("1", (1,)), ("2", (1, 2)), ("4", (1, 4)), ("8", (1, 8)), ("all", (1, 2, 4, 8))),
)
def test_router_v3_filters_training_and_test_pairs_before_use(
    variant: str,
    allowed: tuple[int, ...],
) -> None:
    frame = pd.DataFrame(
        {
            "group_size": [1, 1, 2, 2, 4, 4, 8, 8],
            "pair": list(range(8)),
        }
    )
    filtered = ExperimentRunner._filter_variant_pairs(
        frame,
        allowed,
        context=f"test-{variant}",
    )
    assert set(filtered["group_size"]) == set(allowed)
    assert set(filtered["pair"]).issubset(set(frame["pair"]))
    with pytest.raises(ValueError, match="group sizes differ"):
        ExperimentRunner._filter_variant_pairs(
            frame.loc[frame["group_size"] != allowed[-1]],
            allowed,
            context=f"missing-{variant}",
        )


def test_router_v3_prediction_artifacts_are_isolated_by_backend_and_variant(
    tmp_path: Path,
) -> None:
    runner = _runner()
    runner.paths = SimpleNamespace(gates_dir=tmp_path / "gates")
    for backend in ("lightgbm", "xgboost"):
        paths = {
            runner._prediction_path(backend, variant, "tableeg", "company")
            for variant in ROUTER_V3_VARIANTS
        }
        assert len(paths) == 5
        assert all(f"/{backend}/variant_" in str(path) for path in paths)


def test_llm_only_never_uses_baran_fallback() -> None:
    cell = SafeCell("tableeg", "toy", 0, 0, "name", "0", "dirty")
    action = _singleton_action(str(cell.cell_id))
    failed = ExperimentRunner._compact_llm_only_record(
        action,
        {"status": "failed", "model_matches_request": True},
        cell,
        "clean",
    )
    assert failed["prediction"] is None
    assert failed["final_source"] == "no_repair"
    assert failed["baran_fallback_used"] is False
    success = ExperimentRunner._compact_llm_only_record(
        action,
        {
            "status": "success",
            "model_matches_request": True,
            "items": [
                {
                    "cell_id": str(cell.cell_id),
                    "decision": "propose",
                    "repair": "clean",
                }
            ],
        },
        cell,
        "clean",
    )
    assert success["prediction"] == "clean"
    assert success["correct_repair"] is True
    assert success["final_source"] == "llm"
    assert success["baran_fallback_used"] is False


def test_router_v3_budget_sweep_has_exact_configured_matrix() -> None:
    runner = _sweep_runner()
    assert runner.is_router_v3_budget_sweep
    assert runner._active_gate_backends() == ("lightgbm",)
    assert tuple(runner._router_training_variants()) == ROUTER_V3_SWEEP_VARIANTS
    assert runner._router_training_variants() == {
        "2": (1, 2),
        "4": (1, 4),
    }
    assert runner._router_budget_shares() == ROUTER_V3_SWEEP_BUDGETS
    specs = runner._scenario_specs()
    assert len(specs) == 10
    assert {
        (
            str(spec["group_size_variant"]),
            float(spec["budget_share"]),
        )
        for spec in specs
    } == {
        (variant, budget)
        for variant in ROUTER_V3_SWEEP_VARIANTS
        for budget in ROUTER_V3_SWEEP_BUDGETS
    }


def test_router_v3_budget_sweep_reuses_one_prediction_per_k_not_budget(
    tmp_path: Path,
) -> None:
    runner = _sweep_runner()
    runner.paths = SimpleNamespace(
        gates_dir=tmp_path / "gates",
        selections_dir=tmp_path / "selections",
    )
    prediction_paths = {
        runner._prediction_path("lightgbm", variant, "source", "beers")
        for variant in ROUTER_V3_SWEEP_VARIANTS
        for _budget in ROUTER_V3_SWEEP_BUDGETS
    }
    assert len(prediction_paths) == 2
    selection_paths = {
        runner._selection_path(
            "lightgbm",
            "size_conditioned",
            variant,
            budget,
            "source",
            "beers",
        )
        for variant in ROUTER_V3_SWEEP_VARIANTS
        for budget in ROUTER_V3_SWEEP_BUDGETS
    }
    assert len(selection_paths) == 10
    assert all("/variant_" in str(path) and "pct/" in str(path) for path in selection_paths)


def test_router_v3_budget_sweep_filters_each_k_before_fit_and_predict() -> None:
    runner = _sweep_runner()
    frame = pd.DataFrame(
        {
            "group_size": [1, 2, 4, 8],
            "pair": [1, 2, 4, 8],
        }
    )
    for variant, sizes in runner._router_training_variants().items():
        filtered = runner._filter_variant_pairs(
            frame,
            sizes,
            context=f"sweep-{variant}",
        )
        assert set(filtered["group_size"]) == set(sizes)
        assert 8 not in set(filtered["group_size"])


def test_router_v3_create_allows_fresh_baran_and_llm_runs(tmp_path: Path) -> None:
    runner = ExperimentRunner.create(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT / "data",
        config_path=PROJECT_ROOT / "configs" / "experiment_router_v3.json",
        llm_config_path=PROJECT_ROOT / "configs" / "deepseek_v4.json",
        vendor_root=PROJECT_ROOT / "vendor" / "raha_source",
        runs_root=tmp_path,
        run_id="fresh-v3",
        provider_token_cap=1_000_000,
    )
    assert runner.baran_source_run is None
    assert runner.response_reuse_run is None
    assert runner.provider_token_cap == 1_000_000
    assert not hasattr(runner, "_reuse_router_v3_parent_calibration")


def test_provider_cap_requires_explicit_cap_or_uncapped_choice() -> None:
    runner = _runner()
    runner.provider_token_cap = None
    runner.allow_uncapped_provider_usage = False
    with pytest.raises(ValueError, match="--token-cap"):
        runner._effective_provider_cap(require=True)
    runner.provider_token_cap = 123
    assert runner._effective_provider_cap(require=True) == 123
    runner.allow_uncapped_provider_usage = True
    assert runner._effective_provider_cap(require=True) is None


def test_baran_stage_runs_fresh_when_no_source_is_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = SafeCell("source", "toy", 0, 0, "name", "0", "dirty")
    loaded = SimpleNamespace(
        safe_cells=lambda: [cell],
        oracle_cells=lambda include_annotations=False: [object()],
    )
    runner = object.__new__(ExperimentRunner)
    runner.paths = SimpleNamespace(
        baran_dir=tmp_path / "baran",
        vendor_root=tmp_path / "vendor",
    )
    runner.experiment_config = {
        "baran_labeling_budget": 20,
        "baran_seed": 16,
        "baran_workers": 1,
        "baran_multiprocessing_start_method": "spawn",
    }
    runner.baran_source_run = None
    runner._baran = {}
    runner._dataset = lambda suite, dataset: loaded
    runner.state = SimpleNamespace(update_stage=lambda *args, **kwargs: None)
    calls: list[object] = []

    def fake_run_baran(dataset, cells, *_args, **_kwargs):
        calls.extend(cells)
        return [
            {
                "cell_id": str(cell.cell_id),
                "suite": "source",
                "dataset": "toy",
                "prediction": "fixed",
                "parse_status": "ok_baran",
            }
        ]

    monkeypatch.setattr(router_v3_module, "run_baran", fake_run_baran)
    summary = runner.run_baran_stage((("source", "toy"),))
    assert calls
    assert summary["fresh"] is True
    assert summary["imported"] is False


def test_baran_stage_strictly_imports_a_bound_source(tmp_path: Path) -> None:
    cell = SafeCell("source", "toy", 0, 0, "name", "0", "dirty")
    loaded = SimpleNamespace(safe_cells=lambda: [cell])
    source = tmp_path / "source-run"
    record = {
        "cell_id": str(cell.cell_id),
        "suite": "source",
        "dataset": "toy",
        "prediction": "fixed",
        "parse_status": "ok_baran",
    }
    write_jsonl(source / "baran" / "source__toy.jsonl", [record])
    runner = object.__new__(ExperimentRunner)
    runner.paths = SimpleNamespace(baran_dir=tmp_path / "run" / "baran")
    runner.baran_source_run = source
    runner._baran = {}
    runner._dataset = lambda suite, dataset: loaded
    runner.state = SimpleNamespace(update_stage=lambda *args, **kwargs: None)
    summary = runner.run_baran_stage((("source", "toy"),))
    assert summary["fresh"] is False
    assert summary["imported"] is True
    assert runner._load_baran("source", "toy") == [record]
