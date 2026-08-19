from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from budgeted_group_repair_no_baran import router_reporting_v3, router_v3
from budgeted_group_repair_no_baran.router_v3 import (
    ROUTER_V3_FOUNDATION_MATRIX,
    ROUTER_V3_FOUNDATION_MATRIX_VARIANTS,
    ROUTER_V3_SWEEP_BUDGETS,
    ROUTER_V3_TABICLV2_MATRIX_REVISION,
    ROUTER_V3_TABPFN3_MATRIX_REVISION,
    ExperimentRunner,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("backend", "revision"),
    (
        ("tabiclv2", ROUTER_V3_TABICLV2_MATRIX_REVISION),
        ("tabpfn3", ROUTER_V3_TABPFN3_MATRIX_REVISION),
    ),
)
def test_foundation_matrix_declares_exact_sparse_scenarios(
    backend: str, revision: str
) -> None:
    runner = object.__new__(ExperimentRunner)
    runner.router_revision = revision
    runner.experiment_config = json.loads(
        (
            PROJECT_ROOT / "configs" / f"experiment_router_v3_{backend}_matrix.json"
        ).read_text(encoding="utf-8")
    )

    assert runner.is_router_v3_foundation_matrix
    assert tuple(runner._router_training_variants()) == (
        ROUTER_V3_FOUNDATION_MATRIX_VARIANTS
    )
    assert runner._router_budget_shares() == ROUTER_V3_SWEEP_BUDGETS
    assert runner._router_scenario_matrix() == ROUTER_V3_FOUNDATION_MATRIX
    specs = runner._scenario_specs()
    assert len(specs) == 12
    assert {
        (str(spec["group_size_variant"]), float(spec["budget_share"]))
        for spec in specs
    } == {
        (variant, budget)
        for variant, budgets in ROUTER_V3_FOUNDATION_MATRIX.items()
        for budget in budgets
    }


def test_foundation_matrix_keeps_one_prediction_per_k_and_sparse_selections(
    tmp_path: Path,
) -> None:
    runner = object.__new__(ExperimentRunner)
    runner.router_revision = ROUTER_V3_TABICLV2_MATRIX_REVISION
    runner.experiment_config = json.loads(
        (
            PROJECT_ROOT / "configs" / "experiment_router_v3_tabiclv2_matrix.json"
        ).read_text(encoding="utf-8")
    )
    runner.paths = SimpleNamespace(
        gates_dir=tmp_path / "gates", selections_dir=tmp_path / "selections"
    )
    prediction_paths = {
        runner._prediction_path("tabiclv2", variant, "source", "beers")
        for variant, _budgets in ROUTER_V3_FOUNDATION_MATRIX.items()
    }
    selection_paths = {
        runner._selection_path(
            "tabiclv2",
            "size_conditioned",
            variant,
            budget,
            "source",
            "beers",
        )
        for variant, budgets in ROUTER_V3_FOUNDATION_MATRIX.items()
        for budget in budgets
    }
    assert len(prediction_paths) == 4
    assert len(selection_paths) == 12


@pytest.mark.parametrize(
    ("revision", "backend"),
    (
        (ROUTER_V3_TABICLV2_MATRIX_REVISION, "tabiclv2"),
        (ROUTER_V3_TABPFN3_MATRIX_REVISION, "tabpfn3"),
    ),
)
def test_foundation_matrix_report_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revision: str,
    backend: str,
) -> None:
    expected = {"ok": True, "kind": f"{backend}-matrix"}
    monkeypatch.setattr(
        router_reporting_v3,
        "_build_foundation_matrix_report",
        lambda root, validation, output_path=None, *, backend: expected,
    )
    monkeypatch.setattr(
        router_v3,
        "validate_run",
        lambda run_dir, require_complete=True: {"router_revision": revision},
    )
    assert router_reporting_v3.build_router_v3_report(tmp_path) == expected
