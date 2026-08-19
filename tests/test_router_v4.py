from __future__ import annotations

import json
from pathlib import Path

import pytest

from budgeted_group_repair_no_baran.cli import parse_args
from budgeted_group_repair_no_baran.router_v3 import (
    ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION,
)
from budgeted_group_repair_no_baran.router_v4 import (
    FROZEN_ROUTER_V4_IMPLEMENTATION_SHA256,
    HISTORICAL_COMPARATOR_HASHES,
    RouterV4ExperimentRunner,
    _probability_diagnostics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_router_v4_frozen_config_and_variant_contract() -> None:
    config = json.loads(
        (
            PROJECT_ROOT
            / "configs"
            / "experiment_router_v4_lightgbm_isotonic_k14_budget20.json"
        ).read_text(encoding="utf-8")
    )
    assert config["router_revision"] == ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION
    assert config["gate_backends"] == ["lightgbm"]
    assert config["router_training_variants"] == {"1": [1], "4": [1, 4]}
    assert config["budget_shares"] == [0.2]
    runner = object.__new__(RouterV4ExperimentRunner)
    runner.experiment_config = config
    assert runner._router_training_variants() == {"1": (1,), "4": (1, 4)}
    runner._validate_calibration_configuration()
    assert runner._bgr_method_name("lightgbm") == "budgeted_group_lightgbm_isotonic"


def test_router_v4_probability_diagnostics_are_deterministic() -> None:
    labels = [0, 0, 1, 0, 1, 1]
    probabilities = [0.05, 0.15, 0.55, 0.35, 0.75, 0.95]
    first, first_bins = _probability_diagnostics(labels, probabilities)
    second, second_bins = _probability_diagnostics(labels, probabilities)
    assert first == second
    assert first_bins == second_bins
    assert first["actual_bins"] == 6
    assert sum(int(row["count"]) for row in first_bins) == len(labels)
    assert 0.0 <= float(first["ece"]) <= 1.0


def test_router_v4_cli_source_and_cache_only_contract() -> None:
    args = parse_args(
        [
            "run-router-bgr",
            "--run-id",
            "v4",
            "--cache-only",
            "--calibration-source-run",
            "runs/frozen",
        ]
    )
    assert args.cache_only is True
    assert args.calibration_source_run == Path("runs/frozen")
    with pytest.raises(SystemExit, match="supported only"):
        parse_args(["check-model", "--run-id", "v4", "--cache-only"])


def test_historical_comparator_hash_contract_is_complete() -> None:
    assert set(HISTORICAL_COMPARATOR_HASHES) == {
        "run_manifest.json",
        "bound_experiment_config.json",
        "llm/calibration_pair_labels.csv",
        "metrics/method_metrics.csv",
        "final/all_methods.jsonl",
    }
    assert all(len(value) == 64 for value in HISTORICAL_COMPARATOR_HASHES.values())
    assert FROZEN_ROUTER_V4_IMPLEMENTATION_SHA256 == {
        "7159d9d20fb2197670ee53f1a59b604c8919135d6dc8eea3b49011c7d866f67f"
    }
