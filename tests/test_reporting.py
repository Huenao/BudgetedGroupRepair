from __future__ import annotations

import pandas as pd

from budgeted_group_repair_no_baran.reporting import build_report
from budgeted_group_repair_no_baran.run_state import write_json


def test_portable_report_builder_accepts_canonical_artifact(tmp_path) -> None:
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    write_json(
        tmp_path / "run_manifest.json",
        {"experiment_config": {"primary_group_size": 4}},
    )
    datasets = [f"dataset_{index}" for index in range(9)]
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "N": 300,
                "baran_accuracy": 0.6,
                "singleton_accuracy": 0.62,
                "n11": 150,
                "n10": 30,
                "n01": 36,
                "n00": 84,
                "oracle_upper_bound": 0.72,
                "upper_bound_minus_best": 0.1,
            }
            for dataset in datasets
        ]
    ).to_csv(metrics / "experiment1_by_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "N_cells": 240,
                "singleton_accuracy": 0.62,
                "structured_accuracy": 0.63,
                "delta_accuracy": 0.01,
                "random_accuracy": 0.61,
                "structured_minus_random": 0.02,
                "token_per_cell_saving": 0.3,
                "decision": "B_noninferior_more_efficient",
            }
            for dataset in datasets
        ]
    ).to_csv(metrics / "experiment2_by_dataset.csv", index=False)
    pd.DataFrame(
        [
            {
                "phase": "TOTAL",
                "logical_queries": 3840,
                "physical_query_invocations": 3840,
                "provider_attempts": 3840,
                "observed_total_tokens": 1000,
            }
        ]
    ).to_csv(metrics / "api_cost_audit.csv", index=False)
    write_json(
        metrics / "experiment1_summary.json",
        {"macro": {"upper_bound_minus_best": 0.1}},
    )
    write_json(
        metrics / "experiment2_summary.json",
        {
            "macro": {"delta_accuracy": 0.01, "token_per_cell_saving": 0.3},
            "decision": "B_noninferior_more_efficient",
        },
    )
    write_json(metrics / "record_audit.json", {"ok": True})
    write_json(
        manifests / "partition_matching_audit.json",
        {
            f"source/{dataset}": {
                "structured": {
                    "covered_cell_count": 240,
                    "selected_group_count": 60,
                }
            }
            for dataset in datasets
        },
    )

    result = build_report(tmp_path, deliver=True)
    assert result["validated"] is True
    assert (tmp_path / "report" / "artifact.json").is_file()
    assert (tmp_path / "report" / "report.html").stat().st_size > 1000
