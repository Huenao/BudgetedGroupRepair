from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

from budgeted_group_repair_no_baran.motivation_reporting import (
    _failure_state,
    _overlap_summary,
    build_motivation_report,
)
from budgeted_group_repair_no_baran.sampling import SELECTED_DATASETS


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _synthetic_ledgers(root: Path) -> None:
    complementarity: list[dict[str, object]] = []
    group: list[dict[str, object]] = []
    for suite, dataset in SELECTED_DATASETS:
        for index in range(24):
            cell_id = f"{suite}:{dataset}:{index // 2}:{index}"
            baran_correct = index % 4 in {0, 1}
            llm_correct = index % 4 in {0, 2}
            quadrant = {
                (True, True): "n11",
                (True, False): "n10",
                (False, True): "n01",
                (False, False): "n00",
            }[(baran_correct, llm_correct)]
            complementarity.append(
                {
                    "suite": suite,
                    "dataset": dataset,
                    "base_family": dataset,
                    "cell_id": cell_id,
                    "row_cluster": f"{suite}:{dataset}:row:{index // 2}",
                    "column": "target",
                    "dirty_value": f"dirty-{index}",
                    "clean_value": f"clean-{index}",
                    "baran_prediction": f"clean-{index}" if baran_correct else "wrong",
                    "baran_valid": True,
                    "baran_correct": baran_correct,
                    "llm_prediction": f"clean-{index}" if llm_correct else "wrong",
                    "llm_valid": True,
                    "llm_correct": llm_correct,
                    "llm_status": "success",
                    "llm_parse_status": "ok",
                    "llm_decision": "propose",
                    "llm_observed_input_tokens": 8,
                    "llm_observed_output_tokens": 2,
                    "llm_observed_total_tokens": 10,
                    "outcome_quadrant": quadrant,
                }
            )
            for view in ("pattern", "semantic"):
                for size in (2, 4, 8):
                    structured_correct = index % 4 in {0, 1}
                    random_correct = llm_correct
                    structured_valid = index % 4 != 3
                    structured_parse = "ok" if structured_valid else "abstain"
                    structured_decision = "propose" if structured_valid else "abstain"
                    structured_total = 6 * size + 5
                    random_total = 7 * size + 5
                    singleton_id = f"singleton:{suite}:{dataset}:{index}"
                    structured_logical_id = (
                        f"structured:{suite}:{dataset}:{view}:k{size}:q{index // size}"
                    )
                    structured_physical_id = (
                        f"structured:{suite}:{dataset}:k{size}:q{index // size}"
                    )
                    random_logical_id = (
                        f"random:{suite}:{dataset}:{view}:k{size}:q{index // size}"
                    )
                    random_physical_id = (
                        f"random:{suite}:{dataset}:k{size}:q{index // size}"
                    )
                    group.append(
                        {
                            "suite": suite,
                            "dataset": dataset,
                            "base_family": dataset,
                            "source_view": view,
                            "group_size": size,
                            "cell_id": cell_id,
                            "row_cluster": f"{suite}:{dataset}:row:{index // 2}",
                            "column": "target",
                            "dirty_value": f"dirty-{index}",
                            "clean_value": f"clean-{index}",
                            "member_position": index % size,
                            "singleton_logical_query_id": singleton_id,
                            "singleton_physical_query_id": singleton_id,
                            "structured_logical_query_id": structured_logical_id,
                            "structured_physical_query_id": structured_physical_id,
                            "random_logical_query_id": random_logical_id,
                            "random_physical_query_id": random_physical_id,
                            "singleton_prediction": (
                                f"clean-{index}" if llm_correct else "wrong"
                            ),
                            "singleton_valid": True,
                            "singleton_correct": llm_correct,
                            "singleton_status": "success",
                            "singleton_parse_status": "ok",
                            "singleton_decision": "propose",
                            "structured_prediction": (
                                f"clean-{index}" if structured_correct else "wrong"
                            ),
                            "structured_valid": structured_valid,
                            "structured_correct": structured_correct,
                            "structured_status": "success",
                            "structured_parse_status": structured_parse,
                            "structured_decision": structured_decision,
                            "random_prediction": (
                                f"clean-{index}" if random_correct else "wrong"
                            ),
                            "random_valid": True,
                            "random_correct": random_correct,
                            "random_status": "success",
                            "random_parse_status": "ok",
                            "random_decision": "propose",
                            "structured_rescue": structured_correct and not llm_correct,
                            "structured_interference": llm_correct
                            and not structured_correct,
                            "random_rescue": random_correct and not llm_correct,
                            "random_interference": llm_correct and not random_correct,
                            "structured_query_observed_input_tokens": structured_total - 2,
                            "structured_query_observed_output_tokens": 2,
                            "structured_query_observed_total_tokens": structured_total,
                            "random_query_observed_input_tokens": random_total - 2,
                            "random_query_observed_output_tokens": 2,
                            "random_query_observed_total_tokens": random_total,
                            "singleton_query_observed_input_tokens": 8,
                            "singleton_query_observed_output_tokens": 2,
                            "singleton_query_observed_total_tokens": 10,
                            "singleton_query_attempts": 1,
                            "structured_query_attempts": 1,
                            "random_query_attempts": 1,
                            "singleton_query_latency_seconds": 0.1,
                            "structured_query_latency_seconds": 0.2,
                            "random_query_latency_seconds": 0.3,
                            "singleton_query_usage_observed_attempts": 1,
                            "structured_query_usage_observed_attempts": 1,
                            "random_query_usage_observed_attempts": 1,
                            "singleton_query_unknown_usage_attempts": 0,
                            "structured_query_unknown_usage_attempts": 0,
                            "random_query_unknown_usage_attempts": 0,
                        }
                    )
    _write_csv(root / "records" / "complementarity_cell_outcomes.csv", complementarity)
    _write_csv(root / "records" / "group_cell_outcomes.csv", group)
    logical_by_physical: dict[str, set[str]] = defaultdict(set)
    cost_by_physical: dict[str, dict[str, object]] = {}
    for row in group:
        for arm in ("singleton", "structured", "random"):
            physical_id = str(row[f"{arm}_physical_query_id"])
            logical_by_physical[physical_id].add(str(row[f"{arm}_logical_query_id"]))
            candidate = {
                "physical_query_id": physical_id,
                "provider_request_hash": f"hash:{physical_id}",
                "status": row[f"{arm}_status"],
                "attempts": row[f"{arm}_query_attempts"],
                "observed_input_tokens": row[f"{arm}_query_observed_input_tokens"],
                "observed_output_tokens": row[f"{arm}_query_observed_output_tokens"],
                "observed_total_tokens": row[f"{arm}_query_observed_total_tokens"],
                "latency_seconds": row[f"{arm}_query_latency_seconds"],
                "usage_observed_attempts": row[f"{arm}_query_usage_observed_attempts"],
                "unknown_usage_attempts": row[f"{arm}_query_unknown_usage_attempts"],
                "estimated_prompt_tokens": int(
                    row[f"{arm}_query_observed_input_tokens"]
                )
                + 1,
                "estimated_completion_tokens": int(
                    row[f"{arm}_query_observed_output_tokens"]
                )
                + 1,
                "estimated_total_tokens": int(
                    row[f"{arm}_query_observed_total_tokens"]
                )
                + 2,
            }
            previous = cost_by_physical.get(physical_id)
            assert previous is None or previous == candidate
            cost_by_physical[physical_id] = candidate
    api_cost = []
    for physical_id in sorted(cost_by_physical):
        api_cost.append(
            {
                **cost_by_physical[physical_id],
                "logical_query_mappings": len(logical_by_physical[physical_id]),
            }
        )
    _write_csv(root / "metrics" / "api_cost_audit.csv", api_cost)


def test_build_motivation_report_metrics_families_costs_and_outputs(tmp_path: Path) -> None:
    _synthetic_ledgers(tmp_path)

    result = build_motivation_report(tmp_path, bootstrap_replicates=12, bootstrap_seed=45)

    assert result["network_calls"] == 0
    assert result["complementarity_cells"] == 9 * 24
    assert result["group_cell_incidences"] == 9 * 24 * 6
    assert result["physical_union_calls"] > 0
    assert result["complementarity_metric_rows"] == 13
    assert result["group_metric_rows"] == 80
    assert result["primary_holm_tests"] == 18
    assert result["secondary_dataset_holm_tests"] == 162
    outputs = [Path(path) for path in result["outputs"].values()]
    assert {path.suffix for path in outputs} <= {".pdf", ".svg", ".md", ".csv", ".json"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)

    complementarity = _read_csv(
        tmp_path / "metrics" / "complementarity_by_dataset.csv"
    )
    micro = next(row for row in complementarity if row["scope"] == "micro")
    assert int(micro["n11"]) == 9 * 6
    assert int(micro["n10"]) == 9 * 6
    assert int(micro["n01"]) == 9 * 6
    assert int(micro["n00"]) == 9 * 6
    assert float(micro["baran_accuracy"]) == 0.5
    assert float(micro["llm_accuracy"]) == 0.5
    assert float(micro["oracle_union_upper_bound"]) == 0.75

    group = _read_csv(tmp_path / "metrics" / "group_by_dataset_view_size.csv")
    macro = next(
        row
        for row in group
        if row["scope"] == "macro"
        and row["source_view"] == "pattern"
        and row["group_size"] == "2"
    )
    assert float(macro["singleton_accuracy"]) == 0.5
    assert float(macro["structured_accuracy"]) == 0.5
    assert float(macro["structured_rescue_rate"]) == 0.25
    assert float(macro["structured_interference_rate"]) == 0.25
    assert float(macro["structured_minus_singleton"]) == 0.0
    assert float(macro["random_minus_singleton"]) == 0.0
    assert float(macro["structured_minus_random"]) == 0.0
    assert float(macro["coverage_rate"]) == 1.0
    assert float(macro["overlap_with_k8_rate"]) == 1.0
    assert int(macro["overlap_with_k8_union_cells"]) == 9 * 24
    assert float(macro["overlap_with_k8_common_singleton_accuracy"]) == float(
        macro["singleton_accuracy"]
    )
    assert float(macro["overlap_with_k8_common_structured_minus_singleton"]) == pytest.approx(
        float(macro["overlap_with_k8_common_structured_rescue_rate"])
        - float(macro["overlap_with_k8_common_structured_interference_rate"])
    )
    assert float(macro["paired_view_overlap_rate"]) == 1.0
    assert int(macro["paired_view_overlap_union_cells"]) == 9 * 24

    tests = _read_csv(tmp_path / "metrics" / "statistical_tests.csv")
    assert sum(row["test_family"] == "complementarity_dataset_9" for row in tests) == 9
    assert sum(row["test_family"] == "group_primary_18" for row in tests) == 18
    assert (
        sum(row["test_family"] == "group_secondary_dataset_162" for row in tests)
        == 162
    )
    assert all(
        row["holm_adjusted_p"] != ""
        for row in tests
        if row["test_family"] in {"group_primary_18", "group_secondary_dataset_162"}
    )

    costs = _read_csv(tmp_path / "metrics" / "group_costs.csv")
    structured_cost = next(
        row
        for row in costs
        if row["scope"] == "dataset"
        and row["suite"] == "source"
        and row["dataset"] == "hospital"
        and row["source_view"] == "pattern"
        and row["group_size"] == "2"
        and row["arm"] == "structured"
    )
    assert int(structured_cost["logical_calls"]) == 12
    assert int(structured_cost["physical_calls"]) == 12
    assert int(structured_cost["observed_total_tokens"]) == 12 * 17
    assert int(structured_cost["attempts"]) == 12
    assert int(structured_cost["retries"]) == 0
    assert float(structured_cost["latency_seconds"]) == pytest.approx(2.4)
    assert int(structured_cost["unknown_usage_attempts"]) == 0
    assert int(structured_cost["logical_observed_total_tokens"]) == 12 * 17

    overall_structured = next(
        row
        for row in costs
        if row["scope"] == "micro_all_conditions" and row["arm"] == "structured"
    )
    assert int(overall_structured["logical_calls"]) == 2 * int(
        overall_structured["physical_calls"]
    )
    assert int(overall_structured["logical_observed_total_tokens"]) == 2 * int(
        overall_structured["observed_total_tokens"]
    )

    api_rows = _read_csv(tmp_path / "metrics" / "api_cost_audit.csv")
    union = next(row for row in costs if row["scope"] == "run_physical_union")
    assert int(union["physical_calls"]) == len(api_rows)
    assert int(union["logical_calls"]) == sum(
        int(row["logical_query_mappings"]) for row in api_rows
    )
    assert int(union["observed_total_tokens"]) == sum(
        int(row["observed_total_tokens"]) for row in api_rows
    )
    assert int(union["estimated_total_tokens"]) == sum(
        int(row["estimated_total_tokens"]) for row in api_rows
    )
    assert union["cost_basis"] == "exact_once_per_physical_query_id"

    report = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "offline opportunity upper bound" in report
    assert "18-test Holm family" in report
    assert "unweighted macro over the fixed nine" in report

    first_metrics = (tmp_path / "metrics" / "group_by_dataset_view_size.csv").read_bytes()
    build_motivation_report(tmp_path, bootstrap_replicates=12, bootstrap_seed=45)
    assert (
        tmp_path / "metrics" / "group_by_dataset_view_size.csv"
    ).read_bytes() == first_metrics


def test_report_rejects_group_singleton_control_drift(tmp_path: Path) -> None:
    _synthetic_ledgers(tmp_path)
    path = tmp_path / "records" / "group_cell_outcomes.csv"
    rows = _read_csv(path)
    rows[0]["dirty_value"] = "drifted"
    _write_csv(path, rows)

    with pytest.raises(ValueError, match="singleton control does not match"):
        build_motivation_report(tmp_path, bootstrap_replicates=2)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"structured_status": "failed"}, "provider_failure"),
        (
            {
                "structured_decision": "",
                "structured_prediction": "",
                "structured_parse_status": "no_json_object",
            },
            "parse_failure",
        ),
        ({"structured_decision": "", "structured_parse_status": "partial"}, "missing"),
        ({"structured_decision": "abstain"}, "abstain"),
        ({"structured_prediction": "  "}, "empty"),
        ({"structured_prediction": " dirty&amp;value "}, "unchanged"),
        ({"structured_valid": False}, "other_invalid"),
    ],
)
def test_failure_subtypes_use_raw_per_cell_values_and_are_exclusive(
    overrides: dict[str, object], expected: str
) -> None:
    row: dict[str, object] = {
        "dirty_value": "dirty&value",
        "structured_status": "success",
        "structured_parse_status": "ok",
        "structured_decision": "propose",
        "structured_prediction": "different",
        "structured_valid": True,
    }
    row.update(overrides)
    assert _failure_state(row, "structured") == expected


def test_overlap_rate_is_directional_and_union_is_explicit() -> None:
    large = {f"c{index}" for index in range(10)}
    nested = {"c0", "c1"}

    intersection, union, large_rate, nested_rate, jaccard = _overlap_summary(
        [large], [nested]
    )

    assert (intersection, union) == (2, 10)
    assert large_rate == pytest.approx(0.2)
    assert nested_rate == pytest.approx(1.0)
    assert jaccard == pytest.approx(0.2)


def test_report_recomputes_correctness_instead_of_trusting_boolean(tmp_path: Path) -> None:
    _synthetic_ledgers(tmp_path)
    path = tmp_path / "records" / "complementarity_cell_outcomes.csv"
    rows = _read_csv(path)
    rows[0]["llm_correct"] = "False"
    _write_csv(path, rows)

    with pytest.raises(ValueError, match="stored llm_correct disagrees"):
        build_motivation_report(tmp_path, bootstrap_replicates=2)


def test_report_rejects_group_usage_drift_from_api_union_audit(tmp_path: Path) -> None:
    _synthetic_ledgers(tmp_path)
    path = tmp_path / "records" / "group_cell_outcomes.csv"
    rows = _read_csv(path)
    rows[0]["structured_query_observed_total_tokens"] = str(
        int(rows[0]["structured_query_observed_total_tokens"]) + 1
    )
    _write_csv(path, rows)

    with pytest.raises(ValueError, match="group/API cost total_tokens drift"):
        build_motivation_report(tmp_path, bootstrap_replicates=2)
