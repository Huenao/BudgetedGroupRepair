from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from budgeted_group_repair_no_baran.final_replay_integrity_audit import (
    audit_final_causal_scope,
    compare_reported_micro_metrics,
    execute_integrity_audit,
    final_record_slice,
    recompute_micro_metrics,
    response_content_projection_sha256,
)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(
    cell_id: str,
    *,
    method: str,
    prediction: str,
    clean_value: str = "clean",
) -> dict[str, object]:
    if method == "llm_only":
        scenario, backend, budget, variant = "baseline", "none", None, "1"
    else:
        scenario, backend, budget, variant = (
            "size_conditioned",
            "lightgbm",
            0.2,
            "2",
        )
    return {
        "suite": "source",
        "dataset": "hospital",
        "method": method,
        "scenario": scenario,
        "backend": backend,
        "budget_share": budget,
        "group_size_variant": variant,
        "cell_id": cell_id,
        "prediction": prediction,
        "clean_value": clean_value,
        "parse_status": "ok",
        "final_source": "llm",
        "valid_prediction": True,
        "correct_repair": prediction == clean_value,
    }


def test_causal_scope_rejects_change_without_authority_support(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    _jsonl(
        source,
        [
            _record("approved", method="llm_only", prediction="old"),
            _record("unrelated", method="llm_only", prediction="old"),
        ],
    )
    _jsonl(
        target,
        [
            _record("approved", method="llm_only", prediction="clean"),
            _record("unrelated", method="llm_only", prediction="wrong"),
        ],
    )

    result = audit_final_causal_scope(
        configuration="fixture",
        source_path=source,
        target_path=target,
        query_causes={"q-approved": ("B_supplemental",)},
        query_cell_ids={"q-approved": {"approved"}},
        llm_only_query_by_cell={"approved": "q-approved", "unrelated": "q-old"},
        selected_queries_by_slice={},
    )

    assert result["all_passed"] is False
    assert result["authorized_changed_records"] == 1
    assert result["unauthorized_changed_records"] == 1
    assert result["violations"][0]["cell_id"] == "unrelated"
    assert result["violations"][0]["reason"] == "no_cache_or_supplemental_authority"


def test_causal_scope_requires_changed_query_to_cover_cell_and_slice(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    old = _record("c1", method="budgeted_group_lightgbm", prediction="old")
    new = _record("c1", method="budgeted_group_lightgbm", prediction="clean")
    _jsonl(source, [old])
    _jsonl(target, [new])
    canonical_slice = (
        "lightgbm/size_conditioned/variant_2/20pct/source__hospital"
    )

    allowed = audit_final_causal_scope(
        configuration="fixture",
        source_path=source,
        target_path=target,
        query_causes={"q1": ("A_cache_union",)},
        query_cell_ids={"q1": {"c1"}},
        llm_only_query_by_cell={},
        selected_queries_by_slice={canonical_slice: {"q1"}},
    )
    denied = audit_final_causal_scope(
        configuration="fixture",
        source_path=source,
        target_path=target,
        query_causes={"q1": ("A_cache_union",)},
        query_cell_ids={"q1": {"c1"}},
        llm_only_query_by_cell={},
        selected_queries_by_slice={canonical_slice: set()},
    )

    assert allowed["all_passed"] is True
    assert allowed["authorized_changed_records"] == 1
    assert denied["all_passed"] is False
    assert denied["violations"][0]["reason"] == "no_cache_or_supplemental_authority"


def test_same_cell_in_unselected_slice_has_no_causal_authority(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    old = _record("c1", method="budgeted_group_lightgbm", prediction="old")
    new = _record("c1", method="budgeted_group_lightgbm", prediction="clean")
    _jsonl(source, [old])
    _jsonl(target, [new])

    result = audit_final_causal_scope(
        configuration="fixture",
        source_path=source,
        target_path=target,
        query_causes={"q1": ("B_supplemental",)},
        query_cell_ids={"q1": {"c1"}},
        llm_only_query_by_cell={},
        selected_queries_by_slice={},
    )

    assert result["all_passed"] is False
    assert result["unsupported_changed_records"] == 1


def test_micro_metrics_are_recomputed_and_csv_drift_fails(tmp_path: Path) -> None:
    records = [
        _record("c1", method="llm_only", prediction="clean"),
        _record("c2", method="llm_only", prediction="wrong"),
    ]
    recomputed = recompute_micro_metrics(records)
    row = next(iter(recomputed.values()))
    assert row["true_error_cells"] == 2
    assert row["predicted_repairs"] == 2
    assert row["valid_predictions"] == 2
    assert row["invalid_predictions"] == 0
    assert row["correct_repairs"] == 1
    assert row["f1"] == 0.5

    metrics = tmp_path / "method_metrics.csv"
    with metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow({**row, "correct_repairs": 2})
    comparison = compare_reported_micro_metrics(recomputed, metrics)
    assert comparison["all_passed"] is False
    assert comparison["mismatches"][0]["field"] == "correct_repairs"


def test_response_projection_binds_content_but_not_replay_metadata() -> None:
    base = {
        "query_id": "q1",
        "prompt_hash": "p1",
        "provider_request_hash": "r1",
        "model": "model",
        "model_requested": "model",
        "cell_ids": ["c1"],
        "status": "success",
        "parse_status": "ok",
        "response_text": "answer-a",
        "items": [{"cell_id": "c1", "prediction": "clean"}],
        "metadata": {
            "phase": "online_selected_union",
            "prompt_schema_version": "schema-v1",
        },
    }
    replay_copy = {
        **base,
        "cache_hit": True,
        "checkpoint_hit": False,
        "metadata": {
            **base["metadata"],
            "final_replay_role": "supplemental_missing_query_last_row",
        },
    }
    changed_response = {**base, "response_text": "answer-b"}

    assert response_content_projection_sha256(base) == response_content_projection_sha256(
        replay_copy
    )
    assert response_content_projection_sha256(base) != response_content_projection_sha256(
        changed_response
    )


def test_selection_slice_zero_pads_one_and_five_percent_budgets() -> None:
    row = _record("c1", method="budgeted_group_lightgbm", prediction="old")

    assert final_record_slice({**row, "budget_share": 0.01}) == (
        "lightgbm/size_conditioned/variant_2/01pct/source__hospital"
    )
    assert final_record_slice({**row, "budget_share": 0.05}) == (
        "lightgbm/size_conditioned/variant_2/05pct/source__hospital"
    )
    assert final_record_slice({**row, "budget_share": 0.2}) == (
        "lightgbm/size_conditioned/variant_2/20pct/source__hospital"
    )


def test_integrity_audit_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "existing-audit").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        execute_integrity_audit(
            runs_root=runs,
            output_run_id="existing-audit",
            specs=(),
            expected_changed_final_keys=None,
        )
