from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from budgeted_group_repair_no_baran.final_replay import (
    IDENTITY_FIELDS,
    build_deterministic_response_bank,
    copy_selection_projection,
    load_response_authority,
    load_supplemental_last_rows,
    materialize_selected_execution,
    merge_checkpoint_rows,
    plan_selection_projection,
    run_final_replay_stages,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _response(
    query_id: str,
    *,
    status: str,
    response_text: str,
    phase: str = "online_selected_union",
) -> dict[str, object]:
    return {
        "query_id": query_id,
        "prompt_hash": f"p-{query_id}",
        "provider_request_hash": f"r-{query_id}",
        "model": "deepseek-v4-flash",
        "model_requested": "deepseek-v4-flash",
        "model_matches_request": True,
        "status": status,
        "parse_status": "ok" if status == "success" else "llm_error",
        "response_text": response_text,
        "cache_hit": False,
        "checkpoint_hit": False,
        "metadata": {
            "phase": phase,
            "prompt_schema_version": "schema-v1",
        },
    }


def _authority(query_id: str, source: str, status: str = "success") -> dict[str, str]:
    return {
        "query_id": query_id,
        "prompt_hash": f"p-{query_id}",
        "provider_request_hash": f"r-{query_id}",
        "model": "deepseek-v4-flash",
        "prompt_schema_version": "schema-v1",
        "chosen_source_run": source,
        "chosen_status": status,
        "chosen_phase": "online_selected_union",
    }


def test_response_bank_uses_first_source_success_and_latest_row(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _jsonl(
        first,
        [
            _response("q1", status="success", response_text="old"),
            _response("q1", status="success", response_text="latest"),
            _response("q2", status="failed", response_text="failure"),
        ],
    )
    _jsonl(
        second,
        [
            _response("q1", status="success", response_text="other-source"),
            _response("q2", status="success", response_text="recovered"),
        ],
    )
    bank, audit = build_deterministic_response_bank(
        [_authority("q1", "first"), _authority("q2", "second")],
        [("first", first), ("second", second)],
    )
    assert [row["response_text"] for row in bank] == ["latest", "recovered"]
    assert all(row["cache_hit"] is True for row in bank)
    assert [row["source_run_id"] for row in audit] == ["first", "second"]


def test_authority_and_supplemental_last_row_are_strict(tmp_path: Path) -> None:
    authority_csv = tmp_path / "cache.csv"
    fields = [*IDENTITY_FIELDS, "chosen_source_run", "chosen_status", "chosen_phase"]
    with authority_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(_authority("q1", "first"))
    assert len(load_response_authority(authority_csv, expected_count=1)) == 1

    authority_jsonl = tmp_path / "missing.jsonl"
    _jsonl(authority_jsonl, [_authority("q1", "first")])
    retry = tmp_path / "retry.jsonl"
    _jsonl(
        retry,
        [
            _response("q1", status="failed", response_text="attempt-one"),
            _response("q1", status="success", response_text="last"),
        ],
    )
    rows = load_supplemental_last_rows(
        authority_jsonl, retry, expected_count=1
    )
    assert [row["response_text"] for row in rows] == ["last"]

    drift = _response("q1", status="success", response_text="bad")
    drift["provider_request_hash"] = "wrong"
    _jsonl(retry, [drift])
    with pytest.raises(ValueError, match="identity drift"):
        load_supplemental_last_rows(authority_jsonl, retry, expected_count=1)


def test_checkpoint_merge_allocates_fresh_cost_only_to_central_run() -> None:
    prefix = [_response("prefix", status="success", response_text="p")]
    bank = [_response("bank", status="success", response_text="b")]
    bank[0]["cache_hit"] = True
    supplemental = [_response("retry", status="success", response_text="r")]
    central = merge_checkpoint_rows(
        source_run_id="source",
        source_prefix=prefix,
        response_bank=bank,
        supplemental_rows=supplemental,
        retry_run_id="retry-run",
        central_retry_cost=True,
    )
    imported = merge_checkpoint_rows(
        source_run_id="source",
        source_prefix=prefix,
        response_bank=bank,
        supplemental_rows=supplemental,
        retry_run_id="retry-run",
        central_retry_cost=False,
    )
    assert [row["query_id"] for row in central] == ["prefix", "bank", "retry"]
    assert sum(not bool(row["cache_hit"]) for row in central) == 1
    assert sum(not bool(row["cache_hit"]) for row in imported) == 0
    assert central[-1]["metadata"]["final_replay_cost_allocation"] == "fresh_central"


def test_selection_projection_uses_historical_relative_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    repaired_run = tmp_path / "repaired"
    relative = Path("size_conditioned/variant_2/20pct/source__beers.json")
    old = source / "selections/lightgbm" / relative
    repaired = repaired_run / "selections/lightgbm" / relative
    old_doc = {
        "backend": "lightgbm",
        "selected_query_ids": ["q1"],
    }
    repaired_doc = {
        **old_doc,
        # The canonical reselect document may name the base run even when this
        # relative path is projected into the sweep configuration.
        "old_selection_source_run_id": "base-run",
        "numeric_semantics": "fixed-point",
    }
    _json(old, old_doc)
    _json(repaired, repaired_doc)
    pairs = plan_selection_projection(
        source_run=source,
        reselection_runs={"lightgbm": repaired_run},
        backends=("lightgbm",),
        expected_count=1,
    )
    assert pairs == [(old.resolve(), repaired.resolve())]
    destination = tmp_path / "new-run"
    copy_selection_projection(
        pairs, source_run=source, destination_run=destination
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite selection"):
        copy_selection_projection(
            pairs, source_run=source, destination_run=destination
        )


def test_selected_execution_uses_last_checkpoint_row() -> None:
    old = _response("q1", status="failed", response_text="old")
    new = _response("q1", status="success", response_text="new")
    new["cache_hit"] = True
    rows = materialize_selected_execution(
        query_prompt_pairs=[("q1", "p-q1")], checkpoint_rows=[old, new]
    )
    assert rows[0]["response_text"] == "new"
    assert rows[0]["checkpoint_hit"] is True


def test_stage_wrapper_never_calls_selected_llm_stage() -> None:
    calls: list[str] = []

    class Runner:
        def run_selected_llm_stage(self):
            raise AssertionError("provider-bearing selected stage must not run")

        def build_final_records_stage(self):
            calls.append("final")
            return {"ok": "final"}

        def build_metrics_stage(self):
            calls.append("metrics")
            return {"ok": "metrics"}

        def build_audit_stage(self):
            calls.append("audit")
            return {"ok": "audit"}

    result = run_final_replay_stages(Runner())
    assert calls == ["final", "metrics", "audit"]
    assert set(result) == {"final", "metrics", "audit"}
