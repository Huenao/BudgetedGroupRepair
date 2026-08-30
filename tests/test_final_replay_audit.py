from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from budgeted_group_repair_no_baran.final_replay_audit import (
    ReplayAuditSpec,
    compare_final_records,
    execute_final_replay_audit,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    cell_id: str,
    *,
    prediction: str,
    correct: bool,
    accepted: bool | None,
    method: str = "bgr",
) -> dict[str, object]:
    row: dict[str, object] = {
        "suite": "source",
        "dataset": "hospital",
        "method": method,
        "scenario": "size_conditioned",
        "backend": "lightgbm",
        "budget_share": 0.2,
        "group_size_variant": "2",
        "cell_id": cell_id,
        "prediction": prediction,
        "parse_status": "ok",
        "final_source": "baran",
        "correct_repair": correct,
    }
    if accepted is not None:
        row["accepted_llm"] = accepted
    return row


def _make_pair(tmp_path: Path) -> tuple[Path, ReplayAuditSpec]:
    runs = tmp_path / "runs"
    source = runs / "source-run"
    target = runs / "target-run"
    old_rows = [
        _record("c1", prediction="old", correct=False, accepted=False),
        _record("c2", prediction="right", correct=True, accepted=None),
    ]
    new_rows = [
        {
            **_record("c1", prediction="right", correct=True, accepted=True),
            "parse_status": "ok_llm",
            "final_source": "llm",
        },
        _record("c2", prediction="wrong", correct=False, accepted=False),
    ]
    _jsonl(source / "final/all_methods.jsonl", old_rows)
    _jsonl(target / "final/all_methods.jsonl", new_rows)
    _jsonl(
        target / "llm/selected_execution.jsonl",
        [
            {"query_id": "q1", "prompt_hash": "p1"},
            {"query_id": "q2", "prompt_hash": "p2"},
        ],
    )
    _json(target / "selections/lightgbm/one.json", {"selected_query_ids": ["q1"]})
    stages = {
        stage: {"status": "complete"}
        for stage in ("offline_final_replay_inputs", "final_records", "metrics", "audit")
    }
    replay = {"api_called_by_final_replay": False}
    _json(
        target / "run_manifest.json",
        {
            "run_id": "target-run",
            "status": "complete",
            "stages": stages,
            "final_replay": replay,
        },
    )
    _json(
        target / "provenance/final_replay.json",
        {
            **replay,
            "run_selected_llm_stage_called": False,
            "stages_executed": ["final_records", "metrics", "audit"],
            "copied_repaired_selections": [{}],
        },
    )
    _json(target / "llm/selected_union_plan.json", replay)
    return runs, ReplayAuditSpec(
        "fixture",
        "source-run",
        "target-run",
        ("lightgbm",),
        2,
        1,
        2,
    )


def test_compare_final_records_uses_semantic_keys_not_row_order(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    first = _record("c1", prediction="old", correct=False, accepted=False)
    second = _record("c2", prediction="right", correct=True, accepted=None)
    changed = {
        **_record("c1", prediction="right", correct=True, accepted=True),
        "parse_status": "ok_llm",
        "final_source": "llm",
    }
    _jsonl(source, [first, second])
    _jsonl(target, [second, changed])
    rows, summary = compare_final_records(
        configuration="fixture", source_path=source, target_path=target
    )
    assert len(rows) == 1
    assert summary["semantic_keys_equal"] is True
    assert summary["prediction_changed"] == 1
    assert summary["parse_status_changed"] == 1
    assert summary["final_source_changed"] == 1
    assert summary["accepted_llm_changed"] == 1
    assert summary["correct_repair_changed"] == 1
    assert summary["newly_correct"] == 1
    assert summary["lost_correct"] == 0


def test_compare_reports_key_set_drift_and_rejects_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    row = _record("c1", prediction="old", correct=False, accepted=False)
    _jsonl(source, [row])
    _jsonl(target, [])
    _, summary = compare_final_records(
        configuration="fixture", source_path=source, target_path=target
    )
    assert summary["semantic_keys_equal"] is False
    assert summary["source_only_records"] == 1

    _jsonl(source, [row, row])
    with pytest.raises(ValueError, match="duplicate final-record semantic key"):
        compare_final_records(
            configuration="fixture", source_path=source, target_path=target
        )


def test_audit_writes_new_artifacts_without_modifying_compared_runs(tmp_path: Path) -> None:
    runs, spec = _make_pair(tmp_path)
    source_final = runs / "source-run/final/all_methods.jsonl"
    target_final = runs / "target-run/final/all_methods.jsonl"
    before = (_sha(source_final), _sha(target_final))
    result = execute_final_replay_audit(
        runs_root=runs, output_run_id="audit-run", specs=(spec,)
    )
    assert result["validation_passed"] is True
    assert (_sha(source_final), _sha(target_final)) == before
    output = runs / "audit-run"
    assert (output / "final_result_difference.csv").is_file()
    assert (output / "final_result_difference_summary.csv").is_file()
    assert (output / "final_replay_runs.csv").is_file()
    validation = json.loads(
        (output / "replay_specific_validation.json").read_text(encoding="utf-8")
    )
    assert validation["standard_router_v3_validate_run_used"] is False
    assert validation["source_and_target_runs_read_only"] is True
    assert validation["all_passed"] is True
    with (output / "final_result_difference_summary.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    assert [row["configuration"] for row in summary_rows] == ["fixture", "__all__"]
    assert summary_rows[0]["newly_correct"] == "1"
    assert summary_rows[0]["lost_correct"] == "1"

    with pytest.raises(FileExistsError, match="refusing to overwrite audit run"):
        execute_final_replay_audit(
            runs_root=runs, output_run_id="audit-run", specs=(spec,)
        )


def test_validation_records_offline_flag_failure_instead_of_hiding_it(
    tmp_path: Path,
) -> None:
    runs, spec = _make_pair(tmp_path)
    provenance_path = runs / "target-run/provenance/final_replay.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["api_called_by_final_replay"] = True
    _json(provenance_path, provenance)
    result = execute_final_replay_audit(
        runs_root=runs, output_run_id="failed-audit", specs=(spec,)
    )
    assert result["validation_passed"] is False
    validation = json.loads(
        (runs / "failed-audit/replay_specific_validation.json").read_text(
            encoding="utf-8"
        )
    )
    check = validation["runs"][0]["checks"][
        "provenance_api_called_by_final_replay"
    ]
    assert check == {"expected": False, "observed": True, "passed": False}
