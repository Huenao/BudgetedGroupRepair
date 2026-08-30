"""Read-only, replay-specific validation for the five offline final runs.

This module compares materialised final records against their historical source
runs and writes a separate audit run.  It does not import a provider client,
modify either compared run, or invoke the standard Router-v3 run validator.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .run_state import sha256_file, write_json


SEMANTIC_KEY_FIELDS = (
    "suite",
    "dataset",
    "method",
    "scenario",
    "backend",
    "budget_share",
    "group_size_variant",
    "cell_id",
)
COMPARISON_FIELDS = (
    "prediction",
    "parse_status",
    "final_source",
    "accepted_llm",
    "correct_repair",
)
GROUP_FIELDS = (
    "configuration",
    "backend",
    "scenario",
    "group_size_variant",
    "budget_share",
    "dataset",
    "method",
)
COUNTER_FIELDS = (
    "source_records",
    "target_records",
    "matched_records",
    "source_only_records",
    "target_only_records",
    "prediction_changed",
    "parse_status_changed",
    "final_source_changed",
    "accepted_llm_changed",
    "correct_repair_changed",
    "newly_correct",
    "lost_correct",
    "any_changed_records",
)
REQUIRED_COMPLETE_STAGES = (
    "offline_final_replay_inputs",
    "final_records",
    "metrics",
    "audit",
)
EXPECTED_EXECUTED_STAGES = ("final_records", "metrics", "audit")


@dataclass(frozen=True)
class ReplayAuditSpec:
    configuration: str
    source_run_id: str
    target_run_id: str
    selection_backends: tuple[str, ...]
    expected_final_records: int
    expected_selection_slices: int
    expected_selected_execution: int


DEFAULT_AUDIT_SPECS = (
    ReplayAuditSpec(
        "base",
        "no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_base_final",
        ("lightgbm", "xgboost"),
        266_376,
        90,
        27_621,
    ),
    ReplayAuditSpec(
        "lightgbm_sweep",
        "no_baran_router_v3_deepseek_v4_20260726_budget_sweep_k24_lightgbm",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_lightgbm_sweep_final",
        ("lightgbm",),
        266_376,
        90,
        27_824,
    ),
    ReplayAuditSpec(
        "catboost",
        "no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_catboost_final",
        ("catboost",),
        155_386,
        45,
        25_048,
    ),
    ReplayAuditSpec(
        "tabiclv2",
        "no_baran_router_v3_tabiclv2_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_tabiclv2_final",
        ("tabiclv2",),
        310_772,
        108,
        27_052,
    ),
    ReplayAuditSpec(
        "tabpfn3",
        "no_baran_router_v3_tabpfn3_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_tabpfn3_final",
        ("tabpfn3",),
        310_772,
        108,
        26_319,
    ),
)


@dataclass
class DifferenceCounters:
    source_records: int = 0
    target_records: int = 0
    matched_records: int = 0
    source_only_records: int = 0
    target_only_records: int = 0
    prediction_changed: int = 0
    parse_status_changed: int = 0
    final_source_changed: int = 0
    accepted_llm_changed: int = 0
    correct_repair_changed: int = 0
    newly_correct: int = 0
    lost_correct: int = 0
    any_changed_records: int = 0

    def add(self, other: "DifferenceCounters") -> None:
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name) + getattr(other, field.name))

    def as_dict(self) -> dict[str, int]:
        return {field.name: int(getattr(self, field.name)) for field in fields(self)}


SemanticKey = tuple[str, ...]
ComparableValue = tuple[bool, str]
ComparableRecord = tuple[ComparableValue, ...]
GroupKey = tuple[str, ...]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _canonical_scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def semantic_key(row: Mapping[str, Any]) -> SemanticKey:
    missing = [field for field in SEMANTIC_KEY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"final record is missing semantic-key fields: {missing}")
    return tuple(_canonical_scalar(row[field]) for field in SEMANTIC_KEY_FIELDS)


def _comparable_record(row: Mapping[str, Any]) -> ComparableRecord:
    return tuple(
        (field in row, _canonical_scalar(row[field]) if field in row else "")
        for field in COMPARISON_FIELDS
    )


def _is_true(value: ComparableValue) -> bool:
    return value == (True, "true")


def _display_key_value(value: str) -> str:
    decoded = json.loads(value)
    if decoded is None:
        return "null"
    if isinstance(decoded, bool):
        return "true" if decoded else "false"
    return str(decoded)


def _group_key(configuration: str, semantic: SemanticKey) -> GroupKey:
    values = dict(zip(SEMANTIC_KEY_FIELDS, semantic))
    return (
        configuration,
        _display_key_value(values["backend"]),
        _display_key_value(values["scenario"]),
        _display_key_value(values["group_size_variant"]),
        _display_key_value(values["budget_share"]),
        _display_key_value(values["dataset"]),
        _display_key_value(values["method"]),
    )


def _read_record_index(path: Path) -> dict[SemanticKey, ComparableRecord]:
    result: dict[SemanticKey, ComparableRecord] = {}
    for row in _iter_jsonl(path):
        key = semantic_key(row)
        if key in result:
            raise ValueError(f"duplicate final-record semantic key in {path}: {key}")
        result[key] = _comparable_record(row)
    return result


def _update_matched(
    counters: DifferenceCounters,
    source: ComparableRecord,
    target: ComparableRecord,
) -> None:
    counters.source_records += 1
    counters.target_records += 1
    counters.matched_records += 1
    changed = False
    for index, field in enumerate(COMPARISON_FIELDS):
        if source[index] != target[index]:
            setattr(counters, f"{field}_changed", getattr(counters, f"{field}_changed") + 1)
            changed = True
    if not _is_true(source[-1]) and _is_true(target[-1]):
        counters.newly_correct += 1
    if _is_true(source[-1]) and not _is_true(target[-1]):
        counters.lost_correct += 1
    if changed:
        counters.any_changed_records += 1


def compare_final_records(
    *,
    configuration: str,
    source_path: str | Path,
    target_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare two final ledgers without relying on physical row order."""

    source_file = Path(source_path).resolve()
    target_file = Path(target_path).resolve()
    source = _read_record_index(source_file)
    seen_target: set[SemanticKey] = set()
    grouped: defaultdict[GroupKey, DifferenceCounters] = defaultdict(DifferenceCounters)
    summary = DifferenceCounters()

    for row in _iter_jsonl(target_file):
        key = semantic_key(row)
        if key in seen_target:
            raise ValueError(
                f"duplicate final-record semantic key in {target_file}: {key}"
            )
        seen_target.add(key)
        counters = grouped[_group_key(configuration, key)]
        source_row = source.pop(key, None)
        if source_row is None:
            counters.target_records += 1
            counters.target_only_records += 1
        else:
            _update_matched(counters, source_row, _comparable_record(row))

    for key in sorted(source):
        counters = grouped[_group_key(configuration, key)]
        counters.source_records += 1
        counters.source_only_records += 1

    rows: list[dict[str, Any]] = []
    for group, counters in sorted(grouped.items()):
        summary.add(counters)
        rows.append(
            {
                **dict(zip(GROUP_FIELDS, group)),
                **counters.as_dict(),
            }
        )
    return rows, {
        "configuration": configuration,
        **summary.as_dict(),
        "semantic_keys_equal": not source and summary.target_only_records == 0,
        "source_final_sha256": sha256_file(source_file),
        "target_final_sha256": sha256_file(target_file),
    }


def _count_jsonl(path: Path) -> tuple[int, int]:
    count = 0
    identities: set[tuple[str, str]] = set()
    for row in _iter_jsonl(path):
        count += 1
        identities.add((str(row.get("query_id", "")), str(row.get("prompt_hash", ""))))
    return count, len(identities)


def _selection_count(run_dir: Path, backends: Sequence[str]) -> int:
    return sum(
        1
        for backend in backends
        for path in (run_dir / "selections" / backend).rglob("*.json")
        if path.is_file()
    )


def _stage_status(manifest: Mapping[str, Any], stage: str) -> str:
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        return ""
    document = stages.get(stage)
    return str(document.get("status", "")) if isinstance(document, Mapping) else ""


def _check(expected: Any, observed: Any) -> dict[str, Any]:
    return {
        "passed": type(observed) is type(expected) and observed == expected,
        "expected": expected,
        "observed": observed,
    }


def validate_replay_run(
    *,
    spec: ReplayAuditSpec,
    source_run: Path,
    target_run: Path,
    difference_summary: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _load_json(target_run / "run_manifest.json")
    provenance = _load_json(target_run / "provenance" / "final_replay.json")
    plan = _load_json(target_run / "llm" / "selected_union_plan.json")
    execution_count, execution_unique_count = _count_jsonl(
        target_run / "llm" / "selected_execution.jsonl"
    )
    selection_count = _selection_count(target_run, spec.selection_backends)
    source_records = int(difference_summary["source_records"])
    target_records = int(difference_summary["target_records"])
    copied = provenance.get("copied_repaired_selections")
    copied_count = len(copied) if isinstance(copied, list) else -1
    manifest_replay = manifest.get("final_replay")
    manifest_offline = (
        manifest_replay.get("api_called_by_final_replay")
        if isinstance(manifest_replay, Mapping)
        else None
    )
    checks = {
        "source_final_record_count": _check(spec.expected_final_records, source_records),
        "target_final_record_count": _check(spec.expected_final_records, target_records),
        "semantic_key_sets_equal": _check(True, bool(difference_summary["semantic_keys_equal"])),
        "selection_slice_count": _check(spec.expected_selection_slices, selection_count),
        "copied_selection_provenance_count": _check(
            spec.expected_selection_slices, copied_count
        ),
        "selected_execution_count": _check(
            spec.expected_selected_execution, execution_count
        ),
        "selected_execution_identities_unique": _check(
            execution_count, execution_unique_count
        ),
        "manifest_status": _check("complete", str(manifest.get("status", ""))),
        "manifest_run_id": _check(spec.target_run_id, str(manifest.get("run_id", ""))),
        "manifest_api_called_by_final_replay": _check(False, manifest_offline),
        "provenance_api_called_by_final_replay": _check(
            False, provenance.get("api_called_by_final_replay")
        ),
        "provenance_run_selected_llm_stage_called": _check(
            False, provenance.get("run_selected_llm_stage_called")
        ),
        "plan_api_called_by_final_replay": _check(
            False, plan.get("api_called_by_final_replay")
        ),
        "provenance_stages_executed": _check(
            list(EXPECTED_EXECUTED_STAGES), provenance.get("stages_executed")
        ),
    }
    for stage in REQUIRED_COMPLETE_STAGES:
        checks[f"stage_{stage}"] = _check("complete", _stage_status(manifest, stage))
    return {
        "configuration": spec.configuration,
        "source_run_id": spec.source_run_id,
        "target_run_id": spec.target_run_id,
        "source_run_path": str(source_run),
        "target_run_path": str(target_run),
        "all_passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_list_row(
    spec: ReplayAuditSpec,
    summary: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    checks = validation["checks"]
    return {
        "configuration": spec.configuration,
        "source_run_id": spec.source_run_id,
        "target_run_id": spec.target_run_id,
        "selection_backends": ",".join(spec.selection_backends),
        "expected_final_records": spec.expected_final_records,
        "observed_final_records": summary["target_records"],
        "expected_selection_slices": spec.expected_selection_slices,
        "observed_selection_slices": checks["selection_slice_count"]["observed"],
        "expected_selected_execution": spec.expected_selected_execution,
        "observed_selected_execution": checks["selected_execution_count"]["observed"],
        "manifest_status": checks["manifest_status"]["observed"],
        "semantic_keys_equal": summary["semantic_keys_equal"],
        "api_called_by_final_replay": checks[
            "provenance_api_called_by_final_replay"
        ]["observed"],
        "run_selected_llm_stage_called": checks[
            "provenance_run_selected_llm_stage_called"
        ]["observed"],
        "validation_passed": validation["all_passed"],
        "source_final_sha256": summary["source_final_sha256"],
        "target_final_sha256": summary["target_final_sha256"],
    }


def execute_final_replay_audit(
    *,
    runs_root: str | Path,
    output_run_id: str,
    specs: Sequence[ReplayAuditSpec] = DEFAULT_AUDIT_SPECS,
) -> dict[str, Any]:
    """Audit completed replay runs and create one new, non-overwriting audit run."""

    runs = Path(runs_root).resolve()
    output = runs / output_run_id
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit run: {output}")
    if len({spec.configuration for spec in specs}) != len(specs):
        raise ValueError("audit configurations must be unique")

    detail_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    global_counters = DifferenceCounters()
    for spec in specs:
        source = runs / spec.source_run_id
        target = runs / spec.target_run_id
        rows, summary = compare_final_records(
            configuration=spec.configuration,
            source_path=source / "final" / "all_methods.jsonl",
            target_path=target / "final" / "all_methods.jsonl",
        )
        validation = validate_replay_run(
            spec=spec,
            source_run=source,
            target_run=target,
            difference_summary=summary,
        )
        detail_rows.extend(rows)
        summaries.append(summary)
        validations.append(validation)
        run_rows.append(_run_list_row(spec, summary, validation))
        global_counters.add(
            DifferenceCounters(
                **{field: int(summary[field]) for field in COUNTER_FIELDS}
            )
        )

    global_summary = {
        "configuration": "__all__",
        **global_counters.as_dict(),
        "semantic_keys_equal": all(bool(row["semantic_keys_equal"]) for row in summaries),
        "source_final_sha256": "",
        "target_final_sha256": "",
    }
    summary_rows = [*summaries, global_summary]
    validation_document = {
        "audit_type": "offline_final_replay_specific_validation",
        "standard_router_v3_validate_run_used": False,
        "source_and_target_runs_read_only": True,
        "api_called_by_audit": False,
        "configuration_count": len(specs),
        "all_passed": all(bool(value["all_passed"]) for value in validations),
        "runs": validations,
    }

    output.mkdir(parents=True, exist_ok=False)
    _write_csv(
        output / "final_result_difference.csv",
        detail_rows,
        (*GROUP_FIELDS, *COUNTER_FIELDS),
    )
    _write_csv(
        output / "final_result_difference_summary.csv",
        summary_rows,
        (
            "configuration",
            *COUNTER_FIELDS,
            "semantic_keys_equal",
            "source_final_sha256",
            "target_final_sha256",
        ),
    )
    _write_csv(
        output / "final_replay_runs.csv",
        run_rows,
        tuple(run_rows[0]) if run_rows else (),
    )
    write_json(output / "replay_specific_validation.json", validation_document)
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    manifest = {
        "run_id": output_run_id,
        "run_kind": "read_only_final_replay_difference_audit",
        "status": "complete",
        "source_and_target_runs_read_only": True,
        "api_called": False,
        "standard_router_v3_validate_run_used": False,
        "configuration_count": len(specs),
        "validation_passed": validation_document["all_passed"],
        "artifacts": artifacts,
    }
    write_json(output / "run_manifest.json", manifest)
    return {
        "output_run": str(output),
        "validation_passed": validation_document["all_passed"],
        "configuration_count": len(specs),
        **global_counters.as_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the five completed offline final replays to their source runs"
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument(
        "--output-run-id",
        default="no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_final_replay_index",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = execute_final_replay_audit(
        runs_root=arguments.runs_root,
        output_run_id=arguments.output_run_id,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if bool(result["validation_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPARISON_FIELDS",
    "DEFAULT_AUDIT_SPECS",
    "ReplayAuditSpec",
    "SEMANTIC_KEY_FIELDS",
    "compare_final_records",
    "execute_final_replay_audit",
    "semantic_key",
    "validate_replay_run",
]
