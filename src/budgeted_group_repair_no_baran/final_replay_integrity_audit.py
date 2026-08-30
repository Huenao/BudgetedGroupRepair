"""Additive, offline integrity audit for the five final-replay runs.

The audit is deliberately separate from both the historical runs and the
already-created replay runs.  This module contains no provider client and its
only writes are to a caller-selected, previously non-existent audit directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FINAL_KEY_FIELDS = (
    "suite",
    "dataset",
    "method",
    "scenario",
    "backend",
    "budget_share",
    "group_size_variant",
    "cell_id",
)
FINAL_OUTCOME_FIELDS = (
    "prediction",
    "parse_status",
    "final_source",
    "accepted_llm",
    "correct_repair",
)
RESPONSE_CONTENT_FIELDS = (
    "query_id",
    "prompt_hash",
    "provider_request_hash",
    "model",
    "model_requested",
    "model_returned",
    "model_matches_request",
    "cell_ids",
    "status",
    "parse_status",
    "retryable",
    "response_text",
    "items",
    "missing_cell_ids",
    "unknown_cell_ids",
    "duplicate_cell_ids",
    "invalid_items",
)
RESPONSE_METADATA_CONTENT_FIELDS = ("phase", "prompt_schema_version")
MICRO_KEY_FIELDS = (
    "method",
    "scenario",
    "backend",
    "budget_share",
    "group_size_variant",
)
MICRO_COUNT_FIELDS = (
    "true_error_cells",
    "predicted_repairs",
    "valid_predictions",
    "invalid_predictions",
    "correct_repairs",
    "annotation_mismatches",
)
MICRO_RATE_FIELDS = ("correction_accuracy", "precision", "recall", "f1")


@dataclass(frozen=True)
class IntegrityAuditSpec:
    configuration: str
    source_run_id: str
    target_run_id: str
    selection_backends: tuple[str, ...]


DEFAULT_AUDIT_SPECS = (
    IntegrityAuditSpec(
        "base",
        "no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_base_final",
        ("lightgbm", "xgboost"),
    ),
    IntegrityAuditSpec(
        "lightgbm_sweep",
        "no_baran_router_v3_deepseek_v4_20260726_budget_sweep_k24_lightgbm",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_lightgbm_sweep_final",
        ("lightgbm",),
    ),
    IntegrityAuditSpec(
        "catboost",
        "no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_catboost_final",
        ("catboost",),
    ),
    IntegrityAuditSpec(
        "tabiclv2",
        "no_baran_router_v3_tabiclv2_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_tabiclv2_final",
        ("tabiclv2",),
    ),
    IntegrityAuditSpec(
        "tabpfn3",
        "no_baran_router_v3_tabpfn3_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_tabpfn3_final",
        ("tabpfn3",),
    ),
)

DEFAULT_CACHE_AUDIT_RUN_ID = (
    "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_cache_union"
)
DEFAULT_RETRY_RUN_ID = (
    "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_retry"
)
DEFAULT_OUTPUT_RUN_ID = (
    "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_final_replay_integrity_audit"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            yield value


def response_content_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return response semantics while excluding replay/cost bookkeeping.

    In particular, ``cache_hit``, ``checkpoint_hit``, latency, usage, attempts,
    and ``metadata.final_replay_*`` do not change the repair semantics.  Parsed
    items, status, response text, model response identity, and parse diagnostics
    do, so they are all committed here.
    """

    projection = {
        field: row[field]
        for field in RESPONSE_CONTENT_FIELDS
        if field in row
    }
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        semantic_metadata = {
            field: metadata[field]
            for field in RESPONSE_METADATA_CONTENT_FIELDS
            if field in metadata
        }
        if semantic_metadata:
            projection["metadata"] = semantic_metadata
    return projection


def response_content_projection_sha256(row: Mapping[str, Any]) -> str:
    """Hash the semantic response projection of one checkpoint row."""

    return _canonical_sha256(response_content_projection(row))


def _typed_final_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    missing = [field for field in FINAL_KEY_FIELDS if field not in row]
    if missing:
        raise ValueError(f"final record is missing semantic-key fields: {missing}")
    return tuple(row[field] for field in FINAL_KEY_FIELDS)


def _final_outcome(row: Mapping[str, Any]) -> tuple[tuple[bool, Any], ...]:
    return tuple((field in row, row.get(field)) for field in FINAL_OUTCOME_FIELDS)


def _index_final(path: str | Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        key = _typed_final_key(row)
        if key in result:
            raise ValueError(f"duplicate final semantic key in {path}: {key!r}")
        result[key] = row
    return result


def _budget_percent_token(value: Any) -> str:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"invalid budget_share: {value!r}")
    percent = number * 100.0
    if math.isclose(percent, round(percent), abs_tol=1e-9):
        # The experiment's canonical path contract zero-pads sub-10% budgets
        # (``01pct`` and ``05pct``) while leaving 10/20/50 unchanged.
        return f"{int(round(percent)):02d}pct"
    return f"{format(percent, '.12g')}pct"


def final_record_slice(row: Mapping[str, Any]) -> str:
    """Map one BGR final record to the canonical selection-slice path."""

    return "/".join(
        (
            str(row.get("backend", "")),
            str(row.get("scenario", "")),
            f"variant_{row.get('group_size_variant', '')}",
            _budget_percent_token(row.get("budget_share")),
            f"{row.get('suite', '')}__{row.get('dataset', '')}",
        )
    )


def _changed_fields(source: Mapping[str, Any], target: Mapping[str, Any]) -> list[str]:
    return [
        field
        for field in FINAL_OUTCOME_FIELDS
        if (field in source, source.get(field)) != (field in target, target.get(field))
    ]


def audit_final_causal_scope(
    *,
    configuration: str,
    source_path: str | Path,
    target_path: str | Path,
    query_causes: Mapping[str, Sequence[str]],
    query_cell_ids: Mapping[str, set[str]],
    llm_only_query_by_cell: Mapping[str, str],
    selected_queries_by_slice: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Require every changed final semantic key to have a changed-query cause.

    LLM-only rows are authorized only by their singleton query.  A BGR row is
    authorized only when a changed query covers the cell *and* was selected in
    that exact backend/scenario/variant/budget/dataset slice.  Merely sharing a
    cell with a changed query in another slice is insufficient.
    """

    source = _index_final(source_path)
    target = _index_final(target_path)
    source_keys = set(source)
    target_keys = set(target)
    violations: list[dict[str, Any]] = []
    changed_records: list[dict[str, Any]] = []
    authorized = 0

    for key in sorted(source_keys & target_keys, key=repr):
        old = source[key]
        new = target[key]
        fields = _changed_fields(old, new)
        if not fields:
            continue
        method = str(new.get("method", ""))
        cell_id = str(new.get("cell_id", ""))
        candidates: list[str] = []
        if method == "llm_only":
            query_id = llm_only_query_by_cell.get(cell_id)
            if query_id:
                candidates = [query_id]
        elif method.startswith("budgeted_group_"):
            selected = selected_queries_by_slice.get(final_record_slice(new), set())
            candidates = sorted(
                query_id
                for query_id in selected
                if cell_id in query_cell_ids.get(query_id, set())
            )

        causal_queries = sorted(
            query_id for query_id in candidates if query_causes.get(query_id)
        )
        typed = dict(zip(FINAL_KEY_FIELDS, key))
        record = {
            "configuration": configuration,
            **typed,
            "changed_fields": fields,
            "causal_query_ids": causal_queries,
            "causal_roles": sorted(
                {
                    role
                    for query_id in causal_queries
                    for role in query_causes.get(query_id, ())
                }
            ),
        }
        changed_records.append(record)
        if causal_queries:
            authorized += 1
        else:
            violations.append(
                {
                    **record,
                    "reason": "no_cache_or_supplemental_authority",
                }
            )

    for key in sorted(source_keys ^ target_keys, key=repr):
        row = source.get(key, target.get(key, {}))
        violations.append(
            {
                "configuration": configuration,
                **dict(zip(FINAL_KEY_FIELDS, key)),
                "changed_fields": ["semantic_key_presence"],
                "causal_query_ids": [],
                "causal_roles": [],
                "reason": "source_target_semantic_key_set_drift",
            }
        )

    return {
        "configuration": configuration,
        "all_passed": not violations,
        "source_semantic_key_count": len(source_keys),
        "target_semantic_key_count": len(target_keys),
        "semantic_key_sets_equal": source_keys == target_keys,
        "changed_final_semantic_keys": len(changed_records),
        "authorized_changed_records": authorized,
        "unauthorized_changed_records": len(violations),
        "unsupported_changed_records": len(violations),
        "changed_records": changed_records,
        "violations": violations,
    }


def _normalise_for_match(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"[\t\n ]+", " ", html.unescape(text)).strip("\t\n ")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _micro_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    budget = row.get("budget_share")
    if budget in {None, ""}:
        budget = None
    else:
        budget = float(budget)
    return (
        str(row.get("method", "")),
        str(row.get("scenario") or "main"),
        str(row.get("backend") or "none"),
        budget,
        str(row.get("group_size_variant") or "all"),
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def recompute_micro_metrics(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """Independently recompute core micro counts and rates from final records."""

    groups: defaultdict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[_micro_key(row)].append(row)
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        total = len(rows)
        valid = [str(row.get("parse_status") or "").startswith("ok") for row in rows]
        correct = [
            is_valid
            and row.get("prediction") is not None
            and "clean_value" in row
            and _normalise_for_match(row.get("prediction"))
            == _normalise_for_match(row.get("clean_value"))
            for row, is_valid in zip(rows, valid)
        ]
        predicted = sum(valid)
        correct_count = sum(correct)
        mismatches = sum(
            (
                "correct_repair" in row
                and _coerce_bool(row.get("correct_repair")) != expected_correct
            )
            or (
                "valid_prediction" in row
                and _coerce_bool(row.get("valid_prediction")) != expected_valid
            )
            for row, expected_valid, expected_correct in zip(rows, valid, correct)
        )
        precision = _safe_rate(correct_count, predicted)
        recall = _safe_rate(correct_count, total)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        result[key] = {
            "scope": "micro",
            **dict(zip(MICRO_KEY_FIELDS, key)),
            "suite": "ALL",
            "dataset": "MICRO",
            "true_error_cells": total,
            "predicted_repairs": predicted,
            "valid_predictions": predicted,
            "invalid_predictions": total - predicted,
            "correct_repairs": correct_count,
            "correction_accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "annotation_mismatches": mismatches,
        }
    return result


def _csv_micro_key(row: Mapping[str, str]) -> tuple[Any, ...]:
    return (
        str(row.get("method", "")),
        str(row.get("scenario", "")),
        str(row.get("backend", "")),
        None if row.get("budget_share", "") == "" else float(row["budget_share"]),
        str(row.get("group_size_variant", "")),
    )


def compare_reported_micro_metrics(
    recomputed: Mapping[tuple[Any, ...], Mapping[str, Any]],
    reported_csv: str | Path,
) -> dict[str, Any]:
    """Compare independent core micro metrics with a reported method CSV."""

    with Path(reported_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("scope") == "micro"]
    reported: dict[tuple[Any, ...], dict[str, str]] = {}
    duplicates: list[list[Any]] = []
    for row in rows:
        key = _csv_micro_key(row)
        if key in reported:
            duplicates.append(list(key))
        reported[key] = row
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(recomputed) | set(reported), key=repr):
        expected = recomputed.get(key)
        observed = reported.get(key)
        if expected is None or observed is None:
            mismatches.append(
                {
                    **dict(zip(MICRO_KEY_FIELDS, key)),
                    "field": "row_presence",
                    "expected": expected is not None,
                    "observed": observed is not None,
                }
            )
            continue
        for field in MICRO_COUNT_FIELDS:
            actual = int(observed[field])
            wanted = int(expected[field])
            if actual != wanted:
                mismatches.append(
                    {
                        **dict(zip(MICRO_KEY_FIELDS, key)),
                        "field": field,
                        "expected": wanted,
                        "observed": actual,
                    }
                )
        for field in MICRO_RATE_FIELDS:
            actual = float(observed[field])
            wanted = float(expected[field])
            if not math.isclose(actual, wanted, rel_tol=1e-12, abs_tol=1e-12):
                mismatches.append(
                    {
                        **dict(zip(MICRO_KEY_FIELDS, key)),
                        "field": field,
                        "expected": wanted,
                        "observed": actual,
                    }
                )
    return {
        "all_passed": not mismatches and not duplicates,
        "recomputed_row_count": len(recomputed),
        "reported_row_count": len(reported),
        "duplicate_keys": duplicates,
        "mismatches": mismatches,
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {source}")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(run_dir: Path) -> Path:
    for relative in (
        "llm/shared/group_query_checkpoint.jsonl",
        "llm/group_query_checkpoint.jsonl",
    ):
        candidate = run_dir / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no group-query checkpoint in {run_dir}")


def _safe_run_dir(runs_root: Path, run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError(f"unsafe run ID: {run_id!r}")
    result = (runs_root / run_id).resolve()
    if result.parent != runs_root.resolve():
        raise ValueError(f"run escapes runs root: {run_id!r}")
    return result


def _canonical_row_digest(rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    """Commit an ordered sequence through each row's complete canonical JSON."""

    row_hashes: list[str] = []
    for row in rows:
        row_hashes.append(_canonical_sha256(dict(row)))
    return _canonical_sha256(row_hashes), len(row_hashes)


def _canonical_content_multiset(rows: Iterable[Mapping[str, Any]]) -> tuple[str, int]:
    hashes = sorted(response_content_projection_sha256(row) for row in rows)
    return _canonical_sha256(hashes), len(hashes)


def _role(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    return (
        str(metadata.get("final_replay_role", ""))
        if isinstance(metadata, Mapping)
        else ""
    )


def _phase(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    return str(metadata.get("phase", "")) if isinstance(metadata, Mapping) else ""


def _explicit_five_field_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = row.get("metadata")
    schema = (
        str(metadata.get("prompt_schema_version", ""))
        if isinstance(metadata, Mapping)
        else ""
    ) or str(row.get("prompt_schema_version", ""))
    identity = (
        str(row.get("query_id", "")),
        str(row.get("prompt_hash", "")),
        str(row.get("provider_request_hash", "")),
        str(row.get("model", row.get("model_requested", ""))),
        schema,
    )
    if not all(identity):
        raise ValueError("row lacks an explicit five-field request identity")
    return identity


def _execution_index(path: str | Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in result:
            raise ValueError(f"invalid or duplicate execution query ID in {path}")
        result[query_id] = row
    return result


def _four_field_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("query_id", "")),
        str(row.get("prompt_hash", "")),
        str(row.get("provider_request_hash", "")),
        str(row.get("model", row.get("model_requested", ""))),
    )


def compare_selected_execution(
    source_path: str | Path,
    target_path: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[str, ...]],
    dict[str, set[str]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Find semantic response changes and classify their target replay role."""

    source = _execution_index(source_path)
    target = _execution_index(target_path)
    source_ids = set(source)
    target_ids = set(target)
    query_causes: dict[str, tuple[str, ...]] = {}
    query_cells = {
        query_id: {str(value) for value in row.get("cell_ids", [])}
        for query_id, row in target.items()
    }
    changed: list[dict[str, Any]] = []
    invalid_roles: list[dict[str, str]] = []
    identity_drifts: list[str] = []
    role_counts: defaultdict[str, int] = defaultdict(int)
    role_to_cause = {
        "deterministic_response_bank": "A_cache_union",
        "supplemental_missing_query_last_row": "B_supplemental",
    }
    for query_id in sorted(source_ids & target_ids):
        old = source[query_id]
        new = target[query_id]
        if _four_field_identity(old) != _four_field_identity(new):
            identity_drifts.append(query_id)
        old_projection = response_content_projection(old)
        new_projection = response_content_projection(new)
        if old_projection == new_projection:
            continue
        changed_fields = sorted(
            field
            for field in set(old_projection) | set(new_projection)
            if old_projection.get(field) != new_projection.get(field)
        )
        target_role = _role(new)
        cause = role_to_cause.get(target_role)
        if cause is not None:
            query_causes[query_id] = (cause,)
            role_counts[cause] += 1
        else:
            invalid_roles.append({"query_id": query_id, "target_role": target_role})
        changed.append(
            {
                "query_id": query_id,
                "cause": cause or "unsupported_role",
                "target_role": target_role,
                "source_status": str(old.get("status", "")),
                "target_status": str(new.get("status", "")),
                "source_parse_status": str(old.get("parse_status", "")),
                "target_parse_status": str(new.get("parse_status", "")),
                "changed_fields": changed_fields,
                "cell_count": len(query_cells[query_id]),
            }
        )
    document = {
        "all_passed": (
            source_ids == target_ids and not identity_drifts and not invalid_roles
        ),
        "source_query_count": len(source_ids),
        "target_query_count": len(target_ids),
        "query_id_sets_equal": source_ids == target_ids,
        "source_only_query_count": len(source_ids - target_ids),
        "target_only_query_count": len(target_ids - source_ids),
        "four_field_identity_drift_count": len(identity_drifts),
        "four_field_identity_drift_query_ids": identity_drifts,
        "changed_execution_query_count": len(changed),
        "changed_query_counts_by_cause": dict(sorted(role_counts.items())),
        "unsupported_changed_role_count": len(invalid_roles),
        "unsupported_changed_roles": invalid_roles,
    }
    return document, query_causes, query_cells, target, changed


def load_selection_slices(
    target_run: str | Path,
    backends: Sequence[str],
) -> tuple[dict[str, set[str]], list[str]]:
    run = Path(target_run)
    selected: dict[str, set[str]] = {}
    errors: list[str] = []
    selection_root = run / "selections"
    for backend in backends:
        backend_root = selection_root / backend
        for path in sorted(backend_root.rglob("*.json")):
            relative = path.relative_to(selection_root).with_suffix("").as_posix()
            document = _load_json(path)
            expected = "/".join(
                (
                    str(document.get("backend", "")),
                    str(document.get("scenario", "")),
                    f"variant_{document.get('group_size_variant', '')}",
                    _budget_percent_token(document.get("budget_share")),
                    f"{document.get('suite', '')}__{document.get('dataset', '')}",
                )
            )
            if relative != expected or str(document.get("backend", "")) != backend:
                errors.append(relative)
            query_ids = document.get(
                "selected_query_ids", document.get("selected", [])
            )
            if relative in selected:
                errors.append(f"duplicate:{relative}")
            selected[relative] = {str(value) for value in query_ids}
    return selected, errors


def build_llm_only_cell_map(
    plan: Mapping[str, Any],
    execution: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], list[str]]:
    result: dict[str, str] = {}
    errors: list[str] = []
    for query_id in sorted(str(value) for value in plan.get("llm_only_query_ids", [])):
        row = execution.get(query_id)
        if row is None:
            errors.append(f"missing:{query_id}")
            continue
        cells = [str(value) for value in row.get("cell_ids", [])]
        if len(cells) != 1:
            errors.append(f"not_singleton:{query_id}")
            continue
        previous = result.get(cells[0])
        if previous is not None and previous != query_id:
            errors.append(f"duplicate_cell:{cells[0]}")
            continue
        result[cells[0]] = query_id
    return result, errors


def build_cache_bank_binding(
    *, runs_root: Path, cache_audit_run: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Bind five source checkpoints and rebuild the 32,777-row response bank."""

    from .final_replay import (
        DEFAULT_CACHE_SOURCE_RUN_IDS,
        build_deterministic_response_bank,
        load_response_authority,
    )

    source_audit_path = cache_audit_run / "cache_source_audit.json"
    source_audit = _load_json(source_audit_path)
    recorded_rows = source_audit.get("checkpoint_sources", [])
    if not isinstance(recorded_rows, list):
        raise ValueError("cache source audit lacks checkpoint_sources")
    recorded = {
        str(row.get("source_run_id", "")): row
        for row in recorded_rows
        if isinstance(row, Mapping)
    }
    sources: list[tuple[str, Path]] = []
    source_checks: list[dict[str, Any]] = []
    for run_id in DEFAULT_CACHE_SOURCE_RUN_IDS:
        checkpoint = _checkpoint_path(_safe_run_dir(runs_root, run_id))
        digest = _sha256_file(checkpoint)
        row_count = sum(1 for _ in _iter_jsonl(checkpoint))
        declared = recorded.get(run_id, {})
        passed = (
            digest == str(declared.get("checkpoint_sha256", ""))
            and row_count == int(declared.get("row_count", -1))
        )
        source_checks.append(
            {
                "source_run_id": run_id,
                "checkpoint_relative_path": checkpoint.relative_to(
                    _safe_run_dir(runs_root, run_id)
                ).as_posix(),
                "checkpoint_sha256": digest,
                "recorded_checkpoint_sha256": declared.get("checkpoint_sha256"),
                "row_count": row_count,
                "recorded_row_count": declared.get("row_count"),
                "passed": passed,
            }
        )
        sources.append((run_id, checkpoint))

    authority_path = cache_audit_run / "cache_union_audit.csv"
    authority = load_response_authority(authority_path)
    bank, bank_audit = build_deterministic_response_bank(authority, sources)
    bank_digest, bank_count = _canonical_row_digest(bank)
    explicit_schema_count = sum(
        isinstance(row.get("metadata"), Mapping)
        and "prompt_schema_version" in row["metadata"]
        for row in bank
    )
    expected_schemas = {
        str(row.get("prompt_schema_version", "")) for row in authority
    }
    identity_document = {
        "historical_cache_identity_contract": (
            "4_observed_plus_1_authority_bound_expected"
        ),
        "observed_fields": [
            "query_id",
            "prompt_hash",
            "provider_request_hash",
            "model",
        ],
        "authority_bound_expected_fields": ["prompt_schema_version"],
        "prompt_schema_version_expected": (
            next(iter(expected_schemas)) if len(expected_schemas) == 1 else None
        ),
        "historical_rows_with_explicit_schema": explicit_schema_count,
        "historical_rows_total": bank_count,
    }
    document = {
        "all_passed": (
            len(recorded) == 5
            and all(row["passed"] for row in source_checks)
            and bank_count == 32_777
            and len(expected_schemas) == 1
            and explicit_schema_count == 0
        ),
        "cache_source_audit_sha256": _sha256_file(source_audit_path),
        "cache_union_authority_sha256": _sha256_file(authority_path),
        "source_checkpoint_count": len(source_checks),
        "source_checkpoints": source_checks,
        "response_bank_count": bank_count,
        "response_bank_full_canonical_row_digest": bank_digest,
        "response_bank_identity_sha256": _canonical_sha256(bank_audit),
        "identity_evidence": identity_document,
    }
    return document, bank, bank_audit


def build_retry_binding(
    *, cache_audit_run: Path, retry_run: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Bind retry inputs, explicit identities, outputs, and response content."""

    manifest_path = retry_run / "run_manifest.json"
    summary_path = retry_run / "execution_summary.json"
    authority_path = cache_audit_run / "missing_query_union_enriched.jsonl"
    base_authority_path = cache_audit_run / "missing_query_union.jsonl"
    authorized_path = retry_run / "authorized_requests.jsonl"
    checkpoint_path = _checkpoint_path(retry_run)
    selected_path = retry_run / "llm" / "selected_execution.jsonl"
    config_path = retry_run / "bound_llm_config.json"
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)
    authority = list(_iter_jsonl(authority_path))
    authorized = list(_iter_jsonl(authorized_path))
    checkpoint = list(_iter_jsonl(checkpoint_path))
    selected = list(_iter_jsonl(selected_path))
    identities = {
        "authority": {_explicit_five_field_identity(row) for row in authority},
        "authorized_requests": {
            _explicit_five_field_identity(row) for row in authorized
        },
        "checkpoint": {_explicit_five_field_identity(row) for row in checkpoint},
        "selected_execution": {
            _explicit_five_field_identity(row) for row in selected
        },
    }
    identity_sets_equal = len({frozenset(value) for value in identities.values()}) == 1
    checkpoint_multiset, _ = _canonical_content_multiset(checkpoint)
    selected_multiset, _ = _canonical_content_multiset(selected)
    supplemental_digest, supplemental_count = _canonical_content_multiset(checkpoint)
    success_count = sum(str(row.get("status", "")) == "success" for row in checkpoint)
    failure_count = len(checkpoint) - success_count
    hash_checks = {
        "authority": (
            _sha256_file(authority_path), str(manifest.get("authority_sha256", ""))
        ),
        "base_authority": (
            _sha256_file(base_authority_path),
            str(manifest.get("base_authority_sha256", "")),
        ),
        "bound_llm_config": (
            _sha256_file(config_path), str(manifest.get("llm_config_sha256", ""))
        ),
    }
    artifact_hashes = {
        "run_manifest.json": _sha256_file(manifest_path),
        "execution_summary.json": _sha256_file(summary_path),
        "authorized_requests.jsonl": _sha256_file(authorized_path),
        "bound_llm_config.json": _sha256_file(config_path),
        "llm/group_query_checkpoint.jsonl": _sha256_file(checkpoint_path),
        "llm/selected_execution.jsonl": _sha256_file(selected_path),
    }
    passed = (
        str(manifest.get("status", "")) == "complete"
        and int(manifest.get("request_count", -1)) == 35
        and int(manifest.get("result_count", -1)) == 35
        and len(authority) == len(authorized) == len(checkpoint) == len(selected) == 35
        and identity_sets_equal
        and checkpoint_multiset == selected_multiset
        and success_count == 30
        and failure_count == 5
        and all(current == recorded for current, recorded in hash_checks.values())
        and int(summary.get("result_count", -1)) == 35
    )
    document = {
        "all_passed": passed,
        "retry_identity_contract": "5_fields_observed",
        "retry_rows_with_explicit_schema": len(checkpoint),
        "identity_sets_equal": identity_sets_equal,
        "identity_set_counts": {
            name: len(value) for name, value in identities.items()
        },
        "result_count": len(checkpoint),
        "success_count": success_count,
        "failure_count": failure_count,
        "checkpoint_selected_content_multiset_equal": (
            checkpoint_multiset == selected_multiset
        ),
        "checkpoint_content_multiset_sha256": checkpoint_multiset,
        "selected_content_multiset_sha256": selected_multiset,
        "supplemental_content_multiset_sha256": supplemental_digest,
        "input_hash_checks": {
            name: {
                "current": current,
                "recorded": recorded,
                "passed": current == recorded,
            }
            for name, (current, recorded) in hash_checks.items()
        },
        "artifact_sha256": artifact_hashes,
    }
    return document, checkpoint, supplemental_digest


def _selection_binding(
    *,
    runs_root: Path,
    source_run: Path,
    target_run: Path,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    from .final_replay import DEFAULT_RESELECTION_RUNS

    records = provenance.get("copied_repaired_selections", [])
    if not isinstance(records, list):
        records = []
    checks: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        relative = Path(str(record.get("relative_path", "")))
        parts = relative.parts
        backend = parts[1] if len(parts) > 1 and parts[0] == "selections" else ""
        reselect_id = DEFAULT_RESELECTION_RUNS.get(backend, "")
        source = source_run / relative
        copied = target_run / relative
        repaired = (
            _safe_run_dir(runs_root, reselect_id) / relative
            if reselect_id
            else Path("/__missing_reselection__")
        )
        current = {
            "source_selection_sha256": _sha256_file(source) if source.is_file() else "",
            "repaired_selection_sha256": (
                _sha256_file(repaired) if repaired.is_file() else ""
            ),
            "copied_selection_sha256": _sha256_file(copied) if copied.is_file() else "",
        }
        passed = all(
            current[field] == str(record.get(field, "")) for field in current
        )
        checks.append(
            {
                "relative_path": relative.as_posix(),
                "backend": backend,
                "passed": passed,
                **current,
            }
        )
    tree_digest = _canonical_sha256(
        [
            (row["relative_path"], row["copied_selection_sha256"])
            for row in sorted(checks, key=lambda value: value["relative_path"])
        ]
    )
    return {
        "all_passed": bool(checks) and all(row["passed"] for row in checks),
        "selection_count": len(checks),
        "selection_tree_sha256": tree_digest,
        "mismatch_count": sum(not row["passed"] for row in checks),
        "mismatches": [row for row in checks if not row["passed"]],
    }


def validate_final_binding(
    *,
    runs_root: Path,
    spec: IntegrityAuditSpec,
    expected_bank: Sequence[Mapping[str, Any]],
    expected_bank_audit: Sequence[Mapping[str, Any]],
    expected_supplemental_digest: str,
    cache_audit_run: Path,
    retry_run: Path,
) -> dict[str, Any]:
    source_run = _safe_run_dir(runs_root, spec.source_run_id)
    target_run = _safe_run_dir(runs_root, spec.target_run_id)
    manifest_path = target_run / "run_manifest.json"
    provenance_path = target_run / "provenance" / "final_replay.json"
    manifest = _load_json(manifest_path)
    provenance = _load_json(provenance_path)
    checkpoint_path = _checkpoint_path(target_run)
    selected_path = target_run / "llm" / "selected_execution.jsonl"
    fallback_path = target_run / "llm" / "online_selected_union_baran_fallbacks.jsonl"
    plan = _load_json(target_run / "llm" / "selected_union_plan.json")
    bank_rows: list[dict[str, Any]] = []
    supplemental_rows: list[dict[str, Any]] = []
    role_counts: defaultdict[str, int] = defaultdict(int)
    for row in _iter_jsonl(checkpoint_path):
        role = _role(row)
        role_counts[role] += 1
        if role == "deterministic_response_bank":
            bank_rows.append(row)
        elif role == "supplemental_missing_query_last_row":
            supplemental_rows.append(row)
    expected_bank_digest, expected_bank_count = _canonical_row_digest(expected_bank)
    target_bank_digest, target_bank_count = _canonical_row_digest(bank_rows)
    target_supplemental_digest, target_supplemental_count = (
        _canonical_content_multiset(supplemental_rows)
    )
    selection = _selection_binding(
        runs_root=runs_root,
        source_run=source_run,
        target_run=target_run,
        provenance=provenance,
    )
    hash_checks = {
        "source_run_manifest": (
            _sha256_file(source_run / "run_manifest.json"),
            str(provenance.get("source_run_manifest_sha256", "")),
        ),
        "cache_union_authority": (
            _sha256_file(cache_audit_run / "cache_union_audit.csv"),
            str(provenance.get("cache_union_authority_sha256", "")),
        ),
        "retry_manifest": (
            _sha256_file(retry_run / "run_manifest.json"),
            str(provenance.get("retry_manifest_sha256", "")),
        ),
        "checkpoint": (
            _sha256_file(checkpoint_path),
            str(provenance.get("checkpoint_sha256", "")),
        ),
        "selected_execution": (
            _sha256_file(selected_path),
            str(provenance.get("selected_execution_sha256", "")),
        ),
        "fallback": (
            _sha256_file(fallback_path),
            str(provenance.get("fallback_sha256", "")),
        ),
    }
    stages = manifest.get("stages", {})
    stages_complete = all(
        isinstance(stages.get(name), Mapping)
        and stages[name].get("status") == "complete"
        for name in ("offline_final_replay_inputs", "final_records", "metrics", "audit")
    )
    bank_identity_digest = _canonical_sha256(list(expected_bank_audit))
    checks_passed = (
        str(manifest.get("status", "")) == "complete"
        and stages_complete
        and manifest.get("final_replay") == provenance
        and provenance.get("api_called_by_final_replay") is False
        and provenance.get("run_selected_llm_stage_called") is False
        and plan.get("api_called_by_final_replay") is False
        and all(current == recorded for current, recorded in hash_checks.values())
        and target_bank_count == expected_bank_count == 32_777
        and target_bank_digest == expected_bank_digest
        and target_supplemental_count == 35
        and target_supplemental_digest == expected_supplemental_digest
        and int(provenance.get("response_bank_count", -1)) == 32_777
        and str(provenance.get("response_bank_identity_sha256", ""))
        == bank_identity_digest
        and selection["all_passed"]
    )
    return {
        "configuration": spec.configuration,
        "all_passed": checks_passed,
        "manifest_status": manifest.get("status"),
        "required_stages_complete": stages_complete,
        "manifest_embedded_provenance_equal": manifest.get("final_replay") == provenance,
        "api_called_by_final_replay": provenance.get("api_called_by_final_replay"),
        "run_selected_llm_stage_called": provenance.get(
            "run_selected_llm_stage_called"
        ),
        "hash_checks": {
            name: {
                "current": current,
                "recorded": recorded,
                "passed": current == recorded,
            }
            for name, (current, recorded) in hash_checks.items()
        },
        "checkpoint_role_counts": dict(sorted(role_counts.items())),
        "response_bank_count": target_bank_count,
        "response_bank_full_canonical_row_digest": target_bank_digest,
        "response_bank_matches_rebuilt": target_bank_digest == expected_bank_digest,
        "response_bank_identity_sha256": bank_identity_digest,
        "supplemental_count": target_supplemental_count,
        "supplemental_content_multiset_sha256": target_supplemental_digest,
        "supplemental_matches_retry": (
            target_supplemental_digest == expected_supplemental_digest
        ),
        "selection_binding": selection,
        "final_all_methods_sha256": _sha256_file(
            target_run / "final" / "all_methods.jsonl"
        ),
        "method_metrics_sha256": _sha256_file(
            target_run / "metrics" / "method_metrics.csv"
        ),
    }


def metric_delta_rows(
    *,
    configuration: str,
    source: Mapping[tuple[Any, ...], Mapping[str, Any]],
    target: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(source) | set(target), key=repr):
        old = source.get(key)
        new = target.get(key)
        row: dict[str, Any] = {
            "configuration": configuration,
            **dict(zip(MICRO_KEY_FIELDS, key)),
            "source_present": old is not None,
            "target_present": new is not None,
        }
        for field in (*MICRO_COUNT_FIELDS, *MICRO_RATE_FIELDS):
            source_value = old.get(field) if old is not None else None
            target_value = new.get(field) if new is not None else None
            row[f"source_{field}"] = source_value
            row[f"target_{field}"] = target_value
            row[f"delta_{field}"] = (
                float(target_value) - float(source_value)
                if source_value is not None and target_value is not None
                else None
            )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        "|".join(str(value) for value in value)
                        if isinstance(value, (list, tuple, set))
                        else value
                    )
                    for field, value in row.items()
                }
            )


def _write_json_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def execute_integrity_audit(
    *,
    runs_root: str | Path,
    output_run_id: str,
    cache_audit_run_id: str = DEFAULT_CACHE_AUDIT_RUN_ID,
    retry_run_id: str = DEFAULT_RETRY_RUN_ID,
    specs: Sequence[IntegrityAuditSpec] = DEFAULT_AUDIT_SPECS,
    expected_changed_final_keys: int | None = 985,
) -> dict[str, Any]:
    """Run the additive causal/binding audit and create one new audit run."""

    runs = Path(runs_root).resolve()
    output = _safe_run_dir(runs, output_run_id)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite integrity audit run: {output}")
    cache_audit = _safe_run_dir(runs, cache_audit_run_id)
    retry_run = _safe_run_dir(runs, retry_run_id)
    cache_binding, bank, bank_audit = build_cache_bank_binding(
        runs_root=runs, cache_audit_run=cache_audit
    )
    retry_binding, _, supplemental_digest = build_retry_binding(
        cache_audit_run=cache_audit, retry_run=retry_run
    )

    run_documents: list[dict[str, Any]] = []
    changed_query_rows: list[dict[str, Any]] = []
    changed_final_rows: list[dict[str, Any]] = []
    metric_validation_rows: list[dict[str, Any]] = []
    recomputed_metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    total_changed = 0
    total_unexplained = 0
    for spec in specs:
        source_run = _safe_run_dir(runs, spec.source_run_id)
        target_run = _safe_run_dir(runs, spec.target_run_id)
        binding = validate_final_binding(
            runs_root=runs,
            spec=spec,
            expected_bank=bank,
            expected_bank_audit=bank_audit,
            expected_supplemental_digest=supplemental_digest,
            cache_audit_run=cache_audit,
            retry_run=retry_run,
        )
        execution_doc, causes, query_cells, target_execution, query_rows = (
            compare_selected_execution(
                source_run / "llm" / "selected_execution.jsonl",
                target_run / "llm" / "selected_execution.jsonl",
            )
        )
        for row in query_rows:
            changed_query_rows.append({"configuration": spec.configuration, **row})
        selections, selection_errors = load_selection_slices(
            target_run, spec.selection_backends
        )
        plan = _load_json(target_run / "llm" / "selected_union_plan.json")
        llm_only, llm_only_errors = build_llm_only_cell_map(plan, target_execution)
        causal = audit_final_causal_scope(
            configuration=spec.configuration,
            source_path=source_run / "final" / "all_methods.jsonl",
            target_path=target_run / "final" / "all_methods.jsonl",
            query_causes=causes,
            query_cell_ids=query_cells,
            llm_only_query_by_cell=llm_only,
            selected_queries_by_slice=selections,
        )
        changed_final_rows.extend(causal.pop("changed_records"))
        total_changed += int(causal["changed_final_semantic_keys"])
        total_unexplained += int(causal["unauthorized_changed_records"])

        target_metrics = recompute_micro_metrics(
            _iter_jsonl(target_run / "final" / "all_methods.jsonl")
        )
        source_metrics = recompute_micro_metrics(
            _iter_jsonl(source_run / "final" / "all_methods.jsonl")
        )
        metric_validation = compare_reported_micro_metrics(
            target_metrics, target_run / "metrics" / "method_metrics.csv"
        )
        metric_validation_rows.append(
            {
                "configuration": spec.configuration,
                "all_passed": metric_validation["all_passed"],
                "recomputed_row_count": metric_validation["recomputed_row_count"],
                "reported_row_count": metric_validation["reported_row_count"],
                "mismatch_count": len(metric_validation["mismatches"]),
                "duplicate_key_count": len(metric_validation["duplicate_keys"]),
            }
        )
        for row in target_metrics.values():
            recomputed_metric_rows.append({"configuration": spec.configuration, **row})
        delta_rows.extend(
            metric_delta_rows(
                configuration=spec.configuration,
                source=source_metrics,
                target=target_metrics,
            )
        )
        all_passed = (
            binding["all_passed"]
            and execution_doc["all_passed"]
            and not selection_errors
            and not llm_only_errors
            and causal["all_passed"]
            and metric_validation["all_passed"]
        )
        run_documents.append(
            {
                "configuration": spec.configuration,
                "source_run_id": spec.source_run_id,
                "target_run_id": spec.target_run_id,
                "all_passed": all_passed,
                "artifact_binding": binding,
                "selected_execution_comparison": execution_doc,
                "selection_slice_count": len(selections),
                "selection_schema_errors": selection_errors,
                "llm_only_mapping_count": len(llm_only),
                "llm_only_mapping_errors": llm_only_errors,
                "causal_scope": causal,
                "micro_metrics_validation": metric_validation,
            }
        )

    expected_count_passed = (
        expected_changed_final_keys is None
        or total_changed == expected_changed_final_keys
    )
    validation_passed = (
        cache_binding["all_passed"]
        and retry_binding["all_passed"]
        and all(row["all_passed"] for row in run_documents)
        and expected_count_passed
        and total_unexplained == 0
    )
    audit_document = {
        "audit_type": "offline_final_replay_causal_and_artifact_integrity",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_and_target_runs_read_only": True,
        "historical_artifacts_overwritten": False,
        "api_called_by_audit": False,
        "standard_router_v3_validate_run_used": False,
        "validation_passed": validation_passed,
        "configuration_count": len(specs),
        "expected_changed_final_semantic_keys": expected_changed_final_keys,
        "observed_changed_final_semantic_keys": total_changed,
        "changed_final_count_matches_expected": expected_count_passed,
        "unexplained_changed_final_semantic_keys": total_unexplained,
        "cache_and_response_bank_binding": cache_binding,
        "retry_binding": retry_binding,
        "runs": run_documents,
    }

    output.mkdir(parents=True, exist_ok=False)
    _write_json_new(output / "causal_integrity_audit.json", audit_document)
    _write_csv(output / "changed_execution_queries.csv", changed_query_rows)
    _write_csv(output / "changed_final_semantic_keys.csv", changed_final_rows)
    _write_csv(output / "method_metrics_recomputed_micro.csv", recomputed_metric_rows)
    _write_csv(output / "method_metrics_validation.csv", metric_validation_rows)
    _write_csv(output / "aggregate_metric_delta.csv", delta_rows)
    artifacts = {
        path.name: _sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    binding_root = _canonical_sha256(artifacts)
    manifest = {
        "run_id": output_run_id,
        "run_kind": "offline_final_replay_integrity_audit",
        "status": "complete",
        "created_at_utc": audit_document["created_at_utc"],
        "api_called": False,
        "source_and_target_runs_read_only": True,
        "historical_artifacts_overwritten": False,
        "configuration_count": len(specs),
        "validation_passed": validation_passed,
        "observed_changed_final_semantic_keys": total_changed,
        "unexplained_changed_final_semantic_keys": total_unexplained,
        "artifacts": artifacts,
        "artifact_binding_root_sha256": binding_root,
    }
    _write_json_new(output / "run_manifest.json", manifest)
    return {
        "output_run": str(output),
        "validation_passed": validation_passed,
        "changed_final_semantic_keys": total_changed,
        "unexplained_changed_final_semantic_keys": total_unexplained,
        "method_metric_mismatches": sum(
            int(row["mismatch_count"]) for row in metric_validation_rows
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the additive offline final-replay causal integrity audit"
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-run-id", default=DEFAULT_OUTPUT_RUN_ID)
    parser.add_argument("--cache-audit-run-id", default=DEFAULT_CACHE_AUDIT_RUN_ID)
    parser.add_argument("--retry-run-id", default=DEFAULT_RETRY_RUN_ID)
    parser.add_argument("--expected-changed-final-keys", type=int, default=985)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = execute_integrity_audit(
        runs_root=args.runs_root,
        output_run_id=args.output_run_id,
        cache_audit_run_id=args.cache_audit_run_id,
        retry_run_id=args.retry_run_id,
        expected_changed_final_keys=args.expected_changed_final_keys,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["validation_passed"] else 1


__all__ = [
    "DEFAULT_AUDIT_SPECS",
    "IntegrityAuditSpec",
    "audit_final_causal_scope",
    "build_cache_bank_binding",
    "build_retry_binding",
    "compare_reported_micro_metrics",
    "compare_selected_execution",
    "execute_integrity_audit",
    "final_record_slice",
    "load_selection_slices",
    "recompute_micro_metrics",
    "response_content_projection",
    "response_content_projection_sha256",
]


if __name__ == "__main__":
    raise SystemExit(main())
