"""Independent CARE correction metrics for every BGR experiment slice."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import html
import math
import re
from typing import Any


DIMENSION_FIELDS = ("scenario", "backend", "budget_share", "group_size_variant")
REQUIRED_RECORD_FIELDS = ("cell_id", "method", "dataset", "clean_value", "parse_status")


def normalize_for_match(value: object) -> str:
    """Apply the same entity and ASCII-whitespace normalization as CARE."""

    text = "" if value is None else str(value)
    return re.sub(r"[\t\n ]+", " ", html.unescape(text)).strip("\t\n ")


def is_valid_repair(record: Mapping[str, object]) -> bool:
    """Only schema-valid predictions enter the precision denominator."""

    return str(record.get("parse_status") or "").startswith("ok")


def is_correct_repair(record: Mapping[str, object]) -> bool:
    """Recompute normalized exact-match correctness from raw per-cell fields."""

    if not is_valid_repair(record):
        return False
    if "prediction" not in record or record.get("prediction") is None:
        return False
    if "clean_value" not in record:
        return False
    return normalize_for_match(record.get("prediction")) == normalize_for_match(
        record.get("clean_value")
    )


def slice_dimensions(record: Mapping[str, object]) -> tuple[str, str, float | None, str]:
    """Return canonical ``scenario/backend/budget/variant`` dimensions."""

    raw_budget = record.get("budget_share")
    budget = None if raw_budget in {None, ""} else float(raw_budget)  # type: ignore[comparison-overlap]
    if budget is not None and (not math.isfinite(budget) or budget < 0.0):
        raise ValueError("budget_share must be finite and non-negative")
    return (
        str(record.get("scenario") or "main"),
        str(record.get("backend") or "none"),
        budget,
        str(record.get("group_size_variant") or record.get("variant") or "all"),
    )


def recompute_record(record: Mapping[str, object]) -> dict[str, object]:
    valid = is_valid_repair(record)
    correct = is_correct_repair(record)
    stored_correct_mismatch = (
        "correct_repair" in record
        and _coerce_bool(record.get("correct_repair")) != correct
    )
    stored_valid_mismatch = any(
        field in record and _coerce_bool(record.get(field)) != valid
        for field in ("valid_prediction", "valid_repair")
    )
    scenario, backend, budget, variant = slice_dimensions(record)
    return {
        "cell_id": str(record.get("cell_id") or ""),
        "suite": str(record.get("suite") or ""),
        "dataset": str(record.get("dataset") or ""),
        "method": _method(record),
        "scenario": scenario,
        "backend": backend,
        "budget_share": budget,
        "group_size_variant": variant,
        "valid_repair": valid,
        "correct_repair": correct,
        "stored_correct_mismatch": stored_correct_mismatch,
        "stored_valid_mismatch": stored_valid_mismatch,
    }


def verify_records(
    records: Iterable[Mapping[str, object]],
    *,
    expected_cell_ids: Iterable[str] | Mapping[object, Iterable[str]] | None = None,
) -> dict[str, object]:
    """Audit identities, every slice's coverage, and cached annotations."""

    materialized = [dict(record) for record in records]
    flags = [recompute_record(record) for record in materialized]
    missing_fields: list[str] = []
    for index, record in enumerate(materialized):
        for field in REQUIRED_RECORD_FIELDS:
            present = field in record
            if field == "method":
                present = bool(_method(record))
            elif field == "dataset":
                present = bool(record.get("dataset"))
            if not present:
                missing_fields.append(f"row={index}:{field}")

    identities = [_identity(flag) for flag in flags]
    counts = Counter(identities)
    duplicates = [
        {
            "method": key[0],
            "scenario": key[1],
            "backend": key[2],
            "budget_share": key[3],
            "group_size_variant": key[4],
            "suite": key[5],
            "dataset": key[6],
            "cell_id": key[7],
            "count": count,
        }
        for key, count in sorted(counts.items(), key=lambda item: repr(item[0]))
        if count > 1
    ]
    mismatches = [
        {
            "row": index,
            "method": flag["method"],
            "dataset": flag["dataset"],
            "cell_id": flag["cell_id"],
            "stored_correct_mismatch": flag["stored_correct_mismatch"],
            "stored_valid_mismatch": flag["stored_valid_mismatch"],
        }
        for index, flag in enumerate(flags)
        if flag["stored_correct_mismatch"] or flag["stored_valid_mismatch"]
    ]

    coverage: dict[tuple[object, ...], set[str]] = defaultdict(set)
    for flag in flags:
        coverage[_coverage_key(flag)].add(str(flag["cell_id"]))
    coverage_errors: list[dict[str, object]] = []
    if expected_cell_ids is not None:
        for key, observed in sorted(coverage.items(), key=lambda item: repr(item[0])):
            expected = _expected_for_slice(expected_cell_ids, key[-2], key[-1])
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            if missing or extra:
                coverage_errors.append(
                    {
                        "slice": list(key),
                        "missing_count": len(missing),
                        "extra_count": len(extra),
                        "missing": missing,
                        "extra": extra,
                    }
                )

    return {
        "records": len(materialized),
        "unique_records": len(counts),
        "slices": len(coverage),
        "valid_repairs": sum(bool(flag["valid_repair"]) for flag in flags),
        "correct_repairs": sum(bool(flag["correct_repair"]) for flag in flags),
        "missing_fields": missing_fields,
        "duplicate_records": duplicates,
        "annotation_mismatches": mismatches,
        "coverage_errors": coverage_errors,
        "ok": not (missing_fields or duplicates or mismatches or coverage_errors),
    }


def summarize_records(
    records: Iterable[Mapping[str, object]],
    method: str | Sequence[str] | None = None,
    dataset: str | Sequence[str] | None = None,
    *,
    scenario: str | Sequence[str] | None = None,
    backend: str | Sequence[str] | None = None,
    budget_share: float | Sequence[float] | None = None,
    group_size_variant: str | Sequence[str] | None = None,
    include_aggregates: bool = True,
    strict: bool = True,
) -> list[dict[str, object]]:
    """Compute per-dataset, micro, and macro rows for each full slice."""

    materialized = [dict(record) for record in records]
    selected = []
    for record in materialized:
        dims = slice_dimensions(record)
        if not _selection(_method(record), method):
            continue
        if not _selection(str(record.get("dataset") or ""), dataset):
            continue
        if not _selection(dims[0], scenario) or not _selection(dims[1], backend):
            continue
        if not _budget_selection(dims[2], budget_share):
            continue
        if not _selection(dims[3], group_size_variant):
            continue
        selected.append(record)
    if not selected:
        return []
    audit = verify_records(selected)
    if strict:
        _strict_check(audit)
    flags = [recompute_record(record) for record in selected]

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for flag in flags:
        grouped[
            (
                flag["method"],
                flag["scenario"],
                flag["backend"],
                flag["budget_share"],
                flag["group_size_variant"],
                flag["suite"],
                flag["dataset"],
            )
        ].append(flag)

    dataset_rows = [
        _summarize_flags(group, key, scope="dataset")
        for key, group in grouped.items()
    ]
    if not include_aggregates:
        return sorted(dataset_rows, key=_sort_key)

    by_series: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    flags_by_series: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in dataset_rows:
        by_series[_series_key(row)].append(row)
    for flag in flags:
        flags_by_series[_series_key(flag)].append(flag)

    rows = list(dataset_rows)
    for series, series_flags in flags_by_series.items():
        aggregate_key = (*series, "ALL", "MICRO")
        rows.append(_summarize_flags(series_flags, aggregate_key, scope="micro"))
        children = by_series[series]
        macro: dict[str, object] = {
            "scope": "macro",
            "method": series[0],
            "scenario": series[1],
            "backend": series[2],
            "budget_share": series[3],
            "group_size_variant": series[4],
            "suite": "ALL",
            "dataset": "MACRO",
        }
        for field in (
            "true_error_cells",
            "predicted_repairs",
            "valid_predictions",
            "invalid_predictions",
            "correct_repairs",
            "annotation_mismatches",
        ):
            macro[field] = sum(int(row[field]) for row in children)
        for field in ("precision", "recall", "f1"):
            macro[field] = _safe_rate(
                sum(float(row[field]) for row in children), len(children)
            )
        macro["correction_accuracy"] = macro["recall"]
        rows.append(macro)
    return sorted(rows, key=_sort_key)


def compare_methods(
    records: Iterable[Mapping[str, object]],
    baseline: str = "baran",
    *,
    method: str | Sequence[str] | None = None,
    strict: bool = True,
) -> list[dict[str, object]]:
    """Attach absolute method-minus-baseline deltas to every result row."""

    materialized = [
        dict(record)
        for record in records
        if _method(record) == baseline or _selection(_method(record), method)
    ]
    if not materialized:
        return []
    methods = {_method(record) for record in materialized if _method(record)}
    if baseline not in methods:
        raise ValueError(f"baseline method {baseline!r} is absent")
    if len(methods) == 1:
        return []
    audit = verify_records(materialized)
    if strict:
        _strict_check(audit)

    flags = [recompute_record(record) for record in materialized]
    coverage: dict[tuple[object, ...], set[str]] = defaultdict(set)
    for flag in flags:
        coverage[_coverage_key(flag)].add(str(flag["cell_id"]))
    baseline_coverage = {
        key: cells for key, cells in coverage.items() if key[0] == baseline
    }
    for key, cells in coverage.items():
        if key[0] == baseline:
            continue
        reference = _matching_baseline_coverage(key, baseline_coverage, baseline)
        if strict and cells != reference:
            raise ValueError(
                f"cell-universe mismatch for slice {key!r}: "
                f"missing={len(reference - cells)}, extra={len(cells - reference)}"
            )

    summaries = summarize_records(materialized, strict=strict)
    baseline_rows = [row for row in summaries if row["method"] == baseline]
    comparisons: list[dict[str, object]] = []
    for row in summaries:
        if row["method"] == baseline:
            continue
        reference = _matching_baseline_row(row, baseline_rows)
        result = dict(row)
        result["baseline_method"] = baseline
        for field in (
            "predicted_repairs",
            "correct_repairs",
            "precision",
            "recall",
            "f1",
        ):
            result[f"baseline_{field}"] = reference[field]
            result[f"{field}_delta"] = float(row[field]) - float(reference[field])
        comparisons.append(result)
    return sorted(comparisons, key=_sort_key)


def area_under_budget_curve(
    points: Mapping[float, float] | Sequence[tuple[float, float]],
    *,
    baseline_value: float,
    max_budget: float = 0.5,
) -> float:
    """Normalized trapezoidal area on ``[0,max_budget]`` with a β=0 anchor."""

    maximum = float(max_budget)
    anchor = float(baseline_value)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_budget must be finite and positive")
    if not math.isfinite(anchor):
        raise ValueError("baseline_value must be finite")
    raw = dict(points.items()) if isinstance(points, Mapping) else dict(points)
    if not raw:
        raise ValueError("budget curve must contain at least one nonzero point")
    checked: dict[float, float] = {0.0: anchor}
    for beta, value in raw.items():
        x = float(beta)
        y = float(value)
        if not math.isfinite(x) or not 0.0 <= x <= maximum:
            raise ValueError("budget points must be finite and within the integration interval")
        if not math.isfinite(y):
            raise ValueError("budget metric values must be finite")
        if x == 0.0 and not math.isclose(y, anchor, abs_tol=1e-12):
            raise ValueError("the beta=0 point must equal the baseline anchor")
        checked[x] = y
    if maximum not in checked:
        raise ValueError("budget curve must include max_budget")
    ordered = sorted(checked.items())
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(ordered, ordered[1:])
    )
    return float(area / maximum)


def compute_aubc(
    summary_rows: Iterable[Mapping[str, object]],
    *,
    baseline_method: str = "baran",
    metric: str = "f1",
    max_budget: float = 0.5,
) -> list[dict[str, object]]:
    """Compute one anchored AUBC row per method/series/scope/dataset."""

    rows = [dict(row) for row in summary_rows]
    baseline_rows = [row for row in rows if str(row.get("method")) == baseline_method]
    if not baseline_rows:
        raise ValueError(f"baseline method {baseline_method!r} is absent")
    curves: dict[tuple[object, ...], dict[float, float]] = defaultdict(dict)
    templates: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        if str(row.get("method")) == baseline_method:
            continue
        budget = row.get("budget_share")
        if budget in {None, ""}:
            continue
        if metric not in row:
            raise ValueError(f"summary row is missing metric {metric!r}")
        key = (
            row.get("method"),
            row.get("scenario"),
            row.get("backend"),
            row.get("group_size_variant"),
            row.get("scope"),
            row.get("suite"),
            row.get("dataset"),
        )
        curves[key][float(budget)] = float(row[metric])
        templates[key] = row

    results: list[dict[str, object]] = []
    for key, points in sorted(curves.items(), key=lambda item: repr(item[0])):
        reference = _matching_baseline_row(templates[key], baseline_rows)
        result = {
            "method": key[0],
            "scenario": key[1],
            "backend": key[2],
            "group_size_variant": key[3],
            "scope": key[4],
            "suite": key[5],
            "dataset": key[6],
            "baseline_method": baseline_method,
            "baseline_value": float(reference[metric]),
            "max_budget": float(max_budget),
            f"{metric}_aubc": area_under_budget_curve(
                points,
                baseline_value=float(reference[metric]),
                max_budget=max_budget,
            ),
        }
        results.append(result)
    return results


summarize_budget_metrics = summarize_records


def _summarize_flags(
    flags: Sequence[Mapping[str, object]],
    key: tuple[object, ...],
    *,
    scope: str,
) -> dict[str, object]:
    total = len(flags)
    predicted = sum(bool(flag["valid_repair"]) for flag in flags)
    correct = sum(bool(flag["correct_repair"]) for flag in flags)
    mismatches = sum(
        bool(flag["stored_correct_mismatch"])
        or bool(flag["stored_valid_mismatch"])
        for flag in flags
    )
    precision = _safe_rate(correct, predicted)
    recall = _safe_rate(correct, total)
    return {
        "scope": scope,
        "method": key[0],
        "scenario": key[1],
        "backend": key[2],
        "budget_share": key[3],
        "group_size_variant": key[4],
        "suite": key[5],
        "dataset": key[6],
        "true_error_cells": total,
        "predicted_repairs": predicted,
        "valid_predictions": predicted,
        "invalid_predictions": total - predicted,
        "correct_repairs": correct,
        "correction_accuracy": recall,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "annotation_mismatches": mismatches,
    }


def _identity(flag: Mapping[str, object]) -> tuple[object, ...]:
    return (*_coverage_key(flag), flag["cell_id"])


def _coverage_key(flag: Mapping[str, object]) -> tuple[object, ...]:
    return (
        flag["method"],
        flag["scenario"],
        flag["backend"],
        flag["budget_share"],
        flag["group_size_variant"],
        flag["suite"],
        flag["dataset"],
    )


def _series_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        row["method"],
        row["scenario"],
        row["backend"],
        row["budget_share"],
        row["group_size_variant"],
    )


def _matching_baseline_coverage(
    target_key: tuple[object, ...],
    baseline: Mapping[tuple[object, ...], set[str]],
    baseline_method: str,
) -> set[str]:
    exact = (baseline_method, *target_key[1:])
    if exact in baseline:
        return baseline[exact]
    candidates = [
        cells
        for key, cells in baseline.items()
        if key[-2:] == target_key[-2:]
    ]
    if not candidates:
        raise ValueError(f"missing baseline coverage for {target_key[-2:]}")
    first = candidates[0]
    if any(cells != first for cells in candidates[1:]):
        raise ValueError("baseline slices disagree on their cell universe")
    return first


def _matching_baseline_row(
    target: Mapping[str, object], baseline_rows: Sequence[Mapping[str, object]]
) -> Mapping[str, object]:
    location = (target.get("scope"), target.get("suite"), target.get("dataset"))
    candidates = [
        row
        for row in baseline_rows
        if (row.get("scope"), row.get("suite"), row.get("dataset")) == location
    ]
    if not candidates:
        raise ValueError(f"missing baseline summary for {location}")
    exact = [
        row
        for row in candidates
        if all(row.get(field) == target.get(field) for field in DIMENSION_FIELDS)
    ]
    if exact:
        return exact[0]
    signature_fields = (
        "true_error_cells",
        "predicted_repairs",
        "correct_repairs",
        "precision",
        "recall",
        "f1",
    )
    signature = tuple(candidates[0].get(field) for field in signature_fields)
    if any(
        tuple(row.get(field) for field in signature_fields) != signature
        for row in candidates[1:]
    ):
        raise ValueError("baseline slices disagree on metric values")
    return candidates[0]


def _expected_for_slice(
    expected: Iterable[str] | Mapping[object, Iterable[str]],
    suite: object,
    dataset: object,
) -> set[str]:
    if not isinstance(expected, Mapping):
        return {str(value) for value in expected}
    for key in ((str(suite), str(dataset)), str(dataset), f"{suite}:{dataset}"):
        if key in expected:
            return {str(value) for value in expected[key]}
    raise ValueError(f"expected_cell_ids has no entry for {(suite, dataset)}")


def _strict_check(audit: Mapping[str, object]) -> None:
    problems = []
    for key in (
        "missing_fields",
        "duplicate_records",
        "annotation_mismatches",
        "coverage_errors",
    ):
        value = audit.get(key)
        if value:
            problems.append(f"{key}={len(value)}")  # type: ignore[arg-type]
    if problems:
        raise ValueError("per-cell record verification failed: " + ", ".join(problems))


def _sort_key(row: Mapping[str, object]) -> tuple[object, ...]:
    scope_order = {"dataset": 0, "micro": 1, "macro": 2}
    budget = row.get("budget_share")
    budget_key = -1.0 if budget is None else float(budget)
    return (
        str(row.get("method")),
        str(row.get("scenario")),
        str(row.get("backend")),
        budget_key,
        str(row.get("group_size_variant")),
        scope_order.get(str(row.get("scope")), 9),
        str(row.get("suite")),
        str(row.get("dataset")),
    )


def _method(record: Mapping[str, object]) -> str:
    return str(record.get("method") or record.get("experiment") or "")


def _selection(value: str, selected: str | Sequence[str] | None) -> bool:
    if selected is None:
        return True
    if isinstance(selected, str):
        return value == selected
    return value in {str(item) for item in selected}


def _budget_selection(
    value: float | None, selected: float | Sequence[float] | None
) -> bool:
    if selected is None:
        return True
    requested = (float(selected),) if isinstance(selected, (int, float)) else tuple(float(item) for item in selected)
    return value is not None and any(math.isclose(value, item, abs_tol=1e-12) for item in requested)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_rate(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return (
        0.0
        if precision + recall == 0.0
        else 2.0 * precision * recall / (precision + recall)
    )
