"""Offline analysis and Introduction figure for the motivation experiment.

The report builder consumes only the two finalized, clean-label-bound cell
ledgers.  It does not inspect execution checkpoints, instantiate a model
client, or mutate the ledgers.  All inferential families and aggregation rules
are fixed here so regenerating a report cannot silently change the protocol.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from .data import normalize_for_match
from .sampling import SELECTED_DATASETS
from .statistics import exact_mcnemar, holm_adjust


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REPORT_SCHEMA_VERSION = "motivation-evidence-report-v1"
PRIMARY_VIEWS = ("pattern", "semantic")
GROUP_SIZES = (2, 4, 8)
CONTRASTS = (
    "structured_minus_singleton",
    "random_minus_singleton",
    "structured_minus_random",
)
COMPLEMENTARITY_REQUIRED_COLUMNS = frozenset(
    {
        "suite",
        "dataset",
        "base_family",
        "cell_id",
        "row_cluster",
        "column",
        "dirty_value",
        "clean_value",
        "baran_prediction",
        "baran_valid",
        "baran_correct",
        "llm_prediction",
        "llm_valid",
        "llm_correct",
        "llm_status",
        "llm_parse_status",
        "llm_decision",
        "llm_observed_input_tokens",
        "llm_observed_output_tokens",
        "llm_observed_total_tokens",
        "outcome_quadrant",
    }
)
GROUP_REQUIRED_COLUMNS = frozenset(
    {
        "suite",
        "dataset",
        "base_family",
        "source_view",
        "group_size",
        "cell_id",
        "row_cluster",
        "column",
        "dirty_value",
        "clean_value",
        "member_position",
        "singleton_logical_query_id",
        "singleton_physical_query_id",
        "structured_logical_query_id",
        "structured_physical_query_id",
        "random_logical_query_id",
        "random_physical_query_id",
        "singleton_prediction",
        "singleton_valid",
        "singleton_correct",
        "singleton_status",
        "singleton_parse_status",
        "singleton_decision",
        "structured_prediction",
        "structured_valid",
        "structured_correct",
        "structured_status",
        "structured_parse_status",
        "structured_decision",
        "random_prediction",
        "random_valid",
        "random_correct",
        "random_status",
        "random_parse_status",
        "random_decision",
        "structured_rescue",
        "structured_interference",
        "random_rescue",
        "random_interference",
        "structured_query_observed_input_tokens",
        "structured_query_observed_output_tokens",
        "structured_query_observed_total_tokens",
        "random_query_observed_input_tokens",
        "random_query_observed_output_tokens",
        "random_query_observed_total_tokens",
        "singleton_query_observed_input_tokens",
        "singleton_query_observed_output_tokens",
        "singleton_query_observed_total_tokens",
    }
)
COMPLEMENTARITY_RATE_FIELDS = (
    "baran_accuracy",
    "llm_accuracy",
    "oracle_union_upper_bound",
    "llm_salvage_opportunity",
    "overwrite_risk",
    "disagreement_rate",
    "llm_minus_baran",
)
GROUP_RATE_FIELDS = (
    "singleton_accuracy",
    "structured_accuracy",
    "random_accuracy",
    "structured_rescue_rate",
    "structured_interference_rate",
    "random_rescue_rate",
    "random_interference_rate",
    "structured_minus_singleton",
    "random_minus_singleton",
    "structured_minus_random",
    "singleton_invalid_rate",
    "structured_invalid_rate",
    "random_invalid_rate",
    "singleton_provider_failure_rate",
    "structured_provider_failure_rate",
    "random_provider_failure_rate",
    "singleton_parse_failure_rate",
    "structured_parse_failure_rate",
    "random_parse_failure_rate",
    "singleton_missing_rate",
    "structured_missing_rate",
    "random_missing_rate",
    "singleton_abstain_rate",
    "structured_abstain_rate",
    "random_abstain_rate",
    "singleton_empty_rate",
    "structured_empty_rate",
    "random_empty_rate",
    "singleton_unchanged_rate",
    "structured_unchanged_rate",
    "random_unchanged_rate",
    "singleton_other_invalid_rate",
    "structured_other_invalid_rate",
    "random_other_invalid_rate",
)

API_COST_REQUIRED_COLUMNS = frozenset(
    {
        "physical_query_id",
        "provider_request_hash",
        "status",
        "attempts",
        "observed_input_tokens",
        "observed_output_tokens",
        "observed_total_tokens",
        "latency_seconds",
        "usage_observed_attempts",
        "unknown_usage_attempts",
        "estimated_prompt_tokens",
        "estimated_completion_tokens",
        "estimated_total_tokens",
        "logical_query_mappings",
    }
)

_SUCCESS_STATUSES = frozenset({"success", "ok", "completed"})
_ITEM_BEARING_PARSE_STATUSES = frozenset({"ok", "partial"})
_PARSE_FAILURE_MARKERS = (
    "parse_failure",
    "no_json",
    "invalid_json",
    "malformed",
    "query_id_mismatch",
    "invalid_repairs",
    "no_valid_items",
    "duplicate_cell",
    "unknown_cell",
)


def _read_csv(path: Path, required: frozenset[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"motivation report input is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"motivation report input is empty: {path}")
    return rows


def _boolean(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} must be boolean, observed {value!r}")


def _integer(value: object, *, field: str, blank: int | None = None) -> int:
    text = str(value).strip()
    if not text and blank is not None:
        return int(blank)
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer, observed {value!r}") from exc
    if number < 0:
        raise ValueError(f"{field} must be non-negative, observed {number}")
    return number


def _optional_integer(row: Mapping[str, object], fields: Sequence[str]) -> int | None:
    for field in fields:
        if field in row and str(row.get(field, "")).strip():
            return _integer(row[field], field=field)
    return None


def _optional_float(row: Mapping[str, object], fields: Sequence[str]) -> float | None:
    for field in fields:
        if field in row and str(row.get(field, "")).strip():
            try:
                value = float(row[field])
            except ValueError as exc:
                raise ValueError(f"{field} must be numeric, observed {row[field]!r}") from exc
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field} must be a finite non-negative number")
            return value
    return None


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _is_parse_failure(parse_status: object) -> bool:
    normalized = str(parse_status).strip().lower()
    return any(marker in normalized for marker in _PARSE_FAILURE_MARKERS)


def _derived_llm_validity(
    row: Mapping[str, object],
    *,
    prefix: str,
) -> tuple[bool, bool]:
    """Recompute validity/correctness from raw, clean-label-bound fields."""

    status = str(row[f"{prefix}_status"]).strip().lower()
    parse_status = str(row[f"{prefix}_parse_status"]).strip().lower()
    decision = str(row[f"{prefix}_decision"]).strip().lower()
    prediction = normalize_for_match(row.get(f"{prefix}_prediction", ""))
    dirty = normalize_for_match(row.get("dirty_value", ""))
    clean = normalize_for_match(row.get("clean_value", ""))
    valid = bool(
        status in _SUCCESS_STATUSES
        and parse_status in _ITEM_BEARING_PARSE_STATUSES
        and decision == "propose"
        and prediction
        and prediction != dirty
    )
    correct = bool(valid and prediction == clean)
    return valid, correct


def _require_derived_boolean(
    row: dict[str, object],
    *,
    prefix: str,
    identity: object,
) -> None:
    derived_valid, derived_correct = _derived_llm_validity(row, prefix=prefix)
    if bool(row[f"{prefix}_valid"]) != derived_valid:
        raise ValueError(f"stored {prefix}_valid disagrees with raw fields: {identity}")
    if bool(row[f"{prefix}_correct"]) != derived_correct:
        raise ValueError(f"stored {prefix}_correct disagrees with clean evaluation: {identity}")
    row[f"{prefix}_valid"] = derived_valid
    row[f"{prefix}_correct"] = derived_correct


def _derived_seed(seed: int, *parts: object) -> int:
    material = "|".join((str(int(seed)), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def _interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (math.nan, math.nan)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(finite, alpha)),
        float(np.quantile(finite, 1.0 - alpha)),
    )


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty report table: {path}")
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            safe = _json_safe(row)
            assert isinstance(safe, Mapping)
            writer.writerow(
                {
                    column: "" if safe.get(column) is None else safe.get(column, "")
                    for column in columns
                }
            )
    temporary.replace(path)


def _parse_complementarity(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    identities: set[str] = set()
    for source in rows:
        row: dict[str, object] = dict(source)
        for field in ("baran_valid", "baran_correct", "llm_valid", "llm_correct"):
            row[field] = _boolean(source[field], field=field)
        for field in (
            "llm_observed_input_tokens",
            "llm_observed_output_tokens",
            "llm_observed_total_tokens",
        ):
            row[field] = _integer(source[field], field=field, blank=0)
        cell_id = str(row["cell_id"])
        if not cell_id or cell_id in identities:
            raise ValueError(f"duplicate or empty complementarity cell_id: {cell_id!r}")
        identities.add(cell_id)
        derived_baran_correct = bool(
            row["baran_valid"]
            and normalize_for_match(row["baran_prediction"])
            == normalize_for_match(row["clean_value"])
        )
        if bool(row["baran_correct"]) != derived_baran_correct:
            raise ValueError(f"stored baran_correct disagrees with clean evaluation: {cell_id}")
        row["baran_correct"] = derived_baran_correct
        _require_derived_boolean(row, prefix="llm", identity=cell_id)
        expected_quadrant = (
            f"n{int(derived_baran_correct)}{int(bool(row['llm_correct']))}"
        )
        observed_quadrant = str(row["outcome_quadrant"]).strip().lower().replace("_", "")
        if observed_quadrant != expected_quadrant:
            raise ValueError(f"outcome quadrant disagrees with paired correctness: {cell_id}")
        row["outcome_quadrant"] = expected_quadrant
        parsed.append(row)
    observed = {(str(row["suite"]), str(row["dataset"])) for row in parsed}
    if observed != set(SELECTED_DATASETS):
        raise ValueError("complementarity ledger must cover exactly the frozen nine datasets")
    return parsed


def _parse_group(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    identities: set[tuple[str, str, str, int, str]] = set()
    boolean_fields = (
        "singleton_valid",
        "singleton_correct",
        "structured_valid",
        "structured_correct",
        "random_valid",
        "random_correct",
        "structured_rescue",
        "structured_interference",
        "random_rescue",
        "random_interference",
    )
    token_fields = tuple(
        f"{arm}_query_observed_{kind}_tokens"
        for arm in ("singleton", "structured", "random")
        for kind in ("input", "output", "total")
    )
    for source in rows:
        row: dict[str, object] = dict(source)
        row["group_size"] = _integer(source["group_size"], field="group_size")
        row["member_position"] = _integer(source["member_position"], field="member_position")
        for field in boolean_fields:
            row[field] = _boolean(source[field], field=field)
        for field in token_fields:
            row[field] = _integer(source[field], field=field, blank=0)
        view = str(row["source_view"])
        size = int(row["group_size"])
        if view not in PRIMARY_VIEWS or size not in GROUP_SIZES:
            raise ValueError(f"unexpected primary condition: {view}/k={size}")
        identity = (
            str(row["suite"]),
            str(row["dataset"]),
            view,
            size,
            str(row["cell_id"]),
        )
        if not identity[-1] or identity in identities:
            raise ValueError(f"duplicate or empty group cell incidence: {identity}")
        identities.add(identity)
        for arm in ("singleton", "structured", "random"):
            _require_derived_boolean(row, prefix=arm, identity=identity)
            for identity_field in ("logical_query_id", "physical_query_id"):
                if not str(row[f"{arm}_{identity_field}"]):
                    raise ValueError(f"empty {arm}_{identity_field}: {identity}")
        expected = {
            "structured_rescue": bool(row["structured_correct"])
            and not bool(row["singleton_correct"]),
            "structured_interference": bool(row["singleton_correct"])
            and not bool(row["structured_correct"]),
            "random_rescue": bool(row["random_correct"])
            and not bool(row["singleton_correct"]),
            "random_interference": bool(row["singleton_correct"])
            and not bool(row["random_correct"]),
        }
        for field, value in expected.items():
            if bool(row[field]) != value:
                raise ValueError(f"inconsistent {field}: {identity}")
        parsed.append(row)
    expected_conditions = {
        (*dataset, view, size)
        for dataset in SELECTED_DATASETS
        for view in PRIMARY_VIEWS
        for size in GROUP_SIZES
    }
    observed_conditions = {
        (
            str(row["suite"]),
            str(row["dataset"]),
            str(row["source_view"]),
            int(row["group_size"]),
        )
        for row in parsed
    }
    if observed_conditions != expected_conditions:
        raise ValueError("group ledger must cover all 54 frozen dataset/view/size conditions")
    return parsed


def _complementarity_values(row: Mapping[str, object]) -> np.ndarray:
    baran = bool(row["baran_correct"])
    llm = bool(row["llm_correct"])
    return np.asarray(
        [
            float(baran),
            float(llm),
            float(baran or llm),
            float(llm and not baran),
            float(baran and not llm),
            float(baran != llm),
            float(llm) - float(baran),
        ],
        dtype=float,
    )


def _row_bootstrap_samples(
    rows: Sequence[Mapping[str, object]],
    *,
    value: Callable[[Mapping[str, object]], np.ndarray],
    replicates: int,
    seed: int,
    macro: bool,
) -> np.ndarray:
    if macro:
        datasets: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
        for row in rows:
            datasets[(str(row["suite"]), str(row["dataset"]))].append(row)
        samples = [
            _row_bootstrap_samples(
                subset,
                value=value,
                replicates=replicates,
                seed=_derived_seed(seed, *key),
                macro=False,
            )
            for key, subset in sorted(datasets.items())
        ]
        return np.mean(np.stack(samples, axis=0), axis=0)

    clusters: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        clusters[str(row["row_cluster"])].append(value(row))
    keys = sorted(clusters)
    if not keys:
        return np.full((replicates, len(COMPLEMENTARITY_RATE_FIELDS)), np.nan)
    sums = np.stack([np.sum(clusters[key], axis=0) for key in keys], axis=0)
    counts = np.asarray([len(clusters[key]) for key in keys], dtype=float)
    rng = np.random.default_rng(int(seed))
    samples = np.empty((replicates, sums.shape[1]), dtype=float)
    for replicate in range(replicates):
        indices = rng.integers(0, len(keys), size=len(keys))
        denominator = float(np.sum(counts[indices]))
        samples[replicate] = np.sum(sums[indices], axis=0) / denominator
    return samples


def _complementarity_aggregate(
    rows: Sequence[Mapping[str, object]],
    *,
    scope: str,
    suite: str,
    dataset: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
    macro: bool = False,
) -> dict[str, object]:
    counts = Counter(
        (bool(row["baran_correct"]), bool(row["llm_correct"])) for row in rows
    )
    n11 = counts[(True, True)]
    n10 = counts[(True, False)]
    n01 = counts[(False, True)]
    n00 = counts[(False, False)]
    values = np.stack([_complementarity_values(row) for row in rows], axis=0)
    point = (
        np.mean(
            np.stack(
                [
                    np.mean(
                        values[
                            np.asarray(
                                [
                                    str(row["suite"]) == key[0]
                                    and str(row["dataset"]) == key[1]
                                    for row in rows
                                ],
                                dtype=bool,
                            )
                        ],
                        axis=0,
                    )
                    for key in sorted(
                        {(str(row["suite"]), str(row["dataset"])) for row in rows}
                    )
                ],
                axis=0,
            ),
            axis=0,
        )
        if macro
        else np.mean(values, axis=0)
    )
    samples = _row_bootstrap_samples(
        rows,
        value=_complementarity_values,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        macro=macro,
    )
    result: dict[str, object] = {
        "scope": scope,
        "aggregation": "unweighted_dataset_mean" if macro else "micro_over_cells",
        "quadrant_count_aggregation": (
            "pooled_cell_counts; rate fields use the unweighted nine-dataset macro"
            if macro
            else "pooled_cell_counts"
        ),
        "suite": suite,
        "dataset": dataset,
        "N": len(rows),
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00,
        "mcnemar_p": exact_mcnemar(n10, n01) if not macro else math.nan,
        "mcnemar_p_holm": math.nan,
    }
    for index, field in enumerate(COMPLEMENTARITY_RATE_FIELDS):
        result[field] = float(point[index])
        low, high = _interval(samples[:, index], confidence)
        result[f"{field}_ci_low"] = low
        result[f"{field}_ci_high"] = high
    return result


def _build_complementarity_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
) -> list[dict[str, object]]:
    by_dataset: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    by_suite: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["suite"]), str(row["dataset"]))
        by_dataset[key].append(row)
        by_suite[key[0]].append(row)
    output: list[dict[str, object]] = []
    for suite, dataset in SELECTED_DATASETS:
        output.append(
            _complementarity_aggregate(
                by_dataset[(suite, dataset)],
                scope="dataset",
                suite=suite,
                dataset=dataset,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=_derived_seed(bootstrap_seed, "complementarity", suite, dataset),
                confidence=confidence,
            )
        )
    adjusted = holm_adjust(
        {
            f"{row['suite']}/{row['dataset']}": float(row["mcnemar_p"])
            for row in output
        }
    )
    for row in output:
        row["mcnemar_p_holm"] = adjusted[f"{row['suite']}/{row['dataset']}"]
    for suite in ("source", "tableeg"):
        output.append(
            _complementarity_aggregate(
                by_suite[suite],
                scope="suite",
                suite=suite,
                dataset="ALL",
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=_derived_seed(bootstrap_seed, "complementarity", suite),
                confidence=confidence,
            )
        )
    output.append(
        _complementarity_aggregate(
            rows,
            scope="micro",
            suite="ALL",
            dataset="MICRO",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_derived_seed(bootstrap_seed, "complementarity", "micro"),
            confidence=confidence,
        )
    )
    output.append(
        _complementarity_aggregate(
            rows,
            scope="macro",
            suite="ALL",
            dataset="MACRO",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=_derived_seed(bootstrap_seed, "complementarity", "macro"),
            confidence=confidence,
            macro=True,
        )
    )
    return output


def _failure_state(row: Mapping[str, object], arm: str) -> str:
    """Return one mutually exclusive invalid subtype (or ``valid``)."""

    status = str(row[f"{arm}_status"]).strip().lower()
    parse_status = str(row[f"{arm}_parse_status"]).strip().lower()
    decision = str(row[f"{arm}_decision"]).strip().lower()
    if status not in _SUCCESS_STATUSES or "provider" in parse_status:
        return "provider_failure"
    if _is_parse_failure(parse_status):
        return "parse_failure"
    if not decision or "missing" in parse_status:
        return "missing"
    if decision == "abstain" or "abstain" in parse_status:
        return "abstain"
    prediction = normalize_for_match(row.get(f"{arm}_prediction", ""))
    dirty_value = normalize_for_match(row.get("dirty_value", ""))
    if not prediction:
        return "empty"
    if prediction == dirty_value:
        return "unchanged"
    if not bool(row[f"{arm}_valid"]):
        return "other_invalid"
    return "valid"


def _group_values(row: Mapping[str, object]) -> np.ndarray:
    singleton = bool(row["singleton_correct"])
    structured = bool(row["structured_correct"])
    random = bool(row["random_correct"])
    values: dict[str, float] = {
        "singleton_accuracy": float(singleton),
        "structured_accuracy": float(structured),
        "random_accuracy": float(random),
        "structured_rescue_rate": float(structured and not singleton),
        "structured_interference_rate": float(singleton and not structured),
        "random_rescue_rate": float(random and not singleton),
        "random_interference_rate": float(singleton and not random),
        "structured_minus_singleton": float(structured) - float(singleton),
        "random_minus_singleton": float(random) - float(singleton),
        "structured_minus_random": float(structured) - float(random),
    }
    for arm in ("singleton", "structured", "random"):
        values[f"{arm}_invalid_rate"] = float(not bool(row[f"{arm}_valid"]))
        failure_state = _failure_state(row, arm)
        for state in (
            "provider_failure",
            "parse_failure",
            "missing",
            "abstain",
            "empty",
            "unchanged",
            "other_invalid",
        ):
            values[f"{arm}_{state}_rate"] = float(failure_state == state)
    return np.asarray([values[field] for field in GROUP_RATE_FIELDS], dtype=float)


def _crossed_multiplier_samples(
    rows: Sequence[Mapping[str, object]],
    *,
    field_indices: Sequence[int],
    cluster_keys: Sequence[str],
    replicates: int,
    seed: int,
    macro: bool,
) -> np.ndarray:
    """Vectorized crossed Exp(1) multiplier samples for selected linear metrics."""

    if not rows:
        return np.full((replicates, len(field_indices)), np.nan)
    values = np.stack([_group_values(row)[list(field_indices)] for row in rows], axis=0)
    codes_by_dimension: list[np.ndarray] = []
    levels_by_dimension: list[int] = []
    for key in cluster_keys:
        labels = [str(row[key]) for row in rows]
        levels = sorted(set(labels))
        index = {level: position for position, level in enumerate(levels)}
        codes_by_dimension.append(
            np.fromiter((index[label] for label in labels), dtype=np.intp, count=len(rows))
        )
        levels_by_dimension.append(len(levels))
    dataset_codes: list[np.ndarray] = []
    if macro:
        by_dataset: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            by_dataset[(str(row["suite"]), str(row["dataset"]))].append(index)
        dataset_codes = [np.asarray(indices, dtype=np.intp) for _, indices in sorted(by_dataset.items())]

    rng = np.random.default_rng(int(seed))
    samples = np.empty((replicates, len(field_indices)), dtype=float)
    for replicate in range(replicates):
        weights = np.ones(len(rows), dtype=float)
        for level_count, codes in zip(levels_by_dimension, codes_by_dimension):
            weights *= rng.exponential(1.0, level_count)[codes]
        if macro:
            estimates = []
            for indices in dataset_codes:
                selected_weights = weights[indices]
                denominator = float(np.sum(selected_weights))
                estimates.append(
                    np.dot(selected_weights, values[indices]) / denominator
                    if denominator
                    else np.full(len(field_indices), np.nan)
                )
            samples[replicate] = np.nanmean(np.stack(estimates, axis=0), axis=0)
        else:
            denominator = float(np.sum(weights))
            samples[replicate] = (
                np.dot(weights, values) / denominator
                if denominator
                else np.full(len(field_indices), np.nan)
            )
    return samples


def _group_point_values(
    rows: Sequence[Mapping[str, object]], *, macro: bool
) -> np.ndarray:
    values = np.stack([_group_values(row) for row in rows], axis=0)
    if not macro:
        return np.mean(values, axis=0)
    by_dataset: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_dataset[(str(row["suite"]), str(row["dataset"]))].append(index)
    return np.mean(
        np.stack(
            [np.mean(values[np.asarray(indices, dtype=np.intp)], axis=0) for _, indices in sorted(by_dataset.items())],
            axis=0,
        ),
        axis=0,
    )


def _group_ci(
    rows: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
    macro: bool,
    include_cell_cluster: bool,
) -> dict[str, tuple[float, float]]:
    field_index = {field: index for index, field in enumerate(GROUP_RATE_FIELDS)}
    schemes = (
        (
            (
                "singleton_accuracy",
                "singleton_invalid_rate",
                "singleton_provider_failure_rate",
                "singleton_parse_failure_rate",
                "singleton_missing_rate",
                "singleton_abstain_rate",
                "singleton_empty_rate",
                "singleton_unchanged_rate",
                "singleton_other_invalid_rate",
            ),
            ("row_cluster",),
            "singleton",
        ),
        (
            (
                "structured_accuracy",
                "structured_rescue_rate",
                "structured_interference_rate",
                "structured_minus_singleton",
                "structured_invalid_rate",
                "structured_provider_failure_rate",
                "structured_parse_failure_rate",
                "structured_missing_rate",
                "structured_abstain_rate",
                "structured_empty_rate",
                "structured_unchanged_rate",
                "structured_other_invalid_rate",
            ),
            ("row_cluster", "structured_physical_query_id"),
            "structured",
        ),
        (
            (
                "random_accuracy",
                "random_rescue_rate",
                "random_interference_rate",
                "random_minus_singleton",
                "random_invalid_rate",
                "random_provider_failure_rate",
                "random_parse_failure_rate",
                "random_missing_rate",
                "random_abstain_rate",
                "random_empty_rate",
                "random_unchanged_rate",
                "random_other_invalid_rate",
            ),
            ("row_cluster", "random_physical_query_id"),
            "random",
        ),
        (
            ("structured_minus_random",),
            (
                "row_cluster",
                "structured_physical_query_id",
                "random_physical_query_id",
            ),
            "structured_random",
        ),
    )
    output: dict[str, tuple[float, float]] = {}
    for fields, base_keys, label in schemes:
        cluster_keys = (*base_keys, "cell_id") if include_cell_cluster else base_keys
        samples = _crossed_multiplier_samples(
            rows,
            field_indices=[field_index[field] for field in fields],
            cluster_keys=cluster_keys,
            replicates=bootstrap_replicates,
            seed=_derived_seed(bootstrap_seed, "group", label),
            macro=macro,
        )
        for index, field in enumerate(fields):
            output[field] = _interval(samples[:, index], confidence)
    if set(output) != set(GROUP_RATE_FIELDS):
        raise RuntimeError("group CI schemes do not cover every reported rate")
    return output


def _population_coverage(
    rows: Sequence[Mapping[str, object]],
    *,
    complementarity_counts: Mapping[tuple[str, str], int],
    macro: bool,
) -> tuple[int, int, float]:
    by_dataset: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        by_dataset[(str(row["suite"]), str(row["dataset"]))].add(str(row["cell_id"]))
    unique_cells = sum(len(cells) for cells in by_dataset.values())
    population = sum(complementarity_counts[key] for key in by_dataset)
    if macro:
        coverage = float(
            np.mean(
                [len(cells) / complementarity_counts[key] for key, cells in by_dataset.items()]
            )
        )
    else:
        coverage = _rate(unique_cells, population)
    return unique_cells, population, coverage


def _group_aggregate(
    rows: Sequence[Mapping[str, object]],
    *,
    scope: str,
    suite: str,
    dataset: str,
    source_view: str,
    group_size: int | str,
    complementarity_counts: Mapping[tuple[str, str], int],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
    macro: bool = False,
    include_cell_cluster: bool = False,
) -> dict[str, object]:
    point = _group_point_values(rows, macro=macro)
    intervals = _group_ci(
        rows,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
        macro=macro,
        include_cell_cluster=include_cell_cluster,
    )
    unique_cells, population_cells, coverage = _population_coverage(
        rows, complementarity_counts=complementarity_counts, macro=macro
    )
    result: dict[str, object] = {
        "scope": scope,
        "aggregation": "unweighted_dataset_mean" if macro else "micro_over_cell_incidences",
        "suite": suite,
        "dataset": dataset,
        "source_view": source_view,
        "group_size": group_size,
        "eligible_cell_incidences": len(rows),
        "eligible_unique_cells": unique_cells,
        "population_cells": population_cells,
        "coverage_rate": coverage,
        "cell_cluster_in_ci": include_cell_cluster,
    }
    for index, field in enumerate(GROUP_RATE_FIELDS):
        result[field] = float(point[index])
        result[f"{field}_ci_low"] = intervals[field][0]
        result[f"{field}_ci_high"] = intervals[field][1]
    # Protocol identities; these must hold both pointwise and in every
    # structured/random multiplier replicate because each pair uses one scheme.
    if not math.isclose(
        float(result["structured_minus_singleton"]),
        float(result["structured_rescue_rate"])
        - float(result["structured_interference_rate"]),
        abs_tol=1e-12,
    ):
        raise RuntimeError("structured transition identity failed")
    if not math.isclose(
        float(result["random_minus_singleton"]),
        float(result["random_rescue_rate"]) - float(result["random_interference_rate"]),
        abs_tol=1e-12,
    ):
        raise RuntimeError("random transition identity failed")
    if not math.isclose(
        float(result["structured_minus_singleton"]),
        float(result["random_minus_singleton"])
        + float(result["structured_minus_random"]),
        abs_tol=1e-12,
    ):
        raise RuntimeError("three-arm accuracy decomposition failed")
    return result


def _validate_cross_ledger_identity(
    complementarity: Sequence[Mapping[str, object]],
    group: Sequence[Mapping[str, object]],
) -> None:
    singleton = {
        str(row["cell_id"]): {
            "suite": str(row["suite"]),
            "dataset": str(row["dataset"]),
            "base_family": str(row["base_family"]),
            "row_cluster": str(row["row_cluster"]),
            "column": str(row["column"]),
            "dirty_value": str(row["dirty_value"]),
            "clean_value": str(row["clean_value"]),
            "prediction": str(row["llm_prediction"]),
            "valid": bool(row["llm_valid"]),
            "correct": bool(row["llm_correct"]),
            "status": str(row["llm_status"]),
            "parse_status": str(row["llm_parse_status"]),
            "decision": str(row["llm_decision"]),
            "input_tokens": int(row["llm_observed_input_tokens"]),
            "output_tokens": int(row["llm_observed_output_tokens"]),
            "total_tokens": int(row["llm_observed_total_tokens"]),
        }
        for row in complementarity
    }
    for row in group:
        cell_id = str(row["cell_id"])
        expected = singleton.get(cell_id)
        observed = {
            "suite": str(row["suite"]),
            "dataset": str(row["dataset"]),
            "base_family": str(row["base_family"]),
            "row_cluster": str(row["row_cluster"]),
            "column": str(row["column"]),
            "dirty_value": str(row["dirty_value"]),
            "clean_value": str(row["clean_value"]),
            "prediction": str(row["singleton_prediction"]),
            "valid": bool(row["singleton_valid"]),
            "correct": bool(row["singleton_correct"]),
            "status": str(row["singleton_status"]),
            "parse_status": str(row["singleton_parse_status"]),
            "decision": str(row["singleton_decision"]),
            "input_tokens": int(row["singleton_query_observed_input_tokens"]),
            "output_tokens": int(row["singleton_query_observed_output_tokens"]),
            "total_tokens": int(row["singleton_query_observed_total_tokens"]),
        }
        if expected is None or observed != expected:
            raise ValueError(f"group singleton control does not match complementarity: {cell_id}")


def _parse_api_cost(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    identities: set[str] = set()
    integer_fields = (
        "attempts",
        "observed_input_tokens",
        "observed_output_tokens",
        "observed_total_tokens",
        "usage_observed_attempts",
        "unknown_usage_attempts",
        "estimated_prompt_tokens",
        "estimated_completion_tokens",
        "estimated_total_tokens",
        "logical_query_mappings",
    )
    for source in rows:
        row: dict[str, object] = dict(source)
        physical_id = str(row["physical_query_id"])
        if not physical_id or physical_id in identities:
            raise ValueError(f"duplicate or empty API cost physical_query_id: {physical_id!r}")
        identities.add(physical_id)
        if not str(row["provider_request_hash"]):
            raise ValueError(f"empty provider request hash in API cost audit: {physical_id}")
        for field in integer_fields:
            row[field] = _integer(source[field], field=field, blank=0)
        if int(row["logical_query_mappings"]) <= 0:
            raise ValueError(f"physical request has no logical mappings: {physical_id}")
        row["latency_seconds"] = _optional_float(source, ("latency_seconds",))
        if row["latency_seconds"] is None:
            raise ValueError(f"missing latency_seconds in API cost audit: {physical_id}")
        parsed.append(row)
    return parsed


def _validate_group_cost_against_audit(
    group: Sequence[Mapping[str, object]],
    api_cost: Sequence[Mapping[str, object]],
) -> None:
    audit = {str(row["physical_query_id"]): row for row in api_cost}
    for row in group:
        for arm in ("singleton", "structured", "random"):
            physical_id = str(row[f"{arm}_physical_query_id"])
            expected = audit.get(physical_id)
            if expected is None:
                raise ValueError(f"group ledger references absent API cost row: {physical_id}")
            comparisons: tuple[tuple[str, object, object], ...] = (
                ("status", row[f"{arm}_status"], expected["status"]),
                (
                    "input_tokens",
                    row[f"{arm}_query_observed_input_tokens"],
                    expected["observed_input_tokens"],
                ),
                (
                    "output_tokens",
                    row[f"{arm}_query_observed_output_tokens"],
                    expected["observed_output_tokens"],
                ),
                (
                    "total_tokens",
                    row[f"{arm}_query_observed_total_tokens"],
                    expected["observed_total_tokens"],
                ),
            )
            for field, observed, expected_value in comparisons:
                if str(observed) != str(expected_value):
                    raise ValueError(
                        f"group/API cost {field} drift for physical query {physical_id}"
                    )
            optional_integer_fields = (
                (f"{arm}_query_attempts", "attempts"),
                (f"{arm}_query_usage_observed_attempts", "usage_observed_attempts"),
                (f"{arm}_query_unknown_usage_attempts", "unknown_usage_attempts"),
            )
            for group_field, audit_field in optional_integer_fields:
                observed = _optional_integer(row, (group_field,))
                if observed is not None and observed != int(expected[audit_field]):
                    raise ValueError(
                        f"group/API cost {audit_field} drift for physical query {physical_id}"
                    )
            latency = _optional_float(row, (f"{arm}_query_latency_seconds",))
            if latency is not None and not math.isclose(
                latency, float(expected["latency_seconds"]), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"group/API cost latency drift for physical query {physical_id}")


def _query_cost_entries(
    rows: Sequence[Mapping[str, object]], arm: str
) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    for row in rows:
        physical_id = str(row[f"{arm}_physical_query_id"])
        attempts = _optional_integer(
            row, (f"{arm}_query_attempts", f"{arm}_attempts")
        )
        candidate = {
            "observed_input_tokens": int(row[f"{arm}_query_observed_input_tokens"]),
            "observed_output_tokens": int(row[f"{arm}_query_observed_output_tokens"]),
            "observed_total_tokens": int(row[f"{arm}_query_observed_total_tokens"]),
            "status": str(row[f"{arm}_status"]),
            "attempts": attempts,
            "latency_seconds": _optional_float(
                row,
                (f"{arm}_query_latency_seconds", f"{arm}_latency_seconds"),
            ),
            "usage_observed_attempts": _optional_integer(
                row,
                (
                    f"{arm}_query_usage_observed_attempts",
                    f"{arm}_usage_observed_attempts",
                ),
            ),
            "unknown_usage_attempts": _optional_integer(
                row,
                (
                    f"{arm}_query_unknown_usage_attempts",
                    f"{arm}_unknown_usage_attempts",
                ),
            ),
        }
        previous = entries.get(physical_id)
        if previous is not None and previous != candidate:
            raise ValueError(f"inconsistent usage/status for physical query {physical_id}")
        entries[physical_id] = candidate
    return entries


def _cost_rows_for_scope(
    rows: Sequence[Mapping[str, object]],
    *,
    scope: str,
    suite: str,
    dataset: str,
    source_view: str,
    group_size: int | str,
) -> list[dict[str, object]]:
    costs: dict[str, dict[str, object]] = {}
    for arm in ("singleton", "structured", "random"):
        entries = _query_cost_entries(rows, arm)
        logical_to_physical: dict[str, str] = {}
        for row in rows:
            logical_id = str(row[f"{arm}_logical_query_id"])
            physical_id = str(row[f"{arm}_physical_query_id"])
            previous = logical_to_physical.setdefault(logical_id, physical_id)
            if previous != physical_id:
                raise ValueError(f"logical query maps to multiple physical requests: {logical_id}")
        logical_calls = len(logical_to_physical)
        input_tokens = sum(int(entry["observed_input_tokens"]) for entry in entries.values())
        output_tokens = sum(int(entry["observed_output_tokens"]) for entry in entries.values())
        total_tokens = sum(int(entry["observed_total_tokens"]) for entry in entries.values())
        logical_input_tokens = sum(
            int(entries[physical_id]["observed_input_tokens"])
            for physical_id in logical_to_physical.values()
        )
        logical_output_tokens = sum(
            int(entries[physical_id]["observed_output_tokens"])
            for physical_id in logical_to_physical.values()
        )
        logical_total_tokens = sum(
            int(entries[physical_id]["observed_total_tokens"])
            for physical_id in logical_to_physical.values()
        )
        failures = sum(
            str(entry["status"]).strip().lower() not in {"success", "ok", "completed"}
            for entry in entries.values()
        )
        known_attempts = [entry["attempts"] for entry in entries.values()]
        retries: int | None = None
        if all(value is not None for value in known_attempts):
            retries = sum(max(0, int(value) - 1) for value in known_attempts)
        latency_values = [entry["latency_seconds"] for entry in entries.values()]
        latency = (
            sum(float(value) for value in latency_values)
            if all(value is not None for value in latency_values)
            else None
        )
        observed_attempt_values = [
            entry["usage_observed_attempts"] for entry in entries.values()
        ]
        observed_attempts = (
            sum(int(value) for value in observed_attempt_values)
            if all(value is not None for value in observed_attempt_values)
            else None
        )
        unknown_attempt_values = [
            entry["unknown_usage_attempts"] for entry in entries.values()
        ]
        unknown_attempts = (
            sum(int(value) for value in unknown_attempt_values)
            if all(value is not None for value in unknown_attempt_values)
            else None
        )
        correct = sum(bool(row[f"{arm}_correct"]) for row in rows)
        costs[arm] = {
            "scope": scope,
            "suite": suite,
            "dataset": dataset,
            "source_view": source_view,
            "group_size": group_size,
            "arm": arm,
            "eligible_cell_incidences": len(rows),
            "correct_repairs": correct,
            "logical_calls": logical_calls,
            "physical_calls": len(entries),
            "observed_input_tokens": input_tokens,
            "observed_output_tokens": output_tokens,
            "observed_total_tokens": total_tokens,
            "logical_observed_input_tokens": logical_input_tokens,
            "logical_observed_output_tokens": logical_output_tokens,
            "logical_observed_total_tokens": logical_total_tokens,
            "provider_failures": failures,
            "attempts": sum(int(value) for value in known_attempts)
            if all(value is not None for value in known_attempts)
            else None,
            "retries": retries,
            "latency_seconds": latency,
            "usage_observed_attempts": observed_attempts,
            "unknown_usage_attempts": unknown_attempts,
            "tokens_per_correct_repair": _rate(total_tokens, correct),
            "cost_basis": "physical_dedup_within_arm_scope",
            "attribution_note": (
                "arm/scoped costs are non-additive; use run_physical_union for exact run spend"
            ),
        }
    singleton = costs["singleton"]
    for arm, cost in costs.items():
        if arm == "singleton":
            cost["token_saving_vs_singleton"] = 0.0
            cost["logical_token_saving_vs_singleton"] = 0.0
            cost["request_reduction_vs_singleton"] = 0.0
            cost["logical_request_reduction_vs_singleton"] = 0.0
        else:
            cost["token_saving_vs_singleton"] = 1.0 - _rate(
                int(cost["observed_total_tokens"]), int(singleton["observed_total_tokens"])
            )
            cost["logical_token_saving_vs_singleton"] = 1.0 - _rate(
                int(cost["logical_observed_total_tokens"]),
                int(singleton["logical_observed_total_tokens"]),
            )
            cost["request_reduction_vs_singleton"] = 1.0 - _rate(
                int(cost["physical_calls"]), int(singleton["physical_calls"])
            )
            cost["logical_request_reduction_vs_singleton"] = 1.0 - _rate(
                int(cost["logical_calls"]), int(singleton["logical_calls"])
            )
    return [costs[arm] for arm in ("singleton", "structured", "random")]


def _cost_fields_for_metric(
    metric: dict[str, object], costs: Sequence[Mapping[str, object]]
) -> None:
    indexed = {str(row["arm"]): row for row in costs}
    for arm in ("singleton", "structured", "random"):
        cost = indexed[arm]
        for field in (
            "logical_calls",
            "physical_calls",
            "observed_input_tokens",
            "observed_output_tokens",
            "observed_total_tokens",
            "logical_observed_input_tokens",
            "logical_observed_output_tokens",
            "logical_observed_total_tokens",
            "provider_failures",
            "attempts",
            "retries",
            "latency_seconds",
            "usage_observed_attempts",
            "unknown_usage_attempts",
            "tokens_per_correct_repair",
        ):
            metric[f"{arm}_{field}"] = cost[field]
    for arm in ("structured", "random"):
        metric[f"{arm}_token_saving"] = indexed[arm]["token_saving_vs_singleton"]
        metric[f"{arm}_logical_token_saving"] = indexed[arm][
            "logical_token_saving_vs_singleton"
        ]
        metric[f"{arm}_request_reduction"] = indexed[arm][
            "request_reduction_vs_singleton"
        ]
        metric[f"{arm}_logical_request_reduction"] = indexed[arm][
            "logical_request_reduction_vs_singleton"
        ]


def _condition_cell_sets(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str, int], set[str]]:
    result: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
    for row in rows:
        key = (
            str(row["suite"]),
            str(row["dataset"]),
            str(row["source_view"]),
            int(row["group_size"]),
        )
        result[key].add(f"{key[0]}/{key[1]}/{row['cell_id']}")
    return dict(result)


def _sets_for_scope(
    condition_sets: Mapping[tuple[str, str, str, int], set[str]],
    *,
    scope: str,
    suite: str,
    dataset: str,
    view: str,
    size: int,
) -> list[set[str]]:
    if scope == "dataset":
        return [set(condition_sets[(suite, dataset, view, size)])]
    dataset_order = [
        key
        for key in SELECTED_DATASETS
        if scope in {"micro", "macro"} or key[0] == suite
    ]
    selected = [set(condition_sets[(*key, view, size)]) for key in dataset_order]
    if scope == "macro":
        return selected
    return [set().union(*selected)]


def _dataset_keys_for_scope(
    *, scope: str, suite: str, dataset: str
) -> list[tuple[str, str]]:
    if scope == "dataset":
        return [(suite, dataset)]
    return [
        key
        for key in SELECTED_DATASETS
        if scope in {"micro", "macro"} or key[0] == suite
    ]


def _overlap_summary(
    first: Sequence[set[str]], second: Sequence[set[str]]
) -> tuple[int, int, float, float, float]:
    if len(first) != len(second):
        raise ValueError("population overlap operands have different dataset counts")
    intersections: list[int] = []
    unions: list[int] = []
    first_rates: list[float] = []
    second_rates: list[float] = []
    jaccards: list[float] = []
    for left, right in zip(first, second):
        intersection = len(left & right)
        union = len(left | right)
        intersections.append(intersection)
        unions.append(union)
        first_rates.append(_rate(intersection, len(left)))
        second_rates.append(_rate(intersection, len(right)))
        jaccards.append(_rate(intersection, union))
    return (
        sum(intersections),
        sum(unions),
        float(np.mean(first_rates)),
        float(np.mean(second_rates)),
        float(np.mean(jaccards)),
    )


def _attach_population_overlap(
    metric: dict[str, object],
    condition_sets: Mapping[tuple[str, str, str, int], set[str]],
    rows_by_condition: Mapping[
        tuple[str, str, str, int], Sequence[Mapping[str, object]]
    ],
) -> None:
    if metric["source_view"] not in PRIMARY_VIEWS:
        return
    scope = str(metric["scope"])
    suite = str(metric["suite"])
    dataset = str(metric["dataset"])
    view = str(metric["source_view"])
    size = int(metric["group_size"])
    current = _sets_for_scope(
        condition_sets,
        scope=scope,
        suite=suite,
        dataset=dataset,
        view=view,
        size=size,
    )
    dataset_keys = _dataset_keys_for_scope(
        scope=scope, suite=suite, dataset=dataset
    )
    field_index = {field: index for index, field in enumerate(GROUP_RATE_FIELDS)}
    sensitivity_fields = (
        "singleton_accuracy",
        "structured_accuracy",
        "random_accuracy",
        "structured_rescue_rate",
        "structured_interference_rate",
        "structured_minus_singleton",
    )
    for other_size in GROUP_SIZES:
        other = _sets_for_scope(
            condition_sets,
            scope=scope,
            suite=suite,
            dataset=dataset,
            view=view,
            size=other_size,
        )
        intersection, union, current_rate, other_rate, jaccard = _overlap_summary(
            current, other
        )
        prefix = f"overlap_with_k{other_size}"
        metric[f"{prefix}_cells"] = intersection
        metric[f"{prefix}_union_cells"] = union
        metric[f"{prefix}_rate"] = current_rate
        metric[f"{prefix}_other_population_rate"] = other_rate
        metric[f"overlap_with_k{other_size}_jaccard"] = jaccard
        common_by_dataset = {
            key: condition_sets[(*key, view, size)]
            & condition_sets[(*key, view, other_size)]
            for key in dataset_keys
        }
        common_rows = [
            row
            for key in dataset_keys
            for row in rows_by_condition[(*key, view, size)]
            if f"{key[0]}/{key[1]}/{row['cell_id']}" in common_by_dataset[key]
        ]
        datasets_with_common = sum(bool(common_by_dataset[key]) for key in dataset_keys)
        metric[f"{prefix}_common_population_datasets"] = datasets_with_common
        if common_rows and (scope != "macro" or datasets_with_common == len(dataset_keys)):
            common_point = _group_point_values(common_rows, macro=scope == "macro")
            for field in sensitivity_fields:
                metric[f"{prefix}_common_{field}"] = float(
                    common_point[field_index[field]]
                )
        else:
            for field in sensitivity_fields:
                metric[f"{prefix}_common_{field}"] = math.nan
    paired_view = "semantic" if view == "pattern" else "pattern"
    other = _sets_for_scope(
        condition_sets,
        scope=scope,
        suite=suite,
        dataset=dataset,
        view=paired_view,
        size=size,
    )
    intersection, union, current_rate, other_rate, jaccard = _overlap_summary(
        current, other
    )
    metric["paired_view_overlap_cells"] = intersection
    metric["paired_view_overlap_union_cells"] = union
    metric["paired_view_overlap_rate"] = current_rate
    metric["paired_view_overlap_other_population_rate"] = other_rate
    metric["paired_view_overlap_jaccard"] = jaccard


def _attach_macro_cost_fields(
    macro: dict[str, object], dataset_metrics: Sequence[Mapping[str, object]]
) -> None:
    sum_fields = (
        "logical_calls",
        "physical_calls",
        "observed_input_tokens",
        "observed_output_tokens",
        "observed_total_tokens",
        "logical_observed_input_tokens",
        "logical_observed_output_tokens",
        "logical_observed_total_tokens",
        "provider_failures",
    )
    optional_sum_fields = (
        "attempts",
        "retries",
        "latency_seconds",
        "usage_observed_attempts",
        "unknown_usage_attempts",
    )
    mean_fields = ("tokens_per_correct_repair",)
    for arm in ("singleton", "structured", "random"):
        for field in sum_fields:
            macro[f"{arm}_{field}"] = sum(
                int(row[f"{arm}_{field}"]) for row in dataset_metrics
            )
        for field in optional_sum_fields:
            values = [row.get(f"{arm}_{field}") for row in dataset_metrics]
            macro[f"{arm}_{field}"] = (
                sum(float(value) for value in values)
                if all(value is not None for value in values)
                else None
            )
        for field in mean_fields:
            macro[f"{arm}_{field}"] = float(
                np.mean([float(row[f"{arm}_{field}"]) for row in dataset_metrics])
            )
    for arm in ("structured", "random"):
        macro[f"{arm}_token_saving"] = float(
            np.mean([float(row[f"{arm}_token_saving"]) for row in dataset_metrics])
        )
        macro[f"{arm}_logical_token_saving"] = float(
            np.mean(
                [float(row[f"{arm}_logical_token_saving"]) for row in dataset_metrics]
            )
        )
        macro[f"{arm}_request_reduction"] = float(
            np.mean([float(row[f"{arm}_request_reduction"]) for row in dataset_metrics])
        )
        macro[f"{arm}_logical_request_reduction"] = float(
            np.mean(
                [float(row[f"{arm}_logical_request_reduction"]) for row in dataset_metrics]
            )
        )


def _build_group_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    complementarity_counts: Mapping[tuple[str, str], int],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[tuple[dict[str, object], Sequence[Mapping[str, object]]]],
]:
    by_dataset: dict[
        tuple[str, str, str, int], list[Mapping[str, object]]
    ] = defaultdict(list)
    for row in rows:
        by_dataset[
            (
                str(row["suite"]),
                str(row["dataset"]),
                str(row["source_view"]),
                int(row["group_size"]),
            )
        ].append(row)
    metrics: list[dict[str, object]] = []
    costs: list[dict[str, object]] = []
    metric_inputs: list[tuple[dict[str, object], Sequence[Mapping[str, object]]]] = []
    for view in PRIMARY_VIEWS:
        for size in GROUP_SIZES:
            condition_dataset_metrics: list[dict[str, object]] = []
            condition_rows: list[Mapping[str, object]] = []
            for suite, dataset in SELECTED_DATASETS:
                subset = by_dataset[(suite, dataset, view, size)]
                condition_rows.extend(subset)
                metric = _group_aggregate(
                    subset,
                    scope="dataset",
                    suite=suite,
                    dataset=dataset,
                    source_view=view,
                    group_size=size,
                    complementarity_counts=complementarity_counts,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=_derived_seed(
                        bootstrap_seed, "group", "dataset", suite, dataset, view, size
                    ),
                    confidence=confidence,
                )
                scope_costs = _cost_rows_for_scope(
                    subset,
                    scope="dataset",
                    suite=suite,
                    dataset=dataset,
                    source_view=view,
                    group_size=size,
                )
                _cost_fields_for_metric(metric, scope_costs)
                metrics.append(metric)
                condition_dataset_metrics.append(metric)
                costs.extend(scope_costs)
                metric_inputs.append((metric, subset))
            for suite in ("source", "tableeg"):
                subset = [row for row in condition_rows if str(row["suite"]) == suite]
                metric = _group_aggregate(
                    subset,
                    scope="suite",
                    suite=suite,
                    dataset="ALL",
                    source_view=view,
                    group_size=size,
                    complementarity_counts=complementarity_counts,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=_derived_seed(
                        bootstrap_seed, "group", "suite", suite, view, size
                    ),
                    confidence=confidence,
                )
                scope_costs = _cost_rows_for_scope(
                    subset,
                    scope="suite",
                    suite=suite,
                    dataset="ALL",
                    source_view=view,
                    group_size=size,
                )
                _cost_fields_for_metric(metric, scope_costs)
                metrics.append(metric)
                costs.extend(scope_costs)
                metric_inputs.append((metric, subset))
            micro = _group_aggregate(
                condition_rows,
                scope="micro",
                suite="ALL",
                dataset="MICRO",
                source_view=view,
                group_size=size,
                complementarity_counts=complementarity_counts,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=_derived_seed(bootstrap_seed, "group", "micro", view, size),
                confidence=confidence,
            )
            scope_costs = _cost_rows_for_scope(
                condition_rows,
                scope="micro",
                suite="ALL",
                dataset="MICRO",
                source_view=view,
                group_size=size,
            )
            _cost_fields_for_metric(micro, scope_costs)
            metrics.append(micro)
            costs.extend(scope_costs)
            metric_inputs.append((micro, condition_rows))

            macro = _group_aggregate(
                condition_rows,
                scope="macro",
                suite="ALL",
                dataset="MACRO",
                source_view=view,
                group_size=size,
                complementarity_counts=complementarity_counts,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=_derived_seed(bootstrap_seed, "group", "macro", view, size),
                confidence=confidence,
                macro=True,
            )
            _attach_macro_cost_fields(macro, condition_dataset_metrics)
            metrics.append(macro)
            metric_inputs.append((macro, condition_rows))

    # Descriptive across-condition summaries use cell clustering because each
    # eligible cell can occur in several view/size conditions.
    all_rows = list(rows)
    overall_micro = _group_aggregate(
        all_rows,
        scope="micro_all_conditions",
        suite="ALL",
        dataset="MICRO",
        source_view="ALL",
        group_size="ALL",
        complementarity_counts=complementarity_counts,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=_derived_seed(bootstrap_seed, "group", "all", "micro"),
        confidence=confidence,
        include_cell_cluster=True,
    )
    overall_costs = _cost_rows_for_scope(
        all_rows,
        scope="micro_all_conditions",
        suite="ALL",
        dataset="MICRO",
        source_view="ALL",
        group_size="ALL",
    )
    _cost_fields_for_metric(overall_micro, overall_costs)
    metrics.append(overall_micro)
    costs.extend(overall_costs)
    metric_inputs.append((overall_micro, all_rows))

    overall_macro = _group_aggregate(
        all_rows,
        scope="macro_all_conditions",
        suite="ALL",
        dataset="MACRO",
        source_view="ALL",
        group_size="ALL",
        complementarity_counts=complementarity_counts,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=_derived_seed(bootstrap_seed, "group", "all", "macro"),
        confidence=confidence,
        macro=True,
        include_cell_cluster=True,
    )
    metrics.append(overall_macro)
    metric_inputs.append((overall_macro, all_rows))

    condition_sets = _condition_cell_sets(rows)
    for metric in metrics:
        _attach_population_overlap(metric, condition_sets, by_dataset)
    return metrics, costs, metric_inputs


def _transition_counts(
    rows: Sequence[Mapping[str, object]], contrast: str
) -> tuple[int, int, int, int]:
    if contrast == "structured_minus_singleton":
        treatment, baseline = "structured_correct", "singleton_correct"
    elif contrast == "random_minus_singleton":
        treatment, baseline = "random_correct", "singleton_correct"
    elif contrast == "structured_minus_random":
        treatment, baseline = "structured_correct", "random_correct"
    else:
        raise ValueError(f"unsupported contrast: {contrast}")
    counts = Counter((bool(row[treatment]), bool(row[baseline])) for row in rows)
    return (
        counts[(True, True)],
        counts[(True, False)],
        counts[(False, True)],
        counts[(False, False)],
    )


def _build_group_transitions(
    metric_inputs: Sequence[
        tuple[Mapping[str, object], Sequence[Mapping[str, object]]]
    ],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for metric, rows in metric_inputs:
        for contrast in CONTRASTS:
            both, treatment_only, baseline_only, neither = _transition_counts(rows, contrast)
            family = "descriptive"
            if metric["scope"] == "micro" and metric["source_view"] in PRIMARY_VIEWS:
                family = "group_primary_18"
            elif metric["scope"] == "dataset":
                family = "group_secondary_dataset_162"
            cluster_keys = {
                "structured_minus_singleton": "row_cluster×structured_physical_query_id",
                "random_minus_singleton": "row_cluster×random_physical_query_id",
                "structured_minus_random": (
                    "row_cluster×structured_physical_query_id×random_physical_query_id"
                ),
            }[contrast]
            if bool(metric["cell_cluster_in_ci"]):
                cluster_keys += "×cell_id"
            output.append(
                {
                    "experiment": "group",
                    "scope": metric["scope"],
                    "suite": metric["suite"],
                    "dataset": metric["dataset"],
                    "source_view": metric["source_view"],
                    "group_size": metric["group_size"],
                    "contrast": contrast,
                    "N": len(rows),
                    "both_correct": both,
                    "treatment_only_correct": treatment_only,
                    "baseline_only_correct": baseline_only,
                    "both_wrong": neither,
                    "effect": metric[contrast],
                    "effect_ci_low": metric[f"{contrast}_ci_low"],
                    "effect_ci_high": metric[f"{contrast}_ci_high"],
                    "mcnemar_p": (
                        exact_mcnemar(treatment_only, baseline_only)
                        if "macro" not in str(metric["scope"])
                        else math.nan
                    ),
                    "holm_adjusted_p": math.nan,
                    "test_family": family,
                    "cluster_keys": cluster_keys,
                }
            )
    families = {
        "group_primary_18": 18,
        "group_secondary_dataset_162": 162,
    }
    for family, expected in families.items():
        members = [row for row in output if row["test_family"] == family]
        if len(members) != expected:
            raise RuntimeError(f"{family} must contain {expected} tests, observed {len(members)}")
        adjusted = holm_adjust(
            {
                (
                    f"{row['suite']}/{row['dataset']}/{row['source_view']}/"
                    f"{row['group_size']}/{row['contrast']}"
                ): float(row["mcnemar_p"])
                for row in members
            }
        )
        for row in members:
            key = (
                f"{row['suite']}/{row['dataset']}/{row['source_view']}/"
                f"{row['group_size']}/{row['contrast']}"
            )
            row["holm_adjusted_p"] = adjusted[key]
    return output


def _complementarity_tests(
    metrics: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    tests: list[dict[str, object]] = []
    for row in metrics:
        if row["scope"] != "dataset":
            continue
        tests.append(
            {
                "experiment": "complementarity",
                "scope": "dataset",
                "suite": row["suite"],
                "dataset": row["dataset"],
                "source_view": "",
                "group_size": "",
                "contrast": "singleton_llm_minus_baran",
                "N": row["N"],
                "both_correct": row["n11"],
                "treatment_only_correct": row["n01"],
                "baseline_only_correct": row["n10"],
                "both_wrong": row["n00"],
                "effect": row["llm_minus_baran"],
                "effect_ci_low": row["llm_minus_baran_ci_low"],
                "effect_ci_high": row["llm_minus_baran_ci_high"],
                "mcnemar_p": row["mcnemar_p"],
                "holm_adjusted_p": row["mcnemar_p_holm"],
                "test_family": "complementarity_dataset_9",
                "cluster_keys": "row_cluster",
            }
        )
    if len(tests) != 9:
        raise RuntimeError("complementarity Holm family must contain nine tests")
    return tests


def _number(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _count(value: object) -> str:
    return f"{int(value):,}"


def _markdown_table(
    rows: Sequence[Mapping[str, object]], columns: Sequence[tuple[str, str]]
) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(
            str(row.get(field, "")).replace("|", "\\|").replace("\n", " ")
            for field, _ in columns
        )
        + " |"
        for row in rows
    ]
    return "\n".join((header, divider, *body))


def _figure(
    complementarity_metrics: Sequence[Mapping[str, object]],
    group_metrics: Sequence[Mapping[str, object]],
    *,
    pdf_path: Path,
    svg_path: Path,
) -> None:
    complementarity_micro = next(
        row for row in complementarity_metrics if row["scope"] == "micro"
    )
    macro_rows = [
        row
        for row in group_metrics
        if row["scope"] == "macro" and row["source_view"] in PRIMARY_VIEWS
    ]
    if len(macro_rows) != 6:
        raise RuntimeError("Panel B requires six frozen nine-dataset macro points")

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.hashsalt": REPORT_SCHEMA_VERSION,
        }
    ):
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(14.5, 4.35),
            gridspec_kw={"width_ratios": (1.02, 1.45, 1.2)},
        )

        # Panel A: full-population paired correctness matrix.
        matrix = np.asarray(
            [
                [complementarity_micro["n11"], complementarity_micro["n10"]],
                [complementarity_micro["n01"], complementarity_micro["n00"]],
            ],
            dtype=float,
        )
        total = float(np.sum(matrix))
        image = axes[0].imshow(matrix / total, cmap="Blues", vmin=0.0, vmax=1.0)
        del image
        labels = (("n11", "n10"), ("n01", "n00"))
        for row_index in range(2):
            for column_index in range(2):
                count = int(matrix[row_index, column_index])
                axes[0].text(
                    column_index,
                    row_index,
                    f"{labels[row_index][column_index]}\n{count:,}\n({count / total:.1%})",
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] / total > 0.35 else "black",
                    fontweight="semibold",
                )
        axes[0].set_xticks((0, 1), ("LLM correct", "LLM wrong"))
        axes[0].set_yticks((0, 1), ("Baran correct", "Baran wrong"))
        axes[0].set_title("A  Baran–singleton complementarity", loc="left", fontweight="bold")
        axes[0].set_xlabel(
            "Baran acc. "
            f"{float(complementarity_micro['baran_accuracy']):.1%}  |  "
            "LLM acc. "
            f"{float(complementarity_micro['llm_accuracy']):.1%}\n"
            "Offline opportunity upper bound "
            f"{float(complementarity_micro['oracle_union_upper_bound']):.1%}"
        )

        # Panel B: protocol-fixed unweighted macro over all nine datasets.
        colors = {"pattern": "#2c7fb8", "semantic": "#d95f0e"}
        markers = {2: "o", 4: "s", 8: "D"}
        saving_values = [
            float(row[f"{arm}_token_saving"])
            for row in macro_rows
            for arm in ("structured", "random")
        ]
        minimum_saving = min(saving_values)
        maximum_saving = max(saving_values)

        def bubble_size(saving: float) -> float:
            if math.isclose(minimum_saving, maximum_saving):
                return 170.0
            relative = (saving - minimum_saving) / (maximum_saving - minimum_saving)
            return 55.0 + 260.0 * relative

        maximum = 0.0
        for row in macro_rows:
            view = str(row["source_view"])
            size = int(row["group_size"])
            for arm, prefix, hollow in (
                ("structured", "structured", False),
                ("random", "random", True),
            ):
                x = float(row[f"{prefix}_rescue_rate"])
                y = float(row[f"{prefix}_interference_rate"])
                x_low = float(row[f"{prefix}_rescue_rate_ci_low"])
                x_high = float(row[f"{prefix}_rescue_rate_ci_high"])
                y_low = float(row[f"{prefix}_interference_rate_ci_low"])
                y_high = float(row[f"{prefix}_interference_rate_ci_high"])
                saving = float(row[f"{arm}_token_saving"])
                bubble = bubble_size(saving)
                color = colors[view]
                axes[1].errorbar(
                    x,
                    y,
                    xerr=np.asarray([[max(0.0, x - x_low)], [max(0.0, x_high - x)]]),
                    yerr=np.asarray([[max(0.0, y - y_low)], [max(0.0, y_high - y)]]),
                    fmt="none",
                    ecolor=color,
                    alpha=0.45 if hollow else 0.8,
                    capsize=2,
                    linewidth=0.9,
                )
                axes[1].scatter(
                    [x],
                    [y],
                    s=bubble,
                    marker=markers[size],
                    facecolors="none" if hollow else color,
                    edgecolors=color,
                    linewidths=1.25,
                    alpha=0.75 if hollow else 0.95,
                    label=(f"{view}, {arm}" if size == 2 else None),
                )
                if not hollow:
                    axes[1].annotate(f"k={size}", (x, y), xytext=(4, 4), textcoords="offset points")
                maximum = max(maximum, x_high, y_high)
        limit = max(0.05, maximum * 1.12)
        axes[1].plot((0.0, limit), (0.0, limit), linestyle="--", color="0.45", linewidth=1)
        axes[1].set_xlim(0.0, limit)
        axes[1].set_ylim(0.0, limit)
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].set_xlabel("Rescue rate")
        axes[1].set_ylabel("Interference rate")
        axes[1].set_title(
            "B  Group benefit–interference (9-dataset macro)",
            loc="left",
            fontweight="bold",
        )
        axes[1].legend(frameon=False, fontsize=8, loc="best")
        axes[1].text(
            0.98,
            0.02,
            "Larger bubble = greater token saving",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="0.35",
        )
        axes[1].grid(alpha=0.18)

        # Panel C is deliberately conceptual and contains no Router result.
        axes[2].axis("off")
        axes[2].set_title("C  Joint action-routing motivation", loc="left", fontweight="bold")
        actions = (
            "Baran fallback",
            "Singleton LLM",
            "Pattern groups",
            "Semantic groups",
            "Other structured groups",
        )
        y_positions = np.linspace(0.86, 0.34, len(actions))
        for label, y in zip(actions, y_positions):
            axes[2].text(
                0.08,
                y,
                label,
                transform=axes[2].transAxes,
                ha="left",
                va="center",
                bbox={"boxstyle": "round,pad=0.35", "fc": "#f2f2f2", "ec": "#777777"},
            )
            axes[2].annotate(
                "",
                xy=(0.62, 0.18),
                xytext=(0.48, y),
                xycoords=axes[2].transAxes,
                arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 1.0},
            )
        axes[2].text(
            0.63,
            0.18,
            "Joint budgeted\nquery-action routing",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.55", "fc": "#d9edf7", "ec": "#2c7fb8"},
        )
        axes[2].text(
            0.5,
            0.04,
            "Concept only — no Router outcomes are used",
            transform=axes[2].transAxes,
            ha="center",
            color="0.35",
            fontsize=8,
        )

        figure.tight_layout(w_pad=2.0)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            metadata={"Creator": REPORT_SCHEMA_VERSION, "CreationDate": None, "ModDate": None},
        )
        figure.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            metadata={"Creator": REPORT_SCHEMA_VERSION, "Date": None},
        )
        plt.close(figure)


def _report_markdown(
    complementarity_metrics: Sequence[Mapping[str, object]],
    group_metrics: Sequence[Mapping[str, object]],
    transitions: Sequence[Mapping[str, object]],
    costs: Sequence[Mapping[str, object]],
) -> str:
    complement_rows = []
    for row in complementarity_metrics:
        complement_rows.append(
            {
                "population": (
                    f"{row['suite']}/{row['dataset']}"
                    if row["scope"] == "dataset"
                    else f"{row['scope']}:{row['suite']}"
                ),
                "N": _count(row["N"]),
                "baran": _number(row["baran_accuracy"]),
                "llm": _number(row["llm_accuracy"]),
                "upper": _number(row["oracle_union_upper_bound"]),
                "salvage": _number(row["llm_salvage_opportunity"]),
                "overwrite": _number(row["overwrite_risk"]),
                "holm": _number(row["mcnemar_p_holm"]),
            }
        )
    macro_rows = []
    for row in group_metrics:
        if row["scope"] != "macro" or row["source_view"] not in PRIMARY_VIEWS:
            continue
        macro_rows.append(
            {
                "condition": f"{row['source_view']}/k={row['group_size']}",
                "coverage": _number(row["coverage_rate"]),
                "S": _number(row["singleton_accuracy"]),
                "G": _number(row["structured_accuracy"]),
                "R": _number(row["random_accuracy"]),
                "rescue": _number(row["structured_rescue_rate"]),
                "interference": _number(row["structured_interference_rate"]),
                "G-S": _number(row["structured_minus_singleton"]),
                "R-S": _number(row["random_minus_singleton"]),
                "G-R": _number(row["structured_minus_random"]),
                "saving": _number(row["structured_token_saving"]),
            }
        )
    primary = [row for row in transitions if row["test_family"] == "group_primary_18"]
    primary_rows = [
        {
            "condition": f"{row['source_view']}/k={row['group_size']}",
            "contrast": row["contrast"],
            "effect": _number(row["effect"]),
            "CI": f"[{_number(row['effect_ci_low'])}, {_number(row['effect_ci_high'])}]",
            "Holm p": _number(row["holm_adjusted_p"]),
        }
        for row in primary
    ]
    micro_costs = [
        row
        for row in costs
        if (row["scope"] == "micro" and row["arm"] in {"structured", "random"})
        or row["scope"] == "run_physical_union"
    ]
    cost_rows = [
        {
            "condition": (
                "exact run physical union"
                if row["scope"] == "run_physical_union"
                else f"{row['source_view']}/k={row['group_size']}/{row['arm']}"
            ),
            "logical": _count(row["logical_calls"]),
            "physical": _count(row["physical_calls"]),
            "tokens": _count(row["observed_total_tokens"]),
            "logical tokens": _count(row["logical_observed_total_tokens"]),
            "estimated": (
                _count(row["estimated_total_tokens"])
                if row.get("estimated_total_tokens") is not None
                else "—"
            ),
            "token saving": _number(row["token_saving_vs_singleton"]),
            "request reduction": _number(row["request_reduction_vs_singleton"]),
            "failures": _count(row["provider_failures"]),
            "retries": (
                _count(row["retries"]) if row.get("retries") is not None else "—"
            ),
            "unknown usage": (
                _count(row["unknown_usage_attempts"])
                if row.get("unknown_usage_attempts") is not None
                else "—"
            ),
        }
        for row in micro_costs
    ]
    return "\n".join(
        (
            "# Introduction Motivation Evidence Report",
            "",
            "This report is rebuilt offline from the two finalized paired cell ledgers. "
            "Invalid, missing, abstaining, empty, unchanged, and provider-failed repairs "
            "remain incorrect. The oracle union below is an **offline opportunity upper "
            "bound**, not an executable method result.",
            "",
            "## Experiment 1: Baran–singleton complementarity",
            "",
            _markdown_table(
                complement_rows,
                (
                    ("population", "Population"),
                    ("N", "N"),
                    ("baran", "Baran acc."),
                    ("llm", "LLM acc."),
                    ("upper", "Oracle union upper bound"),
                    ("salvage", "LLM-only opportunity"),
                    ("overwrite", "Overwrite risk"),
                    ("holm", "Dataset Holm p"),
                ),
            ),
            "",
            "The nine dataset-level McNemar tests form one Holm family. Confidence "
            "intervals use dirty-row bootstrap resampling; the macro is the unweighted "
            "mean of the nine dataset rates.",
            "",
            "## Experiment 2: Group benefit, interference, and cost",
            "",
            "Panel B and this main summary use the unweighted macro over the fixed nine "
            "datasets. Different sizes may cover different populations; overlap fields "
            "and same-intersection sensitivity metrics are retained in "
            "`group_by_dataset_view_size.csv`. Each size pair includes intersection and "
            "union counts, directional coverage, Jaccard overlap, and S/G/R outcomes "
            "recomputed on the common population.",
            "",
            _markdown_table(
                macro_rows,
                (
                    ("condition", "Condition"),
                    ("coverage", "Coverage"),
                    ("S", "S acc."),
                    ("G", "G acc."),
                    ("R", "R acc."),
                    ("rescue", "G rescue"),
                    ("interference", "G interference"),
                    ("G-S", "G−S"),
                    ("R-S", "R−S"),
                    ("G-R", "G−R"),
                    ("saving", "G token saving"),
                ),
            ),
            "",
            "### Primary paired tests (18-test Holm family)",
            "",
            _markdown_table(
                primary_rows,
                (
                    ("condition", "Condition"),
                    ("contrast", "Contrast"),
                    ("effect", "Effect"),
                    ("CI", "Clustered CI"),
                    ("Holm p", "Holm p"),
                ),
            ),
            "",
            "The 162 dataset-level tests form a separate secondary Holm family. "
            "G−S uses dirty row × structured physical query multipliers; R−S uses "
            "dirty row × random physical query; G−R uses all three dimensions. "
            "Across-condition summaries additionally cluster by cell ID.",
            "Invalid-state subtypes are mutually exclusive with priority provider "
            "failure, parse failure, missing item, abstain, empty repair, unchanged dirty "
            "value, then other invalid. The overall invalid rate is reported separately and therefore "
            "overlaps its subtype rates by definition.",
            "",
            "### Observed query cost (micro eligible populations)",
            "",
            _markdown_table(
                cost_rows,
                (
                    ("condition", "Condition/arm"),
                    ("logical", "Logical calls"),
                    ("physical", "Physical calls"),
                    ("tokens", "Physical-union observed tokens"),
                    ("logical tokens", "Logical-attributed tokens"),
                    ("estimated", "Estimated tokens"),
                    ("token saving", "Token saving"),
                    ("request reduction", "Request reduction"),
                    ("failures", "Provider failures"),
                    ("retries", "Retries"),
                    ("unknown usage", "Unknown-usage attempts"),
                ),
            ),
            "",
            "Arm/scoped cost rows are non-additive attribution views. Only the exact run "
            "physical-union row is an additive spend total. Logical-attributed tokens are "
            "the no-dedup counterfactual and are not provider spend.",
            "",
            "## Introduction figure",
            "",
            "[PDF](../figures/introduction_motivation.pdf) · "
            "[SVG](../figures/introduction_motivation.svg)",
            "",
            "Panel C is conceptual only and does not use Router experiment outcomes.",
            "",
        )
    )


def _singleton_population_cost(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    input_tokens = sum(int(row["llm_observed_input_tokens"]) for row in rows)
    output_tokens = sum(int(row["llm_observed_output_tokens"]) for row in rows)
    total_tokens = sum(int(row["llm_observed_total_tokens"]) for row in rows)
    correct = sum(bool(row["llm_correct"]) for row in rows)
    failures = sum(
        str(row["llm_status"]).strip().lower() not in {"success", "ok", "completed"}
        for row in rows
    )
    return {
        "scope": "run_singleton_population",
        "suite": "ALL",
        "dataset": "ALL",
        "source_view": "singleton",
        "group_size": 1,
        "arm": "singleton",
        "eligible_cell_incidences": len(rows),
        "correct_repairs": correct,
        "logical_calls": len(rows),
        "physical_calls": len(rows),
        "observed_input_tokens": input_tokens,
        "observed_output_tokens": output_tokens,
        "observed_total_tokens": total_tokens,
        "logical_observed_input_tokens": input_tokens,
        "logical_observed_output_tokens": output_tokens,
        "logical_observed_total_tokens": total_tokens,
        "provider_failures": failures,
        "attempts": None,
        "retries": None,
        "latency_seconds": None,
        "usage_observed_attempts": None,
        "unknown_usage_attempts": None,
        "tokens_per_correct_repair": _rate(total_tokens, correct),
        "token_saving_vs_singleton": 0.0,
        "logical_token_saving_vs_singleton": 0.0,
        "request_reduction_vs_singleton": 0.0,
        "logical_request_reduction_vs_singleton": 0.0,
        "cost_basis": "full_singleton_population_without_cross-query_dedup",
        "attribution_note": (
            "singleton prompts are one-to-one with cells; exact run spend is run_physical_union"
        ),
    }


def _run_physical_union_cost(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Exact once-per-physical-request run cost from the finalized API audit."""

    def total(field: str) -> int:
        return sum(int(row[field]) for row in rows)

    observed_input = total("observed_input_tokens")
    observed_output = total("observed_output_tokens")
    observed_total = total("observed_total_tokens")
    estimated_input = total("estimated_prompt_tokens")
    estimated_output = total("estimated_completion_tokens")
    estimated_total = total("estimated_total_tokens")
    logical_input = sum(
        int(row["observed_input_tokens"]) * int(row["logical_query_mappings"])
        for row in rows
    )
    logical_output = sum(
        int(row["observed_output_tokens"]) * int(row["logical_query_mappings"])
        for row in rows
    )
    logical_total = sum(
        int(row["observed_total_tokens"]) * int(row["logical_query_mappings"])
        for row in rows
    )
    attempts = total("attempts")
    unknown_attempts = total("unknown_usage_attempts")
    return {
        "scope": "run_physical_union",
        "suite": "ALL",
        "dataset": "ALL",
        "source_view": "ALL",
        "group_size": "ALL",
        "arm": "physical_union",
        "eligible_cell_incidences": None,
        "correct_repairs": None,
        "logical_calls": total("logical_query_mappings"),
        "physical_calls": len(rows),
        "observed_input_tokens": observed_input,
        "observed_output_tokens": observed_output,
        "observed_total_tokens": observed_total,
        "logical_observed_input_tokens": logical_input,
        "logical_observed_output_tokens": logical_output,
        "logical_observed_total_tokens": logical_total,
        "provider_failures": sum(
            str(row["status"]).strip().lower() not in _SUCCESS_STATUSES for row in rows
        ),
        "attempts": attempts,
        "retries": sum(max(0, int(row["attempts"]) - 1) for row in rows),
        "latency_seconds": sum(float(row["latency_seconds"]) for row in rows),
        "usage_observed_attempts": total("usage_observed_attempts"),
        "unknown_usage_attempts": unknown_attempts,
        "estimated_input_tokens": estimated_input,
        "estimated_output_tokens": estimated_output,
        "estimated_total_tokens": estimated_total,
        "observed_minus_estimated_total_tokens": observed_total - estimated_total,
        "observed_to_estimated_total_token_ratio": _rate(observed_total, estimated_total),
        "observed_usage_complete": unknown_attempts == 0,
        "observed_tokens_are_lower_bound": unknown_attempts > 0,
        "tokens_per_correct_repair": None,
        "token_saving_vs_singleton": None,
        "logical_token_saving_vs_singleton": None,
        "request_reduction_vs_singleton": None,
        "logical_request_reduction_vs_singleton": None,
        "cost_basis": "exact_once_per_physical_query_id",
        "attribution_note": (
            "authoritative additive run total; arm/scoped rows must not be summed"
        ),
    }


def build_motivation_report(
    run_dir: str | Path,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 45,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Build all offline motivation metrics, the paper figure, and Markdown.

    Parameters are intentionally limited to inferential resampling controls.
    The input paths, registered test families, views, sizes, and datasets are
    fixed by the evidence protocol.
    """

    if int(bootstrap_replicates) <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between zero and one")
    root = Path(run_dir).expanduser().resolve()
    complementarity_path = root / "records" / "complementarity_cell_outcomes.csv"
    group_path = root / "records" / "group_cell_outcomes.csv"
    api_cost_path = root / "metrics" / "api_cost_audit.csv"
    complementarity = _parse_complementarity(
        _read_csv(complementarity_path, COMPLEMENTARITY_REQUIRED_COLUMNS)
    )
    group = _parse_group(_read_csv(group_path, GROUP_REQUIRED_COLUMNS))
    api_cost = _parse_api_cost(_read_csv(api_cost_path, API_COST_REQUIRED_COLUMNS))
    _validate_cross_ledger_identity(complementarity, group)
    _validate_group_cost_against_audit(group, api_cost)

    complementarity_metrics = _build_complementarity_metrics(
        complementarity,
        bootstrap_replicates=int(bootstrap_replicates),
        bootstrap_seed=int(bootstrap_seed),
        confidence=float(confidence),
    )
    complementarity_counts = Counter(
        (str(row["suite"]), str(row["dataset"])) for row in complementarity
    )
    group_metrics, group_costs, metric_inputs = _build_group_metrics(
        group,
        complementarity_counts=complementarity_counts,
        bootstrap_replicates=int(bootstrap_replicates),
        bootstrap_seed=int(bootstrap_seed),
        confidence=float(confidence),
    )
    group_costs.append(_singleton_population_cost(complementarity))
    group_costs.append(_run_physical_union_cost(api_cost))
    group_transitions = _build_group_transitions(metric_inputs)
    statistical_tests = [
        *_complementarity_tests(complementarity_metrics),
        *group_transitions,
    ]

    metrics_dir = root / "metrics"
    figures_dir = root / "figures"
    report_dir = root / "report"
    output_paths = {
        "complementarity_metrics": metrics_dir / "complementarity_by_dataset.csv",
        "complementarity_summary": metrics_dir / "complementarity_summary.json",
        "group_metrics": metrics_dir / "group_by_dataset_view_size.csv",
        "group_summary": metrics_dir / "group_summary.json",
        "group_transitions": metrics_dir / "group_paired_transitions.csv",
        "group_costs": metrics_dir / "group_costs.csv",
        "statistical_tests": metrics_dir / "statistical_tests.csv",
        "figure_pdf": figures_dir / "introduction_motivation.pdf",
        "figure_svg": figures_dir / "introduction_motivation.svg",
        "report": report_dir / "report.md",
    }
    _write_csv(output_paths["complementarity_metrics"], complementarity_metrics)
    _write_csv(output_paths["group_metrics"], group_metrics)
    _write_csv(output_paths["group_transitions"], group_transitions)
    _write_csv(output_paths["group_costs"], group_costs)
    _write_csv(output_paths["statistical_tests"], statistical_tests)

    complementarity_summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "aggregation": {
            "dataset_order": [list(key) for key in SELECTED_DATASETS],
            "macro": "unweighted mean over the frozen nine datasets",
            "micro": "cell-weighted full population",
        },
        "bootstrap": {
            "method": "dirty-row cluster resampling",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "confidence": float(confidence),
        },
        "metrics": complementarity_metrics,
        "oracle_union_interpretation": "offline opportunity upper bound",
        "holm_family": "nine dataset-level paired McNemar tests",
    }
    _write_json(output_paths["complementarity_summary"], complementarity_summary)
    group_summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "aggregation": {
            "dataset_order": [list(key) for key in SELECTED_DATASETS],
            "panel_b": "unweighted macro over all nine frozen datasets",
            "overall_population": "cell incidences across the six registered conditions",
            "cross_size_sensitivity": (
                "each current-size estimand recomputed on its cell intersection with k=2/4/8; "
                "macro remains an unweighted mean over all nine datasets"
            ),
        },
        "bootstrap": {
            "method": "crossed Exp(1) cluster multiplier",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "confidence": float(confidence),
            "g_minus_s_clusters": ["row_cluster", "structured_physical_query_id"],
            "r_minus_s_clusters": ["row_cluster", "random_physical_query_id"],
            "g_minus_r_clusters": [
                "row_cluster",
                "structured_physical_query_id",
                "random_physical_query_id",
            ],
            "overall_extra_cluster": "cell_id",
        },
        "primary_holm_tests": 18,
        "secondary_dataset_holm_tests": 162,
        "cost_accounting": {
            "authoritative_total": "group_costs.csv scope=run_physical_union",
            "physical_union": "each physical_query_id counted once across all arms",
            "logical_tokens": "counterfactual no-dedup attribution, not provider spend",
            "arm_scopes": "non-additive attribution views",
        },
        "macro_metrics": [row for row in group_metrics if row["scope"] == "macro"],
        "micro_metrics": [row for row in group_metrics if row["scope"] == "micro"],
        "overall_metrics": [
            row for row in group_metrics if "all_conditions" in str(row["scope"])
        ],
    }
    _write_json(output_paths["group_summary"], group_summary)
    _figure(
        complementarity_metrics,
        group_metrics,
        pdf_path=output_paths["figure_pdf"],
        svg_path=output_paths["figure_svg"],
    )
    report_text = _report_markdown(
        complementarity_metrics, group_metrics, group_transitions, group_costs
    )
    output_paths["report"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["report"].write_text(report_text, encoding="utf-8")

    suffixes = {path.suffix.lower() for path in output_paths.values()}
    if not suffixes <= {".pdf", ".svg", ".md", ".csv", ".json"}:
        raise RuntimeError(f"unexpected report output type: {suffixes}")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_dir": str(root),
        "network_calls": 0,
        "complementarity_cells": len(complementarity),
        "group_cell_incidences": len(group),
        "physical_union_calls": len(api_cost),
        "complementarity_metric_rows": len(complementarity_metrics),
        "group_metric_rows": len(group_metrics),
        "primary_holm_tests": 18,
        "secondary_dataset_holm_tests": 162,
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }


__all__ = ["REPORT_SCHEMA_VERSION", "build_motivation_report"]
