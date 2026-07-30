"""Materialize and analyze the frozen full-nine Baran/LLM baselines.

This module is deliberately offline.  It only reads a completed Router-v3 run,
projects its two baseline slices, and produces derived research artifacts.  It
never invokes an LLM client and never modifies the source run.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .data import EXPECTED_ORACLE_ERRORS, read_jsonl, write_jsonl
from .metrics import recompute_record, summarize_records, verify_records
from .run_state import read_json, sha256_file, write_json
from .sampling import EXPECTED_SELECTED_ORACLE_ERRORS, SELECTED_DATASETS
from .statistics import exact_mcnemar, holm_adjust


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_RUN = (
    PROJECT_ROOT
    / "runs"
    / "no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all"
)
DEFAULT_BASELINE_DIR = (
    PROJECT_ROOT / "runs" / "baselines" / "no_baran_singleton_deepseek_v4_full9"
)
DEFAULT_ANALYSIS_DIR = (
    PROJECT_ROOT / "runs" / "analyses" / "baran_llm_complementarity_full9"
)

BASELINE_MANIFEST_SCHEMA = "no-baran-singleton-baseline-v1"
ANALYSIS_SCHEMA = "baran-llm-full-complementarity-v1"
TEST_TARGETS = tuple(SELECTED_DATASETS)
TEST_TARGET_CELL_COUNT = EXPECTED_SELECTED_ORACLE_ERRORS
BASELINE_METHODS = ("baran", "llm_only")
EXPECTED_VARIANTS = {"baran": "all", "llm_only": "1"}
TEST_TARGET_SET = frozenset(TEST_TARGETS)
EXPECTED_SOURCE_FILES = (
    "run_manifest.json",
    "final/all_methods.jsonl",
    "llm/group_query_checkpoint.jsonl",
    "llm/selected_union_plan.json",
    "metrics/method_metrics.csv",
)
RATE_FIELDS = (
    "baran_accuracy",
    "llm_accuracy",
    "oracle_upper_bound",
    "upper_bound_minus_baran",
    "upper_bound_minus_best",
    "llm_salvage_rate",
    "baran_rescue_rate",
    "disagreement_rate",
    "llm_proposal_precision",
    "llm_proposal_recall",
    "llm_proposal_f1",
)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    columns: Sequence[str],
) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in materialized:
                values = {column: _json_safe(row.get(column)) for column in columns}
                writer.writerow(
                    {column: "" if value is None else value for column, value in values.items()}
                )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_or_absolute(path: Path, root: Path = PROJECT_ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _source_file_manifest(source_run: Path) -> dict[str, dict[str, object]]:
    files: dict[str, dict[str, object]] = {}
    for relative in EXPECTED_SOURCE_FILES:
        path = source_run / relative
        if not path.is_file():
            raise FileNotFoundError(f"canonical source is missing {relative}: {path}")
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return files


def _is_baseline_slice(row: Mapping[str, object], method: str) -> bool:
    budget = row.get("budget_share")
    return bool(
        str(row.get("method", "")) == method
        and str(row.get("scenario", "")) == "baseline"
        and str(row.get("backend", "")) == "none"
        and budget in {None, ""}
        and str(row.get("group_size_variant", "")) == EXPECTED_VARIANTS[method]
        and (str(row.get("suite", "")), str(row.get("dataset", "")))
        in TEST_TARGET_SET
    )


def _read_projected_baselines(path: Path) -> dict[str, list[dict[str, object]]]:
    """Stream the large all-method ledger while retaining only two slices."""

    projected: dict[str, list[dict[str, object]]] = {
        method: [] for method in BASELINE_METHODS
    }
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            method = str(value.get("method", ""))
            if method in projected and _is_baseline_slice(value, method):
                projected[method].append(value)
    return {method: _ordered(rows) for method, rows in projected.items()}


def _row_cluster(cell_id: str, suite: str, dataset: str) -> str:
    prefix, row, _column = cell_id.rsplit(":", 2)
    if prefix != f"{suite}:{dataset}":
        raise ValueError(f"cell identity disagrees with suite/dataset: {cell_id}")
    int(row)
    return f"{prefix}:{row}"


def _ordered(records: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    target_rank = {key: index for index, key in enumerate(TEST_TARGETS)}
    return sorted(
        (dict(record) for record in records),
        key=lambda row: (
            target_rank.get(
                (str(row["suite"]), str(row["dataset"])), len(target_rank)
            ),
            str(row["suite"]),
            str(row["dataset"]),
            str(row["cell_id"]),
        ),
    )


def _index_unique(
    records: Sequence[Mapping[str, object]], method: str
) -> dict[str, dict[str, object]]:
    indexed: dict[str, dict[str, object]] = {}
    for record in records:
        cell_id = str(record.get("cell_id", ""))
        if not cell_id:
            raise ValueError(f"{method} contains an empty cell_id")
        if cell_id in indexed:
            raise ValueError(f"{method} contains duplicate cell_id {cell_id}")
        indexed[cell_id] = dict(record)
    return indexed


def _validate_population(records: Sequence[Mapping[str, object]], method: str) -> None:
    if len(records) != TEST_TARGET_CELL_COUNT:
        raise ValueError(
            f"{method} must contain {TEST_TARGET_CELL_COUNT} cells, observed {len(records)}"
        )
    counts = Counter((str(row["suite"]), str(row["dataset"])) for row in records)
    expected = {key: int(EXPECTED_ORACLE_ERRORS[key]) for key in TEST_TARGETS}
    if counts != expected:
        raise ValueError(f"{method} does not cover the frozen nine-dataset population")
    audit = verify_records(records)
    if not bool(audit["ok"]):
        raise ValueError(f"{method} record audit failed: {audit}")


def pair_baseline_records(
    baran_records: Sequence[Mapping[str, object]],
    llm_records: Sequence[Mapping[str, object]],
    *,
    require_formal_population: bool = True,
) -> list[dict[str, object]]:
    """Pair exact Baran and pure singleton LLM outcomes by ``cell_id``."""

    baran = _index_unique(baran_records, "baran")
    llm = _index_unique(llm_records, "llm_only")
    if set(baran) != set(llm):
        raise ValueError(
            "Baran/LLM cell universes differ: "
            f"only_baran={len(set(baran) - set(llm))}, "
            f"only_llm={len(set(llm) - set(baran))}"
        )
    if require_formal_population:
        _validate_population(list(baran.values()), "baran")
        _validate_population(list(llm.values()), "llm_only")

    paired: list[dict[str, object]] = []
    for cell_id in sorted(baran):
        baran_row = baran[cell_id]
        llm_row = llm[cell_id]
        if llm_row.get("baran_fallback_used") is not False:
            raise ValueError(f"LLM-only record is not pure No-Baran: {cell_id}")
        if str(llm_row.get("final_source", "")) not in {"llm", "no_repair"}:
            raise ValueError(f"unexpected LLM-only final source for {cell_id}")
        for field in ("suite", "dataset", "clean_value"):
            if baran_row.get(field) != llm_row.get(field):
                raise ValueError(f"paired records disagree on {field}: {cell_id}")

        baran_flag = recompute_record(baran_row)
        llm_flag = recompute_record(llm_row)
        baran_correct = bool(baran_flag["correct_repair"])
        llm_correct = bool(llm_flag["correct_repair"])
        llm_valid = bool(llm_flag["valid_repair"])
        suite = str(baran_row["suite"])
        dataset = str(baran_row["dataset"])
        if baran_correct and llm_correct:
            quadrant = "n11_both_correct"
            oracle_action = "either"
        elif baran_correct:
            quadrant = "n10_baran_only"
            oracle_action = "keep_baran"
        elif llm_correct:
            quadrant = "n01_llm_only"
            oracle_action = "use_llm"
        else:
            quadrant = "n00_both_wrong"
            oracle_action = "neither"
        paired.append(
            {
                "suite": suite,
                "dataset": dataset,
                "cell_id": cell_id,
                "row_cluster": _row_cluster(cell_id, suite, dataset),
                "baran_correct": baran_correct,
                "llm_correct": llm_correct,
                "baran_valid_prediction": bool(baran_flag["valid_repair"]),
                "llm_valid_prediction": llm_valid,
                "llm_help": bool(llm_correct and not baran_correct),
                "llm_harm": bool(baran_correct and not llm_correct),
                "both_correct": bool(baran_correct and llm_correct),
                "both_wrong": bool(not baran_correct and not llm_correct),
                "oracle_correct": bool(baran_correct or llm_correct),
                "outcome_quadrant": quadrant,
                "oracle_action": oracle_action,
                "llm_parse_status": str(llm_row.get("parse_status", "")),
                "llm_final_source": str(llm_row.get("final_source", "")),
                "llm_decision": str(llm_row.get("llm_decision", "")),
                "selected_query_id": str(llm_row.get("selected_query_id", "")),
            }
        )
    return _ordered(paired)


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else math.nan


def _rates_from_vector(vector: Sequence[int | float]) -> dict[str, float]:
    n11, n10, n01, n00, llm_valid = (float(value) for value in vector)
    total = n11 + n10 + n01 + n00
    llm_correct = n11 + n01
    precision = _rate(llm_correct, llm_valid)
    recall = _rate(llm_correct, total)
    return {
        "baran_accuracy": _rate(n11 + n10, total),
        "llm_accuracy": recall,
        "oracle_upper_bound": _rate(n11 + n10 + n01, total),
        "upper_bound_minus_baran": _rate(n01, total),
        "upper_bound_minus_best": _rate(min(n10, n01), total),
        "llm_salvage_rate": _rate(n01, n01 + n00),
        "baran_rescue_rate": _rate(n10, n10 + n00),
        "disagreement_rate": _rate(n10 + n01, total),
        "llm_proposal_precision": precision,
        "llm_proposal_recall": recall,
        "llm_proposal_f1": (
            2.0 * precision * recall / (precision + recall)
            if math.isfinite(precision) and precision + recall > 0.0
            else 0.0
        ),
    }


def _cluster_matrix(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    clusters: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    for row in rows:
        baran_correct = bool(row["baran_correct"])
        llm_correct = bool(row["llm_correct"])
        quadrant = {
            (True, True): 0,
            (True, False): 1,
            (False, True): 2,
            (False, False): 3,
        }[(baran_correct, llm_correct)]
        vector = clusters[str(row["row_cluster"])]
        vector[quadrant] += 1
        vector[4] += int(bool(row["llm_valid_prediction"]))
    return np.asarray([clusters[key] for key in sorted(clusters)], dtype=np.int64)


def _bootstrap_samples(
    rows: Sequence[Mapping[str, object]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    matrix = _cluster_matrix(rows)
    if matrix.size == 0:
        return {field: np.full(replicates, np.nan) for field in RATE_FIELDS}
    rng = np.random.default_rng(int(seed))
    estimates = {field: np.empty(replicates, dtype=float) for field in RATE_FIELDS}
    cluster_count = len(matrix)
    for replicate in range(replicates):
        indices = rng.integers(0, cluster_count, size=cluster_count)
        rates = _rates_from_vector(matrix[indices].sum(axis=0))
        for field in RATE_FIELDS:
            estimates[field][replicate] = rates[field]
    return estimates


def _intervals_from_samples(
    estimates: Mapping[str, np.ndarray], confidence: float
) -> dict[str, tuple[float | None, float | None]]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    alpha = (1.0 - confidence) / 2.0
    output: dict[str, tuple[float | None, float | None]] = {}
    for field, values in estimates.items():
        finite = values[np.isfinite(values)]
        output[field] = (
            (float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha)))
            if finite.size
            else (None, None)
        )
    return output


def _bootstrap_intervals(
    rows: Sequence[Mapping[str, object]],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, tuple[float | None, float | None]]:
    return _intervals_from_samples(
        _bootstrap_samples(rows, replicates=replicates, seed=seed), confidence
    )


def _failure_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = Counter(str(row.get("llm_parse_status", "")) for row in rows)
    return {
        "llm_valid_predictions": sum(bool(row["llm_valid_prediction"]) for row in rows),
        "llm_invalid_predictions": sum(not bool(row["llm_valid_prediction"]) for row in rows),
        "llm_abstain": counts["abstain"],
        "llm_provider_failure": counts["provider_failure"],
        "llm_unchanged_dirty": counts["unchanged_dirty"],
        "llm_other_invalid": sum(
            not bool(row["llm_valid_prediction"])
            and str(row.get("llm_parse_status", ""))
            not in {"abstain", "provider_failure", "unchanged_dirty"}
            for row in rows
        ),
    }


def aggregate_outcomes(
    rows: Sequence[Mapping[str, object]],
    *,
    scope: str,
    suite: str,
    dataset: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, object]:
    counts = Counter(
        (bool(row["baran_correct"]), bool(row["llm_correct"])) for row in rows
    )
    vector = (
        counts[(True, True)],
        counts[(True, False)],
        counts[(False, True)],
        counts[(False, False)],
        sum(bool(row["llm_valid_prediction"]) for row in rows),
    )
    result: dict[str, object] = {
        "scope": scope,
        "aggregation": "micro_over_cells",
        "suite": suite,
        "dataset": dataset,
        "N": len(rows),
        "n11": vector[0],
        "n10": vector[1],
        "n01": vector[2],
        "n00": vector[3],
        **_rates_from_vector(vector),
        **_failure_counts(rows),
        "mcnemar_p": exact_mcnemar(vector[1], vector[2]),
        "mcnemar_p_holm": None,
    }
    intervals = _bootstrap_intervals(
        rows,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    for field, (low, high) in intervals.items():
        result[f"{field}_ci_low"] = low
        result[f"{field}_ci_high"] = high
    return result


def _macro_row(
    dataset_rows: Sequence[Mapping[str, object]],
    intervals: Mapping[str, tuple[float | None, float | None]],
) -> dict[str, object]:
    result: dict[str, object] = {
        "scope": "dataset_macro",
        "aggregation": "unweighted_dataset_mean",
        "suite": "ALL",
        "dataset": "MACRO",
        "N": sum(int(row["N"]) for row in dataset_rows),
        "n11": sum(int(row["n11"]) for row in dataset_rows),
        "n10": sum(int(row["n10"]) for row in dataset_rows),
        "n01": sum(int(row["n01"]) for row in dataset_rows),
        "n00": sum(int(row["n00"]) for row in dataset_rows),
        "mcnemar_p": None,
        "mcnemar_p_holm": None,
    }
    for field in RATE_FIELDS:
        values = [
            float(row[field])
            for row in dataset_rows
            if row[field] is not None and math.isfinite(float(row[field]))
        ]
        result[field] = sum(values) / len(values) if values else None
        result[f"{field}_ci_low"] = intervals[field][0]
        result[f"{field}_ci_high"] = intervals[field][1]
    for field in _failure_counts([]):
        result[field] = sum(int(row[field]) for row in dataset_rows)
    return result


def build_complementarity_rows(
    paired: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
) -> list[dict[str, object]]:
    by_dataset: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    by_family: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in paired:
        key = (str(row["suite"]), str(row["dataset"]))
        by_dataset[key].append(row)
        by_family[key[0]].append(row)
    if set(by_dataset) != set(TEST_TARGETS):
        raise ValueError("paired outcomes do not cover the frozen nine datasets")

    dataset_rows: list[dict[str, object]] = []
    for index, (suite, dataset) in enumerate(TEST_TARGETS):
        dataset_rows.append(
            aggregate_outcomes(
                by_dataset[(suite, dataset)],
                scope="dataset",
                suite=suite,
                dataset=dataset,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + index * 100,
                confidence=confidence,
            )
        )
    adjusted = holm_adjust(
        {str(row["dataset"]): float(row["mcnemar_p"]) for row in dataset_rows}
    )
    for row in dataset_rows:
        row["mcnemar_p_holm"] = adjusted[str(row["dataset"])]

    family_rows = [
        aggregate_outcomes(
            by_family[suite],
            scope="family",
            suite=suite,
            dataset="ALL",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + 10_000 + index * 100,
            confidence=confidence,
        )
        for index, suite in enumerate(("source", "tableeg"))
    ]
    micro = aggregate_outcomes(
        paired,
        scope="micro",
        suite="ALL",
        dataset="MICRO",
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 20_000,
        confidence=confidence,
    )
    macro_samples: dict[str, list[np.ndarray]] = {field: [] for field in RATE_FIELDS}
    for index, key in enumerate(TEST_TARGETS):
        samples = _bootstrap_samples(
            by_dataset[key],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 30_000 + index * 100,
        )
        for field in RATE_FIELDS:
            macro_samples[field].append(samples[field])
    macro_estimates = {
        field: np.nanmean(np.vstack(values), axis=0)
        for field, values in macro_samples.items()
    }
    macro_intervals = _intervals_from_samples(macro_estimates, confidence)
    return [
        *dataset_rows,
        *family_rows,
        micro,
        _macro_row(dataset_rows, macro_intervals),
    ]


def _reconcile_method_metrics(
    source_path: Path,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    computed = summarize_records(records, method=BASELINE_METHODS, strict=True)
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reported = list(csv.DictReader(handle))
    reported_index = {
        (
            row["method"],
            row["scope"],
            row["suite"],
            row["dataset"],
        ): row
        for row in reported
        if row["method"] in BASELINE_METHODS
        and row["scenario"] == "baseline"
        and row["backend"] == "none"
        and row["budget_share"] == ""
        and row["group_size_variant"] == EXPECTED_VARIANTS[row["method"]]
    }
    fields_int = (
        "true_error_cells",
        "predicted_repairs",
        "valid_predictions",
        "invalid_predictions",
        "correct_repairs",
        "annotation_mismatches",
    )
    fields_float = ("precision", "recall", "f1")
    mismatches: list[dict[str, object]] = []
    for row in computed:
        key = (
            str(row["method"]),
            str(row["scope"]),
            str(row["suite"]),
            str(row["dataset"]),
        )
        reference = reported_index.get(key)
        if reference is None:
            mismatches.append({"key": list(key), "reason": "missing_reported_row"})
            continue
        for field in fields_int:
            if int(row[field]) != int(reference[field]):
                mismatches.append(
                    {
                        "key": list(key),
                        "field": field,
                        "computed": row[field],
                        "reported": reference[field],
                    }
                )
        for field in fields_float:
            if not math.isclose(
                float(row[field]), float(reference[field]), rel_tol=0.0, abs_tol=1e-12
            ):
                mismatches.append(
                    {
                        "key": list(key),
                        "field": field,
                        "computed": row[field],
                        "reported": reference[field],
                    }
                )
    if mismatches:
        raise ValueError(f"method metric reconciliation failed: {mismatches[:5]}")
    return {
        "ok": True,
        "rows_checked": len(computed),
        "fields_checked": [*fields_int, *fields_float],
        "mismatches": 0,
    }


def _audit_response_reuse(
    checkpoint_path: Path,
    llm_records: Sequence[Mapping[str, object]],
    *,
    model: str,
) -> dict[str, object]:
    cell_by_query: dict[str, str] = {}
    for row in llm_records:
        query_id = str(row.get("selected_query_id", ""))
        cell_id = str(row["cell_id"])
        if not query_id:
            raise ValueError(f"LLM-only row has no selected_query_id: {cell_id}")
        if query_id in cell_by_query:
            raise ValueError(f"singleton query is shared by multiple cells: {query_id}")
        cell_by_query[query_id] = cell_id

    histories: dict[str, list[dict[str, object]]] = defaultdict(list)
    for response in read_jsonl(checkpoint_path):
        query_id = str(response.get("query_id", ""))
        if query_id in cell_by_query:
            histories[query_id].append(response)
    missing = set(cell_by_query) - set(histories)
    if missing:
        raise ValueError(f"response checkpoint misses {len(missing)} singleton queries")

    latest: dict[str, dict[str, object]] = {}
    for query_id, rows in histories.items():
        prompt_hashes = {str(row.get("prompt_hash", "")) for row in rows}
        request_hashes = {str(row.get("provider_request_hash", "")) for row in rows}
        if "" in prompt_hashes or len(prompt_hashes) != 1:
            raise ValueError(f"prompt identity drift for {query_id}")
        if "" in request_hashes or len(request_hashes) != 1:
            raise ValueError(f"provider request identity drift for {query_id}")
        latest[query_id] = rows[-1]

    statuses = Counter(str(row.get("status", "")) for row in latest.values())
    if set(statuses) - {"success", "failed"}:
        raise ValueError(f"unexpected singleton response statuses: {statuses}")
    for query_id, response in latest.items():
        if str(response.get("model", "")) != model:
            raise ValueError(f"model identity drift for {query_id}")
        if list(response.get("cell_ids", [])) != [cell_by_query[query_id]]:
            raise ValueError(f"checkpoint query is not a singleton for {query_id}")
    final_provider_failures = sum(
        str(row.get("parse_status", "")) == "provider_failure" for row in llm_records
    )
    if statuses["failed"] != final_provider_failures:
        raise ValueError("terminal failure count differs between checkpoint and final ledger")
    return {
        "ok": True,
        "selected_singleton_queries": len(cell_by_query),
        "fixed_response_records": len(latest),
        "status_counts": dict(sorted(statuses.items())),
        "queries_with_history_rows": sum(len(rows) > 1 for rows in histories.values()),
        "matching_fields": [
            "query_id",
            "prompt_hash",
            "provider_request_hash",
            "model",
            "prompt_schema_version",
        ],
        "terminal_failures_frozen_required": True,
        "physical_calls_required_for_exact_reuse": 0,
    }


def _output_file(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_baseline_manifest(baseline_dir: str | Path) -> dict[str, object]:
    root = Path(baseline_dir).expanduser().resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != BASELINE_MANIFEST_SCHEMA:
        raise ValueError("unsupported baseline manifest schema")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("baseline manifest has no output hashes")
    checked = 0
    for relative, metadata in outputs.items():
        if not isinstance(metadata, Mapping):
            raise ValueError(f"invalid baseline output metadata: {relative}")
        path = root / str(relative)
        if (
            not path.is_file()
            or path.stat().st_size != int(metadata.get("bytes", -1))
            or sha256_file(path) != str(metadata.get("sha256", ""))
        ):
            raise ValueError(f"baseline output hash mismatch: {relative}")
        checked += 1
    source_files = manifest.get("source_files")
    if not isinstance(source_files, Mapping):
        raise ValueError("baseline manifest has no source hashes")
    declared_value = str(manifest.get("source_run", ""))
    declared_source = Path(declared_value).expanduser()
    portable_source = (
        declared_source.resolve()
        if declared_source.is_absolute()
        else (PROJECT_ROOT / declared_source).resolve()
    )
    resolved_source = Path(
        str(manifest.get("source_run_resolved", ""))
    ).expanduser().resolve()
    source = portable_source if declared_value and portable_source.is_dir() else resolved_source
    source_checked = 0
    for relative, metadata in source_files.items():
        if not isinstance(metadata, Mapping):
            raise ValueError(f"invalid source metadata: {relative}")
        path = source / str(relative)
        if (
            not path.is_file()
            or path.stat().st_size != int(metadata.get("bytes", -1))
            or sha256_file(path) != str(metadata.get("sha256", ""))
        ):
            raise ValueError(f"canonical source hash mismatch: {relative}")
        source_checked += 1
    return {
        "ok": True,
        "output_files_checked": checked,
        "source_files_checked": source_checked,
        "baseline_dir": str(root),
    }


def _report_markdown(
    source_run: Path,
    baseline_dir: Path,
    rows: Sequence[Mapping[str, object]],
    response_audit: Mapping[str, object],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
) -> str:
    dataset_rows = [row for row in rows if row["scope"] == "dataset"]
    micro = next(row for row in rows if row["scope"] == "micro")
    macro = next(row for row in rows if row["scope"] == "dataset_macro")

    def pct(value: object) -> str:
        number = float(value)
        return f"{100.0 * number:.2f}%" if math.isfinite(number) else "NA"

    def pvalue(value: object) -> str:
        number = float(value)
        return f"{number:.3g}" if number >= 0.001 else f"{number:.2e}"

    table_lines = [
        "| 数据集 | N | Baran | LLM-only | Oracle UB | 仅 LLM 正确 | LLM salvage | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_rows:
        table_lines.append(
            "| {dataset} | {N:,} | {baran} | {llm} | {oracle} | {n01:,} | {salvage} | {p} |".format(
                dataset=row["dataset"],
                N=int(row["N"]),
                baran=pct(row["baran_accuracy"]),
                llm=pct(row["llm_accuracy"]),
                oracle=pct(row["oracle_upper_bound"]),
                n01=int(row["n01"]),
                salvage=pct(row["llm_salvage_rate"]),
                p=pvalue(row["mcnemar_p_holm"]),
            )
        )

    status_counts = response_audit["status_counts"]
    assert isinstance(status_counts, Mapping)
    return "\n".join(
        [
            "# Baran 与 No-Baran Singleton LLM：完整九数据集互补性实验",
            "",
            "## 结论",
            "",
            f"完整配对覆盖 {int(micro['N']):,} 个错误单元格。Baran 的修复成功率为 "
            f"{pct(micro['baran_accuracy'])}，纯 LLM-only 为 {pct(micro['llm_accuracy'])}，"
            f"离线 Oracle 上界为 {pct(micro['oracle_upper_bound'])}。LLM 单独修复了 "
            f"{int(micro['n01']):,} 个 Baran 失败单元格，使 Oracle 相对 Baran 提高 "
            f"{pct(micro['upper_bound_minus_baran'])}。互补性存在，但仅 Baran 正确的 "
            f"{int(micro['n10']):,} 个单元格显著多于仅 LLM 正确的 {int(micro['n01']):,} 个，"
            "因此互补关系是不对称的。",
            "",
            "## 数据与口径",
            "",
            f"- Canonical source：`{_relative_or_absolute(source_run)}`",
            f"- Baseline bundle：`{_relative_or_absolute(baseline_dir)}`",
            f"- LLM-only 请求：{int(response_audit['selected_singleton_queries']):,} 个固定 "
            f"singleton identities；{int(status_counts.get('success', 0)):,} success，"
            f"{int(status_counts.get('failed', 0)):,} terminal failures。",
            "- LLM-only 从未回退到 Baran；abstain、provider failure、unchanged dirty 均计为未修复。",
            "- 正确性使用与正式 Router 指标一致的规范化 exact match 重新计算。",
            "",
            "## 四格结果",
            "",
            "| 组合 | 数量 | 占比 |",
            "|---|---:|---:|",
            f"| 两者都正确（n11） | {int(micro['n11']):,} | {pct(int(micro['n11']) / int(micro['N']))} |",
            f"| 仅 Baran 正确（n10） | {int(micro['n10']):,} | "
            f"{pct(int(micro['n10']) / int(micro['N']))} |",
            f"| 仅 LLM 正确（n01） | {int(micro['n01']):,} | "
            f"{pct(int(micro['n01']) / int(micro['N']))} |",
            f"| 两者都错误（n00） | {int(micro['n00']):,} | {pct(int(micro['n00']) / int(micro['N']))} |",
            "",
            "## 分数据集结果",
            "",
            *table_lines,
            "",
            "dataset-macro 结果：Baran {baran}（95% CI {baran_ci}），LLM-only {llm}"
            "（95% CI {llm_ci}），Oracle UB {oracle}（95% CI {oracle_ci}）。".format(
                baran=pct(macro["baran_accuracy"]),
                baran_ci=(
                    f"{pct(macro['baran_accuracy_ci_low'])}–"
                    f"{pct(macro['baran_accuracy_ci_high'])}"
                ),
                llm=pct(macro["llm_accuracy"]),
                llm_ci=(
                    f"{pct(macro['llm_accuracy_ci_low'])}–"
                    f"{pct(macro['llm_accuracy_ci_high'])}"
                ),
                oracle=pct(macro["oracle_upper_bound"]),
                oracle_ci=(
                    f"{pct(macro['oracle_upper_bound_ci_low'])}–"
                    f"{pct(macro['oracle_upper_bound_ci_high'])}"
                ),
            ),
            "",
            "## 统计方法",
            "",
            f"- 以同一数据行作为 cluster，执行 {bootstrap_replicates:,} 次 row-cluster bootstrap；"
            f"seed={bootstrap_seed}，置信水平={confidence:.0%}。",
            "- 每个数据集使用 exact McNemar 检验比较 Baran 与 LLM-only 的配对正确性，并对九个数据集执行 Holm 校正。",
            "- dataset-macro 是九个数据集指标的非加权平均；micro 和 family 指标按单元格聚合。",
            "",
            "## Router 复用边界",
            "",
            "`singleton_router_labels.csv` 中的正确性、help/harm 和 oracle action 都由 clean label 产生，"
            "只能用于离线训练与评估，不能成为在线特征。未来 singleton-only Router 应通过 manifest 中的 "
            "`response_reuse_run` 读取原始 checkpoint，并冻结 success、abstain 和 terminal failure；"
            "在模型、Prompt schema 或 provider request hash 变化时必须建立新的独立基线。",
            "",
            "## 限制",
            "",
            "- Oracle UB 使用 clean label，是离线上界，不是可部署 Router 的性能。",
            "- 当前结果来自一次固定模型执行，不能用于估计 LLM 跨重复运行的随机性。",
            "- McNemar 检验回答的是两种方法边际正确率是否不同，不证明存在可学习且可泛化的路由策略。",
            "- 原始 `baran/` 目录包含十四个数据集；本分析仅使用最终 ledger 中冻结的九数据集 baseline slices。",
            "",
        ]
    )


def build_full_complementarity(
    source_run: str | Path = DEFAULT_SOURCE_RUN,
    *,
    baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
    output_dir: str | Path = DEFAULT_ANALYSIS_DIR,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 45,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Build the canonical baseline bundle and its full offline analysis."""

    source = Path(source_run).expanduser().resolve()
    baseline = Path(baseline_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if len({source, baseline, output}) != 3:
        raise ValueError("source, baseline, and analysis directories must differ")
    for derived in (baseline, output):
        try:
            derived.relative_to(source)
        except ValueError:
            continue
        raise ValueError("derived output directories cannot be inside the source run")
    source_manifest = read_json(source / "run_manifest.json")
    if source_manifest.get("status") != "complete":
        raise ValueError("canonical source run must be complete")
    source_hashes_before = _source_file_manifest(source)

    projected = _read_projected_baselines(source / "final" / "all_methods.jsonl")
    baran_records = projected["baran"]
    llm_records = projected["llm_only"]
    paired = pair_baseline_records(baran_records, llm_records)
    metric_reconciliation = _reconcile_method_metrics(
        source / "metrics" / "method_metrics.csv",
        [*baran_records, *llm_records],
    )
    response_audit = _audit_response_reuse(
        source / "llm" / "group_query_checkpoint.jsonl",
        llm_records,
        model=str(source_manifest.get("model", "")),
    )
    union_plan = read_json(source / "llm" / "selected_union_plan.json")
    if int(union_plan.get("llm_only_singleton_queries", -1)) != TEST_TARGET_CELL_COUNT:
        raise ValueError("selected union plan does not bind all singleton queries")

    baseline.mkdir(parents=True, exist_ok=True)
    baran_path = baseline / "baran_only.jsonl"
    llm_path = baseline / "llm_only.jsonl"
    labels_path = baseline / "singleton_router_labels.csv"
    write_jsonl(baran_path, baran_records)
    write_jsonl(llm_path, llm_records)
    label_columns = (
        "suite",
        "dataset",
        "cell_id",
        "row_cluster",
        "baran_correct",
        "llm_correct",
        "llm_help",
        "llm_harm",
        "both_correct",
        "both_wrong",
        "oracle_action",
    )
    _write_csv(labels_path, paired, columns=label_columns)

    dataset_counts = Counter((str(row["suite"]), str(row["dataset"])) for row in paired)
    baseline_manifest: dict[str, object] = {
        "schema_version": BASELINE_MANIFEST_SCHEMA,
        "source_run": _relative_or_absolute(source),
        "source_run_resolved": str(source),
        "response_reuse_run": _relative_or_absolute(source),
        "source_completed_at": source_manifest.get("completed_at"),
        "model": source_manifest.get("model"),
        "prompt_schema_version": source_manifest.get("prompt_schema_version"),
        "prompt_schema_sha256": source_manifest.get("prompt_schema_sha256"),
        "data_content_fingerprint": source_manifest.get("data_content_fingerprint"),
        "population": {
            "dataset_count": len(TEST_TARGETS),
            "cell_count": len(paired),
            "datasets": [
                {"suite": suite, "dataset": dataset, "cells": dataset_counts[(suite, dataset)]}
                for suite, dataset in TEST_TARGETS
            ],
        },
        "purity_policy": {
            "llm_prompt_contains_baran": False,
            "baran_fallback_allowed": False,
            "abstain_failure_unchanged_count_as_incorrect": True,
        },
        "response_reuse": response_audit,
        "metric_reconciliation": metric_reconciliation,
        "source_files": source_hashes_before,
        "outputs": {
            "baran_only.jsonl": _output_file(baran_path),
            "llm_only.jsonl": _output_file(llm_path),
            "singleton_router_labels.csv": _output_file(labels_path),
        },
    }
    write_json(baseline / "manifest.json", _json_safe(baseline_manifest))  # type: ignore[arg-type]
    manifest_validation = validate_baseline_manifest(baseline)

    complementarity_rows = build_complementarity_rows(
        paired,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    output.mkdir(parents=True, exist_ok=True)
    paired_path = output / "paired_cell_outcomes.csv"
    metrics_path = output / "complementarity_by_dataset.csv"
    summary_path = output / "complementarity_summary.json"
    report_path = output / "report.md"
    paired_columns = (
        "suite",
        "dataset",
        "cell_id",
        "row_cluster",
        "baran_correct",
        "llm_correct",
        "baran_valid_prediction",
        "llm_valid_prediction",
        "llm_help",
        "llm_harm",
        "both_correct",
        "both_wrong",
        "oracle_correct",
        "outcome_quadrant",
        "oracle_action",
        "llm_parse_status",
        "llm_final_source",
        "llm_decision",
        "selected_query_id",
    )
    metric_columns = (
        "scope",
        "aggregation",
        "suite",
        "dataset",
        "N",
        "n11",
        "n10",
        "n01",
        "n00",
        *RATE_FIELDS,
        *(f"{field}_ci_low" for field in RATE_FIELDS),
        *(f"{field}_ci_high" for field in RATE_FIELDS),
        "llm_valid_predictions",
        "llm_invalid_predictions",
        "llm_abstain",
        "llm_provider_failure",
        "llm_unchanged_dirty",
        "llm_other_invalid",
        "mcnemar_p",
        "mcnemar_p_holm",
    )
    _write_csv(paired_path, paired, columns=paired_columns)
    _write_csv(metrics_path, complementarity_rows, columns=metric_columns)

    micro = next(row for row in complementarity_rows if row["scope"] == "micro")
    quadrant = tuple(int(micro[field]) for field in ("n11", "n10", "n01", "n00"))
    summary: dict[str, object] = {
        "schema_version": ANALYSIS_SCHEMA,
        "source_run": _relative_or_absolute(source),
        "baseline_manifest": _relative_or_absolute(baseline / "manifest.json"),
        "baseline_manifest_sha256": sha256_file(baseline / "manifest.json"),
        "bootstrap": {
            "unit": "row_cluster",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "confidence": confidence,
        },
        "dataset_rows": [row for row in complementarity_rows if row["scope"] == "dataset"],
        "family_rows": [row for row in complementarity_rows if row["scope"] == "family"],
        "micro": micro,
        "dataset_macro": next(
            row for row in complementarity_rows if row["scope"] == "dataset_macro"
        ),
        "response_reuse": response_audit,
        "quality_audit": {
            "paired_cell_ids_unique": len({str(row["cell_id"]) for row in paired}),
            "baran_fallback_rows": sum(
                row.get("baran_fallback_used") is not False for row in llm_records
            ),
            "quadrants": {
                "n11": quadrant[0],
                "n10": quadrant[1],
                "n01": quadrant[2],
                "n00": quadrant[3],
            },
            "method_metrics_reconciled": True,
            "source_immutability_verified": False,
        },
        "interpretation_boundary": {
            "oracle_is_offline_upper_bound": True,
            "router_performance_claimed": False,
            "independent_llm_replicate": False,
        },
    }
    write_json(summary_path, _json_safe(summary))  # type: ignore[arg-type]
    report_path.write_text(
        _report_markdown(
            source,
            baseline,
            complementarity_rows,
            response_audit,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        ),
        encoding="utf-8",
    )

    source_hashes_after = _source_file_manifest(source)
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("canonical source changed while derived artifacts were built")
    summary["quality_audit"]["source_immutability_verified"] = True  # type: ignore[index]
    summary["outputs"] = {
        "complementarity_by_dataset.csv": _output_file(metrics_path),
        "paired_cell_outcomes.csv": _output_file(paired_path),
        "report.md": _output_file(report_path),
    }
    write_json(summary_path, _json_safe(summary))  # type: ignore[arg-type]

    return {
        "ok": True,
        "network_calls": 0,
        "source_run": str(source),
        "baseline_dir": str(baseline),
        "analysis_dir": str(output),
        "baseline_validation": manifest_validation,
        "cells": len(paired),
        "quadrants": list(quadrant),
        "baran_accuracy": micro["baran_accuracy"],
        "llm_accuracy": micro["llm_accuracy"],
        "oracle_upper_bound": micro["oracle_upper_bound"],
        "physical_calls_required_for_exact_reuse": response_audit[
            "physical_calls_required_for_exact_reuse"
        ],
    }


__all__ = [
    "DEFAULT_ANALYSIS_DIR",
    "DEFAULT_BASELINE_DIR",
    "DEFAULT_SOURCE_RUN",
    "aggregate_outcomes",
    "build_complementarity_rows",
    "build_full_complementarity",
    "pair_baseline_records",
    "validate_baseline_manifest",
]
