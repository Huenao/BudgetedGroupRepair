"""Unique-query provider cost accounting for a no-Baran run."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .data import read_jsonl
from .run_state import write_json


def _usage(record: Mapping[str, Any], *keys: str) -> int | None:
    usage = record.get("usage")
    if not isinstance(usage, Mapping):
        return None
    for key in keys:
        if usage.get(key) is not None:
            return int(usage[key])
    return None


def build_cost_audit(
    run_dir: str | Path,
    *,
    estimated_tokens_by_query: Mapping[str, int],
) -> dict[str, Any]:
    """Reconcile logical arm ledgers with cost-bearing shared checkpoints."""

    root = Path(run_dir).resolve()
    response_files = {
        "model_preflight": root / "llm" / "model_preflight.json",
        "singleton": root / "llm" / "singleton_responses.jsonl",
        "structured": root / "llm" / "structured_responses.jsonl",
        "random": root / "llm" / "random_responses.jsonl",
        "bgr": root / "llm" / "bgr_responses.jsonl",
    }
    logical_by_phase: dict[str, int] = {}
    for phase, path in response_files.items():
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            logical_by_phase[phase] = len(read_jsonl(path))
        else:
            logical_by_phase[phase] = 1

    checkpoints = read_jsonl(root / "llm" / "shared" / "group_query_checkpoint.jsonl")
    # A cache-hit checkpoint carries the original usage for logical comparison,
    # but is not a new provider request. The original cost-bearing checkpoint is
    # retained append-only in this same ledger.
    physical = [
        row
        for row in checkpoints
        if not bool(row.get("cache_hit")) and not bool(row.get("checkpoint_hit"))
    ]
    by_phase: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in physical:
        metadata = row.get("metadata")
        phase = str(metadata.get("phase", "unknown")) if isinstance(metadata, Mapping) else "unknown"
        by_phase[phase].append(row)

    rows: list[dict[str, Any]] = []
    for phase, records in sorted(by_phase.items()):
        prompt_known = [_usage(row, "prompt_tokens", "input_tokens") for row in records]
        completion_known = [
            _usage(row, "completion_tokens", "output_tokens") for row in records
        ]
        observed = sum(int(row.get("observed_total_tokens", 0) or 0) for row in records)
        unknown_attempts = sum(int(row.get("unknown_usage_attempts", 0) or 0) for row in records)
        conservative_unknown = sum(
            int(row.get("unknown_usage_attempts", 0) or 0)
            * int(estimated_tokens_by_query.get(str(row.get("query_id", "")), 0))
            for row in records
        )
        attempts = sum(int(row.get("attempts", 0) or 0) for row in records)
        failed_attempts = sum(
            max(0, int(row.get("attempts", 0) or 0) - (1 if row.get("status") == "success" else 0))
            for row in records
        )
        rows.append(
            {
                "phase": phase,
                "logical_queries": logical_by_phase.get(
                    phase.removeprefix("preliminary_"),
                    logical_by_phase.get(phase, len(records)),
                ),
                "physical_query_invocations": len(records),
                "provider_attempts": attempts,
                "failed_attempts": failed_attempts,
                "unknown_usage_attempts": unknown_attempts,
                "prompt_tokens": (
                    sum(int(value) for value in prompt_known if value is not None)
                    if all(value is not None for value in prompt_known)
                    else None
                ),
                "completion_tokens": (
                    sum(int(value) for value in completion_known if value is not None)
                    if all(value is not None for value in completion_known)
                    else None
                ),
                "observed_total_tokens": observed,
                "conservative_unknown_tokens": conservative_unknown,
                "conservative_total_tokens": observed + conservative_unknown,
                "latency_seconds": sum(float(row.get("latency_seconds", 0.0) or 0.0) for row in records),
            }
        )
    totals = {
        "phase": "TOTAL",
        "logical_queries": sum(logical_by_phase.values()),
        "physical_query_invocations": sum(int(row["physical_query_invocations"]) for row in rows),
        "provider_attempts": sum(int(row["provider_attempts"]) for row in rows),
        "failed_attempts": sum(int(row["failed_attempts"]) for row in rows),
        "unknown_usage_attempts": sum(int(row["unknown_usage_attempts"]) for row in rows),
        "prompt_tokens": (
            sum(int(row["prompt_tokens"]) for row in rows)
            if rows and all(row["prompt_tokens"] is not None for row in rows)
            else None
        ),
        "completion_tokens": (
            sum(int(row["completion_tokens"]) for row in rows)
            if rows and all(row["completion_tokens"] is not None for row in rows)
            else None
        ),
        "observed_total_tokens": sum(int(row["observed_total_tokens"]) for row in rows),
        "conservative_unknown_tokens": sum(
            int(row["conservative_unknown_tokens"]) for row in rows
        ),
        "conservative_total_tokens": sum(int(row["conservative_total_tokens"]) for row in rows),
        "latency_seconds": sum(float(row["latency_seconds"]) for row in rows),
    }
    output_rows = [*rows, totals]
    metrics = root / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(metrics / "api_cost_audit.csv", index=False)
    result = {
        "rows": output_rows,
        "totals": totals,
        "checkpoint_records": len(checkpoints),
        "cost_bearing_records": len(physical),
        "deduplication": "append-only shared checkpoint; cache/checkpoint hits excluded",
    }
    write_json(metrics / "api_cost_summary.json", result)
    return result


__all__ = ["build_cost_audit"]
