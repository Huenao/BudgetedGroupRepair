"""Strictly offline final-result replay for the repaired MGreedy selections.

The paid provider work is performed by :mod:`missing_query_retry`.  This module
has no provider client and deliberately does not call ``run_selected_llm_stage``.
It creates five new configuration-level runs, projects only the repaired
selection files belonging to each historical configuration, builds an
append-only response ledger, and executes only final materialisation, metrics,
and audit stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data import write_jsonl
from .protocol import target_order
from .router_v3 import ExperimentRunner, load_json
from .run_state import canonical_json_sha256, sha256_file, write_json


IDENTITY_FIELDS = (
    "query_id",
    "prompt_hash",
    "provider_request_hash",
    "model",
    "prompt_schema_version",
)


@dataclass(frozen=True)
class FinalReplaySpec:
    name: str
    source_run_id: str
    new_run_id: str
    selection_backends: tuple[str, ...]
    expected_selection_slices: int
    central_retry_cost: bool = False
    comparison_spec: str | None = None


DEFAULT_REPLAY_SPECS = (
    FinalReplaySpec(
        "base",
        "no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_base_final",
        ("lightgbm", "xgboost"),
        90,
        central_retry_cost=True,
    ),
    FinalReplaySpec(
        "lightgbm_sweep",
        "no_baran_router_v3_deepseek_v4_20260726_budget_sweep_k24_lightgbm",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_lightgbm_sweep_final",
        ("lightgbm",),
        90,
    ),
    FinalReplaySpec(
        "catboost",
        "no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_catboost_final",
        ("catboost",),
        45,
        comparison_spec="base",
    ),
    FinalReplaySpec(
        "tabiclv2",
        "no_baran_router_v3_tabiclv2_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_tabiclv2_final",
        ("tabiclv2",),
        108,
        comparison_spec="base",
    ),
    FinalReplaySpec(
        "tabpfn3",
        "no_baran_router_v3_tabpfn3_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_missing35_tabpfn3_final",
        ("tabpfn3",),
        108,
        comparison_spec="tabiclv2",
    ),
)

DEFAULT_CACHE_SOURCE_RUN_IDS = (
    "no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all",
    "no_baran_router_v3_deepseek_v4_20260726_budget_sweep_k24_lightgbm",
    "no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost",
    "no_baran_router_v3_tabiclv2_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
    "no_baran_router_v3_tabpfn3_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
)

DEFAULT_RESELECTION_RUNS = {
    "lightgbm": "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_lightgbm_reselect",
    "xgboost": "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_xgboost_reselect",
    "catboost": "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_catboost_reselect",
    "tabiclv2": "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_tabiclv2_reselect",
    "tabpfn3": "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_tabpfn3_reselect",
}

_LINK_FILES = (
    "input_data_manifest.json",
    "gates/split_audit.csv",
    "llm/calibration_execution.jsonl",
    "llm/calibration_pair_labels.csv",
    "llm/calibration_plan.json",
    "llm/model_preflight.json",
    "llm/offline_group_calibration_baran_fallbacks.jsonl",
    "metrics/logical_budget_ledger.csv",
    "metrics/selection_audit.csv",
)


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


def _identity(row: Mapping[str, Any], *, schema: str | None = None) -> tuple[str, ...]:
    metadata = row.get("metadata")
    metadata_schema = (
        str(metadata.get("prompt_schema_version", ""))
        if isinstance(metadata, Mapping)
        else ""
    )
    return (
        str(row.get("query_id", "")),
        str(row.get("prompt_hash", "")),
        str(row.get("provider_request_hash", "")),
        str(row.get("model", row.get("model_requested", ""))),
        metadata_schema or str(schema or row.get("prompt_schema_version", "")),
    )


def _phase(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    return str(metadata.get("phase", "")) if isinstance(metadata, Mapping) else ""


def _phase_compatible(source: str) -> bool:
    return source == "online_selected_union" or source in {
        "offline_group_calibration",
        "preliminary_singleton",
        "preliminary_structured",
        "preliminary_random",
    }


def _checkpoint_path(run_dir: Path) -> Path:
    for relative in (
        "llm/shared/group_query_checkpoint.jsonl",
        "llm/group_query_checkpoint.jsonl",
    ):
        candidate = run_dir / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no group-query checkpoint in {run_dir}")


def load_response_authority(
    cache_union_csv: str | Path,
    *,
    expected_count: int = 32_777,
) -> tuple[dict[str, dict[str, str]], ...]:
    path = Path(cache_union_csv).resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != expected_count:
        raise ValueError(
            f"response authority count drift: expected={expected_count}, observed={len(rows)}"
        )
    identities = [tuple(str(row.get(field, "")) for field in IDENTITY_FIELDS) for row in rows]
    if any(not all(value) for value in identities):
        raise ValueError("response authority contains an incomplete identity")
    if len(set(identities)) != len(identities):
        raise ValueError("response authority contains duplicate identities")
    if len({identity[0] for identity in identities}) != len(identities):
        raise ValueError("response authority contains duplicate query IDs")
    return tuple(rows)


def _imported_row(
    row: Mapping[str, Any],
    *,
    source_run_id: str,
    source_row_index: int,
    role: str,
    fresh: bool,
) -> dict[str, Any]:
    copied = dict(row)
    metadata = row.get("metadata")
    copied["metadata"] = {
        **(dict(metadata) if isinstance(metadata, Mapping) else {}),
        "final_replay_role": role,
        "final_replay_source_run_id": source_run_id,
        "final_replay_source_row_index": source_row_index,
        "final_replay_cost_allocation": "fresh_central" if fresh else "imported",
    }
    copied["cache_hit"] = not fresh
    copied["checkpoint_hit"] = False
    copied["imported_response"] = not fresh
    return copied


def build_deterministic_response_bank(
    authority_rows: Sequence[Mapping[str, str]],
    cache_sources: Sequence[tuple[str, str | Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild the audited fixed-source-order response bank exactly once."""

    authority = {
        tuple(str(row.get(field, "")) for field in IDENTITY_FIELDS): dict(row)
        for row in authority_rows
    }
    by_identity: dict[tuple[str, ...], list[tuple[int, int, str, dict[str, Any]]]] = {
        identity: [] for identity in authority
    }
    by_query = {identity[0]: identity for identity in authority}
    for source_order, (run_id, raw_checkpoint) in enumerate(cache_sources):
        checkpoint = Path(raw_checkpoint).resolve()
        for row_index, row in enumerate(_iter_jsonl(checkpoint), start=1):
            query_id = str(row.get("query_id", ""))
            identity = by_query.get(query_id)
            if identity is None or row.get("model_matches_request", True) is False:
                continue
            phase = _phase(row)
            if not _phase_compatible(phase):
                continue
            if _identity(row, schema=identity[4]) != identity:
                continue
            by_identity[identity].append(
                (source_order, row_index, run_id, dict(row))
            )

    bank: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for identity in sorted(authority):
        candidates = sorted(
            by_identity[identity], key=lambda value: (value[0], -value[1])
        )
        successes = [value for value in candidates if value[3].get("status") == "success"]
        chosen = successes[0] if successes else (candidates[0] if candidates else None)
        if chosen is None:
            raise ValueError(f"response bank is missing {identity[0]}")
        source_order, row_index, source_run_id, row = chosen
        declared = authority[identity]
        if (
            str(declared.get("chosen_source_run", "")) != source_run_id
            or str(declared.get("chosen_status", "")) != str(row.get("status", ""))
            or str(declared.get("chosen_phase", "")) != _phase(row)
        ):
            raise ValueError(f"audited response choice drift for {identity[0]}")
        bank.append(
            _imported_row(
                row,
                source_run_id=source_run_id,
                source_row_index=row_index,
                role="deterministic_response_bank",
                fresh=False,
            )
        )
        audit.append(
            {
                "query_id": identity[0],
                "source_order": source_order,
                "source_run_id": source_run_id,
                "source_row_index": row_index,
                "status": str(row.get("status", "")),
                "phase": _phase(row),
            }
        )
    return bank, audit


def load_supplemental_last_rows(
    authority_jsonl: str | Path,
    retry_checkpoint: str | Path,
    *,
    expected_count: int = 35,
) -> list[dict[str, Any]]:
    authority_rows = list(_iter_jsonl(Path(authority_jsonl).resolve()))
    if len(authority_rows) != expected_count:
        raise ValueError(
            f"supplemental authority count drift: expected={expected_count}, "
            f"observed={len(authority_rows)}"
        )
    authority = {
        tuple(str(row.get(field, "")) for field in IDENTITY_FIELDS): row
        for row in authority_rows
    }
    if len(authority) != len(authority_rows):
        raise ValueError("supplemental authority contains duplicate identities")
    by_query = {identity[0]: identity for identity in authority}
    if len(by_query) != len(authority):
        raise ValueError("supplemental authority contains duplicate query IDs")
    last: dict[tuple[str, ...], tuple[int, dict[str, Any]]] = {}
    for row_index, row in enumerate(_iter_jsonl(Path(retry_checkpoint).resolve()), start=1):
        identity = by_query.get(str(row.get("query_id", "")))
        if identity is None:
            continue
        if _identity(row, schema=identity[4]) != identity:
            raise ValueError(f"supplemental identity drift for {identity[0]}")
        if not _phase_compatible(_phase(row)):
            raise ValueError(f"supplemental phase drift for {identity[0]}")
        last[identity] = (row_index, dict(row))
    if set(last) != set(authority):
        raise ValueError(
            f"retry checkpoint is missing {len(set(authority) - set(last))} supplemental rows"
        )
    return [last[identity][1] for identity in sorted(last)]


def merge_checkpoint_rows(
    *,
    source_run_id: str,
    source_prefix: Sequence[Mapping[str, Any]],
    response_bank: Sequence[Mapping[str, Any]],
    supplemental_rows: Sequence[Mapping[str, Any]],
    retry_run_id: str,
    central_retry_cost: bool,
) -> list[dict[str, Any]]:
    """Return prefix + deterministic bank + supplemental-last-row ledger."""

    merged = [
        _imported_row(
            row,
            source_run_id=source_run_id,
            source_row_index=index,
            role="source_checkpoint_prefix",
            fresh=False,
        )
        for index, row in enumerate(source_prefix, start=1)
    ]
    merged.extend(dict(row) for row in response_bank)
    merged.extend(
        _imported_row(
            row,
            source_run_id=retry_run_id,
            source_row_index=index,
            role="supplemental_missing_query_last_row",
            fresh=central_retry_cost,
        )
        for index, row in enumerate(supplemental_rows, start=1)
    )
    supplemental_ids = [str(row.get("query_id", "")) for row in supplemental_rows]
    if len(supplemental_ids) != len(set(supplemental_ids)):
        raise ValueError("supplemental rows contain duplicate query IDs")
    fresh_rows = [row for row in merged if not bool(row.get("cache_hit"))]
    expected_fresh = len(supplemental_rows) if central_retry_cost else 0
    if len(fresh_rows) != expected_fresh:
        raise AssertionError("fresh retry-cost allocation changed")
    return merged


def plan_selection_projection(
    *,
    source_run: str | Path,
    reselection_runs: Mapping[str, str | Path],
    backends: Sequence[str],
    expected_count: int,
) -> list[tuple[Path, Path]]:
    """Map each historical config-relative selection to its repaired document."""

    source = Path(source_run).resolve()
    pairs: list[tuple[Path, Path]] = []
    for backend in backends:
        old_root = source / "selections" / backend
        repaired_root = Path(reselection_runs[backend]).resolve() / "selections" / backend
        old_paths = sorted(old_root.rglob("*.json"))
        if not old_paths:
            raise FileNotFoundError(f"no source selections for {backend}: {old_root}")
        for old_path in old_paths:
            repaired = repaired_root / old_path.relative_to(old_root)
            if not repaired.is_file():
                raise FileNotFoundError(f"missing repaired selection: {repaired}")
            old_doc = load_json(old_path)
            repaired_doc = load_json(repaired)
            old_ids = old_doc.get("selected_query_ids", old_doc.get("selected", []))
            new_ids = repaired_doc.get(
                "selected_query_ids", repaired_doc.get("selected", [])
            )
            if old_ids != new_ids:
                raise ValueError(f"selection changed unexpectedly: {old_path}")
            if str(repaired_doc.get("backend", "")) != backend:
                raise ValueError(f"repaired selection backend drift: {repaired}")
            pairs.append((old_path, repaired))
    if len(pairs) != expected_count:
        raise ValueError(
            f"selection projection count drift: expected={expected_count}, observed={len(pairs)}"
        )
    return pairs


def copy_selection_projection(
    pairs: Sequence[tuple[Path, Path]],
    *,
    source_run: str | Path,
    destination_run: str | Path,
) -> list[dict[str, Any]]:
    source = Path(source_run).resolve()
    destination = Path(destination_run).resolve()
    records: list[dict[str, Any]] = []
    for old_path, repaired in pairs:
        relative = old_path.resolve().relative_to(source)
        target = destination / relative
        if target.exists():
            raise FileExistsError(f"refusing to overwrite selection: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repaired, target)
        records.append(
            {
                "relative_path": relative.as_posix(),
                "source_selection_sha256": sha256_file(old_path),
                "repaired_selection_sha256": sha256_file(repaired),
                "copied_selection_sha256": sha256_file(target),
            }
        )
    return records


def link_whitelisted_inputs(
    *,
    source_run: str | Path,
    destination_run: str | Path,
    backends: Sequence[str],
) -> list[dict[str, Any]]:
    """Symlink only inputs consumed by final, metrics, and audit stages."""

    source = Path(source_run).resolve()
    destination = Path(destination_run).resolve()
    relative_paths = [Path(value) for value in _LINK_FILES]
    relative_paths.extend(path.relative_to(source) for path in (source / "baran").glob("*.jsonl"))
    relative_paths.extend(
        path.relative_to(source)
        for path in (source / "groups" / "candidates").glob("*.jsonl")
    )
    for backend in backends:
        relative_paths.extend(
            path.relative_to(source)
            for path in (source / "gates" / backend).glob("variant_*/*.csv")
        )
    records: list[dict[str, Any]] = []
    for relative in sorted(set(relative_paths)):
        origin = (source / relative).resolve()
        if not origin.is_file():
            raise FileNotFoundError(f"missing replay input: {origin}")
        target = destination / relative
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"refusing to overwrite replay input: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(origin)
        records.append(
            {
                "relative_path": relative.as_posix(),
                "source_path": str(origin),
                "source_sha256": sha256_file(origin),
                "link_read_only_input": True,
            }
        )
    return records


def final_response_index(
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in checkpoint_rows:
        key = (str(row.get("query_id", "")), str(row.get("prompt_hash", "")))
        if all(key):
            index[key] = dict(row)
    return index


def materialize_selected_execution(
    *,
    query_prompt_pairs: Sequence[tuple[str, str]],
    checkpoint_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    index = final_response_index(checkpoint_rows)
    results: list[dict[str, Any]] = []
    for query_id, prompt_hash in sorted(query_prompt_pairs):
        row = index.get((query_id, prompt_hash))
        if row is None:
            raise ValueError(f"selected execution is missing {query_id}")
        copied = dict(row)
        copied["checkpoint_hit"] = bool(copied.get("cache_hit"))
        results.append(copied)
    if len(results) != len(set(query_prompt_pairs)):
        raise ValueError("selected execution identities are duplicated")
    return results


def run_final_replay_stages(runner: Any) -> dict[str, Any]:
    """The only executable stages allowed in this offline replay."""

    final = runner.build_final_records_stage()
    metrics = runner.build_metrics_stage()
    audit = runner.build_audit_stage()
    return {"final": final, "metrics": metrics, "audit": audit}


def _selection_query_ids(pairs: Sequence[tuple[Path, Path]]) -> set[str]:
    selected: set[str] = set()
    for _, repaired in pairs:
        document = load_json(repaired)
        selected.update(
            str(value)
            for value in document.get(
                "selected_query_ids", document.get("selected", [])
            )
        )
    return selected


def _runner_for_spec(
    *,
    spec: FinalReplaySpec,
    project_root: Path,
    data_root: Path,
    vendor_root: Path,
    runs_root: Path,
    comparison_run: Path | None,
) -> ExperimentRunner:
    source = runs_root / spec.source_run_id
    return ExperimentRunner.create(
        project_root=project_root,
        data_root=data_root,
        config_path=source / "bound_experiment_config.json",
        llm_config_path=source / "bound_llm_config.json",
        vendor_root=vendor_root,
        runs_root=runs_root,
        run_id=spec.new_run_id,
        resume=False,
        baran_source_run=source,
        response_reuse_run=source,
        calibration_source_run=source,
        router_artifact_reuse_run=source,
        router_comparison_run=comparison_run,
    )


def execute_final_replay_matrix(
    *,
    project_root: str | Path,
    data_root: str | Path,
    vendor_root: str | Path,
    runs_root: str | Path,
    cache_audit_run: str | Path,
    retry_run: str | Path,
    specs: Sequence[FinalReplaySpec] = DEFAULT_REPLAY_SPECS,
    reselection_run_ids: Mapping[str, str] = DEFAULT_RESELECTION_RUNS,
) -> dict[str, Any]:
    """Create and execute the five new offline configuration-level runs."""

    project = Path(project_root).resolve()
    data = Path(data_root).resolve()
    vendor = Path(vendor_root).resolve()
    runs = Path(runs_root).resolve()
    cache_audit = Path(cache_audit_run).resolve()
    retry = Path(retry_run).resolve()
    if any((runs / spec.new_run_id).exists() for spec in specs):
        existing = [spec.new_run_id for spec in specs if (runs / spec.new_run_id).exists()]
        raise FileExistsError(f"refusing to overwrite replay runs: {existing}")
    retry_manifest = load_json(retry / "run_manifest.json")
    if (
        str(retry_manifest.get("status")) != "complete"
        or int(retry_manifest.get("result_count", -1)) != 35
        or bool(retry_manifest.get("preflight_called"))
    ):
        raise ValueError("supplemental retry run is not the completed 35-request run")

    authority = load_response_authority(cache_audit / "cache_union_audit.csv")
    cache_sources = [
        (run_id, _checkpoint_path(runs / run_id))
        for run_id in DEFAULT_CACHE_SOURCE_RUN_IDS
    ]
    bank, bank_audit = build_deterministic_response_bank(authority, cache_sources)
    supplemental = load_supplemental_last_rows(
        cache_audit / "missing_query_union_enriched.jsonl",
        _checkpoint_path(retry),
    )
    if sum(row.get("status") == "success" for row in supplemental) != 30:
        raise ValueError("supplemental result status drift from the audited 30/5 outcome")

    reselect_paths = {
        backend: runs / run_id for backend, run_id in reselection_run_ids.items()
    }
    completed: dict[str, Path] = {}
    summaries: list[dict[str, Any]] = []
    for spec in specs:
        source = runs / spec.source_run_id
        pairs = plan_selection_projection(
            source_run=source,
            reselection_runs=reselect_paths,
            backends=spec.selection_backends,
            expected_count=spec.expected_selection_slices,
        )
        source_plan = load_json(source / "llm" / "selected_union_plan.json")
        repaired_bgr = _selection_query_ids(pairs)
        if repaired_bgr != {str(value) for value in source_plan.get("bgr_query_ids", [])}:
            raise ValueError(f"BGR selected union drift for replay spec {spec.name}")
        comparison = (
            completed.get(spec.comparison_spec)
            if spec.comparison_spec is not None
            else None
        )
        if spec.comparison_spec is not None and comparison is None:
            raise RuntimeError(f"comparison replay is not complete: {spec.comparison_spec}")
        runner = _runner_for_spec(
            spec=spec,
            project_root=project,
            data_root=data,
            vendor_root=vendor,
            runs_root=runs,
            comparison_run=comparison,
        )
        run_dir = runner.paths.run_dir
        try:
            linked = link_whitelisted_inputs(
                source_run=source,
                destination_run=run_dir,
                backends=spec.selection_backends,
            )
            copied = copy_selection_projection(
                pairs, source_run=source, destination_run=run_dir
            )
            prefix = list(_iter_jsonl(_checkpoint_path(source)))
            checkpoint = merge_checkpoint_rows(
                source_run_id=spec.source_run_id,
                source_prefix=prefix,
                response_bank=bank,
                supplemental_rows=supplemental,
                retry_run_id=retry.name,
                central_retry_cost=spec.central_retry_cost,
            )
            checkpoint_path = run_dir / "llm" / "group_query_checkpoint.jsonl"
            write_jsonl(checkpoint_path, checkpoint)
            plan_ids = {str(value) for value in source_plan.get("query_ids", [])}
            action_pairs: list[tuple[str, str]] = []
            action_by_id = {}
            for suite, dataset in target_order():
                for action in runner._load_actions(suite, dataset):
                    if action.query_id in plan_ids:
                        action_by_id[action.query_id] = action
                        action_pairs.append((action.query_id, action.prompt_hash))
            if set(action_by_id) != plan_ids:
                raise ValueError(f"selected action coverage drift for {spec.name}")
            execution = materialize_selected_execution(
                query_prompt_pairs=action_pairs,
                checkpoint_rows=checkpoint,
            )
            write_jsonl(run_dir / "llm" / "selected_execution.jsonl", execution)
            fallback = runner._materialize_baran_fallbacks(
                tuple(action_by_id[query_id] for query_id in sorted(action_by_id)),
                execution,
                phase="online_selected_union",
            )
            online_ids = (
                sorted(str(row.get("query_id")) for row in supplemental)
                if spec.central_retry_cost
                else []
            )
            if spec.central_retry_cost and not set(online_ids).issubset(plan_ids):
                raise ValueError("central supplemental identities are outside the base plan")
            status_by_query = {
                str(row.get("query_id")): str(row.get("status")) for row in execution
            }
            failures = sorted(
                query_id for query_id, status in status_by_query.items() if status != "success"
            )
            online_set = set(online_ids)
            cached_failures = sorted(set(failures) - online_set)
            cached_successes = sum(
                status == "success" and query_id not in online_set
                for query_id, status in status_by_query.items()
            )
            online_estimated_tokens = sum(
                int(action_by_id[query_id].estimated_total_tokens)
                for query_id in online_ids
            )
            retry_multiplier = int(runner.llm_config.get("max_retries", 0)) + 1
            replay_plan = {
                **source_plan,
                "online_query_ids": online_ids,
                "online_physical_queries": len(online_ids),
                "online_union_estimated_tokens": online_estimated_tokens,
                "combined_physical_estimated_tokens": online_estimated_tokens,
                "model_preflight_estimated_tokens": 0,
                "retry_multiplier": retry_multiplier,
                "retry_adjusted_token_cap": retry_multiplier
                * online_estimated_tokens,
                "cached_failure_query_ids": cached_failures,
                "cached_terminal_failure_queries_in_union": len(cached_failures),
                "cached_success_queries_in_union": cached_successes,
                "cached_terminal_queries_in_union": len(plan_ids) - len(online_ids),
                "final_failure_query_ids": failures,
                "api_called_by_final_replay": False,
                "supplemental_retry_run": str(retry),
                "supplemental_retry_cost_mode": (
                    "fresh_central" if spec.central_retry_cost else "imported"
                ),
            }
            write_json(run_dir / "llm" / "selected_union_plan.json", replay_plan)
            runner.state.update_stage(
                "offline_final_replay_inputs",
                "complete",
                selected_execution_records=len(execution),
                response_bank_records=len(bank),
                supplemental_records=len(supplemental),
                **fallback,
            )
            stage_results = run_final_replay_stages(runner)
            provenance = {
                "run_kind": "strict_offline_mgreedy_final_replay",
                "source_run_id": spec.source_run_id,
                "source_run_manifest_sha256": sha256_file(source / "run_manifest.json"),
                "cache_audit_run": str(cache_audit),
                "cache_union_authority_sha256": sha256_file(
                    cache_audit / "cache_union_audit.csv"
                ),
                "retry_run": str(retry),
                "retry_manifest_sha256": sha256_file(retry / "run_manifest.json"),
                "central_retry_cost": spec.central_retry_cost,
                "linked_read_only_inputs": linked,
                "copied_repaired_selections": copied,
                "response_bank_count": len(bank),
                "response_bank_identity_sha256": canonical_json_sha256(bank_audit),
                "supplemental_count": len(supplemental),
                "supplemental_successes": sum(
                    row.get("status") == "success" for row in supplemental
                ),
                "supplemental_failures": sum(
                    row.get("status") != "success" for row in supplemental
                ),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "selected_execution_sha256": sha256_file(
                    run_dir / "llm" / "selected_execution.jsonl"
                ),
                "fallback_sha256": sha256_file(
                    run_dir / "llm" / "online_selected_union_baran_fallbacks.jsonl"
                ),
                "api_called_by_final_replay": False,
                "run_selected_llm_stage_called": False,
                "stages_executed": ["final_records", "metrics", "audit"],
            }
            write_json(run_dir / "provenance" / "final_replay.json", provenance)
            runner.state.complete(
                required_stages=("final_records", "metrics", "audit"),
                final_replay=provenance,
            )
            summaries.append(
                {"name": spec.name, "run_dir": str(run_dir), **stage_results}
            )
            completed[spec.name] = run_dir
        except BaseException as error:
            runner.state.update_stage(
                "offline_final_replay",
                "failed",
                failure_class=type(error).__name__,
                failure_message=str(error)[:500],
            )
            raise
    return {"runs": summaries, "api_called": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict offline BGR final replay")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--vendor-root", type=Path, default=Path("vendor/raha_source"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--cache-audit-run", type=Path, required=True)
    parser.add_argument("--retry-run", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = execute_final_replay_matrix(
        project_root=args.project_root,
        data_root=args.data_root,
        vendor_root=args.vendor_root,
        runs_root=args.runs_root,
        cache_audit_run=args.cache_audit_run,
        retry_run=args.retry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
