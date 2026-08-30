"""Offline-only repaired-MGreedy replay and cross-backbone cache audit.

This module deliberately contains no provider-call path.  It rebuilds
selections from frozen gate ledgers, writes fresh run IDs, and reports the
deduplicated request identities that would still require authorization.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .group_llm import DeepSeekGroupClient, GroupClientConfig, GroupLLMJob
from .group_objective import DEFAULT_UPLIFT_SCALE, GroupUpliftObjective
from .group_optimizer import (
    NUMERIC_SEMANTICS,
    eager_gain_cost_greedy,
    exhaustive_optimum,
    lazy_gain_cost_greedy,
)
from .run_state import canonical_json_sha256, sha256_file


@dataclass(frozen=True)
class BackboneSource:
    backend: str
    source_run_id: str
    new_run_id: str


DEFAULT_BACKBONE_SOURCES = (
    BackboneSource(
        "lightgbm",
        "no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_lightgbm_reselect",
    ),
    BackboneSource(
        "lightgbm",
        "no_baran_router_v3_deepseek_v4_20260726_budget_sweep_k24_lightgbm",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_lightgbm_reselect",
    ),
    BackboneSource(
        "xgboost",
        "no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_xgboost_reselect",
    ),
    BackboneSource(
        "catboost",
        "no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_catboost_reselect",
    ),
    BackboneSource(
        "tabiclv2",
        "no_baran_router_v3_tabiclv2_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_tabiclv2_reselect",
    ),
    BackboneSource(
        "tabpfn3",
        "no_baran_router_v3_tabpfn3_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
        "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_tabpfn3_reselect",
    ),
)

DEFAULT_CACHE_SOURCE_RUN_IDS = (
    "no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all",
    "no_baran_router_v3_deepseek_v4_20260726_budget_sweep_k24_lightgbm",
    "no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost",
    "no_baran_router_v3_tabiclv2_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
    "no_baran_router_v3_tabpfn3_deepseek_v4_20260813_matrix_k1248_budget_sweep_k24",
)

DEFAULT_AUDIT_RUN_ID = (
    "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_cache_union"
)

_FREEZE_TERMINAL_FAILURE_REVISIONS = {
    "router_v3_budget_sweep_exact_size_conditioned",
    "router_v3_catboost_exact_size_conditioned",
    "router_v3_tabiclv2_exact_size_conditioned",
    "router_v3_tabpfn3_exact_size_conditioned",
    "router_v3_tabiclv2_k1248_budget_sweep_k24_exact_size_conditioned",
    "router_v3_tabpfn3_k1248_budget_sweep_k24_exact_size_conditioned",
}


def _phase_compatible(source: str, target: str = "online_selected_union") -> bool:
    """Mirror the frozen group-LLM checkpoint phase compatibility contract."""

    if source == target:
        return True
    if source == "model_preflight" and target == "preliminary_singleton":
        return True
    if source == "offline_group_calibration" and target == "online_selected_union":
        return True
    preliminary = source in {
        "preliminary_singleton",
        "preliminary_structured",
        "preliminary_random",
    }
    return preliminary and target in {
        "bgr_selected_union",
        "offline_group_calibration",
        "online_selected_union",
    }


@dataclass(frozen=True)
class _SelectionTask:
    source: BackboneSource
    source_path: Path
    relative_path: Path
    gate_path: Path
    document: Mapping[str, object]
    freezes_terminal_failures: bool

    @property
    def slice_id(self) -> str:
        return "/".join(
            (
                self.source.backend,
                str(self.document["scenario"]),
                f"variant_{self.document['group_size_variant']}",
                f"{int(round(float(self.document['budget_share']) * 100)):02d}pct",
                f"{self.document['suite']}__{self.document['dataset']}",
            )
        )


@dataclass(frozen=True)
class _Ledger:
    objective: GroupUpliftObjective
    costs: Mapping[str, int]
    query_cells: Mapping[str, tuple[str, ...]]
    query_group_size: Mapping[str, int]


@dataclass(frozen=True)
class _ActionIdentity:
    query_id: str
    prompt_hash: str
    provider_request_hash: str
    model: str
    prompt_schema_version: str
    estimated_tokens: int
    group_size: int
    group_view: str
    suite: str
    dataset: str

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.query_id,
            self.prompt_hash,
            self.provider_request_hash,
            self.model,
            self.prompt_schema_version,
        )


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _selection_hash(values: Sequence[str]) -> str:
    return canonical_json_sha256([str(value) for value in values])


def _discover_tasks(runs_root: Path, sources: Sequence[BackboneSource]) -> list[_SelectionTask]:
    tasks_by_slice: dict[tuple[str, str, str, str, str, int], _SelectionTask] = {}
    canonical_gate_run = {
        source.backend: runs_root / source.source_run_id
        for source in reversed(sources)
    }
    checked_new_runs: set[str] = set()
    for source in sources:
        source_run = runs_root / source.source_run_id
        new_run = runs_root / source.new_run_id
        if not source_run.is_dir():
            raise FileNotFoundError(f"missing source run: {source_run}")
        if source.new_run_id not in checked_new_runs and new_run.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {new_run}")
        checked_new_runs.add(source.new_run_id)
        experiment = _load_json(source_run / "bound_experiment_config.json")
        revision = str(experiment.get("router_revision", ""))
        selection_root = source_run / "selections" / source.backend
        paths = sorted(selection_root.rglob("*.json"))
        if not paths:
            raise FileNotFoundError(
                f"no selection JSON files for {source.backend}: {selection_root}"
            )
        for path in paths:
            document = _load_json(path)
            if str(document.get("backend")) != source.backend:
                raise ValueError(f"backend mismatch in {path}")
            relative = path.relative_to(selection_root)
            key = f"{document['suite']}__{document['dataset']}"
            variant = str(document["group_size_variant"])
            gate_path = (
                canonical_gate_run[source.backend]
                / "gates"
                / source.backend
                / f"variant_{variant}"
                / f"{key}.csv"
            )
            if not gate_path.is_file():
                raise FileNotFoundError(f"missing frozen gate ledger: {gate_path}")
            task = _SelectionTask(
                source=source,
                source_path=path,
                relative_path=relative,
                gate_path=gate_path,
                document=document,
                freezes_terminal_failures=(
                    revision in _FREEZE_TERMINAL_FAILURE_REVISIONS
                ),
            )
            slice_key = (
                source.backend,
                str(document["scenario"]),
                str(document["suite"]),
                str(document["dataset"]),
                variant,
                int(round(float(document["budget_share"]) * 100)),
            )
            incumbent = tasks_by_slice.get(slice_key)
            if incumbent is not None:
                incumbent_ids = incumbent.document.get(
                    "selected_query_ids", incumbent.document.get("selected", [])
                )
                current_ids = document.get("selected_query_ids", document.get("selected", []))
                if incumbent_ids != current_ids:
                    raise ValueError(f"duplicate historical slice conflicts: {slice_key}")
                continue
            tasks_by_slice[slice_key] = task
    return list(tasks_by_slice.values())


def _load_gate_ledger(path: Path) -> _Ledger:
    gains: dict[str, dict[str, float]] = defaultdict(dict)
    costs: dict[str, int] = {}
    group_sizes: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "cell_id",
            "query_id",
            "conservative_uplift",
            "estimated_total_tokens",
            "group_size",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"gate ledger schema mismatch: {path}")
        for row_number, row in enumerate(reader, start=2):
            query_id = str(row["query_id"])
            cell_id = str(row["cell_id"])
            if cell_id in gains[query_id]:
                raise ValueError(f"duplicate cell/query at {path}:{row_number}")
            gains[query_id][cell_id] = float(row["conservative_uplift"])
            cost = int(row["estimated_total_tokens"])
            if cost <= 0:
                raise ValueError(f"non-positive token cost at {path}:{row_number}")
            if query_id in costs and costs[query_id] != cost:
                raise ValueError(f"inconsistent query cost at {path}:{row_number}")
            costs[query_id] = cost
            group_size = int(row["group_size"])
            if query_id in group_sizes and group_sizes[query_id] != group_size:
                raise ValueError(f"inconsistent group size at {path}:{row_number}")
            group_sizes[query_id] = group_size
    if not gains:
        raise ValueError(f"empty gate ledger: {path}")
    objective = GroupUpliftObjective(gains)
    return _Ledger(
        objective=objective,
        costs=dict(costs),
        query_cells={query_id: tuple(sorted(values)) for query_id, values in gains.items()},
        query_group_size=dict(group_sizes),
    )


def _representative_queries(ledger: _Ledger, limit: int) -> tuple[str, ...]:
    """Choose a deterministic overlap-rich real-ledger subinstance."""

    ordered = tuple(sorted(ledger.costs))
    seed = min(ordered, key=lambda query_id: (-len(ledger.query_cells[query_id]), query_id))
    selected = [seed]
    selected_set = {seed}
    covered = set(ledger.query_cells[seed])
    while len(selected) < min(limit, len(ordered)):
        overlapping = [
            query_id
            for query_id in ordered
            if query_id not in selected_set
            and covered.intersection(ledger.query_cells[query_id])
        ]
        if overlapping:
            query_id = overlapping[0]
        else:
            query_id = next(value for value in ordered if value not in selected_set)
        selected.append(query_id)
        selected_set.add(query_id)
        covered.update(ledger.query_cells[query_id])
    return tuple(selected)


def _sample_budget(ledger: _Ledger, candidates: Sequence[str]) -> int:
    total = sum(ledger.costs[query_id] for query_id in candidates)
    return min(total, max(min(ledger.costs[query_id] for query_id in candidates), total // 5))


def _audit_real_ledger(path: Path, ledger: _Ledger) -> dict[str, object]:
    sampled = _representative_queries(ledger, 64)
    sampled_budget = _sample_budget(ledger, sampled)
    lazy = lazy_gain_cost_greedy(
        ledger.objective,
        ledger.costs,
        sampled_budget,
        candidates=sampled,
    )
    eager = eager_gain_cost_greedy(
        ledger.objective,
        ledger.costs,
        sampled_budget,
        candidates=sampled,
    )
    lazy_eager_match = (
        lazy.selected_query_ids == eager.selected_query_ids
        and lazy.total_cost == eager.total_cost
        and lazy.objective_units == eager.objective_units
    )
    if not lazy_eager_match:
        raise AssertionError(f"real-ledger lazy/eager mismatch: {path}")

    exact_candidates = sampled[: min(10, len(sampled))]
    exact_budget = _sample_budget(ledger, exact_candidates)
    exact_lazy = lazy_gain_cost_greedy(
        ledger.objective,
        ledger.costs,
        exact_budget,
        candidates=exact_candidates,
    )
    exact = exhaustive_optimum(
        ledger.objective,
        ledger.costs,
        exact_budget,
        candidates=exact_candidates,
        max_candidates=10,
    )
    ratio = (
        exact_lazy.objective_units / exact.objective_units
        if exact.objective_units > 0
        else 1.0
    )
    if ratio < 0.405:
        raise AssertionError(f"real-ledger observed ratio below 0.405: {path}")
    return {
        "gate_path": str(path),
        "candidate_count": len(ledger.costs),
        "cell_count": len(ledger.objective.cell_ids),
        "sampled_candidate_count": len(sampled),
        "sampled_budget": sampled_budget,
        "sampled_lazy_eager_match": True,
        "exact_candidate_count": len(exact_candidates),
        "exact_budget": exact_budget,
        "exact_mgreedy_objective_units": exact_lazy.objective_units,
        "exact_optimum_objective_units": exact.objective_units,
        "observed_ratio": ratio,
        "observed_ratio_is_regression_not_proof": True,
    }


def _reselect_tasks(
    tasks: Sequence[_SelectionTask],
) -> tuple[
    list[dict[str, object]],
    list[tuple[_SelectionTask, dict[str, object]]],
    list[dict[str, object]],
    dict[str, list[dict[str, object]]],
]:
    by_gate: dict[Path, list[_SelectionTask]] = defaultdict(list)
    for task in tasks:
        by_gate[task.gate_path].append(task)

    diff_rows: list[dict[str, object]] = []
    outputs: list[tuple[_SelectionTask, dict[str, object]]] = []
    conformance_rows: list[dict[str, object]] = []
    requirements: dict[str, list[dict[str, object]]] = defaultdict(list)

    for gate_path in sorted(by_gate):
        ledger = _load_gate_ledger(gate_path)
        conformance = _audit_real_ledger(gate_path, ledger)
        conformance_rows.append(conformance)
        gate_sha = sha256_file(gate_path)
        for task in sorted(by_gate[gate_path], key=lambda value: value.slice_id):
            old_ids = tuple(
                str(value)
                for value in task.document.get(
                    "selected_query_ids", task.document.get("selected", [])
                )
            )
            allowed = {int(value) for value in task.document["allowed_group_sizes"]}  # type: ignore[index]
            if any(ledger.query_group_size[query_id] not in allowed for query_id in ledger.costs):
                raise ValueError(f"gate ledger contains a disallowed group size: {gate_path}")
            result = lazy_gain_cost_greedy(
                ledger.objective,
                ledger.costs,
                float(task.document["budget"]),
            )
            new_ids = result.selected_query_ids
            if result.total_cost > int(float(task.document["budget"])):
                raise AssertionError(f"reselection exceeded budget: {task.slice_id}")
            old_set = set(old_ids)
            new_set = set(new_ids)
            added = sorted(new_set - old_set)
            removed = sorted(old_set - new_set)
            kept = old_set & new_set
            union = old_set | new_set
            old_cost = sum(ledger.costs[query_id] for query_id in old_ids)
            old_units = ledger.objective.value_units(old_ids)
            old_raw = ledger.objective.raw_value(old_ids)
            new_raw = ledger.objective.raw_value(new_ids)
            budget = int(float(task.document["budget"]))
            new_document = {
                **dict(task.document),
                **result.as_dict(),
                "selected": list(new_ids),
                "selected_query_ids": list(new_ids),
                "total_cost": result.total_cost,
                "budget": budget,
                "objective_units_decimal": str(result.objective_units),
                "old_selection_source_run_id": task.source.source_run_id,
                "old_selection_sha256": sha256_file(task.source_path),
                "gate_ledger_sha256": gate_sha,
                "numeric_semantics": NUMERIC_SEMANTICS,
                "uplift_scale": ledger.objective.uplift_scale,
                "quantized_gain_mapping_sha256": ledger.objective.quantized_gains_sha256,
                "raw_objective_value_diagnostic": new_raw,
                "selected_cell_incidence": sum(
                    len(ledger.query_cells[query_id]) for query_id in new_ids
                ),
                "unique_covered_cells": len(
                    {
                        cell_id
                        for query_id in new_ids
                        for cell_id in ledger.query_cells[query_id]
                    }
                ),
                "api_called": False,
            }
            outputs.append((task, new_document))
            diff_rows.append(
                {
                    "source_run_id": task.source.source_run_id,
                    "new_run_id": task.source.new_run_id,
                    "backend": task.source.backend,
                    "slice_id": task.slice_id,
                    "suite": task.document["suite"],
                    "dataset": task.document["dataset"],
                    "scenario": task.document["scenario"],
                    "group_size_variant": task.document["group_size_variant"],
                    "budget_share": task.document["budget_share"],
                    "budget": budget,
                    "old_algorithm": task.document["algorithm"],
                    "new_algorithm": result.algorithm,
                    "old_selected_count": len(old_ids),
                    "new_selected_count": len(new_ids),
                    "old_estimated_cost": old_cost,
                    "new_estimated_cost": result.total_cost,
                    "old_slack": budget - old_cost,
                    "new_slack": budget - result.total_cost,
                    "old_objective_recomputed": old_units / ledger.objective.uplift_scale,
                    "new_objective_quantized": result.objective_value,
                    "old_raw_objective_diagnostic": old_raw,
                    "new_raw_objective_diagnostic": new_raw,
                    "raw_objective_delta": new_raw - old_raw,
                    "added_count": len(added),
                    "removed_count": len(removed),
                    "unchanged_count": len(kept),
                    "selection_jaccard": len(kept) / len(union) if union else 1.0,
                    "selection_changed": bool(added or removed),
                    "old_selection_sha256": _selection_hash(old_ids),
                    "new_selection_sha256": _selection_hash(new_ids),
                    "sampled_eager_lazy_match": conformance["sampled_lazy_eager_match"],
                    "sampled_exact_checked": True,
                    "sampled_observed_ratio": conformance["observed_ratio"],
                    "quantized_gain_mapping_sha256": ledger.objective.quantized_gains_sha256,
                }
            )
            for query_id in new_ids:
                requirements[query_id].append(
                    {
                        "backend": task.source.backend,
                        "slice_id": task.slice_id,
                        "freezes_terminal_failures": task.freezes_terminal_failures,
                    }
                )
    return diff_rows, outputs, conformance_rows, dict(requirements)


def _checkpoint_path(run_dir: Path) -> Path:
    shared = run_dir / "llm" / "shared" / "group_query_checkpoint.jsonl"
    direct = run_dir / "llm" / "group_query_checkpoint.jsonl"
    if shared.is_file():
        return shared
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"run has no group-query checkpoint: {run_dir}")


def _load_selected_action_identities(
    runs_root: Path,
    sources: Sequence[BackboneSource],
    required_query_ids: set[str],
) -> tuple[dict[str, _ActionIdentity], dict[str, object]]:
    identities: dict[str, _ActionIdentity] = {}
    config_identity_hashes: set[str] = set()
    config_audit: list[dict[str, object]] = []
    first_client: DeepSeekGroupClient | None = None

    for source in sources:
        run_dir = runs_root / source.source_run_id
        llm_config_path = run_dir / "bound_llm_config.json"
        llm_config = _load_json(llm_config_path)
        config = GroupClientConfig.from_mapping(llm_config)
        client = DeepSeekGroupClient(config, api_key="not-used-for-request-hashing")
        if first_client is None:
            first_client = client
        request_config = {
            "base_url": config.base_url.rstrip("/"),
            "model": config.model,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "extra_body": dict(config.extra_body),
        }
        identity_hash = canonical_json_sha256(request_config)
        config_identity_hashes.add(identity_hash)
        config_audit.append(
            {
                "source_run_id": source.source_run_id,
                "llm_config_sha256": sha256_file(llm_config_path),
                "request_identity_config_sha256": identity_hash,
                "model": config.model,
                "prompt_schema_version": llm_config.get("prompt_schema_version", ""),
            }
        )
    if len(config_identity_hashes) != 1:
        raise ValueError("backbone request-identity LLM configs differ")
    if first_client is None:
        raise ValueError("at least one backbone source is required")
    candidate_run = runs_root / sources[0].source_run_id
    for candidate_path in sorted((candidate_run / "groups" / "candidates").glob("*.jsonl")):
        for row in _iter_jsonl(candidate_path):
            query_id = str(row.get("query_id", ""))
            if query_id not in required_query_ids:
                continue
            job = GroupLLMJob(
                query_id=query_id,
                messages=row["messages"],  # type: ignore[arg-type]
                prompt_hash=str(row["prompt_hash"]),
                expected_cell_ids=tuple(str(value) for value in row["cell_ids"]),  # type: ignore[arg-type]
                max_tokens=int(row["completion_token_ceiling"]),
            )
            identities[query_id] = _ActionIdentity(
                query_id=query_id,
                prompt_hash=job.prompt_hash,
                provider_request_hash=first_client.provider_request_hash(job),
                model=first_client.config.model,
                prompt_schema_version=str(row["prompt_schema_version"]),
                estimated_tokens=int(row["estimated_total_tokens"]),
                group_size=int(row["group_size"]),
                group_view=str(row["group_view"]),
                suite=str(row["suite"]),
                dataset=str(row["dataset"]),
            )

    missing = sorted(required_query_ids - set(identities))
    if missing:
        raise ValueError(f"selected queries missing candidate actions: {missing[:10]}")
    return identities, {
        "request_identity_config_sha256": next(iter(config_identity_hashes)),
        "configs": config_audit,
        "selected_action_count": len(identities),
        "candidate_snapshot_source_run_id": sources[0].source_run_id,
        "selected_actions_found": len(identities),
    }


def _audit_cache_union(
    runs_root: Path,
    cache_source_run_ids: Sequence[str],
    identities: Mapping[str, _ActionIdentity],
    requirements: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    records: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    rejection_counts: dict[str, int] = defaultdict(int)
    checkpoint_audit: list[dict[str, object]] = []

    for source_order, run_id in enumerate(cache_source_run_ids):
        run_dir = runs_root / run_id
        checkpoint = _checkpoint_path(run_dir)
        retained = 0
        total = 0
        for row_index, row in enumerate(_iter_jsonl(checkpoint), start=1):
            total += 1
            query_id = str(row.get("query_id", ""))
            identity = identities.get(query_id)
            if identity is None:
                continue
            if row.get("model_matches_request", True) is False:
                rejection_counts["model_matches_request_false"] += 1
                continue
            metadata = row.get("metadata")
            phase = str(metadata.get("phase", "")) if isinstance(metadata, Mapping) else ""
            if not _phase_compatible(phase):
                rejection_counts[f"phase_incompatible:{phase or '<empty>'}"] += 1
                continue
            observed = (
                query_id,
                str(row.get("prompt_hash", "")),
                str(row.get("provider_request_hash", "")),
                str(row.get("model", "")),
                identity.prompt_schema_version,
            )
            if observed != identity.key:
                rejection_counts["request_identity_mismatch"] += 1
                continue
            response_text = str(row.get("response_text", ""))
            records[identity.key].append(
                {
                    "source_order": source_order,
                    "source_run_id": run_id,
                    "row_index": row_index,
                    "status": str(row.get("status", "")),
                    "phase": phase,
                    "response_content_sha256": hashlib.sha256(
                        response_text.encode("utf-8")
                    ).hexdigest(),
                }
            )
            retained += 1
        checkpoint_audit.append(
            {
                "source_run_id": run_id,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "row_count": total,
                "selected_identity_rows_retained": retained,
            }
        )

    cache_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for query_id in sorted(identities):
        identity = identities[query_id]
        rows = sorted(
            records.get(identity.key, []),
            key=lambda row: (int(row["source_order"]), -int(row["row_index"])),
        )
        successes = [row for row in rows if row["status"] == "success"]
        failures = [row for row in rows if row["status"] != "success"]
        chosen = successes[0] if successes else (failures[0] if failures else None)
        success_hashes = sorted(
            {str(row["response_content_sha256"]) for row in successes}
        )
        needed_by = list(requirements[query_id])
        all_freeze_failures = all(
            bool(row["freezes_terminal_failures"]) for row in needed_by
        )
        reusable = bool(successes) or (bool(failures) and all_freeze_failures)
        cache_state = (
            "success"
            if successes
            else "terminal_failure_frozen"
            if failures and all_freeze_failures
            else "terminal_failure_requires_retry"
            if failures
            else "missing"
        )
        source_runs = sorted({str(row["source_run_id"]) for row in rows})
        backends = sorted({str(row["backend"]) for row in needed_by})
        slices = sorted({str(row["slice_id"]) for row in needed_by})
        retry_slices = sorted(
            {
                str(row["slice_id"])
                for row in needed_by
                if not bool(row["freezes_terminal_failures"])
            }
        )
        frozen_slices = sorted(set(slices) - set(retry_slices))
        retry_backbones = sorted(
            {
                str(row["backend"])
                for row in needed_by
                if not bool(row["freezes_terminal_failures"])
            }
        )
        cache_rows.append(
            {
                "query_id": identity.query_id,
                "prompt_hash": identity.prompt_hash,
                "provider_request_hash": identity.provider_request_hash,
                "model": identity.model,
                "prompt_schema_version": identity.prompt_schema_version,
                "success_count": len(successes),
                "terminal_failure_count": len(failures),
                "response_content_sha256_set": "|".join(success_hashes),
                "success_content_conflict": len(success_hashes) > 1,
                "source_runs": "|".join(source_runs),
                "chosen_source_run": str(chosen["source_run_id"]) if chosen else "",
                "chosen_status": str(chosen["status"]) if chosen else "",
                "chosen_phase": str(chosen["phase"]) if chosen else "",
                "all_required_slices_freeze_terminal_failures": all_freeze_failures,
                "reusable": reusable,
                "cache_state": cache_state,
                "required_by_backends": "|".join(backends),
                "required_by_slice_count": len(slices),
            }
        )
        if not reusable:
            missing_rows.append(
                {
                    "query_id": identity.query_id,
                    "prompt_hash": identity.prompt_hash,
                    "provider_request_hash": identity.provider_request_hash,
                    "model": identity.model,
                    "prompt_schema_version": identity.prompt_schema_version,
                    "estimated_tokens": identity.estimated_tokens,
                    "group_size": identity.group_size,
                    "group_view": identity.group_view,
                    "suite": identity.suite,
                    "dataset": identity.dataset,
                    "required_by_backbones": backends,
                    "required_by_slices": slices,
                    "retry_required_by_backbones": retry_backbones,
                    "retry_required_by_slices": retry_slices,
                    "terminal_failure_reusable_by_slices": frozen_slices,
                    "cache_lookup_result": cache_state,
                    "dedup_request_count": 1,
                }
            )
    return cache_rows, missing_rows, {
        "checkpoint_sources": checkpoint_audit,
        "rejected_rows": dict(sorted(rejection_counts.items())),
    }


def _summarize_diffs(diff_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in diff_rows:
        grouped[str(row["backend"])].append(row)
    grouped["ALL"] = list(diff_rows)
    summaries: list[dict[str, object]] = []
    for backend in sorted(grouped, key=lambda value: (value == "ALL", value)):
        rows = grouped[backend]
        summaries.append(
            {
                "backend": backend,
                "slice_count": len(rows),
                "changed_slice_count": sum(bool(row["selection_changed"]) for row in rows),
                "unchanged_slice_count": sum(not bool(row["selection_changed"]) for row in rows),
                "added_query_occurrences": sum(int(row["added_count"]) for row in rows),
                "removed_query_occurrences": sum(int(row["removed_count"]) for row in rows),
                "mean_selection_jaccard": sum(float(row["selection_jaccard"]) for row in rows)
                / len(rows),
                "min_selection_jaccard": min(float(row["selection_jaccard"]) for row in rows),
                "total_raw_objective_delta": sum(float(row["raw_objective_delta"]) for row in rows),
                "all_sampled_eager_lazy_match": all(
                    bool(row["sampled_eager_lazy_match"]) for row in rows
                ),
                "all_sampled_exact_checked": all(
                    bool(row["sampled_exact_checked"]) for row in rows
                ),
                "minimum_sampled_observed_ratio": min(
                    float(row["sampled_observed_ratio"]) for row in rows
                ),
            }
        )
    return summaries


def _summary_markdown(summary_rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Repaired MGreedy selection difference summary",
        "",
        "All comparisons use the frozen gate ledgers. Old selections are comparison-only.",
        "",
        "| Backbone | Slices | Changed | Added occurrences | Removed occurrences | Mean Jaccard | Min Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {backend} | {slice_count} | {changed_slice_count} | "
            "{added_query_occurrences} | {removed_query_occurrences} | "
            "{mean_selection_jaccard:.9f} | {min_selection_jaccard:.9f} |".format(
                **row
            )
        )
    lines.extend(
        (
            "",
            "`sampled_observed_ratio` is an implementation regression check, not a proof of the 0.405 theorem.",
            "",
        )
    )
    return "\n".join(lines)


def _write_outputs(
    runs_root: Path,
    sources: Sequence[BackboneSource],
    audit_run_id: str,
    outputs: Sequence[tuple[_SelectionTask, Mapping[str, object]]],
    diff_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    conformance_rows: Sequence[Mapping[str, object]],
    cache_rows: Sequence[Mapping[str, object]],
    missing_rows: Sequence[Mapping[str, object]],
    action_audit: Mapping[str, object],
    cache_audit: Mapping[str, object],
) -> dict[str, object]:
    audit_dir = runs_root / audit_run_id
    if audit_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {audit_dir}")
    created_at = datetime.now(timezone.utc).isoformat()
    outputs_by_run: dict[str, list[tuple[_SelectionTask, Mapping[str, object]]]] = defaultdict(list)
    for task, document in outputs:
        outputs_by_run[task.source.new_run_id].append((task, document))
    sources_by_new_run: dict[str, list[BackboneSource]] = defaultdict(list)
    for source in sources:
        sources_by_new_run[source.new_run_id].append(source)

    for new_run_id, grouped_sources in sources_by_new_run.items():
        source = grouped_sources[0]
        new_dir = runs_root / new_run_id
        source_dir = runs_root / source.source_run_id
        if new_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {new_dir}")
        for task, document in outputs_by_run[source.new_run_id]:
            destination = new_dir / "selections" / source.backend / task.relative_path
            _write_json(destination, document)
        _write_json(
            new_dir / "bound_experiment_config.json",
            _load_json(source_dir / "bound_experiment_config.json"),
        )
        _write_json(
            new_dir / "bound_llm_config.json",
            _load_json(source_dir / "bound_llm_config.json"),
        )
        source_rows = [row for row in diff_rows if str(row["new_run_id"]) == new_run_id]
        source_run_ids = [value.source_run_id for value in grouped_sources]
        manifest = {
            "run_id": new_run_id,
            "run_kind": "offline_repaired_mgreedy_reselection",
            "created_at_utc": created_at,
            "backend": source.backend,
            "source_run_ids": source_run_ids,
            "source_run_manifest_sha256": {
                value.source_run_id: sha256_file(
                    runs_root / value.source_run_id / "run_manifest.json"
                )
                for value in grouped_sources
            },
            "selection_count": len(outputs_by_run[new_run_id]),
            "changed_selection_count": sum(
                bool(row["selection_changed"]) for row in source_rows
            ),
            "numeric_semantics": NUMERIC_SEMANTICS,
            "uplift_scale": DEFAULT_UPLIFT_SCALE,
            "api_called": False,
            "historical_artifacts_overwritten": False,
            "scope": "selection-only offline replay; repair replay depends on cache-union decision",
        }
        _write_json(new_dir / "run_manifest.json", manifest)

    diff_fields = (
        "source_run_id",
        "new_run_id",
        "backend",
        "slice_id",
        "suite",
        "dataset",
        "scenario",
        "group_size_variant",
        "budget_share",
        "budget",
        "old_algorithm",
        "new_algorithm",
        "old_selected_count",
        "new_selected_count",
        "old_estimated_cost",
        "new_estimated_cost",
        "old_slack",
        "new_slack",
        "old_objective_recomputed",
        "new_objective_quantized",
        "old_raw_objective_diagnostic",
        "new_raw_objective_diagnostic",
        "raw_objective_delta",
        "added_count",
        "removed_count",
        "unchanged_count",
        "selection_jaccard",
        "selection_changed",
        "old_selection_sha256",
        "new_selection_sha256",
        "sampled_eager_lazy_match",
        "sampled_exact_checked",
        "sampled_observed_ratio",
        "quantized_gain_mapping_sha256",
    )
    summary_fields = tuple(summary_rows[0])
    conformance_fields = tuple(conformance_rows[0])
    cache_fields = tuple(cache_rows[0])
    missing_fields = (
        "query_id",
        "prompt_hash",
        "provider_request_hash",
        "model",
        "prompt_schema_version",
        "estimated_tokens",
        "group_size",
        "group_view",
        "suite",
        "dataset",
        "required_by_backbones",
        "required_by_slices",
        "retry_required_by_backbones",
        "retry_required_by_slices",
        "terminal_failure_reusable_by_slices",
        "cache_lookup_result",
        "dedup_request_count",
    )
    _write_csv(audit_dir / "selection_diff.csv", diff_rows, diff_fields)
    _write_csv(audit_dir / "selection_diff_summary.csv", summary_rows, summary_fields)
    _write_csv(
        audit_dir / "real_ledger_theory_conformance.csv",
        conformance_rows,
        conformance_fields,
    )
    _write_csv(audit_dir / "cache_union_audit.csv", cache_rows, cache_fields)
    csv_missing = [
        {
            **row,
            "required_by_backbones": "|".join(row["required_by_backbones"]),  # type: ignore[arg-type]
            "required_by_slices": "|".join(row["required_by_slices"]),  # type: ignore[arg-type]
            "retry_required_by_backbones": "|".join(row["retry_required_by_backbones"]),  # type: ignore[arg-type]
            "retry_required_by_slices": "|".join(row["retry_required_by_slices"]),  # type: ignore[arg-type]
            "terminal_failure_reusable_by_slices": "|".join(row["terminal_failure_reusable_by_slices"]),  # type: ignore[arg-type]
        }
        for row in missing_rows
    ]
    _write_csv(audit_dir / "missing_query_union.csv", csv_missing, missing_fields)
    _write_jsonl(audit_dir / "missing_query_union.jsonl", missing_rows)
    markdown_path = audit_dir / "selection_difference_table.md"
    if markdown_path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {markdown_path}")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_summary_markdown(summary_rows), encoding="utf-8")

    retry_multipliers = []
    for config_row in action_audit["configs"]:  # type: ignore[index]
        source_dir = runs_root / str(config_row["source_run_id"])
        retry_multipliers.append(
            int(_load_json(source_dir / "bound_llm_config.json").get("max_retries", 0))
            + 1
        )
    retry_multiplier = max(retry_multipliers)
    missing_estimated_tokens = sum(int(row["estimated_tokens"]) for row in missing_rows)
    decision = {
        "api_called": False,
        "decision": "no_api_needed"
        if not missing_rows
        else "explicit_authorization_required_before_paid_api",
        "deduplicated_selected_query_count": len(cache_rows),
        "deduplicated_missing_query_count": len(missing_rows),
        "missing_estimated_total_tokens": missing_estimated_tokens,
        "max_retry_multiplier": retry_multiplier,
        "retry_adjusted_missing_token_cap_excluding_preflight": (
            retry_multiplier * missing_estimated_tokens
        ),
        "terminal_failure_policy": (
            "success reusable for all slices; terminal failure reusable only when every "
            "requiring historical configuration freezes terminal failures"
        ),
        "target_checkpoint_phase": "online_selected_union",
        "cache_conflict_resolution": (
            "fixed cache-source order, then latest checkpoint row within source; first success, "
            "otherwise first phase-compatible terminal failure"
        ),
        "request_identity_fields": [
            "query_id",
            "prompt_hash",
            "provider_request_hash",
            "model",
            "prompt_schema_version",
        ],
    }
    _write_json(audit_dir / "api_decision.json", decision)
    _write_json(audit_dir / "action_identity_audit.json", action_audit)
    _write_json(audit_dir / "cache_source_audit.json", cache_audit)
    manifest = {
        "run_id": audit_run_id,
        "run_kind": "offline_cross_backbone_reselection_cache_union_audit",
        "created_at_utc": created_at,
        "new_backbone_run_ids": sorted(sources_by_new_run),
        "source_run_ids": [source.source_run_id for source in sources],
        "numeric_semantics": NUMERIC_SEMANTICS,
        "uplift_scale": DEFAULT_UPLIFT_SCALE,
        "selection_slice_count": len(diff_rows),
        "changed_selection_slice_count": sum(
            bool(row["selection_changed"]) for row in diff_rows
        ),
        "real_ledger_conformance_count": len(conformance_rows),
        "all_sampled_lazy_eager_match": all(
            bool(row["sampled_lazy_eager_match"]) for row in conformance_rows
        ),
        "minimum_sampled_observed_ratio": min(
            float(row["observed_ratio"]) for row in conformance_rows
        ),
        **decision,
        "historical_artifacts_overwritten": False,
    }
    _write_json(audit_dir / "run_manifest.json", manifest)
    return manifest


def run_offline_reselection(
    runs_root: Path,
    *,
    sources: Sequence[BackboneSource] = DEFAULT_BACKBONE_SOURCES,
    cache_source_run_ids: Sequence[str] = DEFAULT_CACHE_SOURCE_RUN_IDS,
    audit_run_id: str = DEFAULT_AUDIT_RUN_ID,
) -> dict[str, object]:
    """Execute the complete no-network replay, refusing every overwrite."""

    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"runs root does not exist: {root}")
    if (root / audit_run_id).exists():
        raise FileExistsError(f"refusing to overwrite existing run: {root / audit_run_id}")
    tasks = _discover_tasks(root, sources)
    diff_rows, outputs, conformance_rows, requirements = _reselect_tasks(tasks)
    identities, action_audit = _load_selected_action_identities(
        root, sources, set(requirements)
    )
    cache_rows, missing_rows, cache_audit = _audit_cache_union(
        root,
        cache_source_run_ids,
        identities,
        requirements,
    )
    summary_rows = _summarize_diffs(diff_rows)
    return _write_outputs(
        root,
        sources,
        audit_run_id,
        outputs,
        diff_rows,
        summary_rows,
        conformance_rows,
        cache_rows,
        missing_rows,
        action_audit,
        cache_audit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline repaired-MGreedy reselection and cache-union audit. "
            "This command has no API-call path."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    arguments = parser.parse_args(argv)
    manifest = run_offline_reselection(arguments.runs_root)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
