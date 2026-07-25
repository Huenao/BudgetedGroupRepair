"""Frozen primary structured and matched-random partitions."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .data import SafeCell
from .group_context import GroupContextBuilder, compute_query_id
from .group_generator import GroupQueryAction
from .prompt_policy import INFORMATION_POLICY, PROMPT_SCHEMA_VERSION


def select_primary_structured_groups(
    actions: Sequence[GroupQueryAction],
    *,
    group_size: int,
    view_priority: Sequence[str],
    seed: int,
) -> tuple[tuple[GroupQueryAction, ...], dict[str, object]]:
    del seed  # Frozen for protocol identity; deterministic query_id breaks final ties.
    priority = {str(view): index for index, view in enumerate(view_priority)}
    candidates = [
        action
        for action in actions
        if action.arm == "structured"
        and action.group_size == int(group_size)
        and action.group_view in priority
    ]
    candidates.sort(
        key=lambda action: (
            priority[action.group_view],
            -float(action.group_features.get("cohesion", 0.0)),
            action.query_id,
        )
    )
    selected: list[GroupQueryAction] = []
    used: set[str] = set()
    for action in candidates:
        members = set(action.cell_ids)
        if members.isdisjoint(used):
            selected.append(action)
            used.update(members)
    frozen = tuple(sorted(selected, key=lambda action: action.query_id))
    all_members = [cell_id for action in frozen for cell_id in action.cell_ids]
    if len(all_members) != len(set(all_members)):
        raise AssertionError("primary structured partition overlaps")
    return frozen, {
        "candidate_count": len(candidates),
        "selected_group_count": len(frozen),
        "covered_cell_count": len(used),
        "group_size": int(group_size),
        "view_priority": list(view_priority),
        "counts_by_view": dict(sorted(Counter(action.group_view for action in frozen).items())),
        "overlap_count": 0,
    }


def _quartile_bins(singletons: Mapping[str, GroupQueryAction]) -> dict[str, int]:
    ordered = sorted(
        singletons,
        key=lambda cell_id: (singletons[cell_id].estimated_prompt_tokens, cell_id),
    )
    count = len(ordered)
    return {
        cell_id: min(3, math.floor(index * 4 / max(1, count)))
        for index, cell_id in enumerate(ordered)
    }


def _rotation(seed: int, label: str, values: Sequence[str]) -> dict[str, str]:
    ordered = sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{int(seed)}|{label}|{value}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ordered) <= 1:
        return {value: value for value in ordered}
    offset_hash = hashlib.sha256(f"{int(seed)}|{label}|offset".encode("utf-8")).hexdigest()
    offset = 1 + int(offset_hash[:16], 16) % (len(ordered) - 1)
    rotated = ordered[offset:] + ordered[:offset]
    return dict(zip(ordered, rotated))


def build_matched_random_groups(
    structured: Sequence[GroupQueryAction],
    *,
    dataset: Any,
    cells: Sequence[SafeCell],
    singleton_actions: Mapping[str, GroupQueryAction],
    fd_components: Sequence[Any] = (),
    seed: int,
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
    similar_row_count: int = 5,
) -> tuple[tuple[GroupQueryAction, ...], dict[str, object]]:
    if not structured:
        return (), {
            "structured_group_count": 0,
            "random_group_count": 0,
            "covered_cell_count": 0,
            "fixed_slot_count": 0,
            "exact_group_collisions": 0,
        }
    cell_by_id = {str(cell.cell_id): cell for cell in cells}
    structured_members = [cell_id for action in structured for cell_id in action.cell_ids]
    if len(structured_members) != len(set(structured_members)):
        raise ValueError("structured input must be a non-overlapping partition")
    if not set(structured_members).issubset(singleton_actions):
        raise ValueError("every structured cell requires a canonical singleton action")
    bins = _quartile_bins({cell_id: singleton_actions[cell_id] for cell_id in structured_members})
    origin_view: dict[str, str] = {}
    strata: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for action in structured:
        for cell_id in action.cell_ids:
            origin_view[cell_id] = action.group_view
            cell = cell_by_id[cell_id]
            strata[(action.group_view, str(cell.column), bins[cell_id])].append(cell_id)
    reassignment: dict[str, str] = {}
    fixed_slots = 0
    for key in sorted(strata):
        label = "|".join(map(str, key))
        mapping = _rotation(seed, label, strata[key])
        reassignment.update(mapping)
        fixed_slots += sum(left == right for left, right in mapping.items())
    random_actions: list[GroupQueryAction] = []
    collisions = 0
    context = GroupContextBuilder(
        dataset,
        cells,
        None,
        fd_components=fd_components,
        similar_row_count=similar_row_count,
    )
    for source_action in sorted(structured, key=lambda action: action.query_id):
        random_ids = tuple(sorted(reassignment[cell_id] for cell_id in source_action.cell_ids))
        if len(set(random_ids)) != source_action.group_size:
            raise AssertionError("matched random group contains duplicate cells")
        collisions += set(random_ids) == set(source_action.cell_ids)
        query_id = compute_query_id(
            source_action.suite,
            source_action.dataset,
            source_action.group_view,
            random_ids,
            arm="random",
            prompt_schema_version=prompt_schema_version,
            information_policy=INFORMATION_POLICY,
        )
        material = context.build_material(
            query_id,
            source_action.group_view,
            tuple(cell_by_id[cell_id] for cell_id in random_ids),
            prompt_schema_version=prompt_schema_version,
        )
        features = {
            "matched_structured_query_id": source_action.query_id,
            "matched_view": source_action.group_view,
            "matched_group_size": source_action.group_size,
            "length_bin_multiset": sorted(bins[cell_id] for cell_id in source_action.cell_ids),
        }
        random_actions.append(
            GroupQueryAction(
                query_id=query_id,
                suite=source_action.suite,
                dataset=source_action.dataset,
                arm="random",
                group_view=source_action.group_view,
                cell_ids=random_ids,
                group_size=len(random_ids),
                prompt_schema_version=prompt_schema_version,
                prompt_information_policy=INFORMATION_POLICY,
                messages=material.messages,
                prompt_hash=material.prompt_hash,
                estimated_prompt_tokens=material.estimated_prompt_tokens,
                completion_token_ceiling=material.completion_token_ceiling,
                estimated_total_tokens=material.estimated_total_tokens,
                group_features=features,
            )
        )
    random_partition = tuple(sorted(random_actions, key=lambda action: action.query_id))
    random_members = [cell_id for action in random_partition for cell_id in action.cell_ids]
    if Counter(random_members) != Counter(structured_members):
        raise AssertionError("matched random partition changed the evaluation population")
    if Counter(action.group_size for action in random_partition) != Counter(
        action.group_size for action in structured
    ):
        raise AssertionError("matched random partition changed the group-size multiset")
    return random_partition, {
        "structured_group_count": len(structured),
        "random_group_count": len(random_partition),
        "covered_cell_count": len(random_members),
        "fixed_slot_count": fixed_slots,
        "fixed_slot_share": fixed_slots / len(random_members) if random_members else 0.0,
        "exact_group_collisions": int(collisions),
        "stratum_count": len(strata),
        "matching_keys": [
            "structured_view",
            "target_column",
            "payload_length_quartile",
        ],
    }


__all__ = ["build_matched_random_groups", "select_primary_structured_groups"]
