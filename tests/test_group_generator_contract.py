from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from budgeted_group_repair_no_baran.data import SafeCell, SafeDataset
from budgeted_group_repair_no_baran.group_generator import GroupGenerator, exact_windows
from budgeted_group_repair_no_baran.public_fd import FDViolationComponent


def _generator_fixture() -> tuple[
    SafeDataset,
    tuple[SafeCell, ...],
    dict[str, dict[str, object]],
    tuple[FDViolationComponent, ...],
]:
    dirty = pd.DataFrame(
        {
            "row_id": ["0", "1", "2", "3"],
            "code": ["A", "A", "B", "C"],
            "city": ["Paris", "Rome", "Rome", "Milan"],
        }
    )
    cells = (
        SafeCell("source", "toy", 0, 1, "code", "0", "A"),
        SafeCell("source", "toy", 0, 2, "city", "0", "Paris"),
        SafeCell("source", "toy", 1, 2, "city", "1", "Rome"),
        SafeCell("source", "toy", 2, 2, "city", "2", "Rome"),
    )
    dataset = SafeDataset("source", "toy", Path("dirty.csv"), dirty, cells)
    cell_ids = tuple(str(cell.cell_id) for cell in cells)
    baran = {
        cell_id: {
            "prediction": f"base-{index}",
            "candidate_count": 2,
            "top_support": 0.75,
        }
        for index, cell_id in enumerate(cell_ids)
    }
    # Two component buckets with identical membership intentionally exercise
    # deterministic request deduplication after group generation.
    components = tuple(
        FDViolationComponent(
            component_id=f"component-{index}",
            suite="source",
            dataset="toy",
            rule_id=f"rule-{index}",
            cell_ids=cell_ids,
            row_indices=(0, 1, 2),
            violation_pair_count=2,
        )
        for index in range(2)
    )
    return dataset, cells, baran, components


def test_exact_windows_uses_half_stride_and_explicit_tail() -> None:
    assert exact_windows(tuple("abcdefg"), 4) == (
        ("a", "b", "c", "d"),
        ("c", "d", "e", "f"),
        ("d", "e", "f", "g"),
    )


def test_group_generator_freezes_canonical_deduplicated_actions() -> None:
    dataset, cells, baran, components = _generator_fixture()
    forward = GroupGenerator(
        dataset,
        cells,
        baran,
        fd_components=components,
        group_sizes=(1, 2),
        similar_row_count=0,
    ).generate()
    reverse = GroupGenerator(
        dataset,
        tuple(reversed(cells)),
        baran,
        fd_components=tuple(reversed(components)),
        group_sizes=(2, 1),
        similar_row_count=0,
    ).generate()

    assert [action.as_dict() for action in forward.actions] == [
        action.as_dict() for action in reverse.actions
    ]
    assert {action.group_view for action in forward.actions} == {
        "singleton",
        "row",
        "pattern",
        "public_fd",
        "semantic",
    }
    assert all(
        action.cell_ids == tuple(sorted(action.cell_ids))
        for action in forward.actions
    )
    assert {
        action.group_size
        for action in forward.actions
        if action.group_view == "singleton"
    } == {1}
    assert {
        action.group_size
        for action in forward.actions
        if action.group_view != "singleton"
    } == {2}

    expected_cell_ids = {str(cell.cell_id) for cell in cells}
    singleton_ids = [
        action.cell_ids[0]
        for action in forward.actions
        if action.group_view == "singleton"
    ]
    assert len(singleton_ids) == len(expected_cell_ids)
    assert set(singleton_ids) == expected_cell_ids
    assert len({action.query_id for action in forward.actions}) == len(forward.actions)
    assert len({action.prompt_hash for action in forward.actions}) == len(forward.actions)
    assert forward.audit["deduplicated_request_count"] > 0

    for action in forward.actions:
        payload = json.loads(action.messages[1]["content"])
        assert payload["group"]["cell_ids"] == list(action.cell_ids)
        assert action.estimated_total_tokens == (
            action.estimated_prompt_tokens + action.completion_token_ceiling
        )
