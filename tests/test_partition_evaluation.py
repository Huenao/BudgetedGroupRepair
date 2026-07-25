from __future__ import annotations

from pathlib import Path

import pandas as pd

from budgeted_group_repair_no_baran.data import SafeCell, SafeDataset
from budgeted_group_repair_no_baran.evaluation import complementarity_metrics, grouping_metrics
from budgeted_group_repair_no_baran.group_generator import GroupGenerator
from budgeted_group_repair_no_baran.partitioning import (
    build_matched_random_groups,
    select_primary_structured_groups,
)


def _generated():
    dirty = pd.DataFrame(
        {
            "row_id": [str(index) for index in range(16)],
            "value": [f"bad-{index}" for index in range(8)] + ["good-a"] * 4 + ["good-b"] * 4,
            "kind": ["A"] * 4 + ["B"] * 4 + ["A"] * 4 + ["B"] * 4,
        }
    )
    cells = tuple(
        SafeCell("source", "toy", row, 1, "value", str(row), str(dirty.iloc[row, 1]))
        for row in range(8)
    )
    dataset = SafeDataset("source", "toy", Path("dirty.csv"), dirty, cells)
    generator = GroupGenerator(dataset, cells, None, group_sizes=(1, 4)).generate()
    return dataset, cells, generator.actions


def test_primary_and_random_partitions_are_disjoint_and_population_matched() -> None:
    dataset, cells, actions = _generated()
    structured, _ = select_primary_structured_groups(
        actions, group_size=4, view_priority=("pattern",), seed=43
    )
    singletons = {action.cell_ids[0]: action for action in actions if action.arm == "singleton"}
    random, audit = build_matched_random_groups(
        structured,
        dataset=dataset,
        cells=cells,
        singleton_actions=singletons,
        seed=44,
        similar_row_count=1,
    )
    structured_cells = [cell for action in structured for cell in action.cell_ids]
    random_cells = [cell for action in random for cell in action.cell_ids]
    assert len(structured_cells) == len(set(structured_cells))
    assert sorted(structured_cells) == sorted(random_cells)
    assert all(action.group_size == 4 and action.arm == "random" for action in random)
    assert audit["covered_cell_count"] == len(structured_cells)


def _record(dataset: str, cell: str, query: str, correct: bool, tokens: int) -> dict[str, object]:
    return {
        "dataset": dataset,
        "cell_id": cell,
        "query_id": query,
        "correct": correct,
        "valid_prediction": True,
        "parse_valid_item": True,
        "missing_item": False,
        "unchanged_dirty": False,
        "actual_query_tokens": tokens,
    }


def test_complementarity_four_cells_recompute_exactly() -> None:
    records = tuple(
        _record("d", f"c{i}", f"q{i}", llm, 10)
        for i, llm in enumerate((True, False, True, False))
    )
    baran = {"c0": "yes", "c1": "yes", "c2": "no", "c3": "no"}
    clean = {cell: "yes" for cell in baran}
    rows, summary = complementarity_metrics(
        records,
        baran_prediction_by_cell=baran,
        clean_by_cell=clean,
        row_id_by_cell={cell: cell for cell in baran},
        bootstrap_replicates=50,
        bootstrap_seed=45,
    )
    row = rows[0]
    assert (row["n11"], row["n10"], row["n01"], row["n00"]) == (1, 1, 1, 1)
    assert row["oracle_upper_bound"] == 0.75
    assert row["upper_bound_minus_best"] == 0.25
    assert summary["micro"]["N"] == 4


def test_grouping_metrics_use_unique_query_tokens_and_parse_gate() -> None:
    singleton = []
    structured = []
    random = []
    for dataset in ("d1", "d2"):
        for index in range(4):
            cell = f"{dataset}-c{index}"
            singleton.append(_record(dataset, cell, f"{cell}-single", index < 3, 100))
            structured.append(_record(dataset, cell, f"{dataset}-group", index < 3, 200))
            random.append(_record(dataset, cell, f"{dataset}-random", index < 2, 210))
    rows, summary = grouping_metrics(
        singleton,
        structured,
        random,
        bootstrap_replicates=50,
        bootstrap_seed=45,
        confidence=0.95,
        noninferiority_margin=0.01,
        minimum_token_saving=0.15,
        maximum_parse_validity_drop=0.01,
        maximum_missing_item_rate_increase=0.01,
    )
    assert all(row["structured_tokens"] == 200 for row in rows)
    assert all(row["singleton_tokens"] == 400 for row in rows)
    assert all(row["token_per_cell_saving"] == 0.5 for row in rows)
    assert summary["micro"]["N_cells"] == 8
    assert summary["decision"] == "B_noninferior_more_efficient"
