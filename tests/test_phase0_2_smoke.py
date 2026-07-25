from __future__ import annotations

from pathlib import Path

import pandas as pd

from budgeted_group_repair_no_baran.data import SafeCell, SafeDataset
from budgeted_group_repair_no_baran.evaluation import (
    bind_oracle_correctness,
    complementarity_metrics,
    grouping_metrics,
    materialize_arm_results,
)
from budgeted_group_repair_no_baran.group_generator import GroupGenerator
from budgeted_group_repair_no_baran.partitioning import (
    build_matched_random_groups,
    select_primary_structured_groups,
)
from budgeted_group_repair_no_baran.prompt_policy import assert_messages_safe


def _response(action, clean, *, abstain_first: bool = False):
    items = []
    for index, cell_id in enumerate(action.cell_ids):
        items.append(
            {
                "cell_id": cell_id,
                "repair": "" if abstain_first and index == 0 else clean[cell_id],
                "confidence": 0.9,
                "decision": "abstain" if abstain_first and index == 0 else "propose",
                "evidence": "dirty-only mock evidence",
                "affected_constraints": [],
            }
        )
    return {
        "query_id": action.query_id,
        "items": items,
        "parse_status": "ok",
        "model_matches_request": True,
        "observed_total_tokens": 100 + 10 * action.group_size,
        "usage": {"prompt_tokens": 80, "completion_tokens": 20 + 10 * action.group_size},
        "attempts": 1,
    }


def test_mock_smoke_runs_prompt_partition_phase1_and_phase2_without_provider() -> None:
    dirty_frame = pd.DataFrame(
        {
            "row_id": [str(index) for index in range(16)],
            "value": [f"bad-{index}" for index in range(8)] + ["good-a"] * 4 + ["good-b"] * 4,
            "kind": ["A"] * 4 + ["B"] * 4 + ["A"] * 4 + ["B"] * 4,
        }
    )
    cells = tuple(
        SafeCell("source", "smoke", index, 1, "value", str(index), f"bad-{index}")
        for index in range(8)
    )
    dataset = SafeDataset("source", "smoke", Path("dirty.csv"), dirty_frame, cells)
    generator = GroupGenerator(dataset, cells, None, group_sizes=(1, 4))
    actions = generator.generate().actions
    singletons = tuple(action for action in actions if action.arm == "singleton")
    structured, _ = select_primary_structured_groups(
        actions, group_size=4, view_priority=("pattern",), seed=43
    )
    random, _ = build_matched_random_groups(
        structured,
        dataset=dataset,
        cells=cells,
        singleton_actions={action.cell_ids[0]: action for action in singletons},
        seed=44,
        similar_row_count=1,
    )
    for action in (*singletons, *structured, *random):
        assert_messages_safe(action.messages)
    population = {cell_id for action in structured for cell_id in action.cell_ids}
    singleton_subset = tuple(action for action in singletons if action.cell_ids[0] in population)
    dirty = {str(cell.cell_id): cell.dirty_value for cell in cells}
    clean = {str(cell.cell_id): "good-" + str(cell.row) for cell in cells}
    singleton_eval = bind_oracle_correctness(
        materialize_arm_results(
            singleton_subset,
            [_response(action, clean) for action in singleton_subset],
            dirty_by_cell=dirty,
        ),
        clean,
    )
    structured_eval = bind_oracle_correctness(
        materialize_arm_results(
            structured,
            [_response(action, clean) for action in structured],
            dirty_by_cell=dirty,
        ),
        clean,
    )
    random_eval = bind_oracle_correctness(
        materialize_arm_results(
            random,
            [_response(action, clean, abstain_first=True) for action in random],
            dirty_by_cell=dirty,
        ),
        clean,
    )
    baran = {
        cell_id: (clean[cell_id] if index % 2 == 0 else dirty[cell_id])
        for index, cell_id in enumerate(sorted(population))
    }
    exp1, _ = complementarity_metrics(
        singleton_eval,
        baran_prediction_by_cell=baran,
        clean_by_cell=clean,
        row_id_by_cell={str(cell.cell_id): cell.row_id for cell in cells},
        bootstrap_replicates=30,
        bootstrap_seed=45,
    )
    exp2, summary = grouping_metrics(
        singleton_eval,
        structured_eval,
        random_eval,
        bootstrap_replicates=30,
        bootstrap_seed=45,
        confidence=0.95,
        noninferiority_margin=0.01,
        minimum_token_saving=0.15,
        maximum_parse_validity_drop=0.01,
        maximum_missing_item_rate_increase=0.01,
    )
    assert exp1[0]["n10"] == 0
    assert exp1[0]["n01"] > 0
    assert exp2[0]["structured_accuracy"] == 1.0
    assert summary["micro"]["structured_minus_random"] > 0
