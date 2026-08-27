from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from budgeted_group_repair_no_baran.data import SafeCell, SafeDataset
from budgeted_group_repair_no_baran.group_context import (
    GroupContextBuilder,
    compute_ordered_query_id,
    compute_query_id,
)
from budgeted_group_repair_no_baran.prompt_policy import (
    INFORMATION_POLICY,
    PromptPolicyError,
    assert_messages_safe,
    assert_payload_safe,
)
from budgeted_group_repair_no_baran.sampling import sample_dataset_cells


def _dataset() -> tuple[SafeDataset, tuple[SafeCell, ...]]:
    dirty = pd.DataFrame(
        {
            "row_id": [str(index) for index in range(10)],
            "name": [f"dirty-{index}" for index in range(10)],
            "country": ["US"] * 5 + ["GB"] * 5,
        }
    )
    cells = tuple(
        SafeCell("source", "toy", row, 1 if row < 7 else 2, "name" if row < 7 else "country", str(row), str(dirty.iloc[row, 1 if row < 7 else 2]))
        for row in range(10)
    )
    return SafeDataset("source", "toy", Path("dirty.csv"), dirty, cells), cells


def test_safe_cell_physically_excludes_oracle_fields() -> None:
    names = {field.name for field in fields(SafeCell)}
    assert "clean_value" not in names
    assert "error_type" not in names


def test_hamilton_sample_is_deterministic_and_column_stratified() -> None:
    _, cells = _dataset()
    first = sample_dataset_cells(cells, sample_n=5, seed=42)
    second = sample_dataset_cells(tuple(reversed(cells)), sample_n=5, seed=42)
    assert first == second
    assert len(first) == 5
    assert {row.column for row in first} == {"name", "country"}
    assert all(not hasattr(row, "clean_value") for row in first)


def test_prompt_ignores_external_features_and_passes_recursive_audit() -> None:
    dataset, cells = _dataset()
    query_id = compute_query_id(
        "source", "toy", "singleton", (str(cells[0].cell_id),), arm="singleton"
    )
    material = GroupContextBuilder(
        dataset,
        cells,
        {str(cells[0].cell_id): {"baran_prediction": "POISON"}},
        similar_row_count=1,
    ).build_material(query_id, "singleton", (cells[0],))
    assert_messages_safe(material.messages)
    serialized = str(material.messages).lower()
    assert "baran" not in serialized
    assert "poison" not in serialized


def test_information_policy_and_arm_bind_query_identity() -> None:
    base = compute_query_id("source", "toy", "row", ("c1", "c2"), arm="structured")
    random = compute_query_id("source", "toy", "row", ("c1", "c2"), arm="random")
    revised = compute_query_id(
        "source",
        "toy",
        "row",
        ("c1", "c2"),
        arm="structured",
        information_policy=INFORMATION_POLICY + "-revision",
    )
    assert len({base, random, revised}) == 3


def test_ordered_evidence_material_preserves_order_and_uses_neutral_view() -> None:
    dataset, cells = _dataset()
    builder = GroupContextBuilder(dataset, cells, similar_row_count=0)
    frozen = (cells[2], str(cells[0].cell_id), cells[1])
    ordered_ids = tuple(str(value.cell_id) if isinstance(value, SafeCell) else value for value in frozen)
    query_id = compute_ordered_query_id("source", "toy", ordered_ids)
    material = builder.build_ordered_material(query_id, frozen)
    payload = json.loads(material.messages[1]["content"])

    assert payload["group"]["view"] == "matched_multi_target"
    assert payload["group"]["cell_ids"] == list(ordered_ids)
    assert [target["cell_id"] for target in payload["targets"]] == list(ordered_ids)
    assert "pattern" not in material.messages[1]["content"]
    assert "semantic" not in material.messages[1]["content"]

    # The legacy production path retains its canonical cell-id ordering.
    legacy = builder.payload("legacy", "pattern", tuple(reversed(cells[:3])))
    assert legacy["group"]["cell_ids"] == sorted(str(cell.cell_id) for cell in cells[:3])


def test_ordered_query_and_prompt_identities_bind_member_position() -> None:
    dataset, cells = _dataset()
    builder = GroupContextBuilder(dataset, cells, similar_row_count=0)
    forward_ids = tuple(str(cell.cell_id) for cell in cells[:3])
    reverse_ids = tuple(reversed(forward_ids))
    forward_query = compute_ordered_query_id("source", "toy", forward_ids)
    reverse_query = compute_ordered_query_id("source", "toy", reverse_ids)
    forward = builder.build_ordered_material(forward_query, forward_ids)
    reverse = builder.build_ordered_material(reverse_query, reverse_ids)

    assert forward_query != reverse_query
    assert forward.prompt_hash != reverse.prompt_hash
    with pytest.raises(ValueError, match="matched_multi_target"):
        builder.build_ordered_material(forward_query, forward_ids, group_view="pattern")


def test_forbidden_nested_field_and_natural_language_are_rejected() -> None:
    with pytest.raises(PromptPolicyError):
        assert_payload_safe({"safe": [{"clean_value": "x"}]})
    with pytest.raises(PromptPolicyError):
        assert_payload_safe({"note": "candidate support is high"})
