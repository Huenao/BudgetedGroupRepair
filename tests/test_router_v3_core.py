from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from budgeted_group_repair_no_baran.cell_features import CellFeatures
from budgeted_group_repair_no_baran.cli import parse_args
from budgeted_group_repair_no_baran.data import SafeCell, load_dataset
from budgeted_group_repair_no_baran.group_context import canonical_messages
from budgeted_group_repair_no_baran.group_generator import GroupQueryAction
from budgeted_group_repair_no_baran.prompt_policy import (
    INFORMATION_POLICY,
    PROMPT_SCHEMA_VERSION,
    assert_messages_safe,
    assert_payload_safe,
)
import pytest
from budgeted_group_repair_no_baran.protocol import split_for_target
from budgeted_group_repair_no_baran.router_v3 import (
    CALIBRATION_SINGLETON_CELL_COUNT,
    MODEL_FEATURE_COLUMNS,
    TEST_TARGET_CELL_COUNT,
    TEST_TARGETS,
    ExperimentRunner,
    _exact_ledger_integer,
    _nonnegative_finite_uplift,
    generation_order,
    target_order,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonnegative_finite_uplift_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _nonnegative_finite_uplift(value)


def test_nonnegative_finite_uplift_clamps_only_finite_negative_values() -> None:
    assert _nonnegative_finite_uplift(-0.25) == 0.0
    assert _nonnegative_finite_uplift(0.25) == 0.25


def test_router_token_ledgers_preserve_integers_above_binary64_exact_range() -> None:
    large = 2**53 + 1
    rows = pd.DataFrame(
        [
            {
                "cell_id": "c1",
                "group_size": 1,
                "group_view": "singleton",
                "estimated_total_tokens": large,
            },
            {
                "cell_id": "c2",
                "group_size": 1,
                "group_view": "singleton",
                "estimated_total_tokens": 1,
            },
        ]
    )

    assert _exact_ledger_integer(str(large), label="cost", minimum=1) == large
    assert ExperimentRunner._singleton_reference_cost(rows) == large + 1


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf")])
def test_router_token_ledger_integer_parser_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        _exact_ledger_integer(value, label="cost", minimum=1)


def _action(
    query_id: str,
    cell_ids: tuple[str, ...],
    size: int,
    view: str,
    cohesion: float = 0.8,
) -> GroupQueryAction:
    messages = canonical_messages(({"role": "user", "content": "{}"},))
    return GroupQueryAction(
        query_id=query_id,
        suite="tableeg",
        dataset="toy",
        arm="singleton" if size == 1 else "structured",
        group_view=view,
        cell_ids=cell_ids,
        group_size=size,
        prompt_schema_version=PROMPT_SCHEMA_VERSION,
        prompt_information_policy=INFORMATION_POLICY,
        messages=messages,
        prompt_hash=f"hash-{query_id}",
        estimated_prompt_tokens=10,
        completion_token_ceiling=192 if size == 1 else 64 + 192 * size,
        estimated_total_tokens=202 if size == 1 else 74 + 192 * size,
        group_features={
            "cohesion": cohesion,
            "same_row": 0,
            "same_column": 1,
            "dirty_type_count": 1,
            "baran_type_count": 1,
            "baran_changed_share": 1.0,
        },
    )


def test_group_query_action_rejects_fractional_token_ledger_values() -> None:
    action = _action("q", ("tableeg:toy:0:0",), 1, "singleton")
    with pytest.raises(ValueError, match="positive integer"):
        replace(action, estimated_prompt_tokens=10.5)
    with pytest.raises(ValueError, match="positive integer"):
        replace(action, completion_token_ceiling=True)
    with pytest.raises(ValueError, match="positive integer"):
        replace(action, estimated_total_tokens=202.5)


def test_router_v3_frozen_dataset_universes_are_exact() -> None:
    assert len(target_order()) == len(TEST_TARGETS) == 9
    assert len(generation_order()) == 14
    target_cells = sum(
        len(load_dataset(suite, dataset, PROJECT_ROOT / "data").safe_cells())
        for suite, dataset in target_order()
    )
    calibration_cells = sum(
        len(load_dataset(suite, dataset, PROJECT_ROOT / "data").safe_cells())
        for suite, dataset in generation_order()
        if suite == "tableeg"
    )
    assert target_cells == TEST_TARGET_CELL_COUNT == 22_198
    assert calibration_cells == CALIBRATION_SINGLETON_CELL_COUNT == 5_543


def test_full_pair_feature_schema_is_label_free_and_prompt_is_no_baran() -> None:
    cells = (
        SafeCell("tableeg", "toy", 0, 0, "name", "0", "bad-a"),
        SafeCell("tableeg", "toy", 1, 0, "name", "1", "bad-b"),
    )
    ids = tuple(str(cell.cell_id) for cell in cells)
    features = tuple(
        CellFeatures(
            cell_id=cell_id,
            suite="tableeg",
            dataset="toy",
            row=index,
            col=0,
            column="name",
            dirty_type="text",
            dirty_format="TEXT",
            baran_prediction=f"candidate-{index}",
            baran_type="text",
            baran_format="TEXT",
            baran_support=(),
            masked_row_text="name=<TARGET_ERROR_MASKED>",
        )
        for index, cell_id in enumerate(ids)
    )
    actions = (
        _action("s0", (ids[0],), 1, "singleton"),
        _action("s1", (ids[1],), 1, "singleton"),
        _action("g2", ids, 2, "pattern"),
    )
    baran = [
        {
            "cell_id": cell_id,
            "prediction": f"candidate-{index}",
            "candidate_count": 2,
            "top_support": 0.8,
            "support_margin": 0.5,
            "source_agreement": 0.7,
        }
        for index, cell_id in enumerate(ids)
    ]
    rows = ExperimentRunner._pair_feature_rows(actions, features, cells, baran)
    assert len(rows) == 4
    assert len(MODEL_FEATURE_COLUMNS) == 25
    assert set(MODEL_FEATURE_COLUMNS).issubset(rows[0])
    forbidden_labels = {
        "clean_value",
        "right_value",
        "baran_correct",
        "llm_correct_in_query",
        "helpful",
        "harmful",
    }
    assert all(not (forbidden_labels & set(row)) for row in rows)
    for action in actions:
        assert_messages_safe(action.as_dict()["messages"])


def test_prompt_policy_allows_natural_dirty_text_but_rejects_baran_fields() -> None:
    assert_payload_safe({"dirty_value": "Baran is a legitimate surname"})
    with pytest.raises(ValueError, match="dirty-evidence-only"):
        assert_payload_safe({"baran_candidate": "forbidden"})


def test_calibration_sampling_is_deterministic_all_singleton_plus_cap() -> None:
    cells = tuple(f"tableeg:toy:{index}:0" for index in range(8))
    actions = [
        _action(f"s{index}", (cell_id,), 1, "singleton")
        for index, cell_id in enumerate(cells)
    ]
    actions.extend(
        _action(
            f"g{index}",
            (cells[index % 7], cells[index % 7 + 1]),
            2,
            "pattern",
            cohesion=index / 12,
        )
        for index in range(12)
    )
    first = ExperimentRunner._calibration_sample("toy", actions, seed=42, cap=5)
    second = ExperimentRunner._calibration_sample("toy", actions, seed=42, cap=5)
    assert [action.query_id for action in first] == [
        action.query_id for action in second
    ]
    assert sum(action.group_size == 1 for action in first) == 8
    assert sum(action.group_size > 1 for action in first) == 5


def test_family_holdout_has_zero_identity_and_label_leakage() -> None:
    table = pd.DataFrame(
        [
            {
                "suite": "tableeg",
                "dataset": "flight_10",
                "cell_id": "tableeg:flight_10:0:0",
                "row_id": "0",
                "column": "c",
                "query_id": "q-target",
                "group_signature": "target",
            },
            {
                "suite": "tableeg",
                "dataset": "company",
                "cell_id": "tableeg:company:0:0",
                "row_id": "0",
                "column": "c",
                "query_id": "q-company",
                "group_signature": "company",
            },
            {
                "suite": "tableeg",
                "dataset": "flight_20",
                "cell_id": "tableeg:flight_20:0:0",
                "row_id": "1",
                "column": "c",
                "query_id": "q-same-family",
                "group_signature": "same-family",
            },
        ]
    )
    train, test, audit = split_for_target(table, "tableeg", "flight_10")
    assert set(train["dataset"]) == {"company"}
    assert set(test["dataset"]) == {"flight_10"}
    assert audit.train_test_cell_overlap == 0
    assert audit.train_test_base_family_overlap == 0
    assert audit.train_test_query_overlap == 0
    assert audit.train_test_group_signature_overlap == 0
    assert not audit.target_group_label_used
    assert not audit.target_response_used_before_selection


def test_router_cli_has_staged_uncapped_commands() -> None:
    plan = parse_args(["plan-router-run", "--run-id", "router-v3"])
    assert plan.experiment_config.name == "experiment_router_v3.json"
    assert plan.baran_source_run is None
    assert plan.response_reuse_run is None
    calibration = parse_args(
        [
            "run-router-calibration",
            "--run-id",
            "router-v3",
            "--no-token-cap",
        ]
    )
    assert calibration.no_token_cap is True
    train = parse_args(["train-router", "--run-id", "router-v3"])
    assert train.command == "train-router"
    selected = parse_args(
        ["run-router-bgr", "--run-id", "router-v3", "--no-token-cap"]
    )
    assert selected.no_token_cap is True
    baselines = parse_args(
        ["run-full-baselines", "--run-id", "baseline-v3", "--token-cap", "10"]
    )
    assert baselines.token_cap == 10
    validate = parse_args(["validate-run", "--run-id", "router-v3"])
    assert validate.allow_incomplete is False
