from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import budgeted_group_repair_no_baran.cli as cli_module
from budgeted_group_repair_no_baran.cli import parse_args
from budgeted_group_repair_no_baran.data import SafeCell, SafeDataset
from budgeted_group_repair_no_baran.group_context import (
    GroupContextBuilder,
    compute_ordered_query_id,
)
from budgeted_group_repair_no_baran.group_llm import (
    DeepSeekGroupClient,
    GroupClientConfig,
    GroupLLMJob,
    ProviderModelIdentityError,
)
from budgeted_group_repair_no_baran.motivation_evidence import (
    DEFAULT_RUN_ID,
    FORMAL_DATASETS,
    MotivationEvidenceRunner,
    PartitionGroup,
    StreamingCheckpointExecutor,
    _assert_terminal_checkpoint_matches_request,
    _hash_file_set,
    _validate_finalized_ledgers,
    _verify_file_set,
    build_random_partition,
    build_structured_partition,
    deduplicate_physical_schedule,
    physical_query_id,
    round_robin_logical_schedule,
    validate_matched_partitions,
)


def test_frozen_config_and_cli_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "motivation_evidence.json").read_text())
    assert tuple((row["suite"], row["dataset"]) for row in config["formal_datasets"]) == FORMAL_DATASETS
    assert config["expected"]["singleton_cells"] == 22_198
    assert config["expected"]["structured_calls_by_size"] == {
        "2": 22_160,
        "4": 11_036,
        "8": 5_438,
    }
    assert config["expected"]["logical_calls_before_exact_dedup"] == 99_466
    plan = parse_args(["plan-motivation-evidence", "--run-id", DEFAULT_RUN_ID])
    assert plan.mode == "full"
    with pytest.raises(SystemExit):
        parse_args(["run-motivation-queries", "--run-id", DEFAULT_RUN_ID, "--resume"])
    paid = parse_args(
        [
            "run-motivation-queries",
            "--run-id",
            DEFAULT_RUN_ID,
            "--resume",
            "--no-token-cap",
        ]
    )
    assert paid.no_token_cap is True
    assert not hasattr(paid, "response_reuse_run")
    assert not hasattr(paid, "baran_source_run")


def _structured_groups() -> tuple[PartitionGroup, ...]:
    groups, audit = build_structured_partition(
        tuple(f"c{index}" for index in range(9)),
        2,
        suite="source",
        dataset="tiny",
        source_view="pattern",
        column="value",
        minimum_groups=3,
    )
    assert audit["eligible_cells"] == 8
    assert audit["leftover_cells"] == 1
    return groups


def test_greater_equal_three_partition_and_position_matched_derangement() -> None:
    too_small, audit = build_structured_partition(tuple("abcd"), 2, minimum_groups=3)
    assert too_small == ()
    assert audit["exclusion_reason"] == "fewer_than_three_complete_groups"
    structured = _structured_groups()
    first = build_random_partition(structured, seed=43)
    second = build_random_partition(structured, seed=43)
    assert first == second
    matched = validate_matched_partitions(structured, first)
    assert matched == {
        "groups_per_arm": 4,
        "eligible_cells": 8,
        "position_matching": True,
        "within_column": True,
        "random_equals_structured_group": False,
    }
    assert all(group.structured_group_id == "" for group in first)


def test_round_robin_precedes_exact_physical_dedup() -> None:
    digest_a = hashlib.sha256(b"a").hexdigest()
    digest_b = hashlib.sha256(b"b").hexdigest()
    a = {
        "logical_query_id": "l1",
        "physical_query_id": physical_query_id(digest_a),
        "provider_request_hash": digest_a,
        "physical_request": {
            "request_query_id": "r1",
            "messages": [{"role": "user", "content": "a"}],
            "ordered_cell_ids": ["c1"],
            "prompt_hash": "p1",
            "model_requested": "deepseek-v4-flash",
            "max_tokens": 192,
        },
    }
    b = {
        "logical_query_id": "l2",
        "physical_query_id": physical_query_id(digest_b),
        "provider_request_hash": digest_b,
        "physical_request": {
            "request_query_id": "r2",
            "messages": [{"role": "user", "content": "b"}],
            "ordered_cell_ids": ["c2"],
            "prompt_hash": "p2",
            "model_requested": "deepseek-v4-flash",
            "max_tokens": 192,
        },
    }
    schedule = list(round_robin_logical_schedule({("a",): [a, a | {"logical_query_id": "l3"}], ("b",): [b]}, seed=44))
    logical, physical = deduplicate_physical_schedule(schedule)
    assert len(logical) == 3 and len(physical) == 2
    assert physical[0]["messages"]
    assert [row["physical_schedule_index"] for row in physical] == [0, 1]


def test_ordered_neutral_prompt_preserves_frozen_position() -> None:
    frame = pd.DataFrame({"a": ["x"] * 11, "b": ["ok"] * 11})
    cells = (
        SafeCell("source", "tiny", 2, 0, "a", "2", "bad2"),
        SafeCell("source", "tiny", 10, 0, "a", "10", "bad10"),
    )
    dataset = SafeDataset("source", "tiny", Path("dirty.csv"), frame, cells)
    builder = GroupContextBuilder(dataset, cells, known_error_cells=cells, similar_row_count=0)
    ordered = (str(cells[1].cell_id), str(cells[0].cell_id))
    query_id = compute_ordered_query_id("source", "tiny", ordered)
    material = builder.build_ordered_material(query_id, ordered)
    payload = json.loads(material.messages[1]["content"])
    assert payload["group"]["view"] == "matched_multi_target"
    assert payload["query_id"] == query_id
    assert payload["group"]["cell_ids"] == list(ordered)
    assert payload["group"]["size"] == 2
    assert [row["cell_id"] for row in payload["targets"]] == list(ordered)
    assert "source_view" not in payload


class _Response:
    def __init__(self, value: dict[str, object]):
        self.body = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return self.body


def _request(client: DeepSeekGroupClient, index: int = 0) -> dict[str, object]:
    query_id = f"r{index}"
    messages = (
        {"role": "system", "content": "return JSON"},
        {"role": "user", "content": query_id},
    )
    from budgeted_group_repair_no_baran.group_context import canonical_messages, compute_prompt_hash

    canonical = canonical_messages(messages)
    prompt_hash = compute_prompt_hash(canonical, 192)
    job = GroupLLMJob(query_id, canonical, prompt_hash, (f"c{index}",), 192)
    provider_hash = client.provider_request_hash(job)
    return {
        "physical_schedule_index": index,
        "physical_query_id": physical_query_id(provider_hash),
        "request_query_id": query_id,
        "provider_request_hash": provider_hash,
        "prompt_hash": prompt_hash,
        "messages": list(messages),
        "ordered_cell_ids": [f"c{index}"],
        "model_requested": client.config.model,
        "max_tokens": 192,
    }


def test_streaming_executor_fsync_checkpoint_and_terminal_resume(tmp_path) -> None:
    calls: list[str] = []

    def opener(request, timeout):
        payload = json.loads(request.data)
        query_id = json.loads(json.dumps(payload["messages"]))[1]["content"]
        calls.append(query_id)
        content = json.dumps(
            {
                "query_id": query_id,
                "repairs": [
                    {
                        "cell_id": "c0",
                        "repair": "fixed",
                        "confidence": 0.9,
                        "decision": "propose",
                        "evidence": "dirty evidence",
                        "affected_constraints": [],
                    }
                ],
            }
        )
        return _Response(
            {
                "id": "one",
                "model": payload["model"],
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            }
        )

    client = DeepSeekGroupClient(
        GroupClientConfig(max_retries=0, concurrency=4), api_key="test", opener=opener
    )
    request = _request(client)
    executor = StreamingCheckpointExecutor(client, tmp_path / "checkpoint.jsonl", concurrency=4)
    first = executor.execute([request])
    second = executor.execute([request])
    assert first["executed_physical_calls"] == 1
    assert second["resumed_terminal_calls"] == 1
    assert calls == ["r0"]
    row = json.loads((tmp_path / "checkpoint.jsonl").read_text().strip())
    _assert_terminal_checkpoint_matches_request(
        row,
        request,
        model_requested=client.config.model,
    )
    assert row["model_field_present"] is True
    assert row["provider_model_field_present"] is True
    assert row["model_returned_present"] is True
    assert row["observed_total_tokens"] == 7
    tampered = copy.deepcopy(row)
    tampered["items"][0]["repair"] = "checkpoint-only mutation"
    with pytest.raises(AssertionError, match="disagrees with response_text"):
        _assert_terminal_checkpoint_matches_request(
            tampered,
            request,
            model_requested=client.config.model,
        )


def test_model_identity_abort_is_not_a_terminal_checkpoint(tmp_path) -> None:
    def opener(request, timeout):
        payload = json.loads(request.data)
        return _Response(
            {
                "model": "substitute",
                "choices": [{"message": {"content": "{}"}}],
            }
        )

    client = DeepSeekGroupClient(
        GroupClientConfig(max_retries=5), api_key="test", opener=opener
    )
    executor = StreamingCheckpointExecutor(client, tmp_path / "checkpoint.jsonl")
    with pytest.raises(ProviderModelIdentityError):
        executor.execute([_request(client)])
    assert not (tmp_path / "checkpoint.jsonl").exists()
    audit = json.loads((tmp_path / "infrastructure_abort.jsonl").read_text().strip())
    assert audit["status"] == "model_identity_abort"
    assert audit["model_returned"] == "substitute"


def test_invalid_prediction_retains_raw_repair_for_analysis() -> None:
    checkpoint = {
        "status": "completed",
        "parse_status": "ok",
        "items": [{"cell_id": "c", "repair": "dirty", "decision": "propose"}],
    }
    value = MotivationEvidenceRunner._prediction_for_cell(
        checkpoint, {"cell_id": "c", "dirty_value": "dirty"}
    )
    assert value["prediction"] == "dirty"
    assert value["valid"] is False


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    (
        (
            {"status": "terminal_failure", "parse_status": "provider_failure", "items": []},
            "provider_failure",
        ),
        (
            {"status": "completed", "parse_status": "no_json_object", "items": []},
            "parse_failure",
        ),
        (
            {
                "status": "completed",
                "parse_status": "partial",
                "items": [{"cell_id": "other", "repair": "x", "decision": "propose"}],
                "missing_cell_ids": ["c"],
                "invalid_items": [],
            },
            "missing",
        ),
        (
            {
                "status": "completed",
                "parse_status": "partial",
                "items": [{"cell_id": "other", "repair": "x", "decision": "propose"}],
                "missing_cell_ids": ["c"],
                "invalid_items": [{"cell_id": "c", "status": "invalid_repair"}],
            },
            "parse_failure",
        ),
        (
            {
                "status": "completed",
                "parse_status": "partial",
                "items": [{"cell_id": "other", "repair": "x", "decision": "propose"}],
                "missing_cell_ids": ["c"],
                "duplicate_cell_ids": ["c"],
            },
            "parse_failure",
        ),
    ),
)
def test_prediction_distinguishes_provider_parse_and_pure_missing(
    checkpoint: dict[str, object], expected: str
) -> None:
    value = MotivationEvidenceRunner._prediction_for_cell(
        checkpoint,
        {"cell_id": "c", "dirty_value": "dirty"},
    )
    assert value["parse_status"] == expected


def test_lightweight_file_fingerprint_detects_mutation(tmp_path) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    records = _hash_file_set((source,), relative_to=tmp_path)
    _verify_file_set(records, relative_to=tmp_path, label="test source")
    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="fingerprint drift"):
        _verify_file_set(records, relative_to=tmp_path, label="test source")


def test_same_run_commands_prefer_complete_bound_configs(tmp_path, monkeypatch) -> None:
    run_id = DEFAULT_RUN_ID
    bound_root = tmp_path / "runs" / run_id / "configs"
    bound_root.mkdir(parents=True)
    bound_experiment = bound_root / "motivation_evidence.json"
    bound_llm = bound_root / "deepseek_v4.json"
    bound_experiment.write_text("{}\n", encoding="utf-8")
    bound_llm.write_text("{}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_create(cls, **kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setattr(cli_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(MotivationEvidenceRunner, "create", classmethod(fake_create))
    args = SimpleNamespace(
        run_id=run_id,
        experiment_config=tmp_path / "changed_default.json",
        llm_config=tmp_path / "changed_llm_default.json",
        resume=True,
        token_cap=None,
        no_token_cap=True,
    )
    cli_module._motivation_runner(args)
    assert captured["config_path"] == bound_experiment
    assert captured["llm_config_path"] == bound_llm


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_finalized_ledger_validator_recomputes_correctness_and_transitions(tmp_path) -> None:
    complementarity = tmp_path / "complementarity.csv"
    group = tmp_path / "group.csv"
    cost = tmp_path / "cost.csv"
    complementarity_rows = [
        {
            "suite": "source",
            "dataset": "tiny",
            "cell_id": "c",
            "clean_value": "fixed",
            "baran_prediction": "wrong",
            "baran_valid": True,
            "baran_correct": False,
            "llm_prediction": "fixed",
            "llm_valid": True,
            "llm_correct": True,
            "outcome_quadrant": "n_01",
        }
    ]
    group_rows = [
        {
            "suite": "source",
            "dataset": "tiny",
            "source_view": "pattern",
            "group_size": 2,
            "cell_id": "c",
            "clean_value": "fixed",
            "member_position": 0,
            "structured_group_id": "sg",
            "random_group_id": "rg",
            "singleton_prediction": "fixed",
            "singleton_valid": True,
            "singleton_correct": True,
            "structured_prediction": "wrong",
            "structured_valid": True,
            "structured_correct": False,
            "random_prediction": "fixed",
            "random_valid": True,
            "random_correct": True,
            "structured_rescue": False,
            "structured_interference": True,
            "random_rescue": False,
            "random_interference": False,
        }
    ]
    cost_rows = [
        {
            "physical_schedule_index": 0,
            "physical_query_id": "p",
            "provider_request_hash": "h",
            "request_query_id": "q",
            "model_requested": "deepseek-v4-flash",
            "status": "completed",
            "historical_imported_response": False,
        }
    ]
    _write_csv(complementarity, complementarity_rows)
    _write_csv(group, group_rows)
    _write_csv(cost, cost_rows)
    result = _validate_finalized_ledgers(
        complementarity_path=complementarity,
        group_path=group,
        cost_path=cost,
        population_ids={"c"},
        expected_group_memberships={
            ("source", "tiny", "pattern", 2, "c"): {
                "member_position": 0,
                "structured_group_id": "sg",
                "random_group_id": "rg",
            }
        },
        physical_requests={
            "p": {
                "physical_schedule_index": 0,
                "provider_request_hash": "h",
                "request_query_id": "q",
                "model_requested": "deepseek-v4-flash",
            }
        },
    )
    assert result["quadrants"] == {"n_01": 1}
    group_rows[0]["structured_interference"] = False
    _write_csv(group, group_rows)
    with pytest.raises(AssertionError, match="structured_interference identity drift"):
        _validate_finalized_ledgers(
            complementarity_path=complementarity,
            group_path=group,
            cost_path=cost,
            population_ids={"c"},
            expected_group_memberships={
                ("source", "tiny", "pattern", 2, "c"): {
                    "member_position": 0,
                    "structured_group_id": "sg",
                    "random_group_id": "rg",
                }
            },
            physical_requests={
                "p": {
                    "physical_schedule_index": 0,
                    "provider_request_hash": "h",
                    "request_query_id": "q",
                    "model_requested": "deepseek-v4-flash",
                }
            },
        )
