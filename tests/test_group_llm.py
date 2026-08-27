from __future__ import annotations

import json

import pytest

from budgeted_group_repair_no_baran.group_context import canonical_messages, compute_prompt_hash
from budgeted_group_repair_no_baran.group_llm import (
    DeepSeekGroupClient,
    GroupClientConfig,
    GroupLLMJob,
    ProviderModelIdentityError,
    parse_group_response,
    run_group_llm_batch,
)


def _item(cell_id: str, *, confidence: float = 0.9) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "repair": "fixed-" + cell_id,
        "confidence": confidence,
        "decision": "propose",
        "evidence": "dirty row evidence",
        "affected_constraints": [],
    }


def _job(model_phase: str = "preliminary_singleton") -> GroupLLMJob:
    messages = canonical_messages(
        ({"role": "system", "content": "return JSON"}, {"role": "user", "content": "q1"})
    )
    return GroupLLMJob(
        "q1",
        messages,
        compute_prompt_hash(messages, 192),
        ("c1", "c2"),
        192,
        {"phase": model_phase, "model_requested": "deepseek-v4-flash"},
    )


def test_partial_parser_keeps_only_unambiguous_propose_items() -> None:
    payload = {
        "query_id": "q1",
        "repairs": [_item("c1"), _item("c2"), _item("c2"), _item("unknown")],
    }
    parsed = parse_group_response(json.dumps(payload), "q1", ("c1", "c2"))
    assert parsed.parse_status == "partial"
    assert tuple(parsed.item_by_cell) == ("c1",)
    assert parsed.missing_cell_ids == ("c2",)
    assert parsed.duplicate_cell_ids == ("c2",)
    invalid_decision = _item("c1") | {"decision": "use_llm"}
    parsed = parse_group_response(
        json.dumps({"query_id": "q1", "repairs": [invalid_decision]}), "q1", ("c1",)
    )
    assert parsed.parse_status == "no_valid_items"


class _Response:
    def __init__(self, payload: dict[str, object]):
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return self.body


def test_checkpoint_binds_provider_model_and_preflight_can_feed_singleton(tmp_path) -> None:
    calls: list[str] = []

    def opener(request, timeout):
        request_payload = json.loads(request.data)
        calls.append(request_payload["model"])
        content = json.dumps({"query_id": "q1", "repairs": [_item("c1"), _item("c2")]})
        return _Response(
            {
                "id": "r-" + str(len(calls)),
                "model": request_payload["model"],
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

    preflight_job = _job("model_preflight")
    client = DeepSeekGroupClient(GroupClientConfig(max_retries=0), api_key="test", opener=opener)
    run_group_llm_batch(client, (preflight_job,), tmp_path)
    singleton = run_group_llm_batch(client, (_job(),), tmp_path)
    assert singleton[0]["checkpoint_hit"] is True
    assert singleton[0]["provider_model_field_present"] is True
    assert singleton[0]["model_returned_present"] is True
    assert singleton[0]["model_returned"] == "deepseek-v4-flash"
    assert calls == ["deepseek-v4-flash"]

    changed = DeepSeekGroupClient(
        GroupClientConfig(model="different-model", max_retries=0), api_key="test", opener=opener
    )
    run_group_llm_batch(changed, (_job(),), tmp_path)
    assert calls[-1] == "different-model"


def test_missing_provider_model_is_not_replaced_with_requested_model(tmp_path) -> None:
    def opener(request, timeout):
        content = json.dumps({"query_id": "q1", "repairs": [_item("c1"), _item("c2")]})
        return _Response(
            {
                "id": "missing-model",
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

    client = DeepSeekGroupClient(GroupClientConfig(max_retries=0), api_key="test", opener=opener)
    with pytest.raises(ProviderModelIdentityError) as raised:
        run_group_llm_batch(client, (_job(),), tmp_path)
    assert raised.value.model_field_present is False
    assert raised.value.model_returned == ""
    checkpoint = json.loads((tmp_path / "group_query_checkpoint.jsonl").read_text().splitlines()[-1])
    assert checkpoint["model_requested"] == "deepseek-v4-flash"
    assert checkpoint["model_returned"] == ""
    assert checkpoint["provider_model_field_present"] is False
    assert checkpoint["model_matches_request"] is False


def test_provider_model_mismatch_is_a_hard_failure(tmp_path) -> None:
    def opener(request, timeout):
        content = json.dumps({"query_id": "q1", "repairs": [_item("c1"), _item("c2")]})
        return _Response(
            {
                "id": "wrong-model",
                "model": "deepseek-substitute",
                "choices": [{"message": {"content": content}}],
            }
        )

    client = DeepSeekGroupClient(GroupClientConfig(max_retries=5), api_key="test", opener=opener)
    with pytest.raises(ProviderModelIdentityError) as raised:
        run_group_llm_batch(client, (_job(),), tmp_path)
    assert raised.value.model_field_present is True
    assert raised.value.model_returned == "deepseek-substitute"
    checkpoint = json.loads((tmp_path / "group_query_checkpoint.jsonl").read_text().splitlines()[-1])
    assert checkpoint["attempts"] == 1
    assert checkpoint["model_returned"] == "deepseek-substitute"
    assert checkpoint["model_returned_present"] is True
