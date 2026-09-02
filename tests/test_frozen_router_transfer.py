from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

import budgeted_group_repair_no_baran.frozen_router_transfer as transfer
from budgeted_group_repair_no_baran.group_llm import (
    CompletionTruncatedError,
    GroupClientConfig,
    ProviderModelIdentityError,
    run_group_llm_batch,
)


@pytest.fixture(scope="module")
def frozen_plan() -> transfer.FrozenTransferPlan:
    return transfer.build_frozen_transfer_plan(hash_protected_sources=False)


@pytest.fixture(scope="module")
def qwen_plan() -> transfer.FrozenTransferPlan:
    return transfer.build_frozen_transfer_plan(
        profile=transfer.QWEN37_FLASH_PROFILE,
        hash_protected_sources=False,
    )


def _item(cell_id: str, *, decision: str = "propose") -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "repair": "fixed-" + cell_id,
        "confidence": 0.9,
        "decision": decision,
        "evidence": "dirty-only evidence",
        "affected_constraints": [],
    }


class _Response:
    def __init__(self, payload: dict[str, object]):
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return self.body


def _provider_response(
    request,
    *,
    returned_model: str | None = transfer.REPAIR_MODEL,
    finish_reason: str | None = "stop",
):
    payload = json.loads(request.data)
    query_id = json.loads(payload["messages"][-1]["content"])["query_id"]
    cell_ids = json.loads(payload["messages"][-1]["content"])["targets"]
    result: dict[str, object] = {
        "id": "test-response",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": json.dumps(
                        {
                            "query_id": query_id,
                            "repairs": [_item(row["cell_id"]) for row in cell_ids],
                        }
                    )
                }
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 100,
        },
    }
    if returned_model is not None:
        result["model"] = returned_model
    return _Response(result)


def _qwen_provider_response(
    request,
    *,
    returned_model: str = transfer.QWEN_MODEL,
    finish_reason: str | None = "stop",
):
    response = _provider_response(
        request,
        returned_model=returned_model,
        finish_reason=finish_reason,
    )
    payload = json.loads(response.body)
    payload["usage"] = {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 25},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    return _Response(payload)


def test_plan_binds_only_fixed_eight_and_exact_frozen_identities(
    frozen_plan: transfer.FrozenTransferPlan,
) -> None:
    summary = frozen_plan.summary()
    assert summary["request_count"] == 1_994
    assert summary["group_size_counts"] == {"1": 1_559, "4": 435}
    assert summary["estimated_total_tokens"] == 3_604_244
    assert summary["maximum_http_attempts"] == 11_964
    assert summary["worst_case_estimated_tokens"] == 21_625_464
    assert summary["selected_query_set_sha256"] == transfer.EXPECTED_QUERY_SET_SHA256
    assert summary["prompt_identity_sha256"] == transfer.EXPECTED_PROMPT_IDENTITY_SHA256
    assert frozen_plan.preflight_query_id == transfer.EXPECTED_PREFLIGHT_QUERY_ID
    assert all(row["model_requested"] == transfer.REPAIR_MODEL for row in frozen_plan.request_rows)
    assert all(row["dataset"] != "movies_1" for row in frozen_plan.request_rows)
    assert len({row["query_id"] for row in frozen_plan.request_rows}) == 1_994


def test_qwen_profile_changes_only_provider_request_identity(
    frozen_plan: transfer.FrozenTransferPlan,
    qwen_plan: transfer.FrozenTransferPlan,
) -> None:
    assert qwen_plan.profile == transfer.QWEN37_FLASH_PROFILE
    assert qwen_plan.repair_model == "qwen3.7-flash-2026-07-15"
    assert qwen_plan.summary()["request_count"] == 1_994
    assert qwen_plan.summary()["group_size_counts"] == {"1": 1_559, "4": 435}
    assert qwen_plan.summary()["maximum_serialized_message_bytes_per_request"] == 22_980
    assert qwen_plan.identity_audit == frozen_plan.identity_audit
    assert (
        qwen_plan.identity_audit["frozen_message_hash"]
        == transfer.EXPECTED_FROZEN_MESSAGE_SHA256
    )
    assert [row["query_id"] for row in qwen_plan.request_rows] == [
        row["query_id"] for row in frozen_plan.request_rows
    ]
    assert [row["prompt_hash"] for row in qwen_plan.request_rows] == [
        row["prompt_hash"] for row in frozen_plan.request_rows
    ]
    assert qwen_plan.request_plan_sha256 != frozen_plan.request_plan_sha256
    assert (
        qwen_plan.provider_request_hash
        == transfer.EXPECTED_QWEN_PROVIDER_REQUEST_HASH
    )
    assert all(row["model_requested"] == transfer.QWEN_MODEL for row in qwen_plan.request_rows)
    assert all(row["dataset"] != "movies_1" for row in qwen_plan.request_rows)


def test_qwen_bound_config_is_closed_and_preserves_flash_prompt_policy(
    qwen_plan: transfer.FrozenTransferPlan,
) -> None:
    config = transfer._bound_repair_config(qwen_plan)
    assert config["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config["model"] == transfer.QWEN_MODEL
    assert config["temperature"] == 0.0
    assert config["top_p"] == 1.0
    assert config["extra_body"] == {"enable_thinking": False}
    assert "thinking" not in config["extra_body"]
    assert config["planning_completion_token_ceiling"] == {
        "singleton": 192,
        "group_base": 64,
        "per_cell": 192,
        "formula": "size == 1 ? 192 : 64 + 192 * size",
    }
    assert config["completion_token_parameter"] == "max_completion_tokens"
    assert config["stream"] is False
    assert config["provider_max_completion_tokens"] == {
        "singleton": 4096,
        "group_size_4": 16384,
    }
    assert config["documented_max_completion_token_tolerance"] == 10
    assert config["strict_complete_response"] is True


def test_qwen_provider_ceilings_and_stratified_preflight_are_deterministic(
    qwen_plan: transfer.FrozenTransferPlan,
) -> None:
    assert {
        int(row["provider_max_completion_tokens"])
        for row in qwen_plan.request_rows
        if row["group_size"] == 1
    } == {4096}
    assert {
        int(row["provider_max_completion_tokens"])
        for row in qwen_plan.request_rows
        if row["group_size"] == 4
    } == {16384}
    assert all(
        row["provider_completion_token_parameter"]
        == "max_completion_tokens"
        for row in qwen_plan.request_rows
    )
    selected = [qwen_plan.action_by_id[value] for value in qwen_plan.preflight_query_ids]
    assert len(selected) == 15
    assert all(action.group_size == 4 for action in selected[:8])
    assert len({(action.suite, action.dataset) for action in selected[:8]}) == 8
    assert all(action.group_size == 1 for action in selected[8:])
    assert len({(action.suite, action.dataset) for action in selected[8:]}) == 7
    for action in selected:
        peers = [
            candidate
            for candidate in qwen_plan.actions
            if candidate.suite == action.suite
            and candidate.dataset == action.dataset
            and candidate.group_size == action.group_size
        ]
        assert action.estimated_prompt_tokens == max(
            candidate.estimated_prompt_tokens for candidate in peers
        )


def test_dry_run_writes_authorization_without_loading_credentials_or_api(
    frozen_plan: transfer.FrozenTransferPlan,
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        transfer,
        "build_frozen_transfer_plan",
        lambda *args, **kwargs: frozen_plan,
    )
    monkeypatch.setattr(
        transfer,
        "_flash_replay_parity",
        lambda plan: {
            "ok": True,
            "checkpoint_rows_bound": 1_994,
            "cell_rows_replayed": 14_523,
            "cell_mismatches": 0,
            "request_plan_sha256_before_oracle_replay": plan.request_plan_sha256,
            "request_plan_sha256_after_oracle_replay": plan.request_plan_sha256,
        },
    )
    monkeypatch.setattr(
        transfer,
        "load_env_file",
        lambda *_: (_ for _ in ()).throw(AssertionError("env file was loaded")),
    )
    report = transfer.create_dry_run(project_root=tmp_path, run_id="dry-test")
    run_dir = tmp_path / "runs" / "dry-test"
    assert report["api_called"] is False
    assert report["environment_file_loaded"] is False
    assert (run_dir / "authorized_requests.jsonl").is_file()
    assert not (run_dir / "llm" / transfer.MODEL_NAMESPACE / "api_attempt_audit.jsonl").exists()
    assert len(list(transfer._iter_jsonl(run_dir / "authorized_requests.jsonl"))) == 1_994


def test_qwen_dry_run_uses_cny_price_and_never_loads_credentials(
    qwen_plan: transfer.FrozenTransferPlan,
    monkeypatch,
    tmp_path: Path,
) -> None:
    protected = [
        qwen_plan.source_run / "run_manifest.json",
        qwen_plan.project_root
        / "runs"
        / transfer.PRO_TRANSFER_RUN_ID
        / "run_manifest.json",
    ]
    diagnostic_manifest = (
        qwen_plan.project_root
        / "runs"
        / transfer.QWEN_DIAGNOSTIC_RUN_ID
        / "run_manifest.json"
    )
    if diagnostic_manifest.is_file():
        protected.append(diagnostic_manifest)
    before = {str(path): transfer._sha256(path) for path in protected}
    monkeypatch.setattr(
        transfer,
        "build_frozen_transfer_plan",
        lambda *args, **kwargs: qwen_plan,
    )
    monkeypatch.setattr(
        transfer,
        "_flash_replay_parity",
        lambda plan: {
            "ok": True,
            "checkpoint_rows_bound": 1_994,
            "cell_rows_replayed": 14_523,
            "cell_mismatches": 0,
            "request_plan_sha256_before_oracle_replay": plan.request_plan_sha256,
            "request_plan_sha256_after_oracle_replay": plan.request_plan_sha256,
        },
    )
    monkeypatch.setattr(
        transfer,
        "load_env_file",
        lambda *_: (_ for _ in ()).throw(AssertionError("env file was loaded")),
    )
    report = transfer.create_dry_run(
        project_root=tmp_path,
        run_id="qwen-dry-test",
        profile=transfer.QWEN37_FLASH_PROFILE,
    )
    expected = (
        2_942_996 * 0.2
        + transfer.QWEN_PROVIDER_BILLED_OUTPUT_CAP_ESTIMATE * 0.8
    ) / 1_000_000
    assert report["api_called"] is False
    assert report["environment_file_loaded"] is False
    assert report["estimated_cost_cny"][
        "normal_round_all_input_cache_miss"
    ] == pytest.approx(expected)
    manifest = transfer._read_json(tmp_path / "runs" / "qwen-dry-test" / "run_manifest.json")
    assert manifest["profile_id"] == transfer.QWEN_MODEL
    assert manifest["model_namespace"] == transfer.QWEN_MODEL_NAMESPACE
    assert manifest["selection_budget_reference"] == (
        "frozen_deepseek_v4_flash_plan"
    )
    assert manifest["selection_planning_budget_share"] == 0.2
    assert manifest["actual_token_parity_enforced"] is False
    assert manifest["qwen_actual_tokens_used_for_selection"] is False
    assert manifest["preflight_query_count"] == 15
    assert {str(path): transfer._sha256(path) for path in protected} == before


def test_selection_hash_drift_is_fail_closed(monkeypatch) -> None:
    real_hash = transfer._sha256

    def drift(path: str | Path) -> str:
        source = Path(path)
        if source.name == "source__hospital.json" and "selections" in source.parts:
            return "0" * 64
        return real_hash(source)

    monkeypatch.setattr(transfer, "_sha256", drift)
    with pytest.raises(ValueError, match="selection SHA-256 drift"):
        transfer.build_frozen_transfer_plan(hash_protected_sources=False)


@pytest.mark.parametrize("field", ("prompt_hash", "cell_ids"))
def test_candidate_prompt_or_membership_drift_is_fail_closed(
    frozen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
    field: str,
) -> None:
    action = frozen_plan.actions[0]
    candidate_path = transfer._candidate_path(
        frozen_plan.source_run, "source__hospital"
    )
    raw = next(
        row for row in transfer._iter_jsonl(candidate_path) if row["query_id"] == action.query_id
    )
    if field == "prompt_hash":
        raw[field] = "0" * 64
    else:
        raw[field] = [*raw[field][:-1], "source:hospital:0:0"]
    path = tmp_path / "candidate.jsonl"
    path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity drift|prompt drift"):
        transfer._load_selected_actions(path, [action.query_id])


def test_gate_membership_and_cost_drift_are_fail_closed(
    frozen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
) -> None:
    action = frozen_plan.actions[0]
    actions = {action.query_id: action}
    header = (
        "query_id,cell_id,group_size,estimated_total_tokens,conservative_uplift\n"
    )
    membership_path = tmp_path / "membership.csv"
    membership_path.write_text(
        header
        + f"{action.query_id},wrong-cell,{action.group_size},"
        f"{action.estimated_total_tokens},0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="membership drift"):
        transfer._selected_gate_rows(membership_path, actions)

    cost_path = tmp_path / "cost.csv"
    cost_path.write_text(
        header
        + "".join(
            f"{action.query_id},{cell_id},{action.group_size},"
            f"{action.estimated_total_tokens + 1},0.1\n"
            for cell_id in action.cell_ids
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cost drift"):
        transfer._selected_gate_rows(cost_path, actions)


@pytest.mark.parametrize("returned_model", (None, "deepseek-v4-flash"))
def test_model_identity_missing_or_mismatch_is_audited_and_stops(
    frozen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
    returned_model: str | None,
) -> None:
    action = frozen_plan.action_by_id[frozen_plan.preflight_query_id]
    job = transfer._job_for_action(
        action, require_complete=True, transfer_stage="preflight"
    )

    def opener(request, timeout):
        return _provider_response(request, returned_model=returned_model)

    client = transfer.AuditedDeepSeekGroupClient(
        GroupClientConfig(model=transfer.REPAIR_MODEL, max_retries=5),
        api_key="test-only",
        audit_path=tmp_path / "audit.jsonl",
        opener=opener,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(ProviderModelIdentityError):
        run_group_llm_batch(client, [job], tmp_path / "pro")
    audit = list(transfer._iter_jsonl(tmp_path / "audit.jsonl"))
    assert len(audit) == 1
    assert audit[0]["status"] == "model_identity_error"
    assert audit[0]["requested_model"] == transfer.REPAIR_MODEL


@pytest.mark.parametrize("returned_model", ("qwen3.7-flash", "deepseek-v4-pro"))
def test_qwen_snapshot_identity_is_exact_and_fail_closed(
    qwen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
    returned_model: str,
) -> None:
    action = qwen_plan.action_by_id[qwen_plan.preflight_query_id]
    job = transfer._job_for_action(
        action,
        profile=transfer.QWEN37_FLASH_PROFILE,
        require_complete=True,
        transfer_stage="preflight",
    )

    def opener(request, timeout):
        return _qwen_provider_response(request, returned_model=returned_model)

    client = transfer.AuditedOpenAICompatibleGroupClient(
        GroupClientConfig(
            base_url=transfer.QWEN37_FLASH_PROFILE.base_url,
            model=transfer.QWEN_MODEL,
            max_retries=0,
            completion_token_parameter="max_completion_tokens",
            stream=False,
            extra_body={"enable_thinking": False},
        ),
        api_key="test-only",
        audit_path=tmp_path / "audit.jsonl",
        opener=opener,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(ProviderModelIdentityError):
        run_group_llm_batch(client, [job], tmp_path / "qwen")
    audit = list(transfer._iter_jsonl(tmp_path / "audit.jsonl"))
    assert audit[0]["status"] == "model_identity_error"
    assert audit[0]["requested_model"] == transfer.QWEN_MODEL
    assert audit[0]["returned_model"] == returned_model


def test_qwen_complete_response_records_nested_usage_and_isolated_cache(
    qwen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
) -> None:
    action = qwen_plan.action_by_id[qwen_plan.preflight_query_id]
    job = transfer._job_for_action(
        action,
        profile=transfer.QWEN37_FLASH_PROFILE,
        require_complete=True,
        transfer_stage="preflight",
    )

    def opener(request, timeout):
        payload = json.loads(request.data)
        assert payload["enable_thinking"] is False
        assert "thinking" not in payload
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["stream"] is False
        assert payload["max_completion_tokens"] == 16384
        assert "max_tokens" not in payload
        return _qwen_provider_response(request)

    namespace = tmp_path / "llm" / transfer.QWEN_MODEL_NAMESPACE
    client = transfer.AuditedOpenAICompatibleGroupClient(
        GroupClientConfig(
            base_url=transfer.QWEN37_FLASH_PROFILE.base_url,
            model=transfer.QWEN_MODEL,
            max_retries=0,
            completion_token_parameter="max_completion_tokens",
            stream=False,
            extra_body={"enable_thinking": False},
        ),
        api_key="test-only",
        audit_path=namespace / "api_attempt_audit.jsonl",
        opener=opener,
        sleep_fn=lambda _: None,
    )
    rows = run_group_llm_batch(client, [job], namespace, concurrency=1)
    assert rows[0]["status"] == "success"
    assert rows[0]["model_returned"] == transfer.QWEN_MODEL
    attempts = list(transfer._iter_jsonl(namespace / "api_attempt_audit.jsonl"))
    assert attempts[0]["input_tokens"] == 100
    assert attempts[0]["input_cache_hit_tokens"] == 25
    assert attempts[0]["input_cache_miss_tokens"] == 75
    assert attempts[0]["reasoning_tokens"] == 0
    assert (namespace / "group_response_cache.jsonl").is_file()
    assert "deepseek" not in str(namespace)


def test_qwen_finish_reason_length_fails_and_never_enters_success_cache(
    qwen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
) -> None:
    action = qwen_plan.action_by_id[qwen_plan.preflight_query_ids[0]]
    job = transfer._job_for_action(
        action,
        profile=transfer.QWEN37_FLASH_PROFILE,
        transfer_stage="batch",
    )

    def opener(request, timeout):
        return _qwen_provider_response(request, finish_reason="length")

    namespace = tmp_path / "llm" / transfer.QWEN_MODEL_NAMESPACE
    client = transfer.AuditedOpenAICompatibleGroupClient(
        GroupClientConfig(
            base_url=transfer.QWEN37_FLASH_PROFILE.base_url,
            model=transfer.QWEN_MODEL,
            max_retries=0,
            completion_token_parameter="max_completion_tokens",
            stream=False,
            extra_body={"enable_thinking": False},
        ),
        api_key="test-only",
        audit_path=namespace / "api_attempt_audit.jsonl",
        opener=opener,
        sleep_fn=lambda _: None,
    )
    with pytest.raises(CompletionTruncatedError):
        run_group_llm_batch(
            client,
            [job],
            namespace,
            concurrency=1,
            fail_fast_finish_reasons={"length"},
        )
    checkpoint = list(
        transfer._iter_jsonl(namespace / "group_query_checkpoint.jsonl")
    )
    assert checkpoint[-1]["status"] == "failed"
    assert checkpoint[-1]["finish_reason"] == "length"
    assert not (namespace / "group_response_cache.jsonl").exists()
    attempt = next(transfer._iter_jsonl(namespace / "api_attempt_audit.jsonl"))
    assert attempt["finish_reason"] == "length"
    assert attempt["total_tokens"] == 120


def test_qwen_length_retry_preserves_request_and_then_caches_success(
    qwen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
) -> None:
    action = qwen_plan.action_by_id[qwen_plan.preflight_query_ids[0]]
    job = transfer._job_for_action(
        action,
        profile=transfer.QWEN37_FLASH_PROFILE,
        transfer_stage="batch",
    )
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        return _qwen_provider_response(
            request,
            finish_reason="length" if calls == 1 else "stop",
        )

    namespace = tmp_path / "llm" / transfer.QWEN_MODEL_NAMESPACE
    client = transfer.AuditedOpenAICompatibleGroupClient(
        GroupClientConfig(
            base_url=transfer.QWEN37_FLASH_PROFILE.base_url,
            model=transfer.QWEN_MODEL,
            max_retries=1,
            backoff_initial_seconds=0,
            completion_token_parameter="max_completion_tokens",
            stream=False,
            extra_body={"enable_thinking": False},
        ),
        api_key="test-only",
        audit_path=namespace / "api_attempt_audit.jsonl",
        opener=opener,
        sleep_fn=lambda _: None,
    )
    rows = run_group_llm_batch(
        client,
        [job],
        namespace,
        concurrency=1,
        fail_fast_finish_reasons={"length"},
    )
    assert calls == 2
    assert rows[0]["status"] == "success"
    assert rows[0]["finish_reason"] == "stop"
    assert rows[0]["attempts"] == 2
    assert [
        row["finish_reason"]
        for row in transfer._iter_jsonl(namespace / "api_attempt_audit.jsonl")
    ] == ["length", "stop"]
    assert (namespace / "group_response_cache.jsonl").is_file()


def test_only_selected_failed_finish_reason_is_retried_from_checkpoint(
    qwen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
) -> None:
    action = qwen_plan.action_by_id[qwen_plan.preflight_query_ids[0]]
    job = transfer._job_for_action(
        action,
        profile=transfer.QWEN37_FLASH_PROFILE,
        transfer_stage="batch",
    )
    namespace = tmp_path / "llm" / transfer.QWEN_MODEL_NAMESPACE

    first_client = transfer.AuditedOpenAICompatibleGroupClient(
        GroupClientConfig(
            base_url=transfer.QWEN37_FLASH_PROFILE.base_url,
            model=transfer.QWEN_MODEL,
            max_retries=0,
            completion_token_parameter="max_completion_tokens",
            stream=False,
            extra_body={"enable_thinking": False},
        ),
        api_key="test-only",
        audit_path=namespace / "api_attempt_audit.jsonl",
        opener=lambda request, timeout: _qwen_provider_response(
            request, finish_reason="length"
        ),
        sleep_fn=lambda _: None,
    )
    with pytest.raises(CompletionTruncatedError):
        run_group_llm_batch(
            first_client,
            [job],
            namespace,
            concurrency=1,
            fail_fast_finish_reasons={"length"},
        )

    second_client = transfer.AuditedOpenAICompatibleGroupClient(
        GroupClientConfig(
            base_url=transfer.QWEN37_FLASH_PROFILE.base_url,
            model=transfer.QWEN_MODEL,
            max_retries=0,
            completion_token_parameter="max_completion_tokens",
            stream=False,
            extra_body={"enable_thinking": False},
        ),
        api_key="test-only",
        audit_path=namespace / "api_attempt_audit.jsonl",
        opener=lambda request, timeout: _qwen_provider_response(request),
        sleep_fn=lambda _: None,
    )
    rows = run_group_llm_batch(
        second_client,
        [job],
        namespace,
        concurrency=1,
        retry_failed=False,
        retry_failed_finish_reasons={"length"},
        fail_fast_finish_reasons={"length"},
    )
    assert rows[0]["status"] == "success"
    assert rows[0]["checkpoint_hit"] is False
    checkpoints = list(
        transfer._iter_jsonl(namespace / "group_query_checkpoint.jsonl")
    )
    assert [row["finish_reason"] for row in checkpoints] == ["length", "stop"]


def test_failed_preflight_unconditionally_blocks_batch(
    qwen_plan: transfer.FrozenTransferPlan,
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_id = "qwen-failed-preflight"
    run_dir = tmp_path / "runs" / run_id
    transfer._write_json(run_dir / "run_manifest.json", {"status": "preflight_failed"})
    monkeypatch.setattr(transfer, "_reload_bound_plan", lambda *_: qwen_plan)
    with pytest.raises(RuntimeError, match="not allowed"):
        transfer.run_authorized_batch(
            project_root=tmp_path,
            run_id=run_id,
            profile=transfer.QWEN37_FLASH_PROFILE,
            env_file=tmp_path / ".qwen_env",
            resume=True,
            approve_paid_api=True,
        )


def test_any_stratified_preflight_failure_stops_and_blocks_batch(
    qwen_plan: transfer.FrozenTransferPlan,
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_id = "qwen-stratified-preflight-failure"
    run_dir = tmp_path / "runs" / run_id
    transfer._write_json(run_dir / "run_manifest.json", {"status": "planned"})
    monkeypatch.setattr(transfer, "_reload_bound_plan", lambda *_: qwen_plan)
    monkeypatch.setattr(transfer, "_load_paid_environment", lambda *_: ())
    monkeypatch.setattr(
        transfer,
        "_paid_client",
        lambda *_: SimpleNamespace(config=SimpleNamespace(concurrency=1)),
    )
    request_by_id = {
        str(row["query_id"]): row for row in qwen_plan.request_rows
    }
    calls = 0

    def fake_batch(client, jobs, namespace, **kwargs):
        nonlocal calls
        calls += 1
        job = list(jobs)[0]
        action = qwen_plan.action_by_id[job.query_id]
        request = request_by_id[job.query_id]
        failed = calls == 5
        items = [_item(value) for value in action.cell_ids]
        if failed:
            items = items[:-1]
        row = {
            "query_id": job.query_id,
            "prompt_hash": job.prompt_hash,
            "provider_request_hash": request["provider_request_hash"],
            "status": "failed" if failed else "success",
            "parse_status": "partial" if failed else "ok",
            "items": items,
            "missing_cell_ids": [action.cell_ids[-1]] if failed else [],
            "unknown_cell_ids": [],
            "duplicate_cell_ids": [],
            "invalid_items": [],
            "provider_model_field_present": True,
            "model_returned": transfer.QWEN_MODEL,
            "model_requested": transfer.QWEN_MODEL,
            "model_matches_request": True,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "thinking_disabled": True,
        }
        if not failed:
            cache_path = Path(namespace) / "group_response_cache.jsonl"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "provider_request_hash": request[
                                "provider_request_hash"
                            ],
                            "model_returned": transfer.QWEN_MODEL,
                            "finish_reason": "stop",
                        }
                    )
                    + "\n"
                )
        return [row]

    monkeypatch.setattr(transfer, "run_group_llm_batch", fake_batch)
    with pytest.raises(RuntimeError, match="preflight failed"):
        transfer.run_preflight(
            project_root=tmp_path,
            run_id=run_id,
            profile=transfer.QWEN37_FLASH_PROFILE,
            env_file=tmp_path / ".qwen_env",
            approve_paid_api=True,
        )
    assert calls == 5
    report = transfer._read_json(
        run_dir / "llm" / transfer.QWEN_MODEL_NAMESPACE / "preflight.json"
    )
    assert report["planned_query_count"] == 15
    assert report["completed_query_count"] == 5
    assert report["passed"] is False
    assert transfer._read_json(run_dir / "run_manifest.json")["status"] == (
        "preflight_failed"
    )


def test_preflight_env_failure_does_not_mark_run_as_running(
    qwen_plan: transfer.FrozenTransferPlan,
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_id = "qwen-preflight-env-failure"
    run_dir = tmp_path / "runs" / run_id
    transfer._write_json(
        run_dir / "run_manifest.json",
        {
            "status": "planned",
            "api_called": False,
            "paid_api_authorized": False,
        },
    )
    monkeypatch.setattr(transfer, "_reload_bound_plan", lambda *_: qwen_plan)
    monkeypatch.setattr(
        transfer,
        "_load_paid_environment",
        lambda *_: (_ for _ in ()).throw(RuntimeError("missing env file")),
    )

    with pytest.raises(RuntimeError, match="missing env file"):
        transfer.run_preflight(
            project_root=tmp_path,
            run_id=run_id,
            profile=transfer.QWEN37_FLASH_PROFILE,
            env_file=tmp_path / ".qwen_env",
            approve_paid_api=True,
        )

    manifest = transfer._read_json(run_dir / "run_manifest.json")
    assert manifest["status"] == "planned"
    assert manifest["api_called"] is False
    assert manifest["paid_api_authorized"] is False


def test_qwen_batch_never_accepts_partial_success(
    qwen_plan: transfer.FrozenTransferPlan,
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_id = "qwen-partial-is-fatal"
    run_dir = tmp_path / "runs" / run_id
    transfer._write_json(run_dir / "run_manifest.json", {"status": "preflight_complete"})
    monkeypatch.setattr(transfer, "_reload_bound_plan", lambda *_: qwen_plan)
    monkeypatch.setattr(transfer, "_load_paid_environment", lambda *_: ())
    monkeypatch.setattr(
        transfer,
        "_paid_client",
        lambda *_: SimpleNamespace(config=SimpleNamespace(concurrency=4)),
    )

    partial_query_id = next(
        str(row["query_id"])
        for row in qwen_plan.request_rows
        if row["query_id"] not in set(qwen_plan.preflight_query_ids)
    )

    def fake_batch(*args, **kwargs):
        return [
            {
                "query_id": row["query_id"],
                "status": "success",
                "parse_status": (
                    "partial"
                    if row["query_id"] == partial_query_id
                    else "ok"
                ),
                "checkpoint_hit": row["query_id"]
                in set(qwen_plan.preflight_query_ids),
                "cache_hit": False,
                "attempts": 0,
            }
            for row in qwen_plan.request_rows
        ]

    monkeypatch.setattr(transfer, "run_group_llm_batch", fake_batch)
    with pytest.raises(ValueError, match="partial"):
        transfer.run_authorized_batch(
            project_root=tmp_path,
            run_id=run_id,
            profile=transfer.QWEN37_FLASH_PROFILE,
            env_file=tmp_path / ".qwen_env",
            resume=True,
            approve_paid_api=True,
        )
    manifest = transfer._read_json(run_dir / "run_manifest.json")
    assert manifest["status"] == "batch_interrupted"
    assert manifest["failed_preflight_accepted_for_strict_transfer"] is False


def test_complete_preflight_uses_isolated_pro_cache_and_resume_is_free(
    frozen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
) -> None:
    calls = 0
    action = frozen_plan.action_by_id[frozen_plan.preflight_query_id]
    job = transfer._job_for_action(
        action, require_complete=True, transfer_stage="preflight"
    )

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("retry once")
        return _provider_response(request)

    namespace = tmp_path / "llm" / transfer.MODEL_NAMESPACE
    client = transfer.AuditedDeepSeekGroupClient(
        GroupClientConfig(
            model=transfer.REPAIR_MODEL,
            max_retries=1,
            backoff_initial_seconds=0,
            extra_body={"thinking": {"type": "disabled"}},
        ),
        api_key="test-only",
        audit_path=namespace / "api_attempt_audit.jsonl",
        opener=opener,
        sleep_fn=lambda _: None,
    )
    first = run_group_llm_batch(client, [job], namespace, concurrency=1)
    resumed = run_group_llm_batch(
        client, [job], namespace, concurrency=1, retry_failed=False
    )
    assert calls == 2
    assert first[0]["status"] == "success"
    assert first[0]["parse_status"] == "ok"
    assert first[0]["model_returned"] == transfer.REPAIR_MODEL
    assert first[0]["usage"]["total_tokens"] == 120
    assert resumed[0]["checkpoint_hit"] is True
    assert (namespace / "group_response_cache.jsonl").is_file()
    assert "deepseek-v4-flash" not in str(namespace)
    attempts = list(transfer._iter_jsonl(namespace / "api_attempt_audit.jsonl"))
    assert [row["status"] for row in attempts] == ["request_error", "success"]
    assert all(row["prompt_hash"] == action.prompt_hash for row in attempts)


def test_partial_abstain_api_failure_and_verifier_diagnostics() -> None:
    ledger = [
        {
            "suite": "source",
            "dataset": "toy",
            "cell_id": "c1",
            "correct_repair": True,
            "rejected_reasons": ["low_llm_confidence"],
        },
        {
            "suite": "source",
            "dataset": "toy",
            "cell_id": "c2",
            "correct_repair": False,
            "rejected_reasons": ["fallback_decision"],
        },
    ]
    baran = [
        {"suite": "source", "dataset": "toy", "cell_id": "c1", "correct_repair": False},
        {"suite": "source", "dataset": "toy", "cell_id": "c2", "correct_repair": True},
    ]
    responses = {
        ("q1", "p1"): {
            "status": "success",
            "parse_status": "partial",
            "items": [_item("c1", decision="abstain")],
        },
        ("q2", "p2"): {"status": "failed", "parse_status": "no_json_object", "items": []},
        ("q3", "p3"): {"status": "failed", "parse_status": "llm_error", "items": []},
    }
    result = transfer._diagnostics(ledger, responses, baran)
    assert result["rescued_cells_vs_baran"] == 1
    assert result["harmed_cells_vs_baran"] == 1
    assert result["net_gain_vs_baran"] == 0
    assert result["verifier_rejections"] == 2
    assert result["partial_parse_queries"] == 1
    assert result["parse_failures"] == 1
    assert result["abstention_items"] == 1
    assert result["abstention_queries"] == 1
    assert result["api_failure_queries"] == 1


def test_actual_token_and_cache_split_cost_is_recomputed(
    frozen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
) -> None:
    request = frozen_plan.request_rows[0]
    audit_path = (
        tmp_path
        / "llm"
        / transfer.MODEL_NAMESPACE
        / "api_attempt_audit.jsonl"
    )
    transfer._write_jsonl(
        audit_path,
        [
            {
                "query_id": request["query_id"],
                "prompt_hash": request["prompt_hash"],
                "provider_request_hash": request["provider_request_hash"],
                "started_at_utc": "2026-08-30T12:00:00+00:00",
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_cache_hit_tokens": 25,
                "input_cache_miss_tokens": 75,
            }
        ],
    )
    result = transfer._cost_audit(frozen_plan, tmp_path)
    expected = (25 * 0.022 + 75 * 0.66 + 20 * 1.98) / 1_000_000
    assert result["physical_http_attempts"] == 1
    assert result["retry_attempts"] == 0
    assert result["observed_tokens"]["total_tokens"] == 120
    assert result["known_cost_usd"] == pytest.approx(expected)
    assert result["conservative_cost_upper_usd"] == pytest.approx(expected)
    assert result["cost_is_exact"] is True


def test_qwen_nested_cache_cost_and_input_tier_are_recomputed(
    qwen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
) -> None:
    request = qwen_plan.request_rows[0]
    audit_path = (
        tmp_path
        / "llm"
        / transfer.QWEN_MODEL_NAMESPACE
        / "api_attempt_audit.jsonl"
    )
    transfer._write_jsonl(
        audit_path,
        [
            {
                "query_id": request["query_id"],
                "prompt_hash": request["prompt_hash"],
                "provider_request_hash": request["provider_request_hash"],
                "started_at_utc": "2026-08-31T12:00:00+00:00",
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_cache_hit_tokens": 25,
                "input_cache_miss_tokens": 75,
                "reasoning_tokens": 0,
                "finish_reason": "stop",
            },
            {
                "query_id": request["query_id"],
                "prompt_hash": request["prompt_hash"],
                "provider_request_hash": request["provider_request_hash"],
                "started_at_utc": "2026-08-31T12:00:01+00:00",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_cache_hit_tokens": 0,
                "input_cache_miss_tokens": 10,
                "reasoning_tokens": 0,
                "finish_reason": "length",
            }
        ],
    )
    result = transfer._cost_audit(qwen_plan, tmp_path)
    expected = (
        25 * 0.04 + 75 * 0.2 + 20 * 0.8 + 10 * 0.2 + 5 * 0.8
    ) / 1_000_000
    assert result["physical_http_attempts"] == 2
    assert result["retry_attempts"] == 1
    assert result["observed_tokens"]["total_tokens"] == 135
    assert result["attempts_by_input_price_tier"] == {"input_le_32k": 2}
    assert result["finish_reason_counts"] == {"length": 1, "stop": 1}
    assert result["attempts_per_query"][request["query_id"]] == 2
    assert result["known_cost_cny"] == pytest.approx(expected)
    assert result["conservative_cost_upper_cny"] == pytest.approx(expected)
    assert result["cost_is_exact"] is True


def test_metric_deltas_and_f1_win_tie_loss() -> None:
    rows = []
    for method, values in {
        "pro": (0.8, 0.7),
        "flash": (0.7, 0.7),
        "baran": (0.8, 0.6),
    }.items():
        for dataset, f1 in zip(("a", "b"), values):
            rows.append(
                {
                    "comparison_method": method,
                    "scope": "dataset",
                    "dataset": dataset,
                    "precision": f1,
                    "recall": f1,
                    "f1": f1,
                }
            )
    result = transfer._metric_deltas(rows)
    assert result["win_tie_loss"]["flash"] == {
        "win": 1,
        "tie": 1,
        "loss": 0,
        "metric": "per-dataset F1",
        "tolerance": 1e-12,
    }
    assert result["win_tie_loss"]["baran"]["win"] == 1
    assert result["win_tie_loss"]["baran"]["tie"] == 1


def test_qwen_metric_deltas_include_flash_baran_and_pro() -> None:
    rows = []
    for method, values in {
        "qwen": (0.9, 0.5),
        "flash": (0.8, 0.5),
        "baran": (0.7, 0.6),
        "pro": (0.9, 0.4),
    }.items():
        for dataset, f1 in zip(("a", "b"), values):
            rows.append(
                {
                    "comparison_method": method,
                    "scope": "dataset",
                    "dataset": dataset,
                    "precision": f1,
                    "recall": f1,
                    "f1": f1,
                }
            )
    result = transfer._metric_deltas(
        rows,
        transfer_label="qwen",
        baselines=("flash", "baran", "pro"),
    )
    assert result["win_tie_loss"]["flash"]["win"] == 1
    assert result["win_tie_loss"]["flash"]["tie"] == 1
    assert result["win_tie_loss"]["baran"]["win"] == 1
    assert result["win_tie_loss"]["baran"]["loss"] == 1
    assert result["win_tie_loss"]["pro"]["win"] == 1
    assert result["win_tie_loss"]["pro"]["tie"] == 1
    row = result["comparisons"]["pro"][0]
    assert row["qwen_f1"] == row["transfer_f1"]


def test_qwen_markdown_report_contains_three_way_comparison(
    qwen_plan: transfer.FrozenTransferPlan,
    tmp_path: Path,
) -> None:
    flash_rows = list(transfer._official_flash_rows(qwen_plan).values())
    baran_rows = transfer._official_baran_rows(qwen_plan)
    pro_rows = transfer._validated_pro_transfer_rows(qwen_plan)
    summaries = transfer._summary_rows(
        flash_rows,
        flash_rows,
        baran_rows,
        transfer_label="qwen",
        extra_methods=(("pro", pro_rows),),
    )
    deltas = transfer._metric_deltas(
        summaries,
        transfer_label="qwen",
        baselines=("flash", "baran", "pro"),
    )
    transfer._write_csv(tmp_path / "metrics" / "fixed_eight_metrics.csv", summaries)
    transfer._write_json(tmp_path / "metrics" / "metric_deltas.json", deltas)
    transfer._write_json(
        tmp_path / "metrics" / "diagnostics.json",
        {
            "rescued_cells_vs_baran": 0,
            "harmed_cells_vs_baran": 0,
            "net_gain_vs_baran": 0,
            "verifier_rejections": 0,
            "parse_failures": 0,
            "partial_parse_queries": 0,
            "api_failure_queries": 0,
            "abstention_queries": 0,
            "abstention_items": 0,
        },
    )
    transfer._write_json(
        tmp_path / "metrics" / "api_usage_and_cost.json",
        {
            "observed_tokens": {},
            "known_cost_cny": 0,
            "conservative_cost_upper_cny": 0,
        },
    )
    transfer._write_json(
        tmp_path / "metrics" / "model_identity_audit.json",
        {
            "physical_attempts": 1_994,
            "successful_terminal_queries": 1_994,
            "returned_model_counts": {transfer.QWEN_MODEL: 1_994},
        },
    )
    transfer._write_json(
        tmp_path / "dry_run.json",
        {
            "estimated_cost_cny": {
                "normal_round_all_input_cache_miss": 1.1175976
            }
        },
    )
    target = transfer._write_experiment_markdown(
        tmp_path,
        qwen_plan,
        {"validated_at_utc": "2026-08-31T12:00:00+00:00"},
    )
    report = target.read_text(encoding="utf-8")
    assert transfer.QWEN_MODEL in report
    assert "ΔF1 vs Pro" in report
    assert "Qwen、Pro 响应及 clean labels 均未参与" in report
    assert "Movies_1" in report
    assert "./final/integrity_validation.json" in report


def test_flash_checkpoint_offline_replay_is_cell_exact(
    frozen_plan: transfer.FrozenTransferPlan,
) -> None:
    result = transfer._flash_replay_parity(frozen_plan)
    assert result["ok"] is True
    assert result["checkpoint_rows_bound"] == 1_994
    assert result["cell_rows_replayed"] == 14_523
    assert result["cell_mismatches"] == 0
    assert (
        result["request_plan_sha256_before_oracle_replay"]
        == result["request_plan_sha256_after_oracle_replay"]
    )
