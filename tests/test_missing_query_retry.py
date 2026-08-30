from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from budgeted_group_repair_no_baran.group_context import (
    INFORMATION_POLICY,
    PROMPT_SCHEMA_VERSION,
    compute_prompt_hash,
    compute_query_id,
)
from budgeted_group_repair_no_baran.group_llm import (
    DeepSeekGroupClient,
    GroupClientConfig,
    GroupLLMJob,
)
from budgeted_group_repair_no_baran.missing_query_retry import (
    build_audited_retry_plan,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    config_path = tmp_path / "bound_llm_config.json"
    config_values = {
        "base_url": "https://example.invalid",
        "model": "deepseek-v4-flash",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_retries": 5,
        "concurrency": 4,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
    }
    _write_json(config_path, config_values)
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    messages = [
        {"role": "system", "content": "repair"},
        {"role": "user", "content": "one cell"},
    ]
    cell_ids = ["source:movies_1:1:2"]
    query_id = compute_query_id(
        "source",
        "movies_1",
        "singleton",
        cell_ids,
        arm="singleton",
    )
    prompt_hash = compute_prompt_hash(messages, 192)
    action = {
        "query_id": query_id,
        "prompt_hash": prompt_hash,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "prompt_information_policy": INFORMATION_POLICY,
        "messages": messages,
        "cell_ids": cell_ids,
        "completion_token_ceiling": 192,
        "estimated_total_tokens": 250,
        "suite": "source",
        "dataset": "movies_1",
        "group_size": 1,
        "group_view": "singleton",
        "arm": "singleton",
    }
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    _write_json(candidate_dir / "source__movies_1.jsonl", action)
    job = GroupLLMJob(
        query_id=query_id,
        messages=messages,
        prompt_hash=prompt_hash,
        expected_cell_ids=tuple(cell_ids),
        max_tokens=192,
    )
    provider_hash = DeepSeekGroupClient(
        GroupClientConfig.from_mapping(config_values), api_key="unused"
    ).provider_request_hash(job)
    authority_path = tmp_path / "missing.jsonl"
    _write_json(
        authority_path,
        {
            "query_id": query_id,
            "prompt_hash": prompt_hash,
            "provider_request_hash": provider_hash,
            "model": "deepseek-v4-flash",
            "prompt_schema_version": PROMPT_SCHEMA_VERSION,
            "cache_lookup_result": "terminal_failure_requires_retry",
            "dedup_request_count": 1,
            "estimated_tokens": 250,
            "suite": "source",
            "dataset": "movies_1",
            "group_size": 1,
            "group_view": "singleton",
        },
    )
    return authority_path, candidate_dir, config_path, config_sha


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request_config_sha256(config_path: Path) -> str:
    values = json.loads(config_path.read_text(encoding="utf-8"))
    identity = {
        "base_url": str(values["base_url"]).rstrip("/"),
        "model": values["model"],
        "temperature": values.get("temperature", 0.0),
        "top_p": values.get("top_p", 1.0),
        "extra_body": values.get("extra_body", {}),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_build_audited_retry_plan_locks_scope_and_budget(tmp_path: Path) -> None:
    authority, candidates, config, config_sha = _fixture(tmp_path)
    plan = build_audited_retry_plan(
        authority_path=authority,
        base_authority_path=authority,
        candidate_dir=candidates,
        llm_config_path=config,
        expected_requests=1,
        expected_logical_tokens=250,
        retry_adjusted_estimated_token_reference=1_500,
        provider_token_cap=None,
        allow_uncapped_provider_usage=True,
        expected_authority_sha256=_sha256(authority),
        expected_base_authority_sha256=_sha256(authority),
        expected_llm_config_sha256=config_sha,
        expected_request_identity_config_sha256=_request_config_sha256(config),
        expected_max_retries=5,
        expected_concurrency=4,
    )
    assert len(plan.jobs) == 1
    assert plan.jobs[0].metadata["phase"] == "online_selected_union"
    assert plan.retry_adjusted_estimated_token_reference == 1_500
    assert plan.provider_token_cap is None


def test_build_audited_retry_plan_rejects_cap_or_candidate_duplication(
    tmp_path: Path,
) -> None:
    authority, candidates, config, config_sha = _fixture(tmp_path)
    with pytest.raises(ValueError, match="estimated-token reference drift"):
        build_audited_retry_plan(
            authority_path=authority,
            base_authority_path=authority,
            candidate_dir=candidates,
            llm_config_path=config,
            expected_requests=1,
            expected_logical_tokens=250,
            retry_adjusted_estimated_token_reference=1_499,
            provider_token_cap=None,
            allow_uncapped_provider_usage=True,
            expected_authority_sha256=_sha256(authority),
            expected_base_authority_sha256=_sha256(authority),
            expected_llm_config_sha256=config_sha,
            expected_request_identity_config_sha256=_request_config_sha256(config),
            expected_max_retries=5,
            expected_concurrency=4,
        )
    duplicate = next(candidates.glob("*.jsonl")).read_text(encoding="utf-8")
    (candidates / "duplicate.jsonl").write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates query ID"):
        build_audited_retry_plan(
            authority_path=authority,
            base_authority_path=authority,
            candidate_dir=candidates,
            llm_config_path=config,
            expected_requests=1,
            expected_logical_tokens=250,
            retry_adjusted_estimated_token_reference=1_500,
            provider_token_cap=1_500,
            allow_uncapped_provider_usage=False,
            expected_authority_sha256=_sha256(authority),
            expected_base_authority_sha256=_sha256(authority),
            expected_llm_config_sha256=config_sha,
            expected_request_identity_config_sha256=_request_config_sha256(config),
            expected_max_retries=5,
            expected_concurrency=4,
        )
