"""Execute one audited missing-query union without widening provider scope.

This module is intentionally separate from the full Router runner.  It recovers
only the action rows named by an immutable audit ledger, verifies all five
request-identity fields, and then submits that exact deduplicated set.  It never
runs a model preflight or discovers additional work at execution time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cli import load_env_file
from .group_context import (
    INFORMATION_POLICY,
    canonical_messages,
    compute_prompt_hash,
    compute_query_id,
)
from .group_llm import (
    DeepSeekGroupClient,
    GroupClientConfig,
    GroupLLMJob,
    run_group_llm_batch,
)


IDENTITY_FIELDS = (
    "query_id",
    "prompt_hash",
    "provider_request_hash",
    "model",
    "prompt_schema_version",
)


@dataclass(frozen=True)
class AuditedRetryPlan:
    jobs: tuple[GroupLLMJob, ...]
    request_rows: tuple[Mapping[str, Any], ...]
    logical_estimated_tokens: int
    retry_multiplier: int
    retry_adjusted_estimated_token_reference: int
    provider_token_cap: int | None
    authority_sha256: str
    base_authority_sha256: str
    llm_config_sha256: str
    request_identity_config_sha256: str
    candidate_file_sha256: Mapping[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "request_count": len(self.jobs),
            "logical_estimated_tokens": self.logical_estimated_tokens,
            "retry_multiplier": self.retry_multiplier,
            "retry_adjusted_estimated_token_reference": (
                self.retry_adjusted_estimated_token_reference
            ),
            "provider_token_cap": self.provider_token_cap,
            "uncapped_provider_usage": self.provider_token_cap is None,
            "request_identity_fields": list(IDENTITY_FIELDS),
            "authority_sha256": self.authority_sha256,
            "base_authority_sha256": self.base_authority_sha256,
            "llm_config_sha256": self.llm_config_sha256,
            "request_identity_config_sha256": self.request_identity_config_sha256,
            "candidate_file_sha256": dict(self.candidate_file_sha256),
            "preflight_requests": 0,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _iter_jsonl_strict(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON row at {path}:{line_number}")
            yield value


def _read_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl_strict(path))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    temporary.replace(path)


def _authority_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in IDENTITY_FIELDS)


def build_audited_retry_plan(
    *,
    authority_path: str | Path,
    base_authority_path: str | Path,
    candidate_dir: str | Path,
    llm_config_path: str | Path,
    expected_requests: int,
    expected_logical_tokens: int,
    retry_adjusted_estimated_token_reference: int,
    provider_token_cap: int | None,
    allow_uncapped_provider_usage: bool,
    expected_authority_sha256: str,
    expected_base_authority_sha256: str,
    expected_llm_config_sha256: str,
    expected_request_identity_config_sha256: str,
    expected_max_retries: int,
    expected_concurrency: int,
) -> AuditedRetryPlan:
    """Validate and recover the exact provider jobs named by an audit ledger."""

    authority_source = Path(authority_path).resolve()
    base_authority_source = Path(base_authority_path).resolve()
    candidates_root = Path(candidate_dir).resolve()
    config_source = Path(llm_config_path).resolve()
    authority_sha256 = _sha256(authority_source)
    if authority_sha256 != str(expected_authority_sha256):
        raise ValueError("authority digest differs from the authorized digest")
    authority = _read_jsonl_strict(authority_source)
    base_authority_sha256 = _sha256(base_authority_source)
    if base_authority_sha256 != str(expected_base_authority_sha256):
        raise ValueError("base authority digest differs from the authorized digest")
    base_authority = _read_jsonl_strict(base_authority_source)
    base_keys = set(base_authority[0]) if base_authority else set()
    if not base_keys or any(set(row) != base_keys for row in base_authority):
        raise ValueError("base authority rows do not share one stable schema")
    base_projection = [
        {key: row.get(key) for key in sorted(base_keys)}
        for row in sorted(base_authority, key=lambda value: str(value.get("query_id", "")))
    ]
    enriched_projection = [
        {key: row.get(key) for key in sorted(base_keys)}
        for row in sorted(authority, key=lambda value: str(value.get("query_id", "")))
    ]
    if enriched_projection != base_projection:
        raise ValueError("enriched authority differs from the authoritative base projection")
    if len(authority) != int(expected_requests):
        raise ValueError(
            f"authority request count drift: expected={expected_requests}, "
            f"observed={len(authority)}"
        )
    identities = [_authority_identity(row) for row in authority]
    if any(not all(identity) for identity in identities):
        raise ValueError("authority contains an incomplete five-field identity")
    if len(set(identities)) != len(identities):
        raise ValueError("authority contains duplicate five-field identities")
    query_ids = [str(row["query_id"]) for row in authority]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("authority contains duplicate query IDs")
    if any(
        str(row.get("cache_lookup_result"))
        != "terminal_failure_requires_retry"
        for row in authority
    ):
        raise ValueError("authority widened beyond terminal failures requiring retry")
    if any(int(row.get("dedup_request_count", 0)) != 1 for row in authority):
        raise ValueError("authority contains a non-deduplicated request")

    config_values = _read_json(config_source)
    config = GroupClientConfig.from_mapping(config_values)
    request_config = {
        "base_url": config.base_url.rstrip("/"),
        "model": config.model,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "extra_body": dict(config.extra_body),
    }
    request_identity_config_sha256 = hashlib.sha256(
        json.dumps(
            request_config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if request_identity_config_sha256 != str(
        expected_request_identity_config_sha256
    ):
        raise ValueError("provider request-identity configuration digest drift")
    config_sha256 = _sha256(config_source)
    if config_sha256 != str(expected_llm_config_sha256):
        raise ValueError(
            "bound LLM configuration digest differs from the authorized digest"
        )
    if int(config.max_retries) != int(expected_max_retries):
        raise ValueError(
            f"max_retries drift: expected={expected_max_retries}, "
            f"observed={config.max_retries}"
        )
    if int(config.concurrency) != int(expected_concurrency):
        raise ValueError(
            f"concurrency drift: expected={expected_concurrency}, "
            f"observed={config.concurrency}"
        )
    retry_multiplier = int(config.max_retries) + 1
    logical_tokens = sum(int(row.get("estimated_tokens", -1)) for row in authority)
    if logical_tokens != int(expected_logical_tokens):
        raise ValueError(
            f"logical token total drift: expected={expected_logical_tokens}, "
            f"observed={logical_tokens}"
        )
    retry_reference = retry_multiplier * logical_tokens
    if retry_reference != int(retry_adjusted_estimated_token_reference):
        raise ValueError(
            "retry-adjusted estimated-token reference drift: "
            f"expected={retry_adjusted_estimated_token_reference}, "
            f"observed={retry_reference}"
        )
    if bool(allow_uncapped_provider_usage) == (provider_token_cap is not None):
        raise ValueError(
            "choose exactly one of a provider token cap or uncapped provider usage"
        )
    if provider_token_cap is not None and int(provider_token_cap) <= 0:
        raise ValueError("provider token cap must be positive")
    models = {str(row["model"]) for row in authority}
    if models != {config.model}:
        raise ValueError(
            f"authority model differs from bound config: {sorted(models)!r}"
        )
    configured_schema = str(config_values.get("prompt_schema_version", ""))
    schemas = {str(row["prompt_schema_version"]) for row in authority}
    if not configured_schema or schemas != {configured_schema}:
        raise ValueError(
            "authority prompt schema differs from the bound LLM configuration"
        )

    authority_by_query = {str(row["query_id"]): row for row in authority}
    recovered: dict[str, tuple[dict[str, Any], Path]] = {}
    for candidate_path in sorted(candidates_root.glob("*.jsonl")):
        for action in _iter_jsonl_strict(candidate_path):
            query_id = str(action.get("query_id", ""))
            if query_id not in authority_by_query:
                continue
            if query_id in recovered:
                raise ValueError(f"candidate snapshot duplicates query ID {query_id}")
            recovered[query_id] = (action, candidate_path)
    if set(recovered) != set(authority_by_query):
        missing = sorted(set(authority_by_query) - set(recovered))
        raise ValueError(f"candidate snapshot is missing {len(missing)} audited queries")

    hash_client = DeepSeekGroupClient(config, api_key="identity-hash-only")
    jobs: list[GroupLLMJob] = []
    request_rows: list[Mapping[str, Any]] = []
    used_candidate_paths: set[Path] = set()
    for authority_row in sorted(authority, key=lambda row: str(row["query_id"])):
        query_id = str(authority_row["query_id"])
        action, candidate_path = recovered[query_id]
        used_candidate_paths.add(candidate_path)
        raw_messages = action.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError(f"candidate messages are not a list for {query_id}")
        messages = canonical_messages(raw_messages)
        max_tokens = int(action.get("completion_token_ceiling", 0))
        schema = str(action.get("prompt_schema_version", ""))
        information_policy = str(
            action.get("prompt_information_policy", INFORMATION_POLICY)
        )
        recomputed_query_id = compute_query_id(
            str(action.get("suite", "")),
            str(action.get("dataset", "")),
            str(action.get("group_view", "")),
            tuple(str(value) for value in action.get("cell_ids", ())),
            arm=str(action.get("arm", "structured")),
            prompt_schema_version=schema,
            information_policy=information_policy,
        )
        if recomputed_query_id != query_id:
            raise ValueError(f"candidate query ID is invalid for {query_id}")
        recomputed_prompt_hash = compute_prompt_hash(
            messages,
            max_tokens,
            prompt_schema_version=schema,
            information_policy=information_policy,
        )
        if recomputed_prompt_hash != str(action.get("prompt_hash", "")):
            raise ValueError(f"candidate prompt hash is invalid for {query_id}")
        metadata = {
            "phase": "online_selected_union",
            "estimated_total_tokens": int(action.get("estimated_total_tokens", -1)),
            "suite": str(action.get("suite", "")),
            "dataset": str(action.get("dataset", "")),
            "group_size": int(action.get("group_size", 0)),
            "group_view": str(action.get("group_view", "")),
            "prompt_schema_version": schema,
            "model_requested": config.model,
            "strict_missing_union_retry": True,
        }
        job = GroupLLMJob(
            query_id=query_id,
            messages=messages,
            prompt_hash=recomputed_prompt_hash,
            expected_cell_ids=tuple(str(value) for value in action.get("cell_ids", ())),
            max_tokens=max_tokens,
            metadata=metadata,
        )
        provider_hash = hash_client.provider_request_hash(job)
        candidate_identity = (
            query_id,
            job.prompt_hash,
            provider_hash,
            config.model,
            schema,
        )
        if candidate_identity != _authority_identity(authority_row):
            raise ValueError(f"five-field request identity drift for {query_id}")
        if metadata["estimated_total_tokens"] != int(
            authority_row.get("estimated_tokens", -1)
        ):
            raise ValueError(f"estimated token drift for {query_id}")
        if (
            metadata["suite"] != str(authority_row.get("suite", ""))
            or metadata["dataset"] != str(authority_row.get("dataset", ""))
            or metadata["group_size"] != int(authority_row.get("group_size", 0))
            or metadata["group_view"] != str(authority_row.get("group_view", ""))
        ):
            raise ValueError(f"candidate audit metadata drift for {query_id}")
        jobs.append(job)
        request_rows.append(
            {
                **{field: value for field, value in zip(IDENTITY_FIELDS, candidate_identity)},
                "suite": metadata["suite"],
                "dataset": metadata["dataset"],
                "group_size": metadata["group_size"],
                "group_view": metadata["group_view"],
                "estimated_total_tokens": metadata["estimated_total_tokens"],
                "completion_token_ceiling": max_tokens,
                "cell_ids": list(job.expected_cell_ids),
                "candidate_file": candidate_path.name,
            }
        )

    if len(jobs) != int(expected_requests):
        raise AssertionError("validated job count changed unexpectedly")
    if sum(int(row["estimated_total_tokens"]) for row in request_rows) != logical_tokens:
        raise AssertionError("validated request token total changed unexpectedly")
    candidate_hashes = {
        path.name: _sha256(path) for path in sorted(used_candidate_paths)
    }
    return AuditedRetryPlan(
        jobs=tuple(jobs),
        request_rows=tuple(request_rows),
        logical_estimated_tokens=logical_tokens,
        retry_multiplier=retry_multiplier,
        retry_adjusted_estimated_token_reference=retry_reference,
        provider_token_cap=(
            None if provider_token_cap is None else int(provider_token_cap)
        ),
        authority_sha256=authority_sha256,
        base_authority_sha256=base_authority_sha256,
        llm_config_sha256=config_sha256,
        request_identity_config_sha256=request_identity_config_sha256,
        candidate_file_sha256=candidate_hashes,
    )


def execute_audited_retry(
    *,
    plan: AuditedRetryPlan,
    llm_config_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Submit the frozen plan once, writing a new append-only retry run."""

    output = Path(run_dir).resolve()
    if output.exists():
        raise FileExistsError(f"retry run already exists; refusing overwrite: {output}")
    config_source = Path(llm_config_path).resolve()
    config_values = _read_json(config_source)
    config = GroupClientConfig.from_mapping(config_values)
    if config.max_retries + 1 != plan.retry_multiplier:
        raise ValueError("LLM retry configuration changed after plan validation")
    if _sha256(config_source) != plan.llm_config_sha256:
        raise ValueError("LLM configuration changed after plan validation")
    key_name = str(config_values.get("api_key_env", "DEEPSEEK_API_KEY"))
    api_key = os.environ.get(key_name, "")
    if not api_key:
        raise RuntimeError(f"required environment variable is not set: {key_name}")

    output.mkdir(parents=True, exist_ok=False)
    llm_dir = output / "llm"
    llm_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_source, output / "bound_llm_config.json")
    _write_jsonl(output / "authorized_requests.jsonl", plan.request_rows)
    created_at = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "run_id": output.name,
        "run_kind": "strict_nonfreezing_missing_query_union_retry",
        "status": "running",
        "created_at_utc": created_at,
        "api_authorized": True,
        "api_called": False,
        "preflight_called": False,
        **plan.summary(),
    }
    _write_json(output / "run_manifest.json", manifest)
    client = DeepSeekGroupClient(config, api_key=api_key)
    try:
        manifest["api_called"] = True
        manifest["execution_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "run_manifest.json", manifest)
        canary_results = run_group_llm_batch(
            client,
            plan.jobs[:1],
            llm_dir,
            concurrency=1,
            retry_failed=True,
        )
        if len(canary_results) != 1:
            raise ValueError("audited canary did not produce exactly one result")
        canary = canary_results[0]
        if str(canary.get("exception_class", "")) in {
            "PermanentGroupLLMError",
            "ProviderModelIdentityError",
        }:
            raise RuntimeError(
                "audited canary found a permanent provider/configuration error; "
                "remaining authorized requests were not submitted"
            )
        remaining_results = run_group_llm_batch(
            client,
            plan.jobs[1:],
            llm_dir,
            concurrency=config.concurrency,
            retry_failed=True,
        )
        results = [*canary_results, *remaining_results]
        _write_jsonl(llm_dir / "selected_execution.jsonl", results)
        if len(results) != len(plan.jobs):
            raise ValueError(
                f"provider result count drift: expected={len(plan.jobs)}, "
                f"observed={len(results)}"
            )
        planned = {
            tuple(str(row[field]) for field in IDENTITY_FIELDS)
            for row in plan.request_rows
        }
        observed = set()
        for row in results:
            metadata = row.get("metadata")
            schema = (
                str(metadata.get("prompt_schema_version", ""))
                if isinstance(metadata, Mapping)
                else ""
            )
            observed.add(
                (
                    str(row.get("query_id", "")),
                    str(row.get("prompt_hash", "")),
                    str(row.get("provider_request_hash", "")),
                    str(row.get("model_requested", "")),
                    schema,
                )
            )
        if observed != planned or len(observed) != len(results):
            raise ValueError("provider result identities differ from the authorized plan")
        attempts = [int(row.get("attempts", 0) or 0) for row in results]
        if any(value < 1 or value > plan.retry_multiplier for value in attempts):
            raise ValueError("provider result contains an out-of-policy attempt count")
        observed_tokens = sum(
            int(row.get("observed_total_tokens", 0) or 0) for row in results
        )
        unknown_attempts = sum(
            int(row.get("unknown_usage_attempts", 0) or 0) for row in results
        )
        estimated_by_query = {
            str(row["query_id"]): int(row["estimated_total_tokens"])
            for row in plan.request_rows
        }
        conservative_debit = sum(
            int(row.get("observed_total_tokens", 0) or 0)
            + estimated_by_query[str(row["query_id"])]
            * int(row.get("unknown_usage_attempts", 0) or 0)
            for row in results
        )
        estimated_attempt_debit = sum(
            estimated_by_query[str(row["query_id"])]
            * int(row.get("attempts", 0) or 0)
            for row in results
        )
        if (
            plan.provider_token_cap is not None
            and estimated_attempt_debit > plan.provider_token_cap
        ):
            raise ValueError("retry attempt ledger exceeds the authorized hard cap")
        if any(
            bool(row.get("cache_hit")) or bool(row.get("checkpoint_hit"))
            for row in results
        ):
            raise ValueError("fresh retry execution unexpectedly used a cache/checkpoint hit")
        summary = {
            **plan.summary(),
            "run_id": output.name,
            "status": "complete",
            "api_called": True,
            "preflight_called": False,
            "result_count": len(results),
            "success_count": sum(row.get("status") == "success" for row in results),
            "failure_count": sum(row.get("status") != "success" for row in results),
            "attempt_count": sum(attempts),
            "maximum_attempts_per_request": max(attempts, default=0),
            "usage_observed_total_tokens": observed_tokens,
            "unknown_usage_attempts": unknown_attempts,
            "conservative_safety_debit": conservative_debit,
            "estimated_attempt_debit": estimated_attempt_debit,
            "conservative_safety_cap_respected": (
                None
                if plan.provider_token_cap is None
                else conservative_debit <= plan.provider_token_cap
            ),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(output / "execution_summary.json", summary)
        manifest.update(summary)
        _write_json(output / "run_manifest.json", manifest)
        return summary
    except BaseException as error:
        manifest.update(
            {
                "status": "failed",
                "failure_class": type(error).__name__,
                "failure_message": str(error)[:500],
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_json(output / "run_manifest.json", manifest)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute an exact audited missing-query union"
    )
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--base-authority", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--llm-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-requests", type=int, required=True)
    parser.add_argument("--expected-logical-tokens", type=int, required=True)
    parser.add_argument(
        "--retry-adjusted-estimated-token-reference", type=int, required=True
    )
    usage = parser.add_mutually_exclusive_group(required=True)
    usage.add_argument("--provider-token-cap", type=int)
    usage.add_argument("--no-token-cap", action="store_true")
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-base-authority-sha256", required=True)
    parser.add_argument("--expected-llm-config-sha256", required=True)
    parser.add_argument("--expected-request-identity-config-sha256", required=True)
    parser.add_argument("--expected-max-retries", type=int, required=True)
    parser.add_argument("--expected-concurrency", type=int, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="submit the audited jobs; omission performs validation only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_audited_retry_plan(
        authority_path=args.authority,
        base_authority_path=args.base_authority,
        candidate_dir=args.candidate_dir,
        llm_config_path=args.llm_config,
        expected_requests=args.expected_requests,
        expected_logical_tokens=args.expected_logical_tokens,
        retry_adjusted_estimated_token_reference=(
            args.retry_adjusted_estimated_token_reference
        ),
        provider_token_cap=args.provider_token_cap,
        allow_uncapped_provider_usage=bool(args.no_token_cap),
        expected_authority_sha256=args.expected_authority_sha256,
        expected_base_authority_sha256=args.expected_base_authority_sha256,
        expected_llm_config_sha256=args.expected_llm_config_sha256,
        expected_request_identity_config_sha256=(
            args.expected_request_identity_config_sha256
        ),
        expected_max_retries=args.expected_max_retries,
        expected_concurrency=args.expected_concurrency,
    )
    if not args.execute:
        print(json.dumps({"status": "validated", **plan.summary()}, sort_keys=True))
        return 0
    if args.env_file is not None:
        load_env_file(args.env_file)
    summary = execute_audited_retry(
        plan=plan,
        llm_config_path=args.llm_config,
        run_dir=args.run_dir,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AuditedRetryPlan",
    "IDENTITY_FIELDS",
    "build_audited_retry_plan",
    "execute_audited_retry",
    "main",
]
