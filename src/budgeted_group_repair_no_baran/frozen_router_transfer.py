"""Frozen Flash-router cross-repairer transfer experiment.

This entry point is deliberately isolated from :mod:`router_v3`.  It reads the
official TabICLv2 k=4, 20% selection artifacts verbatim, binds their request
identities before any oracle access, and permits only the explicitly requested
DeepSeek V4 Pro repairer.  It never trains, predicts, ranks, or selects Router
actions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .cli import load_env_file
from .data import load_dataset
from .group_context import (
    compute_prompt_hash,
    compute_query_id,
    messages_as_dicts,
)
from .group_generator import GroupQueryAction
from .group_llm import (
    CompletionTruncatedError,
    DeepSeekGroupClient,
    GroupClientConfig,
    GroupLLMJob,
    ProviderModelIdentityError,
    parse_group_response,
    run_group_llm_batch,
)
from .metrics import normalize_for_match, summarize_records, verify_records
from .public_fd import fds_for_dataset, load_public_fds
from .router_v3 import _action_from_dict, _read_csv as _router_read_csv
from .verifier import (
    GroupRepairVerifier,
    RankedRepairCandidate,
    VerifierConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs"
FORMAL_FREEZE_RUN_ID = (
    "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_"
    "official_results_freeze_v1"
)
FROZEN_RUN_ID = (
    "no_baran_router_v3_deepseek_v4_20260830_mgreedyfix_"
    "missing35_tabiclv2_final"
)
DEFAULT_TRANSFER_RUN_ID = (
    "no_baran_frozen_tabiclv2_flash_router_deepseek_v4_pro_20260831"
)
QWEN_DIAGNOSTIC_RUN_ID = (
    "no_baran_frozen_tabiclv2_flash_router_qwen3_7_flash_"
    "complete_v2_20260831"
)
QWEN_TRANSFER_RUN_ID = (
    "no_baran_frozen_tabiclv2_flash_router_qwen3_7_flash_"
    "complete_v3_20260831"
)
QWEN_VERSION_LABEL = "qwen37_flash_complete_output_transfer_v3"
EXPERIMENT_NAME = (
    "BGR + DeepSeek-V4-Pro (frozen Flash-trained TabICLv2 router)"
)
SHORT_EXPERIMENT_NAME = "Frozen Flash-router cross-repairer transfer"
TRANSFER_METHOD = "frozen_flash_router_cross_repairer_deepseek_v4_pro"
FLASH_METHOD = "frozen_flash_router_deepseek_v4_flash"
BARAN_METHOD = "baran"
REPAIR_MODEL = "deepseek-v4-pro"
FLASH_MODEL = "deepseek-v4-flash"
MODEL_NAMESPACE = "deepseek-v4-pro"
QWEN_MODEL = "qwen3.7-flash-2026-07-15"
QWEN_MODEL_NAMESPACE = QWEN_MODEL
QWEN_EXPERIMENT_NAME = (
    "BGR + Qwen3.7-Flash (frozen Flash-trained TabICLv2 router)"
)
QWEN_SHORT_EXPERIMENT_NAME = (
    "Frozen Flash-router cross-repairer transfer (Qwen3.7-Flash)"
)
QWEN_PAPER_DESCRIPTION = (
    "Frozen Flash-router cross-repairer transfer under the same 20% "
    "Flash-reference planning budget, with realized Qwen token usage "
    "reported separately."
)
QWEN_TRANSFER_METHOD = "frozen_flash_router_cross_repairer_qwen3_7_flash"
PROMPT_SCHEMA_VERSION = "bgr-no-baran-v1"
EXPECTED_REQUESTS = 1_994
EXPECTED_SINGLETONS = 1_559
EXPECTED_GROUPS_OF_FOUR = 435
EXPECTED_ERROR_CELLS = 14_523
EXPECTED_INPUT_TOKENS = 2_942_996
EXPECTED_OUTPUT_TOKENS = 661_248
EXPECTED_TOTAL_TOKENS = 3_604_244
EXPECTED_RETRY_MULTIPLIER = 6
EXPECTED_WORST_TOKENS = 21_625_464
EXPECTED_MAX_SERIALIZED_MESSAGE_BYTES = 22_980
QWEN_SINGLETON_MAX_COMPLETION_TOKENS = 4_096
QWEN_GROUP4_MAX_COMPLETION_TOKENS = 16_384
QWEN_DOCUMENTED_COMPLETION_TOKEN_TOLERANCE = 10
QWEN_PROVIDER_OUTPUT_TOKEN_CAP = (
    EXPECTED_SINGLETONS * QWEN_SINGLETON_MAX_COMPLETION_TOKENS
    + EXPECTED_GROUPS_OF_FOUR * QWEN_GROUP4_MAX_COMPLETION_TOKENS
)
QWEN_PROVIDER_BILLED_OUTPUT_CAP_ESTIMATE = (
    QWEN_PROVIDER_OUTPUT_TOKEN_CAP
    + EXPECTED_REQUESTS * QWEN_DOCUMENTED_COMPLETION_TOKEN_TOLERANCE
)
QWEN_PROVIDER_NORMAL_TOKEN_CAP_ESTIMATE = (
    EXPECTED_INPUT_TOKENS + QWEN_PROVIDER_BILLED_OUTPUT_CAP_ESTIMATE
)
QWEN_PROVIDER_WORST_TOKEN_CAP_ESTIMATE = (
    QWEN_PROVIDER_NORMAL_TOKEN_CAP_ESTIMATE * EXPECTED_RETRY_MULTIPLIER
)
EXPECTED_PREFLIGHT_QUERIES = 15
EXPECTED_PREFLIGHT_QUERY_ID = (
    "bgrq_15c97ac909d2c3d19d0d2210da1d294c48c2c38bea0ab9123acd5ddf27c1923e"
)

EXPECTED_FORMAL_MANIFEST_SHA256 = (
    "76071f92597e08cda2ecb47feaedaf46c65ba56e856b223af70367d948f3c806"
)
EXPECTED_FROZEN_MANIFEST_SHA256 = (
    "25a686db2a53edfb5448dc926ace501fa2108df68b4877d16893a4bdc71b8f7b"
)
EXPECTED_SELECTION_MANIFEST_SHA256 = (
    "06fe449dec735ec41f290e16cefb0f88e0f810f4aab60a4de19d9ce13858674a"
)
EXPECTED_QUERY_SET_SHA256 = (
    "c7850fa7a20d2a735d73f47428e2067c8dd920d2b18737973f95ebaac096eefb"
)
EXPECTED_ORDERED_QUERY_SHA256 = (
    "fa43b0c258c9fa602e59c9b648486ded2b3dd375546f6fa3d71b107fd5999239"
)
EXPECTED_MEMBERSHIP_SHA256 = (
    "40db76154e88ce0c3785562379db6987cb38e491443016c9ae5a2c2365661b72"
)
EXPECTED_PROMPT_IDENTITY_SHA256 = (
    "189868f6cee5bcfd18910f7a887587e86359ca507a1bf0e338f8fd7f5cf96113"
)
EXPECTED_FROZEN_MESSAGE_SHA256 = (
    "35877dff4d4e8b864b8a94fd87915b35d5d6f6b5b0c4e86cdcd15c27c43cdbe1"
)
EXPECTED_QWEN_PROVIDER_REQUEST_HASH = (
    "7796e95408f075b1720b73eb2ff57cb411d0ca62e398545dbae795fc79a5d993"
)
EXPECTED_QWEN_REQUEST_PLAN_SHA256 = (
    "27e1ee9a4530be4e22605fd05bd1400d839eb791ff9bf9b34fec79184d73b40b"
)

FIXED_DATASETS = (
    ("Hospital", "source", "hospital", "source__hospital", 69),
    ("Flights", "source", "flights", "source__flights", 680),
    ("Beers", "source", "beers", "source__beers", 727),
    ("Rayyan", "source", "rayyan", "source__rayyan", 83),
    ("Company", "tableeg", "company", "tableeg__company", 41),
    ("Marketing", "tableeg", "marketing", "tableeg__marketing", 72),
    (
        "Restaurant",
        "tableeg",
        "restaurant_20",
        "tableeg__restaurant_20",
        42,
    ),
    ("Soccer", "tableeg", "soccer", "tableeg__soccer", 280),
)

EXPECTED_SELECTION_SHA256 = {
    "source__hospital": "a726ad117c0099b70333afcafa0702299733d7a0127f07934388601b3f774392",
    "source__flights": "396f70cf619679aac1cdc715df941a6bbebf8688928548b7b6709d75ff6e8288",
    "source__beers": "1b5dfdd152d31da9fe87cc59700c0b9652eef8527134287636cf475aa93281d1",
    "source__rayyan": "b4f6c5f55905aa2fb06efeea79a20ee279de1a68b631c336cafc6b1bbae86486",
    "tableeg__company": "cc9902f55bd515f3ee89754f80ce78dc203be523c33a372b0a29b186a79f6d8f",
    "tableeg__marketing": "105e6b88350cd7e68d60f2f7d8681680951a5f744e2526ff5f78a366546fd34d",
    "tableeg__restaurant_20": "f681dd1fcc9fa8f0206bf73e0c5062d7620657c730ae45f8d7ff6ad7b8be9fa4",
    "tableeg__soccer": "961a6cd484bc680c2fdb2979a440fbcd4d43a59479187e5b3a5f340bb062e30f",
}

PRICE_SNAPSHOT = {
    "source": "https://api-docs.deepseek.com/quick_start/pricing/",
    "checked_at_utc": "2026-08-31T00:00:00+00:00",
    "currency": "USD",
    "unit_tokens": 1_000_000,
    "off_peak": {
        "input_cache_hit": 0.022,
        "input_cache_miss": 0.66,
        "output": 1.98,
    },
    "peak": {
        "input_cache_hit": 0.044,
        "input_cache_miss": 1.32,
        "output": 3.96,
    },
    "peak_policy": "Monday-Friday, 01:00-04:00 and 06:00-10:00 UTC",
}

QWEN_PRICE_SNAPSHOT = {
    "source": "https://help.aliyun.com/zh/model-studio/qwen3-7-flash",
    "checked_at_utc": "2026-08-31T00:00:00+00:00",
    "currency": "CNY",
    "unit_tokens": 1_000_000,
    "region": "cn-beijing",
    "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "endpoint_region_binding": "legacy DashScope endpoint for cn-beijing",
    "pricing_strategy": "per_request_input_token_tier",
    "tiers": (
        {
            "name": "input_le_32k",
            "maximum_input_tokens": 32_768,
            "input_cache_hit": 0.04,
            "input_cache_miss": 0.2,
            "output": 0.8,
        },
        {
            "name": "input_le_256k",
            "maximum_input_tokens": 262_144,
            "input_cache_hit": 0.12,
            "input_cache_miss": 0.6,
            "output": 2.4,
        },
        {
            "name": "input_le_1m",
            "maximum_input_tokens": 1_000_000,
            "input_cache_hit": 0.24,
            "input_cache_miss": 1.2,
            "output": 4.8,
        },
    ),
}

PRO_TRANSFER_RUN_ID = DEFAULT_TRANSFER_RUN_ID
EXPECTED_PRO_MANIFEST_SHA256 = (
    "14dcde9b17e4a10f5d0531727dd05d735dde94c4112040fc77c1a078eeb01c1a"
)
EXPECTED_PRO_LEDGER_SHA256 = (
    "6095c812f529596854c63df552ff455b7c820b4d7b809375732fd0c06acdd1d7"
)
EXPECTED_PRO_METRICS_SHA256 = (
    "44cafc5dc1ff69dba1f7dfafa959e1e4e39cfb17a04a8596443b26f1cefacfce"
)


@dataclass(frozen=True)
class TransferProfile:
    profile_id: str
    default_run_id: str
    experiment_name: str
    short_experiment_name: str
    transfer_method: str
    comparison_method: str
    repair_model: str
    model_namespace: str
    provider: str
    base_url: str
    api_key_env: str
    default_env_filename: str
    extra_body: Mapping[str, Any]
    price_snapshot: Mapping[str, Any]
    comparison_baselines: tuple[str, ...]


DEEPSEEK_PRO_PROFILE = TransferProfile(
    profile_id=REPAIR_MODEL,
    default_run_id=DEFAULT_TRANSFER_RUN_ID,
    experiment_name=EXPERIMENT_NAME,
    short_experiment_name=SHORT_EXPERIMENT_NAME,
    transfer_method=TRANSFER_METHOD,
    comparison_method="pro",
    repair_model=REPAIR_MODEL,
    model_namespace=MODEL_NAMESPACE,
    provider="deepseek",
    base_url="https://api.deepseek.com",
    api_key_env="DEEPSEEK_API_KEY",
    default_env_filename=".deepseek_env",
    extra_body={"thinking": {"type": "disabled"}},
    price_snapshot=PRICE_SNAPSHOT,
    comparison_baselines=("flash", "baran"),
)

QWEN37_FLASH_PROFILE = TransferProfile(
    profile_id=QWEN_MODEL,
    default_run_id=QWEN_TRANSFER_RUN_ID,
    experiment_name=QWEN_EXPERIMENT_NAME,
    short_experiment_name=QWEN_SHORT_EXPERIMENT_NAME,
    transfer_method=QWEN_TRANSFER_METHOD,
    comparison_method="qwen",
    repair_model=QWEN_MODEL,
    model_namespace=QWEN_MODEL_NAMESPACE,
    provider="alibaba-cloud-model-studio",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key_env="QWEN_API_KEY",
    default_env_filename=".qwen_env",
    extra_body={"enable_thinking": False},
    price_snapshot=QWEN_PRICE_SNAPSHOT,
    comparison_baselines=("flash", "baran", "pro"),
)

TRANSFER_PROFILES = {
    DEEPSEEK_PRO_PROFILE.profile_id: DEEPSEEK_PRO_PROFILE,
    QWEN37_FLASH_PROFILE.profile_id: QWEN37_FLASH_PROFILE,
}
DEFAULT_PROFILE_ID = DEEPSEEK_PRO_PROFILE.profile_id


def _profile(value: str | TransferProfile | None = None) -> TransferProfile:
    if isinstance(value, TransferProfile):
        if TRANSFER_PROFILES.get(value.profile_id) != value:
            raise ValueError("unregistered frozen-transfer profile")
        return value
    profile_id = DEFAULT_PROFILE_ID if value is None else str(value)
    try:
        return TRANSFER_PROFILES[profile_id]
    except KeyError as error:
        raise ValueError(f"unsupported frozen-transfer profile: {profile_id!r}") from error

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected JSON object")
            yield value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
                + "\n"
            )


def _append_jsonl(
    path: Path,
    row: Mapping[str, Any],
    lock: threading.Lock,
) -> None:
    encoded = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _safe_run_id(value: str) -> str:
    if not _RUN_ID.fullmatch(value):
        raise ValueError("invalid run ID")
    return value


def _frozen_run(project_root: Path) -> Path:
    return project_root / "runs" / FROZEN_RUN_ID


def _formal_freeze(project_root: Path) -> Path:
    return project_root / "runs" / FORMAL_FREEZE_RUN_ID / "run_manifest.json"


def _selection_path(source_run: Path, stem: str) -> Path:
    return (
        source_run
        / "selections"
        / "tabiclv2"
        / "size_conditioned"
        / "variant_4"
        / "20pct"
        / f"{stem}.json"
    )


def _candidate_path(source_run: Path, stem: str) -> Path:
    return source_run / "groups" / "candidates" / f"{stem}.jsonl"


def _gate_path(source_run: Path, stem: str) -> Path:
    return source_run / "gates" / "tabiclv2" / "variant_4" / f"{stem}.csv"


def _baran_path(source_run: Path, stem: str) -> Path:
    return source_run / "baran" / f"{stem}.jsonl"


def _validate_llm_config(
    values: Mapping[str, Any],
    profile: TransferProfile,
) -> dict[str, Any]:
    profile = _profile(profile)
    expected = {
        "base_url": "https://api.deepseek.com",
        "model": FLASH_MODEL,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_retries": 5,
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
    }
    for key, expected_value in expected.items():
        if values.get(key) != expected_value:
            raise ValueError(
                f"frozen Flash LLM configuration drift for {key}: "
                f"expected={expected_value!r}, observed={values.get(key)!r}"
            )
    if values.get("extra_body") != {"thinking": {"type": "disabled"}}:
        raise ValueError("frozen LLM thinking configuration drift")
    ceiling = values.get("completion_token_ceiling")
    if not isinstance(ceiling, Mapping) or dict(ceiling) != {
        "singleton": 192,
        "group_base": 64,
        "per_cell": 192,
        "formula": "size == 1 ? 192 : 64 + 192 * size",
    }:
        raise ValueError("frozen completion-token ceiling drift")
    derived = dict(values)
    derived["base_url"] = profile.base_url
    derived["model"] = profile.repair_model
    derived["api_key_env"] = profile.api_key_env
    derived["extra_body"] = dict(profile.extra_body)
    if profile.profile_id == QWEN37_FLASH_PROFILE.profile_id:
        derived["completion_token_parameter"] = "max_completion_tokens"
        derived["stream"] = False
        derived["strict_complete_response"] = True
        derived["provider_max_completion_tokens"] = {
            "singleton": QWEN_SINGLETON_MAX_COMPLETION_TOKENS,
            "group_size_4": QWEN_GROUP4_MAX_COMPLETION_TOKENS,
        }
    else:
        derived["completion_token_parameter"] = "max_tokens"
        derived["strict_complete_response"] = False
    derived["derived_from_frozen_flash_config"] = True
    derived["repairer_only_override"] = {
        "model": profile.repair_model,
        "provider_completion_parameter": derived["completion_token_parameter"],
    }
    return derived


def _provider_max_completion_tokens(
    profile: TransferProfile,
    group_size: int,
    planning_ceiling: int,
) -> int:
    if profile.profile_id != QWEN37_FLASH_PROFILE.profile_id:
        return int(planning_ceiling)
    if int(group_size) == 1:
        return QWEN_SINGLETON_MAX_COMPLETION_TOKENS
    if int(group_size) == 4:
        return QWEN_GROUP4_MAX_COMPLETION_TOKENS
    raise ValueError(f"unsupported Qwen group size: {group_size}")


def _load_selected_actions(
    candidate_path: Path,
    selected_ids: Sequence[str],
) -> dict[str, GroupQueryAction]:
    requested = set(selected_ids)
    actions: dict[str, GroupQueryAction] = {}
    for raw in _iter_jsonl(candidate_path):
        query_id = str(raw.get("query_id", ""))
        if query_id not in requested:
            continue
        if query_id in actions:
            raise ValueError(f"candidate file duplicates selected query {query_id}")
        action = _action_from_dict(raw)
        computed_query = compute_query_id(
            action.suite,
            action.dataset,
            action.group_view,
            action.cell_ids,
            arm=action.arm,
            prompt_schema_version=action.prompt_schema_version,
            information_policy=action.prompt_information_policy,
        )
        if computed_query != action.query_id:
            raise ValueError(f"selected action query identity drift: {action.query_id}")
        computed_prompt = compute_prompt_hash(
            action.messages,
            action.completion_token_ceiling,
            prompt_schema_version=action.prompt_schema_version,
            information_policy=action.prompt_information_policy,
        )
        if computed_prompt != action.prompt_hash:
            raise ValueError(f"selected action prompt drift: {action.query_id}")
        if action.group_size not in {1, 4}:
            raise ValueError(f"selected action has disallowed group size: {action.query_id}")
        actions[query_id] = action
    if set(actions) != requested:
        raise ValueError(
            f"candidate file is missing {len(requested - set(actions))} selected actions"
        )
    return actions


def _selected_gate_rows(
    gate_path: Path,
    actions: Mapping[str, GroupQueryAction],
) -> dict[tuple[str, str], float]:
    pairs: dict[tuple[str, str], float] = {}
    for row in _router_read_csv(gate_path).itertuples(index=False):
        query_id = str(row.query_id)
        if query_id not in actions:
            continue
        cell_id = str(row.cell_id)
        action = actions[query_id]
        if cell_id not in action.cell_ids:
            raise ValueError(f"gate row membership drift for {query_id}/{cell_id}")
        if int(row.group_size) != action.group_size:
            raise ValueError(f"gate row size drift for {query_id}")
        if int(row.estimated_total_tokens) != action.estimated_total_tokens:
            raise ValueError(f"gate row cost drift for {query_id}")
        key = (cell_id, query_id)
        if key in pairs:
            raise ValueError(f"gate ledger duplicates pair {key}")
        pairs[key] = float(row.conservative_uplift)
    expected = {
        (cell_id, action.query_id)
        for action in actions.values()
        for cell_id in action.cell_ids
    }
    if set(pairs) != expected:
        raise ValueError(
            f"gate ledger selected-pair coverage drift: "
            f"missing={len(expected - set(pairs))}, extra={len(set(pairs) - expected)}"
        )
    return pairs


@dataclass(frozen=True)
class FrozenTransferPlan:
    project_root: Path
    formal_manifest_path: Path
    source_run: Path
    profile: TransferProfile
    repair_model: str
    derived_llm_config: Mapping[str, Any]
    actions: tuple[GroupQueryAction, ...]
    request_rows: tuple[Mapping[str, Any], ...]
    selection_audit: tuple[Mapping[str, Any], ...]
    identity_audit: Mapping[str, Any]
    protected_source_hashes: Mapping[str, str]
    preflight_query_ids: tuple[str, ...]

    @property
    def request_plan_sha256(self) -> str:
        return _canonical_sha256([dict(row) for row in self.request_rows])

    @property
    def provider_request_hash(self) -> str:
        return _canonical_sha256(
            [
                {
                    "query_id": str(row["query_id"]),
                    "provider_request_hash": str(row["provider_request_hash"]),
                }
                for row in self.request_rows
            ]
        )

    @property
    def preflight_query_id(self) -> str:
        """Compatibility alias for the historical single-query preflight."""

        return self.preflight_query_ids[0]

    @property
    def action_by_id(self) -> dict[str, GroupQueryAction]:
        return {action.query_id: action for action in self.actions}

    def summary(self) -> dict[str, Any]:
        sizes = Counter(action.group_size for action in self.actions)
        prompt_tokens = sum(action.estimated_prompt_tokens for action in self.actions)
        planning_completion_tokens = sum(
            action.completion_token_ceiling for action in self.actions
        )
        planning_total_tokens = sum(
            action.estimated_total_tokens for action in self.actions
        )
        provider_completion_tokens = sum(
            int(row["provider_max_completion_tokens"])
            for row in self.request_rows
        )
        documented_tolerance = (
            len(self.actions) * QWEN_DOCUMENTED_COMPLETION_TOKEN_TOLERANCE
            if self.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
            else 0
        )
        billed_output_cap = provider_completion_tokens + documented_tolerance
        provider_total_tokens = prompt_tokens + billed_output_cap
        return {
            "profile_id": self.profile.profile_id,
            "experiment_name": self.profile.experiment_name,
            "short_experiment_name": self.profile.short_experiment_name,
            "provider": self.profile.provider,
            "source_run": str(self.source_run),
            "formal_freeze_manifest": str(self.formal_manifest_path),
            "repair_model": self.repair_model,
            "router": "tabiclv2",
            "router_training_source": "historical DeepSeek V4 Flash responses",
            "group_size_variant": "4",
            "selection_budget_reference": "frozen_deepseek_v4_flash_plan",
            "selection_planning_budget_share": 0.2,
            "planning_budget_share": 0.2,
            "actual_token_parity_enforced": False,
            "qwen_actual_tokens_used_for_selection": False,
            "rho": 1.0,
            "gamma": 1.0,
            "request_count": len(self.actions),
            "group_size_counts": {str(key): value for key, value in sorted(sizes.items())},
            "estimated_input_tokens": prompt_tokens,
            "planning_completion_token_ceiling_tokens": planning_completion_tokens,
            "completion_token_ceiling_tokens": provider_completion_tokens,
            "provider_completion_token_ceiling_tokens": provider_completion_tokens,
            "provider_documented_output_token_tolerance": documented_tolerance,
            "provider_billed_output_token_cap_estimate": billed_output_cap,
            "estimated_total_tokens": planning_total_tokens,
            "provider_normal_round_token_cap_estimate": provider_total_tokens,
            "retry_multiplier": EXPECTED_RETRY_MULTIPLIER,
            "maximum_http_attempts": len(self.actions) * EXPECTED_RETRY_MULTIPLIER,
            "worst_case_estimated_tokens": provider_total_tokens
            * EXPECTED_RETRY_MULTIPLIER,
            "maximum_estimated_input_tokens_per_request": max(
                action.estimated_prompt_tokens for action in self.actions
            ),
            "maximum_completion_tokens_per_request": max(
                int(row["provider_max_completion_tokens"])
                for row in self.request_rows
            ),
            "maximum_estimated_total_tokens_per_request": max(
                int(row["estimated_input_tokens"])
                + int(row["provider_max_completion_tokens"])
                for row in self.request_rows
            ),
            "maximum_serialized_message_bytes_per_request": max(
                len(
                    json.dumps(
                        messages_as_dicts(action.messages),
                        ensure_ascii=False,
                    ).encode("utf-8")
                )
                for action in self.actions
            ),
            "maximum_retry_adjusted_input_tokens_per_request": max(
                action.estimated_prompt_tokens for action in self.actions
            )
            * EXPECTED_RETRY_MULTIPLIER,
            "maximum_retry_adjusted_output_tokens_per_request": max(
                int(row["provider_max_completion_tokens"])
                for row in self.request_rows
            )
            * EXPECTED_RETRY_MULTIPLIER,
            "maximum_retry_adjusted_total_tokens_per_request": max(
                int(row["estimated_input_tokens"])
                + int(row["provider_max_completion_tokens"])
                for row in self.request_rows
            )
            * EXPECTED_RETRY_MULTIPLIER,
            "preflight_query_ids": list(self.preflight_query_ids),
            "preflight_query_count": len(self.preflight_query_ids),
            "request_plan_sha256": self.request_plan_sha256,
            "provider_request_hash": self.provider_request_hash,
            **dict(self.identity_audit),
        }


def _protected_files(
    project_root: Path,
    source_run: Path,
    source_manifest: Mapping[str, Any],
    profile: TransferProfile,
) -> tuple[Path, ...]:
    files = [
        _formal_freeze(project_root),
        source_run / "run_manifest.json",
        source_run / "bound_experiment_config.json",
        source_run / "bound_llm_config.json",
        source_run / "llm" / "group_query_checkpoint.jsonl",
        source_run / "final" / "all_methods.jsonl",
        project_root / "configs" / "public_fds.json",
    ]
    # The whole formal Flash result run is immutable input to this transfer.
    # Hash every file, not just the fixed-eight slice consumed below, so the
    # closing audit can substantiate that no historical artifact was rewritten.
    files.extend(path for path in source_run.rglob("*") if path.is_file())
    parent_value = source_manifest.get("baran_source_run")
    if not isinstance(parent_value, str) or not parent_value:
        raise ValueError("frozen run manifest is missing baran_source_run")
    parent = Path(parent_value).resolve()
    files.extend(
        [
            parent / "run_manifest.json",
            parent / "llm" / "group_query_checkpoint.jsonl",
            parent / "llm" / "group_response_cache.jsonl",
        ]
    )
    for _, _, _, stem, _ in FIXED_DATASETS:
        files.extend(
            [
                _selection_path(source_run, stem),
                _candidate_path(source_run, stem),
                _gate_path(source_run, stem),
                _baran_path(source_run, stem),
                source_run / "final" / "per_dataset" / f"{stem}.jsonl",
            ]
        )
    if profile.profile_id == QWEN37_FLASH_PROFILE.profile_id:
        pro_run = project_root / "runs" / PRO_TRANSFER_RUN_ID
        pro_files = {
            pro_run / "run_manifest.json": EXPECTED_PRO_MANIFEST_SHA256,
            pro_run / "final" / "fixed_eight_cell_ledger.jsonl": (
                EXPECTED_PRO_LEDGER_SHA256
            ),
            pro_run / "metrics" / "fixed_eight_metrics.csv": (
                EXPECTED_PRO_METRICS_SHA256
            ),
        }
        for path, expected_sha in pro_files.items():
            if not path.is_file():
                raise FileNotFoundError(f"protected Pro transfer artifact is missing: {path}")
            if _sha256(path) != expected_sha:
                raise ValueError(f"protected Pro transfer artifact SHA-256 drift: {path}")
            files.append(path)
        diagnostic_run = project_root / "runs" / QWEN_DIAGNOSTIC_RUN_ID
        if diagnostic_run.is_dir():
            files.extend(
                path for path in diagnostic_run.rglob("*") if path.is_file()
            )
    unique: dict[str, Path] = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"protected frozen artifact is missing: {path}")
        unique[str(path)] = path
    return tuple(unique[key] for key in sorted(unique))


def build_frozen_transfer_plan(
    project_root: str | Path = PROJECT_ROOT,
    *,
    repair_model: str | None = None,
    profile: str | TransferProfile | None = None,
    hash_protected_sources: bool = True,
) -> FrozenTransferPlan:
    selected_profile = _profile(profile or repair_model)
    if repair_model is not None and repair_model != selected_profile.repair_model:
        raise ValueError("repair model/profile mismatch")
    project = Path(project_root).resolve()
    source_run = _frozen_run(project)
    formal_path = _formal_freeze(project)
    if _sha256(formal_path) != EXPECTED_FORMAL_MANIFEST_SHA256:
        raise ValueError("formal freeze manifest SHA-256 drift")
    if _sha256(source_run / "run_manifest.json") != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen TabICLv2 run manifest SHA-256 drift")
    formal = _read_json(formal_path)
    source_manifest = _read_json(source_run / "run_manifest.json")
    bound = next(
        (
            row
            for row in formal.get("bound_final_runs", [])
            if isinstance(row, Mapping) and row.get("configuration") == "tabiclv2"
        ),
        None,
    )
    if not isinstance(bound, Mapping) or str(bound.get("run_id")) != FROZEN_RUN_ID:
        raise ValueError("formal freeze no longer binds the expected TabICLv2 run")
    artifacts = bound.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get("run_manifest.json") != (
        EXPECTED_FROZEN_MANIFEST_SHA256
    ):
        raise ValueError("formal freeze TabICLv2 manifest binding drift")
    submission = formal.get("freeze_scope", {}).get("submission_datasets", [])
    expected_submission = [f"{suite}/{dataset}" for _, suite, dataset, _, _ in FIXED_DATASETS]
    if (
        len(submission) != len(expected_submission)
        or set(submission) != set(expected_submission)
        or "source/movies_1" in submission
    ):
        raise ValueError("formal fixed-eight dataset scope drift")
    config = source_manifest.get("experiment_config")
    if not isinstance(config, Mapping):
        raise ValueError("frozen run is missing experiment_config")
    if (
        float(config.get("primary_budget_share", -1)) != 0.2
        or float(config.get("harm_penalty_rho", -1)) != 1.0
        or float(config.get("uncertainty_penalty_gamma", -1)) != 1.0
    ):
        raise ValueError("frozen budget/rho/gamma configuration drift")
    llm_values = _read_json(source_run / "bound_llm_config.json")
    derived_config = _validate_llm_config(llm_values, selected_profile)
    client_config = GroupClientConfig.from_mapping(derived_config)
    if client_config.max_retries + 1 != EXPECTED_RETRY_MULTIPLIER:
        raise ValueError("finite retry policy drift")
    hash_client = DeepSeekGroupClient(client_config, api_key="identity-hash-only")

    actions_ordered: list[GroupQueryAction] = []
    request_rows: list[Mapping[str, Any]] = []
    selection_audit: list[Mapping[str, Any]] = []
    selection_files: list[Mapping[str, str]] = []
    ordered_ids: list[str] = []
    memberships: list[Mapping[str, Any]] = []
    prompt_identities: list[Mapping[str, str]] = []
    message_identities: list[Mapping[str, str]] = []
    global_ids: set[str] = set()

    for label, suite, dataset, stem, expected_count in FIXED_DATASETS:
        selection_path = _selection_path(source_run, stem)
        selection_sha = _sha256(selection_path)
        if selection_sha != EXPECTED_SELECTION_SHA256[stem]:
            raise ValueError(f"frozen selection SHA-256 drift for {stem}")
        document = _read_json(selection_path)
        selected_ids = [str(value) for value in document.get("selected_query_ids", [])]
        if len(selected_ids) != expected_count or len(set(selected_ids)) != expected_count:
            raise ValueError(
                f"selected-query count drift for {label}: "
                f"expected={expected_count}, observed={len(selected_ids)}"
            )
        if document.get("selected") != document.get("selected_query_ids"):
            raise ValueError(f"selection alias drift for {label}")
        step_ids = [str(row.get("query_id", "")) for row in document.get("steps", [])]
        if step_ids != selected_ids:
            raise ValueError(f"selection step order drift for {label}")
        if (
            document.get("backend") != "tabiclv2"
            or document.get("scenario") != "size_conditioned"
            or str(document.get("group_size_variant")) != "4"
            or float(document.get("budget_share", -1)) != 0.2
            or document.get("allowed_group_sizes") != [1, 4]
        ):
            raise ValueError(f"selection configuration drift for {label}")
        overlap = global_ids.intersection(selected_ids)
        if overlap:
            raise ValueError(f"selected query IDs overlap across datasets: {sorted(overlap)[:3]}")
        global_ids.update(selected_ids)
        candidate_path = _candidate_path(source_run, stem)
        actions = _load_selected_actions(candidate_path, selected_ids)
        gate_path = _gate_path(source_run, stem)
        gate_sha = _sha256(gate_path)
        if gate_sha != str(document.get("gate_ledger_sha256", "")):
            raise ValueError(f"gate ledger SHA-256 drift for {label}")
        _selected_gate_rows(gate_path, actions)
        steps = document.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError(f"selection steps are not an array for {label}")
        spent = 0
        for step in steps:
            if not isinstance(step, Mapping):
                raise ValueError(f"selection step is not an object for {label}")
            query_id = str(step.get("query_id", ""))
            action = actions[query_id]
            cost = int(step.get("cost_tokens", step.get("cost", 0)))
            if cost != action.estimated_total_tokens:
                raise ValueError(f"selection/action cost drift for {query_id}")
            spent += cost
            actions_ordered.append(action)
            ordered_ids.append(query_id)
            memberships.append(
                {"dataset": stem, "query_id": query_id, "cell_ids": list(action.cell_ids)}
            )
            prompt_identities.append(
                {"dataset": stem, "query_id": query_id, "prompt_hash": action.prompt_hash}
            )
            frozen_message_hash = _canonical_sha256(
                messages_as_dicts(action.messages)
            )
            message_identities.append(
                {
                    "dataset": stem,
                    "query_id": query_id,
                    "frozen_message_hash": frozen_message_hash,
                }
            )
            provider_max_completion_tokens = _provider_max_completion_tokens(
                selected_profile,
                action.group_size,
                action.completion_token_ceiling,
            )
            job = GroupLLMJob(
                query_id=action.query_id,
                messages=action.messages,
                prompt_hash=action.prompt_hash,
                expected_cell_ids=tuple(action.cell_ids),
                max_tokens=provider_max_completion_tokens,
                metadata={
                    "phase": "online_selected_union",
                    "suite": suite,
                    "dataset": dataset,
                    "group_size": action.group_size,
                    "group_view": action.group_view,
                    "estimated_total_tokens": action.estimated_total_tokens,
                    "prompt_schema_version": action.prompt_schema_version,
                    "model_requested": selected_profile.repair_model,
                    "frozen_flash_router_transfer": True,
                    "completion_token_parameter": derived_config[
                        "completion_token_parameter"
                    ],
                },
            )
            request_rows.append(
                {
                    "query_id": query_id,
                    "prompt_hash": action.prompt_hash,
                    "frozen_selection_prompt_hash": action.prompt_hash,
                    "frozen_message_hash": frozen_message_hash,
                    "provider_request_hash": hash_client.provider_request_hash(job),
                    "model_requested": selected_profile.repair_model,
                    "suite": suite,
                    "dataset": dataset,
                    "dataset_stem": stem,
                    "group_view": action.group_view,
                    "group_size": action.group_size,
                    "cell_ids": list(action.cell_ids),
                    "prompt_schema_version": action.prompt_schema_version,
                    "prompt_information_policy": action.prompt_information_policy,
                    "messages_sha256": frozen_message_hash,
                    "estimated_input_tokens": action.estimated_prompt_tokens,
                    "planning_completion_token_ceiling": (
                        action.completion_token_ceiling
                    ),
                    "completion_token_ceiling": (
                        action.completion_token_ceiling
                    ),
                    "provider_max_completion_tokens": (
                        provider_max_completion_tokens
                    ),
                    "provider_completion_token_parameter": derived_config[
                        "completion_token_parameter"
                    ],
                    "estimated_total_tokens": action.estimated_total_tokens,
                    "selection_budget_reference": (
                        "frozen_deepseek_v4_flash_plan"
                    ),
                    "selection_planning_budget_share": 0.2,
                    "qwen_actual_tokens_used_for_selection": False,
                    "selection_order": len(ordered_ids) - 1,
                }
            )
        budget = int(document.get("budget", -1))
        if spent > budget:
            raise ValueError(f"frozen selection exceeds its budget for {label}")
        selection_files.append(
            {
                "path": str(selection_path.relative_to(source_run)),
                "sha256": selection_sha,
            }
        )
        selection_audit.append(
            {
                "label": label,
                "suite": suite,
                "dataset": dataset,
                "stem": stem,
                "selected_queries": len(selected_ids),
                "selected_query_ids_sha256": _canonical_sha256(selected_ids),
                "selection_sha256": selection_sha,
                "candidate_sha256": _sha256(candidate_path),
                "gate_ledger_sha256": gate_sha,
                "budget": budget,
                "selected_estimated_tokens": spent,
                "budget_slack": budget - spent,
                "selected_cell_incidence": int(document.get("selected_cell_incidence", 0)),
                "group_size_counts": {
                    str(size): count
                    for size, count in sorted(
                        Counter(actions[query_id].group_size for query_id in selected_ids).items()
                    )
                },
            }
        )

    if len(actions_ordered) != EXPECTED_REQUESTS:
        raise ValueError(
            f"fixed-eight request count drift: expected={EXPECTED_REQUESTS}, "
            f"observed={len(actions_ordered)}"
        )
    sizes = Counter(action.group_size for action in actions_ordered)
    if sizes != {1: EXPECTED_SINGLETONS, 4: EXPECTED_GROUPS_OF_FOUR}:
        raise ValueError(f"selected group-size distribution drift: {dict(sizes)}")
    identity_audit = {
        "formal_freeze_manifest_sha256": _sha256(formal_path),
        "frozen_run_manifest_sha256": _sha256(source_run / "run_manifest.json"),
        "selection_file_manifest_sha256": _canonical_sha256(
            sorted(selection_files, key=lambda row: row["path"])
        ),
        "selected_query_set_sha256": _canonical_sha256(sorted(ordered_ids)),
        "selected_query_ordered_by_dataset_sha256": _canonical_sha256(ordered_ids),
        "selected_membership_identity_sha256": _canonical_sha256(memberships),
        "prompt_identity_sha256": _canonical_sha256(prompt_identities),
        "frozen_selection_prompt_hash": _canonical_sha256(prompt_identities),
        "frozen_message_hash": _canonical_sha256(message_identities),
    }
    expected_identities = {
        "selection_file_manifest_sha256": EXPECTED_SELECTION_MANIFEST_SHA256,
        "selected_query_set_sha256": EXPECTED_QUERY_SET_SHA256,
        "selected_query_ordered_by_dataset_sha256": EXPECTED_ORDERED_QUERY_SHA256,
        "selected_membership_identity_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "prompt_identity_sha256": EXPECTED_PROMPT_IDENTITY_SHA256,
        "frozen_message_hash": EXPECTED_FROZEN_MESSAGE_SHA256,
    }
    for key, expected in expected_identities.items():
        if identity_audit[key] != expected:
            raise ValueError(
                f"frozen identity drift for {key}: "
                f"expected={expected}, observed={identity_audit[key]}"
            )
    provider_request_hash = _canonical_sha256(
        [
            {
                "query_id": str(row["query_id"]),
                "provider_request_hash": str(row["provider_request_hash"]),
            }
            for row in request_rows
        ]
    )
    request_plan_sha256 = _canonical_sha256(request_rows)
    if selected_profile.profile_id == QWEN37_FLASH_PROFILE.profile_id:
        if provider_request_hash != EXPECTED_QWEN_PROVIDER_REQUEST_HASH:
            raise ValueError("Qwen provider-request identity drift")
        if request_plan_sha256 != EXPECTED_QWEN_REQUEST_PLAN_SHA256:
            raise ValueError("Qwen authorized request-plan identity drift")
    prompt_tokens = sum(action.estimated_prompt_tokens for action in actions_ordered)
    output_tokens = sum(action.completion_token_ceiling for action in actions_ordered)
    total_tokens = sum(action.estimated_total_tokens for action in actions_ordered)
    if (prompt_tokens, output_tokens, total_tokens) != (
        EXPECTED_INPUT_TOKENS,
        EXPECTED_OUTPUT_TOKENS,
        EXPECTED_TOTAL_TOKENS,
    ):
        raise ValueError("frozen request token totals drift")
    if total_tokens * EXPECTED_RETRY_MULTIPLIER != EXPECTED_WORST_TOKENS:
        raise AssertionError("retry-adjusted token cap drift")
    provider_output_tokens = sum(
        int(row["provider_max_completion_tokens"]) for row in request_rows
    )
    if (
        selected_profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
        and provider_output_tokens != QWEN_PROVIDER_OUTPUT_TOKEN_CAP
    ):
        raise ValueError("Qwen provider completion-token cap drift")
    maximum_message_bytes = max(
        len(
            json.dumps(
                messages_as_dicts(action.messages),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        for action in actions_ordered
    )
    if maximum_message_bytes != EXPECTED_MAX_SERIALIZED_MESSAGE_BYTES:
        raise ValueError("frozen serialized message byte bound drift")
    if selected_profile.profile_id == QWEN37_FLASH_PROFILE.profile_id:
        group_preflight: list[str] = []
        singleton_preflight: list[str] = []
        for _, suite, dataset, _, _ in FIXED_DATASETS:
            dataset_actions = [
                action
                for action in actions_ordered
                if action.suite == suite and action.dataset == dataset
            ]

            def longest(group_size: int) -> GroupQueryAction | None:
                candidates = [
                    action
                    for action in dataset_actions
                    if action.group_size == group_size
                ]
                if not candidates:
                    return None
                return max(
                    candidates,
                    key=lambda action: (
                        action.estimated_prompt_tokens,
                        len(
                            json.dumps(
                                messages_as_dicts(action.messages),
                                ensure_ascii=False,
                            ).encode("utf-8")
                        ),
                        action.query_id,
                    ),
                )

            group_action = longest(4)
            if group_action is None:
                raise ValueError(
                    f"Qwen preflight has no group-size-4 query for {suite}/{dataset}"
                )
            group_preflight.append(group_action.query_id)
            singleton_action = longest(1)
            if singleton_action is not None:
                singleton_preflight.append(singleton_action.query_id)
        preflight_query_ids = tuple(group_preflight + singleton_preflight)
        if (
            len(preflight_query_ids) != EXPECTED_PREFLIGHT_QUERIES
            or len(set(preflight_query_ids)) != EXPECTED_PREFLIGHT_QUERIES
        ):
            raise ValueError(
                "deterministic Qwen stratified preflight query-count drift"
            )
    else:
        preflight = max(
            actions_ordered,
            key=lambda action: (
                action.group_size,
                action.estimated_total_tokens,
                action.query_id,
            ),
        )
        if (
            preflight.query_id != EXPECTED_PREFLIGHT_QUERY_ID
            or preflight.group_size != 4
        ):
            raise ValueError("deterministic maximum-group preflight query drift")
        preflight_query_ids = (preflight.query_id,)
    protected_hashes: dict[str, str] = {}
    if hash_protected_sources:
        for path in _protected_files(
            project, source_run, source_manifest, selected_profile
        ):
            protected_hashes[str(path)] = _sha256(path)
    return FrozenTransferPlan(
        project_root=project,
        formal_manifest_path=formal_path,
        source_run=source_run,
        profile=selected_profile,
        repair_model=selected_profile.repair_model,
        derived_llm_config=derived_config,
        actions=tuple(actions_ordered),
        request_rows=tuple(request_rows),
        selection_audit=tuple(selection_audit),
        identity_audit=identity_audit,
        protected_source_hashes=protected_hashes,
        preflight_query_ids=preflight_query_ids,
    )


def _response_index(
    checkpoint_path: Path,
    plan: FrozenTransferPlan,
    *,
    expected_model: str | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    authorized = {
        (str(row["query_id"]), str(row["prompt_hash"]))
        for row in plan.request_rows
    }
    index: dict[tuple[str, str], dict[str, Any]] = {}
    prompt_by_query: dict[str, str] = {}
    for row in _iter_jsonl(checkpoint_path):
        query_id = str(row.get("query_id", ""))
        prompt_hash = str(row.get("prompt_hash", ""))
        if not query_id or not prompt_hash:
            continue
        previous = prompt_by_query.setdefault(query_id, prompt_hash)
        if previous != prompt_hash:
            raise ValueError(f"checkpoint prompt drift for {query_id}")
        key = (query_id, prompt_hash)
        if key in authorized:
            index[key] = dict(row)
    if set(index) != authorized:
        missing = sorted(authorized - set(index))
        raise ValueError(
            f"checkpoint does not cover the frozen request plan: missing={missing[:3]} "
            f"(count={len(missing)})"
        )
    if expected_model is not None:
        for key, row in index.items():
            if row.get("status") != "success":
                continue
            returned = str(row.get("model_returned", row.get("model", "")))
            requested = str(row.get("model_requested", expected_model))
            if returned != expected_model or requested != expected_model:
                raise ValueError(f"checkpoint model identity drift for {key[0]}")
            if row.get("model_matches_request", True) is False:
                raise ValueError(f"checkpoint declares model mismatch for {key[0]}")
            if (
                plan.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
                and expected_model == plan.profile.repair_model
            ):
                usage = row.get("usage")
                fields = _usage_fields(
                    {"usage": usage if isinstance(usage, Mapping) else {}}
                )
                expected_cells = set(plan.action_by_id[key[0]].cell_ids)
                returned_cells = {
                    str(item.get("cell_id", ""))
                    for item in row.get("items", [])
                    if isinstance(item, Mapping)
                }
                if not (
                    row.get("finish_reason") == "stop"
                    and row.get("parse_status") == "ok"
                    and returned_cells == expected_cells
                    and len(row.get("items", [])) == len(expected_cells)
                    and not row.get("missing_cell_ids")
                    and not row.get("unknown_cell_ids")
                    and not row.get("duplicate_cell_ids")
                    and not row.get("invalid_items")
                    and all(
                        fields[name] is not None
                        for name in ("input_tokens", "output_tokens", "total_tokens")
                    )
                    and fields["reasoning_tokens"] in {None, 0}
                    and row.get("thinking_disabled") is True
                ):
                    raise ValueError(
                        f"strict Qwen response-integrity drift for {key[0]}"
                    )
    return index


def _official_flash_rows(
    plan: FrozenTransferPlan,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    fixed = {(suite, dataset) for _, suite, dataset, _, _ in FIXED_DATASETS}
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in _iter_jsonl(plan.source_run / "final" / "all_methods.jsonl"):
        suite = str(row.get("suite", ""))
        dataset = str(row.get("dataset", ""))
        if (suite, dataset) not in fixed:
            continue
        if not (
            row.get("method") == "budgeted_group_tabiclv2"
            and row.get("scenario") == "size_conditioned"
            and row.get("backend") == "tabiclv2"
            and float(row.get("budget_share", -1)) == 0.2
            and str(row.get("group_size_variant", "")) == "4"
        ):
            continue
        key = (suite, dataset, str(row.get("cell_id", "")))
        if key in rows:
            raise ValueError(f"official Flash final ledger duplicates {key}")
        rows[key] = dict(row)
    if len(rows) != EXPECTED_ERROR_CELLS:
        raise ValueError(
            "official fixed-eight Flash ledger count drift: "
            f"expected={EXPECTED_ERROR_CELLS}, observed={len(rows)}"
        )
    return rows


def _evaluate_ledger(
    plan: FrozenTransferPlan,
    responses: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    method: str,
) -> list[dict[str, Any]]:
    """Replay the frozen final-cell construction using the existing verifier."""

    source_manifest = _read_json(plan.source_run / "run_manifest.json")
    data_root = Path(str(source_manifest.get("data_root", plan.project_root / "data")))
    experiment_config = source_manifest.get("experiment_config")
    if not isinstance(experiment_config, Mapping):
        raise ValueError("source experiment_config is missing")
    verifier_raw = experiment_config.get("verifier", {})
    verifier_config = VerifierConfig(
        **dict(verifier_raw) if isinstance(verifier_raw, Mapping) else {}
    )
    fd_registry = load_public_fds(plan.project_root / "configs" / "public_fds.json")
    actions_by_dataset: dict[tuple[str, str], dict[str, GroupQueryAction]] = defaultdict(dict)
    for action in plan.actions:
        actions_by_dataset[(action.suite, action.dataset)][action.query_id] = action

    records: list[dict[str, Any]] = []
    for _, suite, dataset, stem, _ in FIXED_DATASETS:
        loaded = load_dataset(suite, dataset, data_root)
        safe = loaded.safe_view()
        cells = tuple(safe.cells)
        cell_by_id = {str(cell.cell_id): cell for cell in cells}
        clean = {
            str(cell.cell_id): normalize_for_match(
                loaded.clean.iloc[cell.row, cell.col]
            )
            for cell in cells
        }
        baran = {
            str(row["cell_id"]): row for row in _iter_jsonl(_baran_path(plan.source_run, stem))
        }
        if set(baran) != set(cell_by_id):
            raise ValueError(f"Baran coverage drift for {suite}/{dataset}")
        verifier = GroupRepairVerifier(
            safe.dirty,
            cells,
            fds_for_dataset(fd_registry, suite, dataset),
            verifier_config,
        )
        actions = actions_by_dataset[(suite, dataset)]
        selected_by_cell: dict[str, list[str]] = defaultdict(list)
        for action in actions.values():
            for cell_id in action.cell_ids:
                selected_by_cell[cell_id].append(action.query_id)
        uplift = _selected_gate_rows(_gate_path(plan.source_run, stem), actions)

        for cell_id in sorted(cell_by_id):
            candidates: list[RankedRepairCandidate] = []
            covering_responses: list[Mapping[str, Any]] = []
            for query_id in selected_by_cell.get(cell_id, []):
                action = actions[query_id]
                response = responses.get((query_id, action.prompt_hash), {})
                covering_responses.append(response)
                response_usable = (
                    response.get("status") == "success"
                    and response.get("model_matches_request", True) is not False
                )
                raw_items = response.get("items", []) if response_usable else []
                item = (
                    next(
                        (
                            raw
                            for raw in raw_items
                            if isinstance(raw, Mapping)
                            and str(raw.get("cell_id")) == cell_id
                        ),
                        None,
                    )
                    if isinstance(raw_items, list)
                    else None
                )
                if isinstance(item, Mapping):
                    item = {**dict(item), "parse_status": "ok_item"}
                candidates.append(
                    RankedRepairCandidate(
                        query_id=query_id,
                        item=item or {"parse_status": "missing_item"},
                        conservative_uplift=uplift[(cell_id, query_id)],
                        cost=action.estimated_total_tokens,
                        group_size=action.group_size,
                    )
                )
            arbitration = verifier.arbitrate(cell_by_id[cell_id], baran[cell_id], candidates)
            decision = arbitration.decision
            parse_status = (
                "ok_llm"
                if decision.accept_llm
                else str(baran[cell_id].get("parse_status", "no_prediction"))
            )
            prediction = decision.final_prediction
            correct = bool(
                parse_status.startswith("ok")
                and normalize_for_match(prediction) == normalize_for_match(clean[cell_id])
            )
            records.append(
                {
                    "cell_id": cell_id,
                    "suite": suite,
                    "dataset": dataset,
                    "method": method,
                    "scenario": "size_conditioned",
                    "backend": "tabiclv2",
                    "budget_share": 0.2,
                    "group_size_variant": "4",
                    "prediction": prediction,
                    "clean_value": clean[cell_id],
                    "parse_status": parse_status,
                    "valid_prediction": parse_status.startswith("ok"),
                    "correct_repair": correct,
                    "final_source": decision.final_source,
                    "accepted_llm": decision.accept_llm,
                    "selected_query_id": decision.query_id,
                    "verification_reason": decision.reason,
                    "verification_score": decision.score,
                    "conservative_uplift": decision.conservative_uplift,
                    "selected_queries_covering_cell": len(candidates),
                    "attempted_query_count": len(arbitration.attempted_query_ids),
                    "attempted_query_ids": list(arbitration.attempted_query_ids),
                    "rejected_candidate_count": len(arbitration.rejected_reasons),
                    "rejected_reasons": list(arbitration.rejected_reasons),
                    "covering_query_parse_statuses": [
                        str(response.get("parse_status", "missing_response"))
                        for response in covering_responses
                    ],
                }
            )
    if len(records) != EXPECTED_ERROR_CELLS:
        raise ValueError(
            f"fixed-eight cell ledger count drift: expected={EXPECTED_ERROR_CELLS}, "
            f"observed={len(records)}"
        )
    return records


def _flash_replay_parity(plan: FrozenTransferPlan) -> dict[str, Any]:
    before = plan.request_plan_sha256
    responses = _response_index(
        plan.source_run / "llm" / "group_query_checkpoint.jsonl",
        plan,
        expected_model=FLASH_MODEL,
    )
    replay = _evaluate_ledger(plan, responses, method="budgeted_group_tabiclv2")
    official = _official_flash_rows(plan)
    compare_fields = (
        "prediction",
        "clean_value",
        "parse_status",
        "valid_prediction",
        "correct_repair",
        "final_source",
        "accepted_llm",
        "selected_query_id",
        "verification_reason",
        "verification_score",
        "conservative_uplift",
        "selected_queries_covering_cell",
        "attempted_query_count",
        "rejected_candidate_count",
    )
    mismatches: list[dict[str, Any]] = []
    for row in replay:
        key = (str(row["suite"]), str(row["dataset"]), str(row["cell_id"]))
        expected = official[key]
        differing = [field for field in compare_fields if row.get(field) != expected.get(field)]
        if differing:
            mismatches.append({"key": list(key), "fields": differing})
            if len(mismatches) >= 20:
                break
    if mismatches:
        raise ValueError(f"frozen Flash offline final replay drift: {mismatches}")
    if plan.request_plan_sha256 != before:
        raise AssertionError("clean-label replay changed the authorized request plan")
    return {
        "ok": True,
        "checkpoint_rows_bound": len(responses),
        "cell_rows_replayed": len(replay),
        "cell_mismatches": 0,
        "request_plan_sha256_before_oracle_replay": before,
        "request_plan_sha256_after_oracle_replay": plan.request_plan_sha256,
    }


def _normal_round_cost(
    summary: Mapping[str, Any],
    profile: TransferProfile,
    price_key: str | None = None,
) -> float:
    profile = _profile(profile)
    snapshot = profile.price_snapshot
    if profile.provider == "deepseek":
        if price_key not in {"off_peak", "peak"}:
            raise ValueError("DeepSeek cost estimate requires a price period")
        prices = snapshot[str(price_key)]
    else:
        tiers = snapshot.get("tiers")
        if not isinstance(tiers, Sequence) or not tiers:
            raise ValueError("Qwen price snapshot has no tiers")
        prices = tiers[0]
    assert isinstance(prices, Mapping)
    return (
        int(summary["estimated_input_tokens"]) * float(prices["input_cache_miss"])
        + int(summary["provider_billed_output_token_cap_estimate"])
        * float(prices["output"])
    ) / int(snapshot["unit_tokens"])


def _bound_repair_config(plan: FrozenTransferPlan) -> dict[str, Any]:
    config = GroupClientConfig.from_mapping(plan.derived_llm_config)
    return {
        "base_url": config.base_url,
        "model": config.model,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "timeout_seconds": config.timeout_seconds,
        "max_retries": config.max_retries,
        "backoff_initial_seconds": config.backoff_initial_seconds,
        "backoff_max_seconds": config.backoff_max_seconds,
        "backoff_jitter": config.backoff_jitter,
        "concurrency": config.concurrency,
        "stream": config.stream,
        "extra_body": dict(config.extra_body),
        "response_format": {"type": "json_object"},
        "completion_token_parameter": config.completion_token_parameter,
        "planning_completion_token_ceiling": dict(
            plan.derived_llm_config["completion_token_ceiling"]
        ),
        "provider_max_completion_tokens": dict(
            plan.derived_llm_config.get(
                "provider_max_completion_tokens",
                {
                    "singleton": 192,
                    "group_size_4": 832,
                },
            )
        ),
        "documented_max_completion_token_tolerance": (
            QWEN_DOCUMENTED_COMPLETION_TOKEN_TOLERANCE
            if plan.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
            else 0
        ),
        "strict_complete_response": bool(
            plan.derived_llm_config.get("strict_complete_response", False)
        ),
        "prompt_schema_version": plan.derived_llm_config["prompt_schema_version"],
        "repairer_only_override": dict(
            plan.derived_llm_config["repairer_only_override"]
        ),
    }


def _old_qwen_diagnostic_overhead(
    plan: FrozenTransferPlan,
) -> dict[str, Any] | None:
    if plan.profile.profile_id != QWEN37_FLASH_PROFILE.profile_id:
        return None
    diagnostic_run = plan.project_root / "runs" / QWEN_DIAGNOSTIC_RUN_ID
    if not diagnostic_run.is_dir():
        return None
    cost = _cost_audit(plan, diagnostic_run)
    manifest = _read_json(diagnostic_run / "run_manifest.json")
    return {
        "captured_at_utc": _utc_now(),
        "accounting_scope": (
            "diagnostic_engineering_overhead_excluded_from_new_valid_experiment"
        ),
        "old_run_id": QWEN_DIAGNOSTIC_RUN_ID,
        "old_run_status": manifest.get("status"),
        "physical_http_attempts": cost["physical_http_attempts"],
        "queries_with_physical_attempts": cost[
            "queries_with_physical_attempts"
        ],
        "retry_attempts": cost["retry_attempts"],
        "usage_incomplete_attempts": cost["usage_incomplete_attempts"],
        "observed_tokens": cost["observed_tokens"],
        "known_itemized_cost_lower_bound_cny": cost["known_cost_cny"],
        "conservative_cache_miss_cost_upper_cny": cost[
            "conservative_cost_upper_cny"
        ],
    }


def create_dry_run(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_id: str | None = None,
    profile: str | TransferProfile | None = None,
) -> dict[str, Any]:
    """Create a fail-closed request authorization without loading credentials."""

    selected_profile = _profile(profile)
    run_name = _safe_run_id(run_id or selected_profile.default_run_id)
    project = Path(project_root).resolve()
    run_dir = project / "runs" / run_name
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run_dir}")
    plan = build_frozen_transfer_plan(
        project,
        profile=selected_profile,
        hash_protected_sources=True,
    )
    plan_summary = plan.summary()
    parity = _flash_replay_parity(plan)
    diagnostic_overhead = (
        _old_qwen_diagnostic_overhead(plan)
        if project == plan.project_root
        else None
    )
    if selected_profile.provider == "deepseek":
        off_peak = _normal_round_cost(plan_summary, selected_profile, "off_peak")
        peak = _normal_round_cost(plan_summary, selected_profile, "peak")
        estimated_cost = {
            "normal_round_all_input_cache_miss_off_peak": off_peak,
            "normal_round_all_input_cache_miss_peak": peak,
            "six_attempt_extreme_off_peak": off_peak * EXPECTED_RETRY_MULTIPLIER,
            "six_attempt_extreme_peak": peak * EXPECTED_RETRY_MULTIPLIER,
        }
        estimated_cost_field = "estimated_cost_usd"
    else:
        normal = _normal_round_cost(plan_summary, selected_profile)
        estimated_cost = {
            "normal_round_all_input_cache_miss": normal,
            "six_attempt_extreme": normal * EXPECTED_RETRY_MULTIPLIER,
        }
        estimated_cost_field = "estimated_cost_cny"
    dry_run = {
        "created_at_utc": _utc_now(),
        "api_called": False,
        "environment_file_loaded": False,
        "paid_api_authorized": False,
        "run_dir": str(run_dir),
        "plan": plan_summary,
        "per_dataset": [dict(row) for row in plan.selection_audit],
        "flash_offline_final_replay": parity,
        "price_snapshot": dict(selected_profile.price_snapshot),
        estimated_cost_field: estimated_cost,
        "token_caps": {
            "normal_round_input_token_estimate": plan_summary[
                "estimated_input_tokens"
            ],
            "normal_round_provider_output_token_cap": plan_summary[
                "provider_completion_token_ceiling_tokens"
            ],
            "provider_documented_output_token_tolerance": plan_summary[
                "provider_documented_output_token_tolerance"
            ],
            "normal_round_billed_output_token_cap_estimate": plan_summary[
                "provider_billed_output_token_cap_estimate"
            ],
            "normal_round_total_token_cap_estimate": plan_summary[
                "provider_normal_round_token_cap_estimate"
            ],
            "six_attempt_total_token_cap_estimate": plan_summary[
                "worst_case_estimated_tokens"
            ],
        },
        "provider_payload_contract": {
            "model": selected_profile.repair_model,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False if selected_profile.provider != "deepseek" else None,
            "response_format": {"type": "json_object"},
            "enable_thinking": (
                False if selected_profile.provider != "deepseek" else None
            ),
            "completion_token_parameter": plan.derived_llm_config[
                "completion_token_parameter"
            ],
            "singleton_provider_max_completion_tokens": (
                plan_summary["maximum_completion_tokens_per_request"]
                if selected_profile.provider == "deepseek"
                else QWEN_SINGLETON_MAX_COMPLETION_TOKENS
            ),
            "group_size_4_provider_max_completion_tokens": (
                832
                if selected_profile.provider == "deepseek"
                else QWEN_GROUP4_MAX_COMPLETION_TOKENS
            ),
            "deprecated_max_tokens_sent_for_qwen": False,
        },
        "data_sent_to_provider": (
            "benchmark query prompts from the frozen selection, including "
            "their context and candidate values"
        ),
        "implementation_files": [
            "src/budgeted_group_repair_no_baran/frozen_router_transfer.py",
            "src/budgeted_group_repair_no_baran/group_llm.py",
            "tests/test_frozen_router_transfer.py",
        ],
        "stage_gate": "EXPLICIT_PAID_API_APPROVAL_REQUIRED_FOR_PREFLIGHT",
    }
    if diagnostic_overhead is not None:
        dry_run["old_diagnostic_overhead_artifact"] = (
            "provenance/old_qwen_diagnostic_overhead.json"
        )
    manifest = {
        "run_id": run_name,
        "profile_id": selected_profile.profile_id,
        "experiment_name": selected_profile.experiment_name,
        "short_experiment_name": selected_profile.short_experiment_name,
        "experiment_type": "frozen_router_repairer_replacement",
        "version_label": (
            QWEN_VERSION_LABEL
            if selected_profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
            else "frozen_flash_router_pro_transfer_v1"
        ),
        "paper_description": (
            QWEN_PAPER_DESCRIPTION
            if selected_profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
            else selected_profile.short_experiment_name
        ),
        "status": "planned",
        "created_at_utc": dry_run["created_at_utc"],
        "api_called": False,
        "paid_api_authorized": False,
        "router_retrained": False,
        "router_recalibrated": False,
        "router_predicted": False,
        "queries_reselected": False,
        "selection_budget_reference": "frozen_deepseek_v4_flash_plan",
        "selection_planning_budget_share": 0.20,
        "actual_token_parity_enforced": False,
        "qwen_actual_tokens_used_for_selection": False,
        "clean_labels_used_for_authorization": False,
        "movies_1_authorized": False,
        "source_run": str(plan.source_run),
        "formal_freeze_manifest": str(plan.formal_manifest_path),
        "provider": selected_profile.provider,
        "repair_model": selected_profile.repair_model,
        "model_namespace": selected_profile.model_namespace,
        "endpoint_region": selected_profile.price_snapshot.get("region"),
        "strict_complete_response_mode": (
            selected_profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
        ),
        "preflight_query_ids": list(plan.preflight_query_ids),
        "preflight_query_count": len(plan.preflight_query_ids),
        "request_plan_sha256": plan.request_plan_sha256,
        "qwen_provider_request_hash": (
            plan.provider_request_hash
            if selected_profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
            else None
        ),
        "identity_audit": dict(plan.identity_audit),
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "provenance").mkdir()
    (run_dir / "llm" / selected_profile.model_namespace).mkdir(parents=True)
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "bound_repair_config.json", _bound_repair_config(plan))
    _write_jsonl(run_dir / "authorized_requests.jsonl", plan.request_rows)
    if diagnostic_overhead is not None:
        _write_json(
            run_dir / "provenance" / "old_qwen_diagnostic_overhead.json",
            diagnostic_overhead,
        )
    _write_json(
        run_dir / "preflight_plan.json",
        {
            "selection_rule": (
                "longest estimated input per dataset and group size; all eight "
                "group-size-4 strata followed by every available singleton stratum"
            ),
            "query_count": len(plan.preflight_query_ids),
            "query_ids": list(plan.preflight_query_ids),
            "queries": [
                dict(row)
                for row in plan.request_rows
                if row["query_id"] in set(plan.preflight_query_ids)
            ],
        },
    )
    _write_json(
        run_dir / "provenance" / "frozen_input_audit.json",
        {
            "captured_at_utc": dry_run["created_at_utc"],
            "protected_source_hashes": dict(plan.protected_source_hashes),
            "identity_audit": dict(plan.identity_audit),
            "request_plan_sha256": plan.request_plan_sha256,
            "provider_request_hash": plan.provider_request_hash,
        },
    )
    _write_json(run_dir / "dry_run.json", dry_run)
    return dry_run


def _run_dir(project_root: str | Path, run_id: str) -> Path:
    return Path(project_root).resolve() / "runs" / _safe_run_id(run_id)


def _update_manifest(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    manifest = _read_json(path)
    manifest.update(updates)
    manifest["updated_at_utc"] = _utc_now()
    _write_json(path, manifest)
    return manifest


def _reload_bound_plan(
    run_dir: Path,
    expected_profile: str | TransferProfile | None = None,
) -> FrozenTransferPlan:
    """Rebuild every frozen identity and compare it with the dry authorization."""

    if not run_dir.is_dir():
        raise FileNotFoundError(f"transfer run does not exist: {run_dir}")
    manifest = _read_json(run_dir / "run_manifest.json")
    manifest_profile = _profile(
        str(manifest.get("profile_id") or manifest.get("repair_model") or "")
    )
    if expected_profile is not None and manifest_profile != _profile(expected_profile):
        raise ValueError("run manifest transfer profile drift")
    if manifest.get("repair_model") != manifest_profile.repair_model:
        raise ValueError("run manifest repair model drift")
    plan = build_frozen_transfer_plan(
        run_dir.parents[1],
        profile=manifest_profile,
        hash_protected_sources=True,
    )
    if manifest.get("request_plan_sha256") != plan.request_plan_sha256:
        raise ValueError("run manifest request-plan binding drift")
    if manifest.get("preflight_query_ids") != list(plan.preflight_query_ids):
        raise ValueError("run manifest preflight-plan binding drift")
    if (
        manifest_profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
        and manifest.get("qwen_provider_request_hash")
        != plan.provider_request_hash
    ):
        raise ValueError("run manifest Qwen provider-request hash drift")
    authorized = list(_iter_jsonl(run_dir / "authorized_requests.jsonl"))
    if authorized != [dict(row) for row in plan.request_rows]:
        raise ValueError("authorized request ledger drift")
    if _canonical_sha256(authorized) != plan.request_plan_sha256:
        raise ValueError("authorized request ledger hash drift")
    if _read_json(run_dir / "bound_repair_config.json") != _bound_repair_config(plan):
        raise ValueError("bound repair configuration drift")
    frozen_audit = _read_json(run_dir / "provenance" / "frozen_input_audit.json")
    captured = frozen_audit.get("protected_source_hashes")
    if not isinstance(captured, Mapping) or dict(captured) != dict(
        plan.protected_source_hashes
    ):
        raise ValueError("protected frozen source artifacts changed after dry run")
    return plan


def _job_for_action(
    action: GroupQueryAction,
    *,
    profile: TransferProfile = DEEPSEEK_PRO_PROFILE,
    require_complete: bool | None = None,
    transfer_stage: str,
) -> GroupLLMJob:
    profile = _profile(profile)
    strict_qwen = profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
    complete = strict_qwen if require_complete is None else bool(require_complete)
    provider_max = _provider_max_completion_tokens(
        profile,
        action.group_size,
        action.completion_token_ceiling,
    )
    return GroupLLMJob(
        query_id=action.query_id,
        messages=action.messages,
        prompt_hash=action.prompt_hash,
        expected_cell_ids=tuple(action.cell_ids),
        max_tokens=provider_max,
        metadata={
            "phase": "online_selected_union",
            "transfer_stage": transfer_stage,
            "suite": action.suite,
            "dataset": action.dataset,
            "group_size": action.group_size,
            "group_view": action.group_view,
            "estimated_total_tokens": action.estimated_total_tokens,
            "prompt_schema_version": action.prompt_schema_version,
            "model_requested": profile.repair_model,
            "frozen_flash_router_transfer": True,
            "require_complete_response": complete,
            "require_complete_json": strict_qwen,
            "required_finish_reason": "stop" if strict_qwen else "",
            "retry_finish_reasons": ["length"] if strict_qwen else [],
            "require_complete_usage": strict_qwen,
            "require_zero_reasoning_tokens": strict_qwen,
            "require_thinking_disabled": strict_qwen,
            "thinking_disabled": True if strict_qwen else None,
            "completion_token_parameter": (
                "max_completion_tokens" if strict_qwen else "max_tokens"
            ),
            "planning_completion_token_ceiling": (
                action.completion_token_ceiling
            ),
            "provider_max_completion_tokens": provider_max,
        },
    )


def _usage_fields(raw: Mapping[str, Any]) -> dict[str, int | None]:
    usage = raw.get("usage")
    values = usage if isinstance(usage, Mapping) else {}

    def number(*keys: str) -> int | None:
        for key in keys:
            value = values.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return None

    prompt = number("prompt_tokens", "input_tokens")
    completion = number("completion_tokens", "output_tokens")
    total = number("total_tokens")
    hit = number("prompt_cache_hit_tokens", "input_cache_hit_tokens")
    miss = number("prompt_cache_miss_tokens", "input_cache_miss_tokens")
    prompt_details = values.get("prompt_tokens_details")
    if hit is None and isinstance(prompt_details, Mapping):
        cached = prompt_details.get("cached_tokens")
        if isinstance(cached, (int, float)) and not isinstance(cached, bool):
            hit = int(cached)
    if miss is None and prompt is not None and hit is not None:
        miss = max(0, prompt - hit)
    completion_details = values.get("completion_tokens_details")
    reasoning: int | None = None
    if isinstance(completion_details, Mapping):
        value = completion_details.get("reasoning_tokens")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            reasoning = int(value)
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": total,
        "input_cache_hit_tokens": hit,
        "input_cache_miss_tokens": miss,
        "reasoning_tokens": reasoning,
    }


class AuditedOpenAICompatibleGroupClient(DeepSeekGroupClient):
    """Record every physical HTTP attempt without altering client retry logic."""

    def __init__(
        self,
        config: GroupClientConfig,
        *,
        api_key: str,
        audit_path: Path,
        opener: Any = None,
        sleep_fn: Any = None,
        random_fn: Any = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if opener is not None:
            kwargs["opener"] = opener
        if sleep_fn is not None:
            kwargs["sleep_fn"] = sleep_fn
        if random_fn is not None:
            kwargs["random_fn"] = random_fn
        super().__init__(config, api_key=api_key, **kwargs)
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self._audit_lock = threading.Lock()
        self._local = threading.local()

    def chat(self, job: GroupLLMJob):  # type: ignore[no-untyped-def]
        self._local.job = job
        self._local.attempt = 0
        try:
            return super().chat(job)
        finally:
            self._local.job = None

    def _request_once(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        job = getattr(self._local, "job", None)
        if not isinstance(job, GroupLLMJob):
            raise RuntimeError("audited request lacks a bound GroupLLMJob")
        self._local.attempt = int(getattr(self._local, "attempt", 0)) + 1
        attempt = int(self._local.attempt)
        started = _utc_now()
        try:
            raw = super()._request_once(payload)
        except Exception as error:
            _append_jsonl(
                self.audit_path,
                {
                    "query_id": job.query_id,
                    "prompt_hash": job.prompt_hash,
                    "provider_request_hash": self.provider_request_hash(job),
                    "attempt": attempt,
                    "started_at_utc": started,
                    "ended_at_utc": _utc_now(),
                    "requested_model": self.config.model,
                    "returned_model": "",
                    "provider_model_field_present": False,
                    "status": "request_error",
                    "finish_reason": "",
                    "parse_status": "not_available",
                    "error_type": type(error).__name__,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "input_cache_hit_tokens": None,
                    "input_cache_miss_tokens": None,
                    "reasoning_tokens": None,
                },
                self._audit_lock,
            )
            raise
        returned_raw = raw.get("model")
        returned = returned_raw if isinstance(returned_raw, str) else ""
        model_present = "model" in raw and bool(returned)
        finish_reason = self._finish_reason(raw)
        error_type = ""
        status = "success"
        if not model_present:
            status = "model_identity_error"
            error_type = "MissingProviderModelIdentity"
        elif returned != self.config.model:
            status = "model_identity_error"
            error_type = "ProviderModelIdentityMismatch"
        parse_status = "not_available"
        try:
            content = self._response_content(raw)
            parse_status = parse_group_response(
                content,
                job.query_id,
                job.expected_cell_ids,
                require_complete_json=bool(
                    job.metadata.get("require_complete_json", False)
                ),
            ).parse_status
        except Exception as error:
            if not error_type:
                error_type = type(error).__name__
            if status == "success":
                status = "response_shape_error"
        required_finish = str(job.metadata.get("required_finish_reason", ""))
        if status == "success" and required_finish and (
            finish_reason != required_finish
        ):
            status = "finish_reason_error"
            error_type = (
                "CompletionTruncated"
                if finish_reason == "length"
                else "UnexpectedFinishReason"
            )
        fields = _usage_fields(raw)
        if status == "success" and bool(
            job.metadata.get("require_complete_response")
        ) and parse_status != "ok":
            status = "response_integrity_error"
            error_type = "IncompleteStructuredResponse"
        if status == "success" and bool(
            job.metadata.get("require_complete_usage")
        ) and any(
            fields[key] is None
            for key in ("input_tokens", "output_tokens", "total_tokens")
        ):
            status = "response_integrity_error"
            error_type = "IncompleteUsage"
        if status == "success" and bool(
            job.metadata.get("require_zero_reasoning_tokens")
        ) and fields["reasoning_tokens"] not in {None, 0}:
            status = "response_integrity_error"
            error_type = "UnexpectedReasoningTokens"
        _append_jsonl(
            self.audit_path,
            {
                "query_id": job.query_id,
                "prompt_hash": job.prompt_hash,
                "provider_request_hash": self.provider_request_hash(job),
                "attempt": attempt,
                "started_at_utc": started,
                "ended_at_utc": _utc_now(),
                "requested_model": self.config.model,
                "returned_model": returned,
                "provider_model_field_present": model_present,
                "status": status,
                "finish_reason": finish_reason,
                "parse_status": parse_status,
                "error_type": error_type,
                **fields,
            },
            self._audit_lock,
        )
        return raw


# Preserve the original isolated runner's import surface.
AuditedDeepSeekGroupClient = AuditedOpenAICompatibleGroupClient


def _paid_client(
    plan: FrozenTransferPlan,
    run_dir: Path,
) -> AuditedOpenAICompatibleGroupClient:
    api_key = os.environ.get(plan.profile.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"{plan.profile.api_key_env} is not set")
    config = GroupClientConfig.from_mapping(plan.derived_llm_config)
    if config.model != plan.profile.repair_model:
        raise ValueError("paid client model is not the bound repair model")
    return AuditedOpenAICompatibleGroupClient(
        config,
        api_key=api_key,
        audit_path=(
            run_dir
            / "llm"
            / plan.profile.model_namespace
            / "api_attempt_audit.jsonl"
        ),
    )


def _load_paid_environment(
    env_file: str | Path,
    profile: TransferProfile,
) -> tuple[str, ...]:
    loaded = load_env_file(env_file)
    if not os.environ.get(profile.api_key_env):
        raise RuntimeError(
            f"{profile.api_key_env} is unavailable after environment loading"
        )
    return loaded


def _preflight_row_passed(
    row: Mapping[str, Any],
    action: GroupQueryAction,
    plan: FrozenTransferPlan,
) -> bool:
    usage = row.get("usage")
    usage_fields = _usage_fields(
        {"usage": usage if isinstance(usage, Mapping) else {}}
    )
    strict_qwen = plan.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
    return bool(
        row.get("status") == "success"
        and row.get("parse_status") == "ok"
        and not row.get("missing_cell_ids")
        and not row.get("unknown_cell_ids")
        and not row.get("duplicate_cell_ids")
        and not row.get("invalid_items")
        and len(row.get("items", [])) == action.group_size
        and {
            str(item.get("cell_id", ""))
            for item in row.get("items", [])
            if isinstance(item, Mapping)
        }
        == set(action.cell_ids)
        and row.get("provider_model_field_present") is True
        and row.get("model_returned") == plan.profile.repair_model
        and row.get("model_requested") == plan.profile.repair_model
        and row.get("model_matches_request") is True
        and all(
            usage_fields[key] is not None
            for key in ("input_tokens", "output_tokens", "total_tokens")
        )
        and usage_fields.get("reasoning_tokens") in {None, 0}
        and (not strict_qwen or row.get("finish_reason") == "stop")
        and (not strict_qwen or row.get("thinking_disabled") is True)
    )


def run_preflight(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_id: str | None = None,
    profile: str | TransferProfile | None = None,
    env_file: str | Path,
    approve_paid_api: bool,
) -> dict[str, Any]:
    if not approve_paid_api:
        raise PermissionError("preflight requires explicit --approve-paid-api")
    selected_profile = _profile(profile)
    run_dir = _run_dir(project_root, run_id or selected_profile.default_run_id)
    plan = _reload_bound_plan(run_dir, selected_profile)
    manifest = _read_json(run_dir / "run_manifest.json")
    report_path = (
        run_dir / "llm" / plan.profile.model_namespace / "preflight.json"
    )
    if manifest.get("status") == "preflight_complete":
        return _read_json(report_path)
    if manifest.get("status") != "planned":
        raise RuntimeError(
            f"preflight is not allowed from status {manifest.get('status')!r}"
        )
    _load_paid_environment(env_file, plan.profile)
    _update_manifest(
        run_dir,
        status="preflight_running",
        paid_api_authorized=True,
        api_called=False,
    )
    client = _paid_client(plan, run_dir)
    actions = [plan.action_by_id[value] for value in plan.preflight_query_ids]
    namespace = run_dir / "llm" / plan.profile.model_namespace
    rows: list[dict[str, Any]] = []
    caught_error: Exception | None = None
    try:
        for action in actions:
            job = _job_for_action(
                action,
                profile=plan.profile,
                require_complete=True,
                transfer_stage="preflight",
            )
            try:
                current = run_group_llm_batch(
                    client,
                    [job],
                    namespace,
                    concurrency=1,
                    retry_failed=True,
                    fail_fast_finish_reasons=(
                        {"length"}
                        if plan.profile.profile_id
                        == QWEN37_FLASH_PROFILE.profile_id
                        else ()
                    ),
                )
                row = dict(current[0])
            except CompletionTruncatedError as error:
                caught_error = error
                checkpoint_rows = list(
                    _iter_jsonl(namespace / "group_query_checkpoint.jsonl")
                )
                matching = [
                    row
                    for row in checkpoint_rows
                    if row.get("query_id") == action.query_id
                    and row.get("prompt_hash") == action.prompt_hash
                ]
                if matching:
                    rows.append(dict(matching[-1]))
                break
            rows.append(row)
            if not _preflight_row_passed(row, action, plan):
                break
    except Exception as error:
        caught_error = error

    cache_path = namespace / "group_response_cache.jsonl"
    cache_rows = list(_iter_jsonl(cache_path)) if cache_path.is_file() else []
    cache_hashes = {
        str(row.get("provider_request_hash", ""))
        for row in cache_rows
        if row.get("model_returned", row.get("model"))
        == plan.profile.repair_model
        and (
            plan.profile.profile_id != QWEN37_FLASH_PROFILE.profile_id
            or row.get("finish_reason") == "stop"
        )
    }
    action_by_id = plan.action_by_id
    results: list[dict[str, Any]] = []
    for row in rows:
        query_id = str(row.get("query_id", ""))
        action = action_by_id[query_id]
        usage = row.get("usage")
        fields = _usage_fields(
            {"usage": usage if isinstance(usage, Mapping) else {}}
        )
        cache_written = str(row.get("provider_request_hash", "")) in cache_hashes
        passed = _preflight_row_passed(row, action, plan) and cache_written
        results.append(
            {
                "query_id": query_id,
                "dataset": action.dataset,
                "suite": action.suite,
                "group_size": action.group_size,
                "frozen_selection_prompt_hash": action.prompt_hash,
                "provider_request_hash": row.get("provider_request_hash"),
                "finish_reason": row.get("finish_reason"),
                "parse_status": row.get("parse_status"),
                "requested_model": row.get("model_requested"),
                "returned_model": row.get("model_returned"),
                "usage": dict(usage) if isinstance(usage, Mapping) else {},
                "usage_complete": all(
                    fields[key] is not None
                    for key in ("input_tokens", "output_tokens", "total_tokens")
                ),
                "reasoning_tokens": fields.get("reasoning_tokens"),
                "thinking_disabled": row.get("thinking_disabled") is True,
                "cache_written": cache_written,
                "passed": passed,
            }
        )
    passed = bool(
        caught_error is None
        and len(results) == len(actions)
        and all(row["passed"] for row in results)
    )
    report = {
        "completed_at_utc": _utc_now(),
        "passed": passed,
        "planned_query_count": len(actions),
        "completed_query_count": len(results),
        "query_ids": list(plan.preflight_query_ids),
        "results": results,
        "finish_reason_counts": dict(
            sorted(Counter(str(row.get("finish_reason", "")) for row in results).items())
        ),
        "cache_namespace": str(namespace),
        "all_successes_cached": bool(results) and all(
            row["cache_written"] for row in results
        ),
        "strict_complete_response_mode": (
            plan.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
        ),
        "error_type": type(caught_error).__name__ if caught_error else "",
    }
    _write_json(report_path, report)
    if not passed:
        _update_manifest(
            run_dir,
            status=(
                "model_identity_failed"
                if isinstance(caught_error, ProviderModelIdentityError)
                else "preflight_failed"
            ),
            api_called=True,
            last_error_type=(
                type(caught_error).__name__
                if caught_error
                else "PreflightIntegrityFailure"
            ),
        )
        if caught_error is not None:
            raise caught_error
        raise RuntimeError(
            f"{plan.profile.repair_model} preflight failed; batch remains blocked"
        )
    _update_manifest(run_dir, status="preflight_complete", api_called=True)
    return report


def run_authorized_batch(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_id: str | None = None,
    profile: str | TransferProfile | None = None,
    env_file: str | Path,
    resume: bool,
    approve_paid_api: bool,
) -> dict[str, Any]:
    if not resume:
        raise PermissionError("batch execution requires --resume")
    if not approve_paid_api:
        raise PermissionError("batch execution requires explicit --approve-paid-api")
    selected_profile = _profile(profile)
    run_dir = _run_dir(project_root, run_id or selected_profile.default_run_id)
    plan = _reload_bound_plan(run_dir, selected_profile)
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest.get("status") == "batch_complete":
        return _read_json(
            run_dir
            / "llm"
            / plan.profile.model_namespace
            / "selected_execution.json"
        )
    status_before_batch = str(manifest.get("status", ""))
    if status_before_batch not in {
        "preflight_complete",
        "batch_interrupted",
        "batch_stopped_truncation",
    }:
        raise RuntimeError(
            f"batch is not allowed from status {manifest.get('status')!r}"
        )
    _update_manifest(
        run_dir,
        status="batch_running",
        paid_api_authorized=True,
        failed_preflight_accepted_for_strict_transfer=False,
        truncation_recovery_enabled=(
            status_before_batch == "batch_stopped_truncation"
        ),
        recovery_from_status=(
            status_before_batch
            if status_before_batch == "batch_stopped_truncation"
            else ""
        ),
    )
    _load_paid_environment(env_file, plan.profile)
    client = _paid_client(plan, run_dir)
    jobs = [
        _job_for_action(
            action,
            profile=plan.profile,
            require_complete=(
                plan.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
            ),
            transfer_stage="batch",
        )
        for action in plan.actions
    ]
    try:
        rows = run_group_llm_batch(
            client,
            jobs,
            run_dir / "llm" / plan.profile.model_namespace,
            concurrency=client.config.concurrency,
            retry_failed=False,
            retry_failed_finish_reasons=(
                {"length"}
                if status_before_batch == "batch_stopped_truncation"
                else ()
            ),
            fail_fast_finish_reasons=(
                {"length"}
                if plan.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
                else ()
            ),
        )
        if len(rows) != EXPECTED_REQUESTS:
            raise ValueError("selected execution did not return every authorized query")
        by_status = Counter(str(row.get("status", "")) for row in rows)
        report = {
            "completed_at_utc": _utc_now(),
            "authorized_queries": EXPECTED_REQUESTS,
            "terminal_queries": len(rows),
            "success_queries": by_status.get("success", 0),
            "failed_queries": by_status.get("failed", 0),
            "checkpoint_hits": sum(bool(row.get("checkpoint_hit")) for row in rows),
            "cache_hits": sum(bool(row.get("cache_hit")) for row in rows),
            "preflight_reused_query_ids": [
                str(row.get("query_id"))
                for row in rows
                if row.get("query_id") in set(plan.preflight_query_ids)
                and bool(row.get("checkpoint_hit"))
            ],
            "failed_preflight_accepted_for_strict_transfer": False,
            "partial_success_queries": sum(
                row.get("status") == "success"
                and row.get("parse_status") == "partial"
                for row in rows
            ),
            "attempts_reported_by_terminal_rows": sum(
                int(row.get("attempts", 0) or 0) for row in rows
            ),
        }
        report["preflight_reused"] = set(
            report["preflight_reused_query_ids"]
        ) == set(plan.preflight_query_ids)
        if not report["preflight_reused"]:
            raise ValueError("all preflight responses were not reused by the batch")
        if (
            plan.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
            and report["partial_success_queries"]
        ):
            raise ValueError("strict Qwen batch accepted a partial response")
        _write_json(
            run_dir
            / "llm"
            / plan.profile.model_namespace
            / "selected_execution.json",
            report,
        )
        _update_manifest(run_dir, status="batch_complete", api_called=True)
        return report
    except Exception as error:
        status = (
            "model_identity_failed"
            if isinstance(error, ProviderModelIdentityError)
            else (
                "batch_stopped_truncation"
                if isinstance(error, CompletionTruncatedError)
                else "batch_interrupted"
            )
        )
        _update_manifest(
            run_dir,
            status=status,
            api_called=True,
            last_error_type=type(error).__name__,
            truncated_query_ids=(
                list(error.query_ids)
                if isinstance(error, CompletionTruncatedError)
                else []
            ),
        )
        raise


def _official_baran_rows(plan: FrozenTransferPlan) -> list[dict[str, Any]]:
    fixed = {(suite, dataset) for _, suite, dataset, _, _ in FIXED_DATASETS}
    rows = [
        dict(row)
        for row in _iter_jsonl(plan.source_run / "final" / "all_methods.jsonl")
        if (str(row.get("suite", "")), str(row.get("dataset", ""))) in fixed
        and row.get("method") == "baran"
        and row.get("scenario") == "baseline"
    ]
    if len(rows) != EXPECTED_ERROR_CELLS:
        raise ValueError(
            "official fixed-eight Baran ledger count drift: "
            f"expected={EXPECTED_ERROR_CELLS}, observed={len(rows)}"
        )
    return rows


def _validated_pro_transfer_rows(plan: FrozenTransferPlan) -> list[dict[str, Any]]:
    if plan.profile.profile_id != QWEN37_FLASH_PROFILE.profile_id:
        return []
    pro_run = plan.project_root / "runs" / PRO_TRANSFER_RUN_ID
    manifest_path = pro_run / "run_manifest.json"
    ledger_path = pro_run / "final" / "fixed_eight_cell_ledger.jsonl"
    metrics_path = pro_run / "metrics" / "fixed_eight_metrics.csv"
    expected = {
        manifest_path: EXPECTED_PRO_MANIFEST_SHA256,
        ledger_path: EXPECTED_PRO_LEDGER_SHA256,
        metrics_path: EXPECTED_PRO_METRICS_SHA256,
    }
    for path, expected_sha in expected.items():
        if _sha256(path) != expected_sha:
            raise ValueError(f"validated Pro comparison artifact drift: {path}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "validated" or manifest.get("validation_ok") is not True:
        raise ValueError("Pro comparison run is not validated")
    rows = list(_iter_jsonl(ledger_path))
    if len(rows) != EXPECTED_ERROR_CELLS:
        raise ValueError("Pro comparison cell ledger row count drift")
    allowed = {(suite, dataset) for _, suite, dataset, _, _ in FIXED_DATASETS}
    observed = {
        (str(row.get("suite", "")), str(row.get("dataset", ""))) for row in rows
    }
    if observed != allowed or any(row.get("dataset") == "movies_1" for row in rows):
        raise ValueError("Pro comparison dataset scope drift")
    return rows


def _summary_rows(
    transfer_rows: Sequence[Mapping[str, Any]],
    flash_rows: Sequence[Mapping[str, Any]],
    baran_rows: Sequence[Mapping[str, Any]],
    *,
    transfer_label: str = "pro",
    extra_methods: Sequence[
        tuple[str, Sequence[Mapping[str, Any]]]
    ] = (),
) -> list[dict[str, Any]]:
    material = [
        (transfer_label, transfer_rows),
        ("flash", flash_rows),
        ("baran", baran_rows),
        *extra_methods,
    ]
    result: list[dict[str, Any]] = []
    for label, records in material:
        rows = summarize_records(records, strict=True)
        if len(rows) != len(FIXED_DATASETS) + 2:
            raise ValueError(f"unexpected {label} metric row count: {len(rows)}")
        for row in rows:
            result.append({"comparison_method": label, **dict(row)})
    return result


def _metric_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("scope", "")), str(row.get("dataset", ""))


def _metric_deltas(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    transfer_label: str = "pro",
    baselines: Sequence[str] = ("flash", "baran"),
) -> dict[str, Any]:
    by_method: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = defaultdict(dict)
    for row in summary_rows:
        by_method[str(row["comparison_method"])][_metric_key(row)] = row
    transfer = by_method[transfer_label]
    comparisons: dict[str, list[dict[str, Any]]] = {}
    win_tie_loss: dict[str, dict[str, int]] = {}
    for baseline in baselines:
        baseline_rows = by_method[baseline]
        deltas: list[dict[str, Any]] = []
        outcome = Counter()
        for key in sorted(transfer):
            current = transfer[key]
            reference = baseline_rows[key]
            row: dict[str, Any] = {
                "scope": key[0],
                "dataset": key[1],
                "baseline": baseline,
            }
            for field in ("precision", "recall", "f1"):
                if transfer_label == "pro":
                    row[f"pro_{field}"] = float(current[field])
                else:
                    row[f"{transfer_label}_{field}"] = float(current[field])
                    row[f"transfer_{field}"] = float(current[field])
                row[f"baseline_{field}"] = float(reference[field])
                row[f"{field}_delta"] = float(current[field]) - float(
                    reference[field]
                )
            deltas.append(row)
            if key[0] == "dataset":
                difference = float(current["f1"]) - float(reference["f1"])
                if difference > 1e-12:
                    outcome["win"] += 1
                elif difference < -1e-12:
                    outcome["loss"] += 1
                else:
                    outcome["tie"] += 1
        comparisons[baseline] = deltas
        win_tie_loss[baseline] = {
            "win": outcome["win"],
            "tie": outcome["tie"],
            "loss": outcome["loss"],
            "metric": "per-dataset F1",
            "tolerance": 1e-12,
        }
    return {"comparisons": comparisons, "win_tie_loss": win_tie_loss}


def _diagnostics(
    ledger: Sequence[Mapping[str, Any]],
    responses: Mapping[tuple[str, str], Mapping[str, Any]],
    baran_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baran_by_cell = {
        (str(row["suite"]), str(row["dataset"]), str(row["cell_id"])): row
        for row in baran_rows
    }
    rescued = harmed = 0
    rejection_reasons: Counter[str] = Counter()
    for row in ledger:
        key = (str(row["suite"]), str(row["dataset"]), str(row["cell_id"]))
        pro_correct = bool(row.get("correct_repair"))
        baran_correct = bool(baran_by_cell[key].get("correct_repair"))
        rescued += int(pro_correct and not baran_correct)
        harmed += int(baran_correct and not pro_correct)
        for reason in row.get("rejected_reasons", []):
            rejection_reasons[str(reason)] += 1
    response_rows = list(responses.values())
    parse_failures = sum(
        str(row.get("parse_status", "")) not in {"ok", "partial", "llm_error"}
        for row in response_rows
    )
    partial = sum(row.get("parse_status") == "partial" for row in response_rows)
    partial_used = sum(
        row.get("parse_status") == "partial" and row.get("status") == "success"
        for row in response_rows
    )
    finish_reasons = Counter(
        str(row.get("finish_reason", "")) for row in response_rows
    )
    abstentions = sum(
        str(item.get("decision", "")) == "abstain"
        for row in response_rows
        for item in row.get("items", [])
        if isinstance(item, Mapping)
    )
    abstention_queries = sum(
        any(
            isinstance(item, Mapping)
            and str(item.get("decision", "")) == "abstain"
            for item in row.get("items", [])
        )
        for row in response_rows
    )
    api_failures = sum(
        row.get("status") == "failed" and row.get("parse_status") == "llm_error"
        for row in response_rows
    )
    terminal_failures = sum(row.get("status") != "success" for row in response_rows)
    return {
        "rescued_cells_vs_baran": rescued,
        "harmed_cells_vs_baran": harmed,
        "net_gain_vs_baran": rescued - harmed,
        "verifier_rejections": sum(rejection_reasons.values()),
        "verifier_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "parse_failures": parse_failures,
        "partial_parse_queries": partial,
        "partial_responses_used": partial_used,
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
        "truncated_queries": finish_reasons.get("length", 0),
        "abstention_items": abstentions,
        "abstention_queries": abstention_queries,
        "api_failure_queries": api_failures,
        "successful_queries": sum(row.get("status") == "success" for row in response_rows),
        "terminal_failed_queries": terminal_failures,
        "terminal_query_count": len(response_rows),
    }


def _price_period(started_at_utc: str) -> str:
    instant = datetime.fromisoformat(started_at_utc.replace("Z", "+00:00")).astimezone(
        timezone.utc
    )
    peak = instant.weekday() < 5 and (
        1 <= instant.hour < 4 or 6 <= instant.hour < 10
    )
    return "peak" if peak else "off_peak"


def _cost_audit(
    plan: FrozenTransferPlan,
    run_dir: Path,
) -> dict[str, Any]:
    profile = plan.profile
    snapshot = profile.price_snapshot
    audit_path = (
        run_dir / "llm" / profile.model_namespace / "api_attempt_audit.jsonl"
    )
    attempts = list(_iter_jsonl(audit_path)) if audit_path.is_file() else []
    authorized = {str(row["query_id"]): row for row in plan.request_rows}
    observed = Counter()
    known_cost = 0.0
    conservative_cost = 0.0
    unknown_usage_attempts = 0
    cost_inexact_attempts = 0
    price_bucket_counts: Counter[str] = Counter()
    finish_reason_counts: Counter[str] = Counter()
    per_dataset_tokens: defaultdict[str, Counter[str]] = defaultdict(Counter)
    output_token_values: list[int] = []
    reasoning_detail_reported_attempts = 0
    reasoning_detail_nonzero_attempts = 0
    for attempt in attempts:
        query_id = str(attempt.get("query_id", ""))
        if query_id not in authorized:
            raise ValueError(f"attempt audit includes unauthorized query {query_id!r}")
        prompt = attempt.get("input_tokens")
        completion = attempt.get("output_tokens")
        total = attempt.get("total_tokens")
        hit = attempt.get("input_cache_hit_tokens")
        miss = attempt.get("input_cache_miss_tokens")
        reasoning = attempt.get("reasoning_tokens")
        if isinstance(reasoning, int):
            reasoning_detail_reported_attempts += 1
            if reasoning != 0:
                reasoning_detail_nonzero_attempts += 1
        finish_reason_counts[str(attempt.get("finish_reason", ""))] += 1
        dataset_key = (
            f"{authorized[query_id]['suite']}/{authorized[query_id]['dataset']}"
        )
        if profile.provider == "deepseek":
            bucket = _price_period(str(attempt["started_at_utc"]))
            prices = snapshot[bucket]
        else:
            input_for_tier = (
                prompt
                if isinstance(prompt, int)
                else int(authorized[query_id]["estimated_input_tokens"])
            )
            tiers = snapshot.get("tiers")
            if not isinstance(tiers, Sequence):
                raise ValueError("Qwen price snapshot has no tiers")
            prices = next(
                (
                    tier
                    for tier in tiers
                    if isinstance(tier, Mapping)
                    and input_for_tier <= int(tier["maximum_input_tokens"])
                ),
                None,
            )
            if not isinstance(prices, Mapping):
                raise ValueError(
                    f"Qwen input token count exceeds the audited price table: {input_for_tier}"
                )
            bucket = str(prices["name"])
        price_bucket_counts[bucket] += 1
        assert isinstance(prices, Mapping)
        unit = int(snapshot["unit_tokens"])
        for name, value in (
            ("input_tokens", prompt),
            ("output_tokens", completion),
            ("total_tokens", total),
            ("input_cache_hit_tokens", hit),
            ("input_cache_miss_tokens", miss),
            ("reasoning_tokens", reasoning),
        ):
            if isinstance(value, int):
                observed[name] += value
                per_dataset_tokens[dataset_key][name] += value
        if isinstance(completion, int):
            output_token_values.append(completion)
        output_known = isinstance(completion, int)
        split_known = isinstance(hit, int) and isinstance(miss, int)
        if output_known:
            known_cost += completion * float(prices["output"]) / unit
            conservative_cost += completion * float(prices["output"]) / unit
        else:
            conservative_cost += (
                (
                    int(authorized[query_id]["provider_max_completion_tokens"])
                    + (
                        QWEN_DOCUMENTED_COMPLETION_TOKEN_TOLERANCE
                        if profile.profile_id
                        == QWEN37_FLASH_PROFILE.profile_id
                        else 0
                    )
                )
                * float(prices["output"])
                / unit
            )
        if split_known:
            known_input_cost = (
                hit * float(prices["input_cache_hit"])
                + miss * float(prices["input_cache_miss"])
            ) / unit
            known_cost += known_input_cost
            conservative_cost += known_input_cost
        elif isinstance(prompt, int):
            conservative_cost += prompt * float(prices["input_cache_miss"]) / unit
        else:
            conservative_cost += (
                int(authorized[query_id]["estimated_input_tokens"])
                * float(prices["input_cache_miss"])
                / unit
            )
        if not all(isinstance(value, int) for value in (prompt, completion, total)):
            unknown_usage_attempts += 1
        if not (output_known and split_known):
            cost_inexact_attempts += 1
    per_query_attempts = Counter(str(row.get("query_id", "")) for row in attempts)

    def percentile(values: Sequence[int], probability: float) -> int | None:
        if not values:
            return None
        ordered = sorted(int(value) for value in values)
        index = max(0, math.ceil(float(probability) * len(ordered)) - 1)
        return ordered[index]

    result = {
        "price_snapshot": dict(snapshot),
        "currency": str(snapshot["currency"]),
        "physical_http_attempts": len(attempts),
        "queries_with_physical_attempts": len(per_query_attempts),
        "retry_attempts": sum(max(0, count - 1) for count in per_query_attempts.values()),
        "attempts_by_price_bucket": dict(sorted(price_bucket_counts.items())),
        "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
        "attempts_per_query": dict(sorted(per_query_attempts.items())),
        "tokens_by_dataset": {
            key: dict(value)
            for key, value in sorted(per_dataset_tokens.items())
        },
        "output_token_distribution": {
            "count": len(output_token_values),
            "median": percentile(output_token_values, 0.50),
            "p90": percentile(output_token_values, 0.90),
            "p95": percentile(output_token_values, 0.95),
            "max": max(output_token_values) if output_token_values else None,
        },
        "observed_tokens": dict(observed),
        "reasoning_token_detail_reported_attempts": (
            reasoning_detail_reported_attempts
        ),
        "reasoning_token_detail_missing_attempts": (
            len(attempts) - reasoning_detail_reported_attempts
        ),
        "reasoning_token_nonzero_attempts": reasoning_detail_nonzero_attempts,
        "usage_incomplete_attempts": unknown_usage_attempts,
        "cost_inexact_attempts": cost_inexact_attempts,
        "cost_is_exact": cost_inexact_attempts == 0,
    }
    if profile.provider == "deepseek":
        result.update(
            {
                "attempts_by_price_period": dict(
                    sorted(price_bucket_counts.items())
                ),
                "known_cost_usd": known_cost,
                "conservative_cost_upper_usd": conservative_cost,
            }
        )
    else:
        flash_responses = _response_index(
            plan.source_run / "llm" / "group_query_checkpoint.jsonl",
            plan,
            expected_model=FLASH_MODEL,
        )
        flash_actual_tokens = sum(
            int(row.get("usage", {}).get("total_tokens", 0) or 0)
            for row in flash_responses.values()
            if isinstance(row.get("usage"), Mapping)
        )
        pro_cost = _read_json(
            plan.project_root
            / "runs"
            / PRO_TRANSFER_RUN_ID
            / "metrics"
            / "api_usage_and_cost.json"
        )
        pro_observed = pro_cost.get("observed_tokens")
        pro_actual_tokens = (
            int(pro_observed.get("total_tokens", 0) or 0)
            if isinstance(pro_observed, Mapping)
            else 0
        )
        qwen_total = int(observed.get("total_tokens", 0))
        result.update(
            {
                "attempts_by_input_price_tier": dict(
                    sorted(price_bucket_counts.items())
                ),
                "known_cost_cny": known_cost,
                "conservative_cost_upper_cny": conservative_cost,
                "reference_actual_tokens": {
                    "deepseek_v4_flash": flash_actual_tokens,
                    "deepseek_v4_pro": pro_actual_tokens,
                },
                "actual_total_token_ratios": {
                    "qwen_over_flash": (
                        qwen_total / flash_actual_tokens
                        if flash_actual_tokens
                        else None
                    ),
                    "qwen_over_deepseek_v4_pro": (
                        qwen_total / pro_actual_tokens
                        if pro_actual_tokens
                        else None
                    ),
                },
            }
        )
    return result


def _model_identity_audit(
    plan: FrozenTransferPlan,
    run_dir: Path,
    responses: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    expected_model = plan.profile.repair_model
    attempts_path = (
        run_dir
        / "llm"
        / plan.profile.model_namespace
        / "api_attempt_audit.jsonl"
    )
    attempts = list(_iter_jsonl(attempts_path)) if attempts_path.is_file() else []
    authorized = {
        (
            str(row["query_id"]),
            str(row["prompt_hash"]),
            str(row["provider_request_hash"]),
        )
        for row in plan.request_rows
    }
    bad_attempts = []
    for row in attempts:
        identity = (
            str(row.get("query_id", "")),
            str(row.get("prompt_hash", "")),
            str(row.get("provider_request_hash", "")),
        )
        returned = str(row.get("returned_model", ""))
        if (
            identity not in authorized
            or row.get("requested_model") != expected_model
            or (returned and returned != expected_model)
            or row.get("status") == "model_identity_error"
        ):
            bad_attempts.append(
                {
                    "query_id": row.get("query_id"),
                    "attempt": row.get("attempt"),
                    "requested_model": row.get("requested_model"),
                    "returned_model": returned,
                    "status": row.get("status"),
                }
            )
    successful = [row for row in responses.values() if row.get("status") == "success"]
    bad_terminal = [
        str(row.get("query_id", ""))
        for row in successful
        if not (
            row.get("model_requested") == expected_model
            and row.get("model_returned") == expected_model
            and row.get("provider_model_field_present") is True
            and row.get("model_matches_request") is True
        )
    ]
    return {
        "requested_model": expected_model,
        "physical_attempts": len(attempts),
        "returned_model_counts": dict(
            sorted(Counter(str(row.get("returned_model", "")) for row in attempts).items())
        ),
        "finish_reason_counts": dict(
            sorted(Counter(str(row.get("finish_reason", "")) for row in attempts).items())
        ),
        "truncated_physical_attempts": sum(
            row.get("finish_reason") == "length" for row in attempts
        ),
        "successful_terminal_queries": len(successful),
        "bad_physical_attempts": bad_attempts,
        "bad_successful_terminal_queries": bad_terminal,
        "ok": not bad_attempts and not bad_terminal,
    }


def finalize_run(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_id: str | None = None,
    profile: str | TransferProfile | None = None,
) -> dict[str, Any]:
    selected_profile = _profile(profile)
    run_dir = _run_dir(project_root, run_id or selected_profile.default_run_id)
    plan = _reload_bound_plan(run_dir, selected_profile)
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest.get("status") not in {"batch_complete", "finalized", "validated"}:
        raise RuntimeError(
            f"finalize is not allowed from status {manifest.get('status')!r}"
        )
    checkpoint = (
        run_dir
        / "llm"
        / plan.profile.model_namespace
        / "group_query_checkpoint.jsonl"
    )
    responses = _response_index(
        checkpoint, plan, expected_model=plan.profile.repair_model
    )
    ledger = _evaluate_ledger(
        plan, responses, method=plan.profile.transfer_method
    )
    flash_by_key = _official_flash_rows(plan)
    flash_rows = list(flash_by_key.values())
    baran_rows = _official_baran_rows(plan)
    pro_rows = _validated_pro_transfer_rows(plan)
    extra_methods = (("pro", pro_rows),) if pro_rows else ()
    summaries = _summary_rows(
        ledger,
        flash_rows,
        baran_rows,
        transfer_label=plan.profile.comparison_method,
        extra_methods=extra_methods,
    )
    deltas = _metric_deltas(
        summaries,
        transfer_label=plan.profile.comparison_method,
        baselines=plan.profile.comparison_baselines,
    )
    diagnostics = _diagnostics(ledger, responses, baran_rows)
    cost = _cost_audit(plan, run_dir)
    diagnostics["physical_retry_attempts"] = cost["retry_attempts"]
    model_audit = _model_identity_audit(plan, run_dir, responses)
    if not model_audit["ok"]:
        raise ValueError("model identity audit failed")
    if plan.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id:
        if diagnostics["truncated_queries"] != 0:
            raise ValueError("Qwen canonical responses include finish_reason=length")
        if diagnostics["partial_responses_used"] != 0:
            raise ValueError("Qwen canonical responses used a partial parse")
        if int(cost["observed_tokens"].get("reasoning_tokens", 0)) != 0:
            raise ValueError("Qwen physical audit includes reasoning tokens")
    expected_ids = {
        (suite, dataset): {
            str(row["cell_id"])
            for row in ledger
            if row["suite"] == suite and row["dataset"] == dataset
        }
        for _, suite, dataset, _, _ in FIXED_DATASETS
    }
    ledger_audit = verify_records(ledger, expected_cell_ids=expected_ids)
    if not ledger_audit.get("ok"):
        raise ValueError("final transfer cell ledger failed independent record audit")

    final_dir = run_dir / "final"
    metrics_dir = run_dir / "metrics"
    _write_jsonl(final_dir / "fixed_eight_cell_ledger.jsonl", ledger)
    _write_csv(metrics_dir / "fixed_eight_metrics.csv", summaries)
    _write_json(metrics_dir / "metric_deltas.json", deltas)
    _write_json(metrics_dir / "diagnostics.json", diagnostics)
    _write_json(metrics_dir / "api_usage_and_cost.json", cost)
    _write_json(metrics_dir / "model_identity_audit.json", model_audit)
    _write_json(metrics_dir / "ledger_audit.json", ledger_audit)
    report = {
        "completed_at_utc": _utc_now(),
        "cell_ledger": str(final_dir / "fixed_eight_cell_ledger.jsonl"),
        "cell_rows": len(ledger),
        "metrics": str(metrics_dir / "fixed_eight_metrics.csv"),
        "diagnostics": diagnostics,
        "cost": cost,
        "model_identity_audit": model_audit,
        "metric_deltas": deltas,
    }
    _write_json(final_dir / "finalize_report.json", report)
    if manifest.get("status") != "validated":
        _update_manifest(run_dir, status="finalized")
    return report


def _write_experiment_markdown(
    run_dir: Path,
    plan: FrozenTransferPlan,
    integrity: Mapping[str, Any],
) -> Path:
    metrics = list(
        csv.DictReader(
            (run_dir / "metrics" / "fixed_eight_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    deltas = _read_json(run_dir / "metrics" / "metric_deltas.json")
    diagnostics = _read_json(run_dir / "metrics" / "diagnostics.json")
    cost = _read_json(run_dir / "metrics" / "api_usage_and_cost.json")
    model_audit = _read_json(run_dir / "metrics" / "model_identity_audit.json")
    dry_run = _read_json(run_dir / "dry_run.json")
    diagnostic_overhead_path = (
        run_dir / "provenance" / "old_qwen_diagnostic_overhead.json"
    )
    diagnostic_overhead = (
        _read_json(diagnostic_overhead_path)
        if diagnostic_overhead_path.is_file()
        else {}
    )
    diagnostic_tokens = diagnostic_overhead.get("observed_tokens", {})
    if not isinstance(diagnostic_tokens, Mapping):
        diagnostic_tokens = {}
    diagnostic_total_tokens = int(diagnostic_tokens.get("total_tokens", 0) or 0)
    diagnostic_cost_cny = float(
        diagnostic_overhead.get("known_itemized_cost_lower_bound_cny", 0.0)
        or 0.0
    )
    current_label = plan.profile.comparison_method
    labels = {dataset: label for label, _, dataset, _, _ in FIXED_DATASETS}
    current_rows = {
        (str(row["scope"]), str(row["dataset"])): row
        for row in metrics
        if row.get("comparison_method") == current_label
    }
    delta_rows = {
        (baseline, str(row["scope"]), str(row["dataset"])): row
        for baseline, rows in deltas.get("comparisons", {}).items()
        for row in rows
        if isinstance(row, Mapping)
    }

    def metric(value: Any) -> str:
        return f"{float(value):.6f}"

    def signed(value: Any) -> str:
        return f"{float(value):+.6f}"

    lines = [
        f"# {plan.profile.experiment_name}",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: `experiment-agent`",
        "- Origin Mode: `run`（已完成独立 `validate`）",
        f"- Origin Date: `{str(integrity['validated_at_utc'])[:10]}`",
        "- Verification Status: `VERIFIED`",
        f"- Version Label: `{QWEN_VERSION_LABEL}`",
        f"- Experiment ID: `{run_dir.name}`",
        f"- Requested model: `{plan.profile.repair_model}`",
        "- Status: `validated`",
        "",
        "> **解释边界：**本实验只替换 repairer，完全冻结 Flash-trained "
        "TabICLv2 Router；不能描述为针对 Qwen 重新训练后的完整 BGR。",
        f"> **论文描述：** `{QWEN_PAPER_DESCRIPTION}`",
        "",
        "## 1. 冻结设定",
        "",
        "- Router：TabICLv2；训练来源：历史 DeepSeek V4 Flash 响应。",
        "- Selection：`variant_4 / 20pct / rho=1 / gamma=1`。",
        "- 八数据集；1,994 queries；1,559 singleton；435 size-4 groups。",
        "- Qwen、Pro 响应及 clean labels 均未参与 Router 选择或预算。",
        "- 20% 是冻结 Flash-reference planning share；Qwen 实际 tokens 单独报告，"
        "不参与选择，也不声称 actual-token parity。",
        "- `Movies_1` 未进入请求、ledger 或 aggregate。",
        "",
        "| 冻结身份 | SHA-256 |",
        "|---|---|",
        f"| Selection 文件集合 | `{plan.identity_audit['selection_file_manifest_sha256']}` |",
        f"| Selected-query 集合 | `{plan.identity_audit['selected_query_set_sha256']}` |",
        f"| Query 顺序 | `{plan.identity_audit['selected_query_ordered_by_dataset_sha256']}` |",
        f"| Group membership | `{plan.identity_audit['selected_membership_identity_sha256']}` |",
        f"| Prompt identity | `{plan.identity_audit['prompt_identity_sha256']}` |",
        f"| Frozen message identity | `{plan.identity_audit['frozen_message_hash']}` |",
        f"| Qwen provider request identity | `{plan.provider_request_hash}` |",
        f"| Qwen request plan | `{plan.request_plan_sha256}` |",
        "",
        "## 2. API 与执行审计",
        "",
        f"- Provider：`{plan.profile.provider}`。",
        f"- Base URL：`{plan.profile.base_url}`。",
        "- Thinking：关闭；temperature `0`；top-p `1`；stream `false`；JSON Object 输出。",
        f"- 请求均显式发送 `enable_thinking=false`；provider 在 "
        f"{cost.get('reasoning_token_detail_reported_attempts', 0)} / "
        f"{cost.get('physical_http_attempts', model_audit['physical_attempts'])} "
        "次物理响应中提供 reasoning-token 明细，"
        f"其余 {cost.get('reasoning_token_detail_missing_attempts', model_audit['physical_attempts'])} "
        "次未报告，因此不声称 "
        "provider-reported reasoning tokens 为 0。",
        "- Qwen 使用 `max_completion_tokens`：singleton `4096`，group-size-4 `16384`；"
        "不发送 `max_tokens`。",
        "- 每 query 最多 5 次重试；preflight checkpoint 在 batch 中复用。",
        f"- 物理 HTTP attempts：{model_audit['physical_attempts']}。",
        f"- 成功终态 queries：{model_audit['successful_terminal_queries']}。",
        f"- 严格成功 / 格式失败终态："
        f"{diagnostics.get('successful_queries', model_audit['successful_terminal_queries'])} / "
        f"{diagnostics.get('terminal_failed_queries', 0)}；"
        "失败响应不进入修复，保留既定 fallback。",
        f"- Returned model 计数：`{json.dumps(model_audit['returned_model_counts'], ensure_ascii=False, sort_keys=True)}`。",
        "",
        "## 3. 每数据集结果",
        "",
        "| 数据集 | Precision | Recall | F1 | ΔF1 vs Flash | ΔF1 vs Baran | ΔF1 vs Pro |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, _, dataset, _, _ in FIXED_DATASETS:
        row = current_rows[("dataset", dataset)]
        values = []
        for baseline in ("flash", "baran", "pro"):
            delta = delta_rows.get((baseline, "dataset", dataset))
            values.append(signed(delta["f1_delta"]) if delta else "—")
        lines.append(
            f"| {labels[dataset]} | {metric(row['precision'])} | "
            f"{metric(row['recall'])} | {metric(row['f1'])} | "
            f"{values[0]} | {values[1]} | {values[2]} |"
        )
    lines.extend(
        [
            "",
            "## 4. Micro / Macro aggregate",
            "",
            "| Scope | System | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for scope, dataset in (("micro", "MICRO"), ("macro", "MACRO")):
        for label in (current_label, "flash", "baran", "pro"):
            row = next(
                item
                for item in metrics
                if item.get("comparison_method") == label
                and item.get("scope") == scope
                and item.get("dataset") == dataset
            )
            lines.append(
                f"| {scope.title()} | {label} | {metric(row['precision'])} | "
                f"{metric(row['recall'])} | {metric(row['f1'])} |"
            )
    lines.extend(
        [
            "",
            "| Baseline | Win / Tie / Loss（八数据集 F1） |",
            "|---|---|",
        ]
    )
    for baseline in plan.profile.comparison_baselines:
        outcome = deltas["win_tie_loss"][baseline]
        lines.append(
            f"| {baseline} | {outcome['win']} / {outcome['tie']} / {outcome['loss']} |"
        )
    lines.extend(
        [
            "",
            "## 5. 诊断",
            "",
            f"- Rescued / harmed / net gain vs Baran：{diagnostics['rescued_cells_vs_baran']} / "
            f"{diagnostics['harmed_cells_vs_baran']} / {diagnostics['net_gain_vs_baran']:+d}。",
            f"- Verifier rejections：{diagnostics['verifier_rejections']}。",
            f"- Parse failures / partial parses / API failures：{diagnostics['parse_failures']} / "
            f"{diagnostics['partial_parse_queries']} / {diagnostics['api_failure_queries']}。",
            f"- 被拒收的物理截断响应：{model_audit.get('truncated_physical_attempts', 0)}；"
            f"最终采用的截断 query：{diagnostics.get('truncated_queries', 0)}。",
            f"- Abstention queries / items：{diagnostics['abstention_queries']} / "
            f"{diagnostics['abstention_items']}。",
            "",
            "## 6. Token 与成本",
            "",
            f"- Input / output / total tokens：{cost['observed_tokens'].get('input_tokens', 0):,} / "
            f"{cost['observed_tokens'].get('output_tokens', 0):,} / "
            f"{cost['observed_tokens'].get('total_tokens', 0):,}。",
            f"- 先前 v2 失败预检诊断开销（不计入正式 v3 指标）："
            f"{diagnostic_total_tokens:,} tokens，¥{diagnostic_cost_cny:.8f}。",
            f"- 本次任务 Qwen API 合计："
            f"{int(cost['observed_tokens'].get('total_tokens', 0)) + diagnostic_total_tokens:,} "
            f"tokens，¥{float(cost.get('known_cost_cny', 0.0)) + diagnostic_cost_cny:.8f}。",
            f"- Cache hit / miss input tokens：{cost['observed_tokens'].get('input_cache_hit_tokens', 0):,} / "
            f"{cost['observed_tokens'].get('input_cache_miss_tokens', 0):,}。",
            f"- 已知成本：¥{float(cost.get('known_cost_cny', 0.0)):.8f}；"
            f"保守上限：¥{float(cost.get('conservative_cost_upper_cny', 0.0)):.8f}。",
            f"- Dry-run 正常一轮全 cache miss 估算：¥{float(dry_run['estimated_cost_cny']['normal_round_all_input_cache_miss']):.8f}。",
            "",
            "## 7. 完整性结论",
            "",
            "- 14,523-row ledger 已独立复算，selection/query/member/prompt hash 前后一致。",
            "- 历史 Flash 与 Pro artifacts 未改写；Qwen cache 与其他模型隔离。",
            "- Requested/returned model 身份审计通过；没有自动回退。",
            "- 本结果是描述性固定 benchmark；未进行随机重复或显著性检验。",
            "",
            "## 8. 主要产物",
            "",
            "- [Run manifest](./run_manifest.json)",
            "- [Dry run](./dry_run.json)",
            "- [Authorized requests](./authorized_requests.jsonl)",
            f"- [Preflight](./llm/{plan.profile.model_namespace}/preflight.json)",
            f"- [API attempt audit](./llm/{plan.profile.model_namespace}/api_attempt_audit.jsonl)",
            f"- [Checkpoint](./llm/{plan.profile.model_namespace}/group_query_checkpoint.jsonl)",
            "- [Final cell ledger](./final/fixed_eight_cell_ledger.jsonl)",
            "- [Metrics](./metrics/fixed_eight_metrics.csv)",
            "- [Metric deltas](./metrics/metric_deltas.json)",
            "- [Diagnostics](./metrics/diagnostics.json)",
            "- [Token and cost audit](./metrics/api_usage_and_cost.json)",
            "- [Model identity audit](./metrics/model_identity_audit.json)",
            "- [Integrity validation](./final/integrity_validation.json)",
            "- [Earlier diagnostic overhead](./provenance/old_qwen_diagnostic_overhead.json)",
            "",
        ]
    )
    target = run_dir / "EXPERIMENT_REPORT.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def validate_run(
    *,
    project_root: str | Path = PROJECT_ROOT,
    run_id: str | None = None,
    profile: str | TransferProfile | None = None,
) -> dict[str, Any]:
    selected_profile = _profile(profile)
    run_dir = _run_dir(project_root, run_id or selected_profile.default_run_id)
    plan = _reload_bound_plan(run_dir, selected_profile)
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest.get("status") not in {"finalized", "validated"}:
        raise RuntimeError(
            f"validate is not allowed from status {manifest.get('status')!r}"
        )
    ledger = list(_iter_jsonl(run_dir / "final" / "fixed_eight_cell_ledger.jsonl"))
    if len(ledger) != EXPECTED_ERROR_CELLS:
        raise ValueError("final cell ledger row count drift")
    allowed = {(suite, dataset) for _, suite, dataset, _, _ in FIXED_DATASETS}
    observed = {(str(row.get("suite")), str(row.get("dataset"))) for row in ledger}
    if observed != allowed or any(row.get("dataset") == "movies_1" for row in ledger):
        raise ValueError("final ledger dataset scope drift")
    stored_metrics = list(
        csv.DictReader(
            (run_dir / "metrics" / "fixed_eight_metrics.csv").open(
                "r", encoding="utf-8", newline=""
            )
        )
    )
    flash_rows = list(_official_flash_rows(plan).values())
    baran_rows = _official_baran_rows(plan)
    pro_rows = _validated_pro_transfer_rows(plan)
    extra_methods = (("pro", pro_rows),) if pro_rows else ()
    recomputed = _summary_rows(
        ledger,
        flash_rows,
        baran_rows,
        transfer_label=plan.profile.comparison_method,
        extra_methods=extra_methods,
    )
    numeric_fields = {
        "budget_share",
        "true_error_cells",
        "predicted_repairs",
        "valid_predictions",
        "invalid_predictions",
        "correct_repairs",
        "annotation_mismatches",
        "precision",
        "recall",
        "f1",
        "correction_accuracy",
    }
    if len(stored_metrics) != len(recomputed):
        raise ValueError("stored metric row count drift")
    for stored, fresh in zip(stored_metrics, recomputed):
        for key, value in fresh.items():
            observed_value: Any = stored.get(key)
            if key in numeric_fields and value is not None:
                if not math.isclose(float(observed_value), float(value), abs_tol=1e-15):
                    raise ValueError(f"stored metric drift for {key}")
            elif str(observed_value) != ("" if value is None else str(value)):
                raise ValueError(f"stored metric identity drift for {key}")
    responses = _response_index(
        run_dir
        / "llm"
        / plan.profile.model_namespace
        / "group_query_checkpoint.jsonl",
        plan,
        expected_model=plan.profile.repair_model,
    )
    model_audit = _model_identity_audit(plan, run_dir, responses)
    if not model_audit["ok"]:
        raise ValueError("model identity re-audit failed")
    diagnostics = _read_json(run_dir / "metrics" / "diagnostics.json")
    if plan.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id and not (
        diagnostics.get("truncated_queries") == 0
        and diagnostics.get("partial_responses_used") == 0
    ):
        raise ValueError("strict Qwen completion-integrity re-audit failed")
    request_plan_hash_after = _canonical_sha256(
        list(_iter_jsonl(run_dir / "authorized_requests.jsonl"))
    )
    integrity = {
        "validated_at_utc": _utc_now(),
        "ok": True,
        "frozen_source_hashes_unchanged": True,
        "selected_query_and_prompt_hashes_unchanged": (
            request_plan_hash_after == plan.request_plan_sha256
        ),
        "metrics_independently_recomputed": True,
        "cell_ledger_rows": len(ledger),
        "movies_1_absent": True,
        "authorized_query_count": len(plan.request_rows),
        "terminal_response_count": len(responses),
        "profile_id": plan.profile.profile_id,
        "repair_model": plan.profile.repair_model,
        "model_identity_audit": model_audit,
        "clean_labels_used_for_router_selection": False,
        "pro_responses_used_for_router_selection": False,
        "qwen_responses_used_for_router_selection": False,
        "router_training_calibration_prediction_selection_executed": False,
        "historical_flash_artifacts_modified": False,
        "historical_pro_artifacts_modified": False,
        "historical_qwen_diagnostic_artifacts_modified": False,
        "finish_reason_length_canonical_responses": diagnostics.get(
            "truncated_queries"
        ),
        "rejected_finish_reason_length_physical_attempts": model_audit.get(
            "truncated_physical_attempts"
        ),
        "partial_responses_used": diagnostics.get("partial_responses_used"),
        "frozen_selection_prompt_hash": plan.identity_audit[
            "frozen_selection_prompt_hash"
        ],
        "frozen_message_hash": plan.identity_audit["frozen_message_hash"],
        "qwen_provider_request_hash": (
            plan.provider_request_hash
            if plan.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id
            else None
        ),
        "request_plan_sha256": plan.request_plan_sha256,
        "request_plan_sha256_after_finalize": request_plan_hash_after,
        "protected_source_hashes_after": dict(plan.protected_source_hashes),
    }
    if not integrity["selected_query_and_prompt_hashes_unchanged"]:
        raise ValueError("authorized request plan changed after finalization")
    if plan.profile.profile_id == QWEN37_FLASH_PROFILE.profile_id:
        report_path = _write_experiment_markdown(run_dir, plan, integrity)
        integrity["experiment_report"] = str(report_path)
    _write_json(run_dir / "final" / "integrity_validation.json", integrity)
    _update_manifest(run_dir, status="validated", validation_ok=True)
    return integrity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen Flash-router closed-profile repairer transfer experiment"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(TRANSFER_PROFILES)),
        default=DEFAULT_PROFILE_ID,
    )
    parser.add_argument("--run-id")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="offline authorization and Flash replay only")
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument(
            "--env-file",
            type=Path,
        )
        command.add_argument("--approve-paid-api", action="store_true")
        if name == "run":
            command.add_argument("--resume", action="store_true")
    commands.add_parser("finalize")
    commands.add_parser("validate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected_profile = _profile(args.profile)
    common = {
        "project_root": args.project_root,
        "run_id": args.run_id or selected_profile.default_run_id,
        "profile": selected_profile,
    }
    if args.command == "plan":
        result = create_dry_run(**common)
    elif args.command == "preflight":
        env_file = args.env_file or (
            PROJECT_ROOT.parent / selected_profile.default_env_filename
        )
        result = run_preflight(
            **common,
            env_file=env_file,
            approve_paid_api=args.approve_paid_api,
        )
    elif args.command == "run":
        env_file = args.env_file or (
            PROJECT_ROOT.parent / selected_profile.default_env_filename
        )
        result = run_authorized_batch(
            **common,
            env_file=env_file,
            resume=args.resume,
            approve_paid_api=args.approve_paid_api,
        )
    elif args.command == "finalize":
        result = finalize_run(**common)
    else:
        result = validate_run(**common)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
