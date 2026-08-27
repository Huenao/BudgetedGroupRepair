"""DeepSeek JSON client and append-only group-query checkpoint ledger."""

from __future__ import annotations

import hashlib
import json
import math
import random
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from .group_context import (
    CanonicalMessages,
    PROMPT_SCHEMA_VERSION,
    canonical_messages,
    compute_prompt_hash,
    messages_as_dicts,
)


JSONDict = dict[str, Any]


class GroupLLMError(RuntimeError):
    """Base error whose message is safe for the experiment ledger."""


class RetryableGroupLLMError(GroupLLMError):
    pass


class PermanentGroupLLMError(GroupLLMError):
    pass


class ProviderModelIdentityError(PermanentGroupLLMError):
    """The provider omitted or changed the requested model identity."""

    def __init__(
        self,
        *,
        model_requested: str,
        model_returned: str,
        model_field_present: bool,
    ) -> None:
        self.model_requested = str(model_requested)
        self.model_returned = str(model_returned)
        self.model_field_present = bool(model_field_present)
        if not self.model_field_present or not self.model_returned:
            message = "model endpoint response omitted a usable model identity"
        else:
            message = (
                "model endpoint identity mismatch: requested "
                f"{self.model_requested!r}, returned {self.model_returned!r}"
            )
        super().__init__(message)


@dataclass(frozen=True)
class ParsedRepairItem:
    cell_id: str
    repair: str
    confidence: float
    decision: str
    evidence: str
    affected_constraints: tuple[str, ...] = ()

    def as_dict(self) -> JSONDict:
        return {
            "cell_id": self.cell_id,
            "repair": self.repair,
            "confidence": self.confidence,
            "decision": self.decision,
            "evidence": self.evidence,
            "affected_constraints": list(self.affected_constraints),
        }


@dataclass(frozen=True)
class GroupParseResult:
    query_id: str
    expected_cell_ids: tuple[str, ...]
    items: tuple[ParsedRepairItem, ...]
    parse_status: str
    missing_cell_ids: tuple[str, ...] = ()
    unknown_cell_ids: tuple[str, ...] = ()
    duplicate_cell_ids: tuple[str, ...] = ()
    invalid_items: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "invalid_items",
            tuple(MappingProxyType(dict(item)) for item in self.invalid_items),
        )

    @property
    def item_by_cell(self) -> dict[str, ParsedRepairItem]:
        return {item.cell_id: item for item in self.items}

    @property
    def is_complete(self) -> bool:
        return self.parse_status == "ok"

    @property
    def has_valid_items(self) -> bool:
        return bool(self.items)

    def as_dict(self) -> JSONDict:
        return {
            "query_id": self.query_id,
            "expected_cell_ids": list(self.expected_cell_ids),
            "items": [item.as_dict() for item in self.items],
            "parse_status": self.parse_status,
            "missing_cell_ids": list(self.missing_cell_ids),
            "unknown_cell_ids": list(self.unknown_cell_ids),
            "duplicate_cell_ids": list(self.duplicate_cell_ids),
            "invalid_items": [dict(item) for item in self.invalid_items],
        }


def _first_json_object(text: str) -> Mapping[str, Any] | None:
    decoder = json.JSONDecoder()
    start = str(text).find("{")
    while start >= 0:
        try:
            value, _ = decoder.raw_decode(str(text)[start:])
        except json.JSONDecodeError:
            start = str(text).find("{", start + 1)
            continue
        if isinstance(value, Mapping):
            return value
        start = str(text).find("{", start + 1)
    return None


def _invalid(index: int, cell_id: str, status: str) -> Mapping[str, Any]:
    return {"index": int(index), "cell_id": str(cell_id), "status": str(status)}


def _parse_item(raw: Mapping[str, Any], index: int) -> tuple[ParsedRepairItem | None, Mapping[str, Any] | None]:
    cell_id = str(raw.get("cell_id", "")).strip()
    required = ("repair", "confidence", "decision", "evidence")
    missing = [key for key in required if key not in raw]
    if missing:
        return None, _invalid(index, cell_id, "missing_fields:" + ",".join(missing))
    if not isinstance(raw.get("repair"), str):
        return None, _invalid(index, cell_id, "invalid_repair")
    confidence_value = raw.get("confidence")
    if isinstance(confidence_value, bool):
        return None, _invalid(index, cell_id, "invalid_confidence")
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        return None, _invalid(index, cell_id, "invalid_confidence")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None, _invalid(index, cell_id, "confidence_out_of_range")
    decision = str(raw.get("decision", "")).strip().lower()
    if decision not in {"propose", "abstain"}:
        return None, _invalid(index, cell_id, "invalid_decision")
    if not isinstance(raw.get("evidence"), str):
        return None, _invalid(index, cell_id, "invalid_evidence")
    affected = raw.get("affected_constraints", [])
    if affected is None:
        affected = []
    if not isinstance(affected, list) or any(not isinstance(value, str) for value in affected):
        return None, _invalid(index, cell_id, "invalid_affected_constraints")
    return (
        ParsedRepairItem(
            cell_id=cell_id,
            repair=str(raw["repair"]),
            confidence=confidence,
            decision=decision,
            evidence=str(raw["evidence"]),
            affected_constraints=tuple(value for value in affected if value.strip()),
        ),
        None,
    )


def parse_group_response(
    text: str,
    expected_query_id: str,
    expected_cell_ids: Sequence[str],
) -> GroupParseResult:
    """Parse independently valid items while rejecting identity ambiguity."""

    expected = tuple(sorted(str(identifier) for identifier in expected_cell_ids))
    expected_set = set(expected)
    payload = _first_json_object(str(text))
    if payload is None:
        return GroupParseResult(
            query_id="",
            expected_cell_ids=expected,
            items=(),
            parse_status="no_json_object",
            missing_cell_ids=expected,
        )
    actual_query_id = str(payload.get("query_id", ""))
    if actual_query_id != str(expected_query_id):
        return GroupParseResult(
            query_id=actual_query_id,
            expected_cell_ids=expected,
            items=(),
            parse_status="query_id_mismatch",
            missing_cell_ids=expected,
        )
    repairs = payload.get("repairs")
    if not isinstance(repairs, list):
        return GroupParseResult(
            query_id=actual_query_id,
            expected_cell_ids=expected,
            items=(),
            parse_status="invalid_repairs_array",
            missing_cell_ids=expected,
        )

    valid: dict[str, ParsedRepairItem] = {}
    occurrences: Counter[str] = Counter()
    unknown: set[str] = set()
    duplicates: set[str] = set()
    invalid_items: list[Mapping[str, Any]] = []
    for index, raw in enumerate(repairs):
        if not isinstance(raw, Mapping):
            invalid_items.append(_invalid(index, "", "item_not_object"))
            continue
        cell_id = str(raw.get("cell_id", "")).strip()
        if not cell_id:
            invalid_items.append(_invalid(index, "", "missing_cell_id"))
            continue
        if cell_id not in expected_set:
            unknown.add(cell_id)
            invalid_items.append(_invalid(index, cell_id, "unknown_cell_id"))
            continue
        occurrences[cell_id] += 1
        if occurrences[cell_id] > 1:
            duplicates.add(cell_id)
            valid.pop(cell_id, None)
            invalid_items.append(_invalid(index, cell_id, "duplicate_cell_id"))
            continue
        item, invalid = _parse_item(raw, index)
        if invalid is not None:
            invalid_items.append(invalid)
        elif item is not None:
            valid[cell_id] = item

    # A duplicated identity invalidates every occurrence, including a valid
    # first item, because there is no safe deterministic choice between them.
    for cell_id in duplicates:
        valid.pop(cell_id, None)
    items = tuple(valid[cell_id] for cell_id in sorted(valid))
    missing = tuple(sorted(expected_set.difference(valid)))
    if (
        len(items) == len(expected)
        and not unknown
        and not duplicates
        and not invalid_items
    ):
        status = "ok"
    elif items:
        status = "partial"
    else:
        status = "no_valid_items"
    return GroupParseResult(
        query_id=actual_query_id,
        expected_cell_ids=expected,
        items=items,
        parse_status=status,
        missing_cell_ids=missing,
        unknown_cell_ids=tuple(sorted(unknown)),
        duplicate_cell_ids=tuple(sorted(duplicates)),
        invalid_items=tuple(invalid_items),
    )


@dataclass(frozen=True)
class GroupClientConfig:
    """Non-secret OpenAI-compatible endpoint settings."""

    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.0
    top_p: float = 1.0
    timeout_seconds: float = 120.0
    max_retries: int = 3
    backoff_initial_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    backoff_jitter: float = 0.2
    concurrency: int = 4
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        reserved = {"model", "messages", "max_tokens", "response_format"}
        overlap = reserved.intersection(self.extra_body)
        if overlap:
            raise ValueError("extra_body cannot override request identity fields: " + ",".join(sorted(overlap)))
        object.__setattr__(self, "extra_body", MappingProxyType(dict(self.extra_body)))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "GroupClientConfig":
        allowed = {
            "base_url",
            "model",
            "temperature",
            "top_p",
            "timeout_seconds",
            "max_retries",
            "backoff_initial_seconds",
            "backoff_max_seconds",
            "backoff_jitter",
            "concurrency",
            "extra_body",
        }
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass(frozen=True)
class GroupLLMJob:
    query_id: str
    messages: CanonicalMessages
    prompt_hash: str
    expected_cell_ids: tuple[str, ...]
    max_tokens: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.query_id):
            raise ValueError("query_id must not be empty")
        expected = tuple(sorted(str(identifier) for identifier in self.expected_cell_ids))
        if not expected or len(set(expected)) != len(expected):
            raise ValueError("expected_cell_ids must be non-empty and unique")
        if int(self.max_tokens) <= 0:
            raise ValueError("max_tokens must be positive")
        if not str(self.prompt_hash):
            raise ValueError("prompt_hash must not be empty")
        object.__setattr__(self, "messages", canonical_messages(self.messages))
        object.__setattr__(self, "expected_cell_ids", expected)
        object.__setattr__(self, "max_tokens", int(self.max_tokens))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_action(
        cls,
        action: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "GroupLLMJob":
        return cls(
            query_id=str(action.query_id),
            messages=action.messages,
            prompt_hash=str(action.prompt_hash),
            expected_cell_ids=tuple(action.cell_ids),
            max_tokens=int(action.completion_token_ceiling),
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class GroupLLMResult:
    content: str
    parsed: GroupParseResult
    model: str
    usage: Mapping[str, Any]
    latency_seconds: float
    prompt_hash: str
    provider_request_hash: str
    attempts: int
    response_id: str = ""
    usage_observed_attempts: int = 0
    unknown_usage_attempts: int = 0
    observed_total_tokens: int = 0
    model_requested: str = ""
    provider_model_field_present: bool = False

    @property
    def model_returned(self) -> str:
        """Actual provider model ID; empty means that none was returned."""

        return self.model

    @property
    def model_returned_present(self) -> bool:
        return bool(self.provider_model_field_present and self.model_returned)


class DeepSeekGroupClient:
    """Small JSON-mode client with an injected credential and finite retry."""

    RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        config: GroupClientConfig,
        *,
        api_key: str,
        opener: Callable[..., Any] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        self.config = config
        self._api_key = str(api_key)
        self._opener = opener or urllib.request.urlopen
        self._sleep = sleep_fn
        self._random = random_fn

    @property
    def endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    def provider_request_hash(self, job: GroupLLMJob) -> str:
        payload = {
            "base_url": self.config.base_url.rstrip("/"),
            "model": self.config.model,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": job.max_tokens,
            "extra_body": dict(self.config.extra_body),
            "messages": messages_as_dicts(job.messages),
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _payload(self, job: GroupLLMJob) -> JSONDict:
        payload: JSONDict = dict(self.config.extra_body)
        payload.update(
            {
                "model": self.config.model,
                "messages": messages_as_dicts(job.messages),
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": int(job.max_tokens),
                "response_format": {"type": "json_object"},
            }
        )
        return payload

    def _request_once(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self._api_key:
            raise PermanentGroupLLMError("an API credential must be injected by the caller")
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._api_key,
            },
            method="POST",
        )
        try:
            response_context = self._opener(request, timeout=self.config.timeout_seconds)
            with response_context as response:
                response_text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            message = f"HTTP {error.code} from model endpoint"
            if error.code in self.RETRYABLE_HTTP_STATUS:
                raise RetryableGroupLLMError(message) from error
            raise PermanentGroupLLMError(message) from error
        except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as error:
            raise RetryableGroupLLMError("model endpoint connection failed") from error
        try:
            decoded = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise RetryableGroupLLMError("model endpoint returned non-JSON data") from error
        if not isinstance(decoded, Mapping):
            raise RetryableGroupLLMError("model endpoint returned a non-object response")
        return decoded

    @staticmethod
    def _response_content(raw: Mapping[str, Any]) -> str:
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RetryableGroupLLMError("unexpected chat completion response shape") from error
        if isinstance(content, str):
            return content
        if isinstance(content, Mapping):
            return json.dumps(dict(content), ensure_ascii=False)
        raise RetryableGroupLLMError("chat completion content is not JSON text")

    def _provider_model(self, raw: Mapping[str, Any]) -> tuple[str, bool]:
        field_present = "model" in raw
        value = raw.get("model")
        returned = value if isinstance(value, str) else ""
        if not field_present or not returned:
            raise ProviderModelIdentityError(
                model_requested=self.config.model,
                model_returned=returned,
                model_field_present=field_present,
            )
        if returned != self.config.model:
            raise ProviderModelIdentityError(
                model_requested=self.config.model,
                model_returned=returned,
                model_field_present=True,
            )
        return returned, True

    @staticmethod
    def _merge_usage(target: JSONDict, source: Mapping[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                target[str(key)] = target.get(str(key), 0) + value

    def _backoff(self, failed_attempt: int) -> None:
        base = min(
            self.config.backoff_max_seconds,
            self.config.backoff_initial_seconds * (2 ** max(0, failed_attempt - 1)),
        )
        if base > 0:
            self._sleep(base * (1.0 + self.config.backoff_jitter * self._random()))

    def chat(self, job: GroupLLMJob) -> GroupLLMResult:
        provider_hash = self.provider_request_hash(job)
        started = time.monotonic()
        usage_total: JSONDict = {}
        attempts = self.config.max_retries + 1
        last_parsed: GroupParseResult | None = None
        last_content = ""
        last_raw: Mapping[str, Any] = {}
        last_model_returned = ""
        last_model_field_present = False
        last_error: RetryableGroupLLMError | None = None
        usage_observed_attempts = 0
        unknown_usage_attempts = 0
        observed_total_tokens = 0
        for attempt in range(1, attempts + 1):
            usage_observed_this_attempt = False
            usage_status_recorded = False
            try:
                raw = self._request_once(self._payload(job))
                last_raw = raw
                model_returned, model_field_present = self._provider_model(raw)
                last_model_returned = model_returned
                last_model_field_present = model_field_present
                usage = raw.get("usage") or {}
                if isinstance(usage, Mapping):
                    self._merge_usage(usage_total, usage)
                    numeric = {
                        str(key)
                        for key, value in usage.items()
                        if isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                    }
                    usage_observed_this_attempt = (
                        "total_tokens" in numeric
                        or bool(numeric.intersection({"prompt_tokens", "input_tokens"}))
                        and bool(
                            numeric.intersection(
                                {"completion_tokens", "output_tokens"}
                            )
                        )
                    )
                    if usage_observed_this_attempt:
                        if "total_tokens" in numeric:
                            observed_total_tokens += int(usage["total_tokens"])
                        else:
                            prompt_key = (
                                "prompt_tokens"
                                if "prompt_tokens" in numeric
                                else "input_tokens"
                            )
                            completion_key = (
                                "completion_tokens"
                                if "completion_tokens" in numeric
                                else "output_tokens"
                            )
                            observed_total_tokens += int(usage[prompt_key]) + int(
                                usage[completion_key]
                            )
                if usage_observed_this_attempt:
                    usage_observed_attempts += 1
                else:
                    unknown_usage_attempts += 1
                usage_status_recorded = True
                content = self._response_content(raw)
                parsed = parse_group_response(content, job.query_id, job.expected_cell_ids)
                last_content = content
                last_parsed = parsed
                if parsed.parse_status in {"ok", "partial"} or attempt == attempts:
                    return GroupLLMResult(
                        content=content,
                        parsed=parsed,
                        model=model_returned,
                        usage=MappingProxyType(dict(usage_total)),
                        latency_seconds=time.monotonic() - started,
                        prompt_hash=job.prompt_hash,
                        provider_request_hash=provider_hash,
                        attempts=attempt,
                        response_id=str(raw.get("id") or ""),
                        usage_observed_attempts=usage_observed_attempts,
                        unknown_usage_attempts=unknown_usage_attempts,
                        observed_total_tokens=observed_total_tokens,
                        model_requested=self.config.model,
                        provider_model_field_present=model_field_present,
                    )
                last_error = RetryableGroupLLMError(
                    "invalid structured group response: " + parsed.parse_status
                )
            except RetryableGroupLLMError as error:
                last_error = error
                if not usage_status_recorded:
                    unknown_usage_attempts += 1
            if attempt < attempts:
                self._backoff(attempt)
        if last_parsed is not None:
            return GroupLLMResult(
                content=last_content,
                parsed=last_parsed,
                model=last_model_returned,
                usage=MappingProxyType(dict(usage_total)),
                latency_seconds=time.monotonic() - started,
                prompt_hash=job.prompt_hash,
                provider_request_hash=provider_hash,
                attempts=attempts,
                response_id=str(last_raw.get("id") or ""),
                usage_observed_attempts=usage_observed_attempts,
                unknown_usage_attempts=unknown_usage_attempts,
                observed_total_tokens=observed_total_tokens,
                model_requested=self.config.model,
                provider_model_field_present=last_model_field_present,
            )
        if usage_total or usage_observed_attempts:
            return GroupLLMResult(
                content=last_content,
                parsed=parse_group_response(
                    last_content, job.query_id, job.expected_cell_ids
                ),
                model=last_model_returned,
                usage=MappingProxyType(dict(usage_total)),
                latency_seconds=time.monotonic() - started,
                prompt_hash=job.prompt_hash,
                provider_request_hash=provider_hash,
                attempts=attempts,
                response_id=str(last_raw.get("id") or ""),
                usage_observed_attempts=usage_observed_attempts,
                unknown_usage_attempts=unknown_usage_attempts,
                observed_total_tokens=observed_total_tokens,
                model_requested=self.config.model,
                provider_model_field_present=last_model_field_present,
            )
        assert last_error is not None
        raise RetryableGroupLLMError(
            f"request failed after {attempts} attempts: {last_error}"
        ) from last_error


@dataclass(frozen=True)
class _IndexedJob:
    index: int
    job: GroupLLMJob


def _normalise_jobs(jobs: Iterable[Any]) -> list[_IndexedJob]:
    normalised: list[_IndexedJob] = []
    for index, raw in enumerate(jobs):
        if isinstance(raw, GroupLLMJob):
            job = raw
        elif hasattr(raw, "query_id") and hasattr(raw, "completion_token_ceiling"):
            job = GroupLLMJob.from_action(raw)
        elif isinstance(raw, Mapping):
            messages = raw.get("messages")
            if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
                raise TypeError("group job messages must be a sequence")
            max_tokens = int(raw.get("max_tokens", raw.get("completion_token_ceiling", 0)))
            prompt_digest = str(raw.get("prompt_hash", ""))
            if not prompt_digest:
                prompt_digest = compute_prompt_hash(
                    messages,
                    max_tokens,
                    prompt_schema_version=str(raw.get("prompt_schema_version", PROMPT_SCHEMA_VERSION)),
                )
            metadata = raw.get("metadata", {})
            job = GroupLLMJob(
                query_id=str(raw.get("query_id", "")),
                messages=canonical_messages(messages),
                prompt_hash=prompt_digest,
                expected_cell_ids=tuple(raw.get("expected_cell_ids", raw.get("cell_ids", ()))),
                max_tokens=max_tokens,
                metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
            )
        else:
            raise TypeError("each group job must be an action, GroupLLMJob, or mapping")
        normalised.append(_IndexedJob(index=index, job=job))
    return normalised


def _read_jsonl(path: Path) -> list[JSONDict]:
    if not path.exists():
        return []
    rows: list[JSONDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any], lock: threading.Lock) -> None:
    encoded = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()


def _record_from_result(job: GroupLLMJob, result: GroupLLMResult, *, cache_hit: bool) -> JSONDict:
    parsed = result.parsed
    require_complete = bool(job.metadata.get("require_complete_response"))
    model_requested = str(
        job.metadata.get("model_requested") or result.model_requested
    )
    model_matches = bool(
        result.model_returned_present
        and model_requested
        and result.model_returned == model_requested
    )
    successful = model_matches and (
        parsed.is_complete
        if require_complete
        else parsed.parse_status in {"ok", "partial"} and parsed.has_valid_items
    )
    return {
        "query_id": job.query_id,
        "prompt_hash": job.prompt_hash,
        "provider_request_hash": result.provider_request_hash,
        "status": "success" if successful else "failed",
        "retryable": not successful,
        "parse_status": parsed.parse_status,
        "items": [item.as_dict() for item in parsed.items],
        "missing_cell_ids": list(parsed.missing_cell_ids),
        "unknown_cell_ids": list(parsed.unknown_cell_ids),
        "duplicate_cell_ids": list(parsed.duplicate_cell_ids),
        "invalid_items": [dict(item) for item in parsed.invalid_items],
        "response_text": result.content,
        "model": result.model_returned,
        "model_requested": model_requested,
        "model_returned": result.model_returned,
        "provider_model_field_present": result.provider_model_field_present,
        "model_returned_present": result.model_returned_present,
        "model_matches_request": model_matches,
        "usage": dict(result.usage),
        "latency_seconds": result.latency_seconds,
        "attempts": result.attempts,
        "usage_observed_attempts": result.usage_observed_attempts,
        "unknown_usage_attempts": result.unknown_usage_attempts,
        "observed_total_tokens": result.observed_total_tokens,
        "cache_hit": bool(cache_hit),
        "checkpoint_hit": False,
        "response_id": result.response_id,
        "max_tokens": job.max_tokens,
        "cell_ids": list(job.expected_cell_ids),
        "metadata": dict(job.metadata),
    }


def _cache_row(result: GroupLLMResult, job: GroupLLMJob) -> JSONDict:
    return {
        "prompt_hash": result.prompt_hash,
        "provider_request_hash": result.provider_request_hash,
        "content": result.content,
        "model": result.model_returned,
        "model_requested": result.model_requested,
        "model_returned": result.model_returned,
        "provider_model_field_present": result.provider_model_field_present,
        "model_returned_present": result.model_returned_present,
        "usage": dict(result.usage),
        "latency_seconds": result.latency_seconds,
        "attempts": result.attempts,
        "usage_observed_attempts": result.usage_observed_attempts,
        "unknown_usage_attempts": result.unknown_usage_attempts,
        "observed_total_tokens": result.observed_total_tokens,
        "response_id": result.response_id,
        "metadata": dict(job.metadata),
    }


def _result_from_cache(row: Mapping[str, Any], job: GroupLLMJob) -> GroupLLMResult | None:
    content = str(row.get("content", ""))
    parsed = parse_group_response(content, job.query_id, job.expected_cell_ids)
    require_complete = bool(job.metadata.get("require_complete_response"))
    valid = (
        parsed.is_complete
        if require_complete
        else parsed.parse_status in {"ok", "partial"} and parsed.has_valid_items
    )
    if not valid:
        return None
    model_returned = str(row.get("model_returned", row.get("model", "")))
    presence_marker = row.get("provider_model_field_present")
    # Legacy same-run cache rows predate the explicit marker.  Their non-empty
    # model value remains readable, while every newly written row records the
    # provider field separately and never substitutes the requested model.
    model_field_present = (
        bool(model_returned) if presence_marker is None else bool(presence_marker)
    )
    return GroupLLMResult(
        content=content,
        parsed=parsed,
        model=model_returned,
        usage=MappingProxyType(dict(row.get("usage") or {})),
        latency_seconds=float(row.get("latency_seconds", 0.0) or 0.0),
        prompt_hash=job.prompt_hash,
        provider_request_hash=str(row.get("provider_request_hash", "")),
        attempts=int(row.get("attempts", 0) or 0),
        response_id=str(row.get("response_id", "")),
        usage_observed_attempts=int(row.get("usage_observed_attempts", 0) or 0),
        unknown_usage_attempts=int(row.get("unknown_usage_attempts", 0) or 0),
        observed_total_tokens=int(row.get("observed_total_tokens", 0) or 0),
        model_requested=str(row.get("model_requested", "")),
        provider_model_field_present=model_field_present,
    )


def run_group_llm_batch(
    client: DeepSeekGroupClient,
    jobs: Iterable[Any],
    run_dir: str | Path,
    *,
    concurrency: int | None = None,
    retry_failed: bool = True,
) -> list[JSONDict]:
    """Execute unique physical requests and resume by query_id + prompt_hash."""

    normalised = _normalise_jobs(jobs)
    if not normalised:
        return []
    run_path = Path(run_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    cache_path = run_path / "group_response_cache.jsonl"
    checkpoint_path = run_path / "group_query_checkpoint.jsonl"
    lock = threading.Lock()

    def phase_of(row: Mapping[str, Any]) -> str:
        metadata = row.get("metadata")
        return str(metadata.get("phase", "")) if isinstance(metadata, Mapping) else ""

    def phase_compatible(source: str, target: str) -> bool:
        if source == target:
            return True
        if source == "model_preflight" and target == "preliminary_singleton":
            return True
        preliminary = source in {
            "preliminary_singleton",
            "preliminary_structured",
            "preliminary_random",
        }
        return preliminary and target in {
            "bgr_selected_union",
            "offline_group_calibration",
            "online_selected_union",
        }

    checkpoints: dict[tuple[str, str], list[JSONDict]] = defaultdict(list)
    known_hash_by_query: dict[str, str] = {}
    for row in _read_jsonl(checkpoint_path):
        query_id = str(row.get("query_id", ""))
        prompt_digest = str(row.get("prompt_hash", ""))
        if not query_id or not prompt_digest:
            continue
        previous_hash = known_hash_by_query.setdefault(query_id, prompt_digest)
        if previous_hash != prompt_digest:
            raise ValueError(f"checkpoint contains prompt drift for query_id {query_id}")
        checkpoints[(query_id, prompt_digest)].append(row)

    input_hash_by_query: dict[str, str] = {}
    for indexed in normalised:
        previous_hash = input_hash_by_query.setdefault(indexed.job.query_id, indexed.job.prompt_hash)
        if previous_hash != indexed.job.prompt_hash:
            raise ValueError(f"input contains prompt drift for query_id {indexed.job.query_id}")
        checkpoint_hash = known_hash_by_query.get(indexed.job.query_id)
        if checkpoint_hash is not None and checkpoint_hash != indexed.job.prompt_hash:
            raise ValueError(f"checkpoint prompt drift for query_id {indexed.job.query_id}")

    cache: dict[str, list[JSONDict]] = defaultdict(list)
    for row in _read_jsonl(cache_path):
        if row.get("provider_request_hash"):
            cache[str(row.get("provider_request_hash"))].append(row)
    output: dict[int, JSONDict] = {}
    pending: dict[tuple[str, str], list[_IndexedJob]] = defaultdict(list)
    for indexed in normalised:
        job = indexed.job
        target_phase = str(job.metadata.get("phase", ""))
        provider_request_hash = client.provider_request_hash(job)
        compatible_checkpoints = [
            row
            for row in checkpoints.get((job.query_id, job.prompt_hash), [])
            if phase_compatible(phase_of(row), target_phase)
            and str(row.get("provider_request_hash", "")) == provider_request_hash
        ]
        checkpoint = compatible_checkpoints[-1] if compatible_checkpoints else None
        if checkpoint is not None and (checkpoint.get("status") == "success" or not retry_failed):
            resumed = dict(checkpoint)
            resumed["checkpoint_hit"] = True
            output[indexed.index] = resumed
            continue
        compatible_cache = [
            row
            for row in cache.get(provider_request_hash, [])
            if phase_compatible(phase_of(row), target_phase)
        ]
        cached = _result_from_cache(compatible_cache[-1], job) if compatible_cache else None
        if cached is not None:
            record = _record_from_result(job, cached, cache_hit=True)
            _append_jsonl(checkpoint_path, record, lock)
            output[indexed.index] = record
            continue
        pending[(job.query_id, job.prompt_hash)].append(indexed)

    worker_count = max(1, int(concurrency or client.config.concurrency))
    future_to_key: dict[Any, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for key, grouped in pending.items():
            future_to_key[pool.submit(client.chat, grouped[0].job)] = key
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            grouped = pending[key]
            representative = grouped[0].job
            try:
                result = future.result()
            except Exception as error:
                retryable = isinstance(error, RetryableGroupLLMError)
                identity_error = (
                    error if isinstance(error, ProviderModelIdentityError) else None
                )
                for indexed in grouped:
                    job = indexed.job
                    model_requested = str(
                        job.metadata.get("model_requested") or client.config.model
                    )
                    model_returned = (
                        identity_error.model_returned if identity_error is not None else ""
                    )
                    model_field_present = bool(
                        identity_error is not None
                        and identity_error.model_field_present
                    )
                    record = {
                        "query_id": job.query_id,
                        "prompt_hash": job.prompt_hash,
                        "provider_request_hash": client.provider_request_hash(job),
                        "status": "failed",
                        "retryable": retryable,
                        "parse_status": "llm_error",
                        "items": [],
                        "missing_cell_ids": list(job.expected_cell_ids),
                        "unknown_cell_ids": [],
                        "duplicate_cell_ids": [],
                        "invalid_items": [],
                        "response_text": "",
                        "model": model_returned,
                        "model_requested": model_requested,
                        "model_returned": model_returned,
                        "provider_model_field_present": model_field_present,
                        "model_returned_present": bool(
                            model_field_present and model_returned
                        ),
                        "model_matches_request": False,
                        "usage": {},
                        "latency_seconds": 0.0,
                        "attempts": client.config.max_retries + 1 if retryable else 1,
                        "usage_observed_attempts": 0,
                        "unknown_usage_attempts": (
                            client.config.max_retries + 1 if retryable else 1
                        ),
                        "observed_total_tokens": 0,
                        "cache_hit": False,
                        "checkpoint_hit": False,
                        "exception_class": type(error).__name__,
                        "error": str(error)[:500],
                        "max_tokens": job.max_tokens,
                        "cell_ids": list(job.expected_cell_ids),
                        "metadata": dict(job.metadata),
                    }
                    _append_jsonl(checkpoint_path, record, lock)
                    output[indexed.index] = record
                if identity_error is not None:
                    # Identity drift invalidates the experiment rather than a
                    # single query.  Cancel work that has not begun and let the
                    # permanent error abort the caller after in-flight workers
                    # have left the executor safely.
                    for other_future in future_to_key:
                        if other_future is not future:
                            other_future.cancel()
                    raise identity_error
                continue

            emitted: list[JSONDict] = []
            for indexed in grouped:
                record = _record_from_result(indexed.job, result, cache_hit=False)
                _append_jsonl(checkpoint_path, record, lock)
                output[indexed.index] = record
                emitted.append(record)
            # Persist the cost-bearing checkpoint before the convenience cache
            # so a crash can never turn a paid response into an uncharged hit.
            if emitted and all(row.get("status") == "success" for row in emitted):
                cache_record = _cache_row(result, representative)
                _append_jsonl(cache_path, cache_record, lock)
                cache[result.provider_request_hash].append(cache_record)

    return [output[index] for index in range(len(normalised))]


# Compatibility aliases for callers that use the shorter names.
ClientConfig = GroupClientConfig
DeepSeekClient = DeepSeekGroupClient
LLMResult = GroupLLMResult
parse_repair_response = parse_group_response
run_llm_batch = run_group_llm_batch


__all__ = [
    "ClientConfig",
    "DeepSeekClient",
    "DeepSeekGroupClient",
    "GroupClientConfig",
    "GroupLLMError",
    "GroupLLMJob",
    "GroupLLMResult",
    "GroupParseResult",
    "LLMResult",
    "ParsedRepairItem",
    "PermanentGroupLLMError",
    "ProviderModelIdentityError",
    "RetryableGroupLLMError",
    "parse_group_response",
    "parse_repair_response",
    "run_group_llm_batch",
    "run_llm_batch",
]
