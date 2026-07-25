"""No-Baran prompt policy and recursive leakage audits."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


INFORMATION_POLICY = "dirty_evidence_only_no_baran"
PROMPT_SCHEMA_VERSION = "bgr-no-baran-v1"

FORBIDDEN_PROMPT_FIELDS = frozenset(
    {
        "baran",
        "baran_candidate",
        "baran_prediction",
        "candidate_support",
        "corrector_support",
        "source_agreement",
        "baran_confidence",
        "baran_correct",
        "clean",
        "clean_value",
        "right_value",
        "correct_repair",
        "llm_correct",
        "error_type",
        "missing_value",
        "tuple_pairs",
    }
)

_FORBIDDEN_TEXT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"baran[_ -]?(candidate|prediction|support|confidence|correct)",
        r"clean[_ -]?value",
        r"right[_ -]?value",
        r"correct[_ -]?repair",
        r"llm[_ -]?correct",
        r"candidate[_ -]?support",
        r"corrector[_ -]?support",
        r"source[_ -]?agreement",
        r"error[_ -]?type",
        r"missing[_ -]?value",
        r"tuple[_ -]?pairs",
    )
)


class PromptPolicyError(ValueError):
    """Raised when an action would expose forbidden information to the LLM."""


def _walk_keys(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], str]]:
    found: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().lower()
            next_path = (*path, str(raw_key))
            if key in FORBIDDEN_PROMPT_FIELDS:
                found.append((next_path, key))
            found.extend(_walk_keys(nested, next_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            found.extend(_walk_keys(nested, (*path, str(index))))
    return found


def audit_payload(payload: Any) -> dict[str, Any]:
    """Return a deterministic audit without echoing source values."""

    forbidden_keys = _walk_keys(payload)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    text_hits = sorted(
        {pattern.pattern for pattern in _FORBIDDEN_TEXT_PATTERNS if pattern.search(encoded)}
    )
    return {
        "prompt_information_policy": INFORMATION_POLICY,
        "forbidden_field_count": len(forbidden_keys),
        "forbidden_fields": [".".join(path) for path, _ in forbidden_keys],
        "forbidden_text_pattern_count": len(text_hits),
        "forbidden_text_patterns": text_hits,
        "ok": not forbidden_keys and not text_hits,
    }


def assert_payload_safe(payload: Any) -> None:
    audit = audit_payload(payload)
    if not audit["ok"]:
        raise PromptPolicyError(
            "prompt payload violates dirty-evidence-only policy: "
            + json.dumps(audit, ensure_ascii=False, sort_keys=True)
        )


def assert_messages_safe(messages: Sequence[Mapping[str, Any]]) -> None:
    canonical = [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in messages
    ]
    assert_payload_safe(canonical)
    for message in canonical:
        try:
            decoded = json.loads(message["content"])
        except json.JSONDecodeError:
            continue
        assert_payload_safe(decoded)


__all__ = [
    "FORBIDDEN_PROMPT_FIELDS",
    "INFORMATION_POLICY",
    "PROMPT_SCHEMA_VERSION",
    "PromptPolicyError",
    "assert_messages_safe",
    "assert_payload_safe",
    "audit_payload",
]
