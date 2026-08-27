"""Standalone runner for the Introduction motivation-evidence experiments.

The module intentionally does not import or instantiate a Router, calibrator,
optimizer, verifier, or Baran fallback.  Planning is label blind except for the
isolated simulated-label boundary inside :func:`baran.run_baran`; clean repair
values are bound only by :meth:`MotivationEvidenceRunner.finalize` after every
frozen physical request has reached a terminal checkpoint state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import tempfile
import threading
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .data import (
    EXPECTED_ORACLE_ERRORS,
    SafeCell,
    append_jsonl,
    load_dataset,
    normalize_for_match,
    read_jsonl,
    sha256_file,
    validate_manifest,
    write_jsonl,
)
from .group_context import (
    GroupContextBuilder,
    PROMPT_SCHEMA_VERSION,
    completion_token_ceiling,
    compute_ordered_query_id,
    compute_prompt_hash,
    compute_query_id,
    messages_as_dicts,
)
from .group_generator import GroupGenerator, stable_average_linkage_order
from .group_llm import (
    DeepSeekGroupClient,
    GroupClientConfig,
    GroupLLMJob,
    PermanentGroupLLMError,
    RetryableGroupLLMError,
    parse_group_response,
)
from .prompt_policy import assert_messages_safe


DEFAULT_RUN_ID = "motivation_evidence_deepseek_v4_flash_20260822_full"
DEFAULT_CONFIG_NAME = "motivation_evidence.json"
DEFAULT_LLM_CONFIG_NAME = "deepseek_v4.json"
FORMAL_DATASETS: tuple[tuple[str, str], ...] = (
    ("source", "hospital"),
    ("source", "flights"),
    ("source", "beers"),
    ("source", "rayyan"),
    ("source", "movies_1"),
    ("tableeg", "company"),
    ("tableeg", "marketing"),
    ("tableeg", "restaurant_20"),
    ("tableeg", "soccer"),
)
PRIMARY_VIEWS = ("pattern", "semantic")
PRIMARY_GROUP_SIZES = (2, 4, 8)
TERMINAL_EXECUTION_STATUSES = frozenset({"completed", "terminal_failure"})
FORBIDDEN_PRE_LABEL_KEYS = frozenset(
    {
        "clean",
        "clean_value",
        "correct",
        "correctness",
        "baran_correct",
        "llm_correct",
        "right_value",
        "oracle_value",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _hash_file_set(paths: Iterable[Path], *, relative_to: Path) -> dict[str, str]:
    """Hash a small, explicit artifact set without traversing weights or runs."""

    root = Path(relative_to).resolve()
    records: dict[str, str] = {}
    for raw_path in sorted({Path(path).resolve() for path in paths}):
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        if root not in raw_path.parents:
            raise ValueError(f"fingerprinted file escaped its root: {raw_path}")
        relative = raw_path.relative_to(root).as_posix()
        records[relative] = sha256_file(raw_path)
    if not records:
        raise ValueError("fingerprinted file set must not be empty")
    return records


def _verify_file_set(
    records: Mapping[str, object],
    *,
    relative_to: Path,
    label: str,
) -> None:
    root = Path(relative_to).resolve()
    if not records:
        raise AssertionError(f"{label} fingerprint is empty")
    for raw_relative, expected in records.items():
        relative = Path(str(raw_relative))
        if relative.is_absolute() or ".." in relative.parts:
            raise AssertionError(f"unsafe {label} fingerprint path: {raw_relative!r}")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise AssertionError(f"missing {label} fingerprint file: {raw_relative}")
        if sha256_file(path) != str(expected):
            raise AssertionError(f"{label} fingerprint drift: {raw_relative}")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl_fsync(path: Path, row: Mapping[str, object], lock: threading.Lock | None = None) -> None:
    encoded = _canonical_json(dict(row)).decode("utf-8")
    guard = lock or threading.Lock()
    with guard:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _walk_keys(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key).lower()
            yield from _walk_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_keys(nested)


def assert_label_blind_artifact(value: object) -> None:
    leaked = FORBIDDEN_PRE_LABEL_KEYS.intersection(_walk_keys(value))
    if leaked:
        raise AssertionError(f"pre-label artifact contains forbidden keys: {sorted(leaked)}")


def _stable_seed(seed: int, *parts: object) -> int:
    payload = {"seed": int(seed), "parts": [str(part) for part in parts]}
    return int(_digest(payload)[:16], 16)


def _derangement(size: int, rng: random.Random) -> tuple[int, ...]:
    if size < 2:
        raise ValueError("a derangement requires at least two groups")
    values = list(range(size))
    # Rejection is fast at these sizes and preserves a simple, auditable
    # uniform draw over permutations conditional on having no fixed point.
    for _ in range(10_000):
        rng.shuffle(values)
        if all(index != value for index, value in enumerate(values)):
            return tuple(values)
    raise RuntimeError(f"failed to construct a derangement of {size} groups")


@dataclass(frozen=True, slots=True)
class PartitionGroup:
    suite: str
    dataset: str
    source_view: str
    group_size: int
    column: str
    arm: str
    group_index: int
    ordered_cell_ids: tuple[str, ...]
    structured_group_id: str = ""

    def __post_init__(self) -> None:
        ordered = tuple(str(value) for value in self.ordered_cell_ids)
        if len(ordered) != int(self.group_size) or len(set(ordered)) != len(ordered):
            raise ValueError("partition group must contain exact-size unique ordered cells")
        if self.arm not in {"structured", "random"}:
            raise ValueError("partition arm must be structured or random")
        object.__setattr__(self, "ordered_cell_ids", ordered)

    @property
    def condition_key(self) -> tuple[str, str, str, int, str]:
        return (self.suite, self.dataset, self.source_view, self.group_size, self.column)

    @property
    def group_id(self) -> str:
        payload = {
            "condition": self.condition_key,
            "arm": self.arm,
            "group_index": self.group_index,
            "ordered_cell_ids": self.ordered_cell_ids,
        }
        return "mevg_" + _digest(payload)

    def as_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "suite": self.suite,
            "dataset": self.dataset,
            "source_view": self.source_view,
            "group_size": self.group_size,
            "column": self.column,
            "arm": self.arm,
            "group_index": self.group_index,
            "ordered_cell_ids": list(self.ordered_cell_ids),
            "structured_group_id": self.structured_group_id,
        }


def build_structured_partition(
    ordered_cell_ids: Sequence[str],
    group_size: int,
    *,
    suite: str = "",
    dataset: str = "",
    source_view: str = "pattern",
    column: str = "",
    minimum_groups: int = 3,
) -> tuple[tuple[PartitionGroup, ...], dict[str, object]]:
    """Cut a frozen leaf order into non-overlapping exact-size blocks.

    The registered revision requires at least three groups (``g >= 3``), not
    the two-group threshold in the earlier prose specification.
    """

    ordered = tuple(str(value) for value in ordered_cell_ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("leaf order contains duplicate cells")
    width = int(group_size)
    threshold = int(minimum_groups)
    if width <= 1:
        raise ValueError("group_size must be greater than one")
    if threshold < 3:
        raise ValueError("motivation evidence eligibility is frozen at g >= 3")
    available_groups = len(ordered) // width
    eligible_groups = available_groups if available_groups >= threshold else 0
    eligible_count = eligible_groups * width
    exclusion_reason = ""
    if not eligible_groups:
        exclusion_reason = "fewer_than_three_complete_groups"
    elif eligible_count < len(ordered):
        exclusion_reason = "incomplete_tail_excluded"
    groups = tuple(
        PartitionGroup(
            suite=str(suite),
            dataset=str(dataset),
            source_view=str(source_view),
            group_size=width,
            column=str(column),
            arm="structured",
            group_index=index,
            ordered_cell_ids=ordered[index * width : (index + 1) * width],
        )
        for index in range(eligible_groups)
    )
    audit: dict[str, object] = {
        "suite": str(suite),
        "dataset": str(dataset),
        "source_view": str(source_view),
        "group_size": width,
        "column": str(column),
        "raw_error_cells": len(ordered),
        "eligible_cells": eligible_count,
        "leftover_cells": len(ordered) - eligible_count,
        "structured_groups": eligible_groups,
        "random_groups": eligible_groups,
        "coverage_rate": eligible_count / len(ordered) if ordered else 0.0,
        "eligibility_rule": "floor(raw_error_cells / group_size) >= 3",
        "exclusion_reason": exclusion_reason,
    }
    return groups, audit


def build_random_partition(
    structured_groups: Sequence[PartitionGroup],
    *,
    seed: int = 43,
    max_attempts: int = 10_000,
) -> tuple[PartitionGroup, ...]:
    """Build position-preserving within-condition matched derangements."""

    structured = tuple(structured_groups)
    if len(structured) < 3:
        raise ValueError("strict random partition requires at least three structured groups")
    condition = structured[0].condition_key
    if any(group.condition_key != condition or group.arm != "structured" for group in structured):
        raise ValueError("random repartition input must be one structured condition")
    width = structured[0].group_size
    if any(group.group_size != width for group in structured):
        raise ValueError("structured groups have inconsistent sizes")
    structured_sets = {frozenset(group.ordered_cell_ids) for group in structured}
    all_cells = [cell for group in structured for cell in group.ordered_cell_ids]
    if len(set(all_cells)) != len(all_cells):
        raise ValueError("structured condition is not a partition")

    for attempt in range(int(max_attempts)):
        rng = random.Random(_stable_seed(seed, *condition, attempt))
        permutations = [_derangement(len(structured), rng) for _ in range(width)]
        rows = tuple(
            tuple(
                structured[permutations[position][target_group]].ordered_cell_ids[position]
                for position in range(width)
            )
            for target_group in range(len(structured))
        )
        if any(len(set(row)) != width for row in rows):
            continue
        if any(frozenset(row) in structured_sets for row in rows):
            continue
        random_groups = tuple(
            PartitionGroup(
                suite=structured[index].suite,
                dataset=structured[index].dataset,
                source_view=structured[index].source_view,
                group_size=width,
                column=structured[index].column,
                arm="random",
                group_index=index,
                ordered_cell_ids=row,
                # A random query draws its positions from several structured
                # queries.  Cell-level pairing is recovered by cell identity,
                # not by claiming a false one-query-to-one-query mapping.
                structured_group_id="",
            )
            for index, row in enumerate(rows)
        )
        random_cells = [cell for group in random_groups for cell in group.ordered_cell_ids]
        if Counter(random_cells) != Counter(all_cells):  # pragma: no cover - defensive
            continue
        structured_position = {
            cell: position
            for group in structured
            for position, cell in enumerate(group.ordered_cell_ids)
        }
        if any(
            structured_position[cell] != position
            for group in random_groups
            for position, cell in enumerate(group.ordered_cell_ids)
        ):
            continue
        return random_groups
    raise RuntimeError(
        "unable to construct a legal within-column random derangement; "
        "cross-column fallback is forbidden"
    )


def validate_matched_partitions(
    structured_groups: Sequence[PartitionGroup],
    random_groups: Sequence[PartitionGroup],
) -> dict[str, object]:
    structured = tuple(structured_groups)
    randomised = tuple(random_groups)
    if len(structured) != len(randomised) or not structured:
        raise AssertionError("structured/random group counts do not match")
    structured_cells = [cell for group in structured for cell in group.ordered_cell_ids]
    random_cells = [cell for group in randomised for cell in group.ordered_cell_ids]
    if Counter(structured_cells) != Counter(random_cells):
        raise AssertionError("structured/random cell multisets do not match")
    if len(set(structured_cells)) != len(structured_cells):
        raise AssertionError("structured groups overlap")
    positions = {
        cell: position
        for group in structured
        for position, cell in enumerate(group.ordered_cell_ids)
    }
    for group in randomised:
        if len(set(group.ordered_cell_ids)) != group.group_size:
            raise AssertionError("random group contains duplicate cells")
        if any(positions[cell] != position for position, cell in enumerate(group.ordered_cell_ids)):
            raise AssertionError("random repartition changed a cell's member position")
    structured_sets = {frozenset(group.ordered_cell_ids) for group in structured}
    if any(frozenset(group.ordered_cell_ids) in structured_sets for group in randomised):
        raise AssertionError("a random group equals a structured group")
    return {
        "groups_per_arm": len(structured),
        "eligible_cells": len(structured_cells),
        "position_matching": True,
        "within_column": True,
        "random_equals_structured_group": False,
    }


def compute_logical_query_id(
    *,
    suite: str,
    dataset: str,
    arm: str,
    source_view: str,
    group_size: int,
    ordered_cell_ids: Sequence[str],
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
) -> str:
    return "mevl_" + _digest(
        {
            "suite": suite,
            "dataset": dataset,
            "arm": arm,
            "source_view": source_view,
            "group_size": int(group_size),
            "ordered_cell_ids": list(ordered_cell_ids),
            "prompt_schema_version": prompt_schema_version,
        }
    )


def compute_request_query_id(
    *,
    suite: str,
    dataset: str,
    ordered_cell_ids: Sequence[str],
    prompt_group_view: str,
    prompt_schema_version: str = PROMPT_SCHEMA_VERSION,
) -> str:
    """Identity echoed inside the neutral Prompt, deliberately excluding provenance."""

    return compute_ordered_query_id(
        suite,
        dataset,
        ordered_cell_ids,
        group_view=prompt_group_view,
        prompt_schema_version=prompt_schema_version,
    )


def physical_query_id(provider_request_hash: str) -> str:
    digest = str(provider_request_hash)
    if len(digest) != 64:
        raise ValueError("provider_request_hash must be a SHA-256 hex digest")
    return "mevp_" + digest


def round_robin_logical_schedule(
    records_by_condition: Mapping[tuple[object, ...], Iterable[Mapping[str, object]]],
    *,
    seed: int = 44,
) -> Iterator[dict[str, object]]:
    """Interleave condition streams before first-occurrence physical dedup."""

    keys = sorted(records_by_condition, key=lambda value: tuple(str(part) for part in value))
    random.Random(int(seed)).shuffle(keys)
    iterators = {key: iter(records_by_condition[key]) for key in keys}
    active = deque(keys)
    logical_index = 0
    while active:
        key = active.popleft()
        try:
            row = dict(next(iterators[key]))
        except StopIteration:
            continue
        row["logical_schedule_index"] = logical_index
        logical_index += 1
        yield row
        active.append(key)


def deduplicate_physical_schedule(
    logical_schedule: Iterable[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return logical mapping and first-occurrence physical schedule."""

    logical_rows: list[dict[str, object]] = []
    physical_rows: list[dict[str, object]] = []
    seen_hash_by_id: dict[str, str] = {}
    emitted: set[str] = set()
    for row in logical_schedule:
        logical = dict(row)
        identifier = str(logical.get("physical_query_id", ""))
        request_hash = str(logical.get("provider_request_hash", ""))
        if not identifier or not request_hash:
            raise ValueError("logical schedule row is missing physical_query_id")
        if identifier != physical_query_id(request_hash):
            raise ValueError("physical_query_id is not derived from provider_request_hash")
        logical_rows.append(logical)
        previous_hash = seen_hash_by_id.setdefault(identifier, request_hash)
        if previous_hash != request_hash:
            raise ValueError("physical identity maps to multiple provider request hashes")
        if identifier in emitted:
            continue
        emitted.add(identifier)
        embedded = logical.get("physical_request")
        if embedded is not None and not isinstance(embedded, Mapping):
            raise TypeError("physical_request must be a mapping")
        request = dict(embedded) if isinstance(embedded, Mapping) else {
            key: logical[key]
            for key in (
                "physical_query_id", "request_query_id", "provider_request_hash",
                "prompt_hash", "messages", "ordered_cell_ids", "model_requested",
                "max_tokens", "estimated_prompt_tokens",
                "estimated_completion_tokens", "estimated_total_tokens",
            )
            if key in logical
        }
        request.update(
            {
                "physical_schedule_index": len(physical_rows),
                "physical_query_id": identifier,
                "provider_request_hash": request_hash,
                "first_logical_query_id": str(logical.get("logical_query_id", "")),
                "first_logical_schedule_index": int(logical.get("logical_schedule_index", -1)),
            }
        )
        physical_rows.append(request)
    return logical_rows, physical_rows


@dataclass(frozen=True, slots=True)
class EvidencePaths:
    project_root: Path
    data_root: Path
    vendor_root: Path
    runs_root: Path
    run_dir: Path
    configs: Path
    provenance: Path
    evidence: Path
    llm: Path
    records: Path
    metrics: Path
    figures: Path
    report: Path

    @classmethod
    def create(
        cls,
        *,
        project_root: str | Path,
        data_root: str | Path,
        vendor_root: str | Path,
        runs_root: str | Path,
        run_id: str,
    ) -> "EvidencePaths":
        project = Path(project_root).resolve()
        runs = Path(runs_root).resolve()
        run_dir = (runs / str(run_id)).resolve()
        if runs not in run_dir.parents:
            raise ValueError("run directory escaped runs_root")
        return cls(
            project_root=project,
            data_root=Path(data_root).resolve(),
            vendor_root=Path(vendor_root).resolve(),
            runs_root=runs,
            run_dir=run_dir,
            configs=run_dir / "configs",
            provenance=run_dir / "provenance",
            evidence=run_dir / "evidence",
            llm=run_dir / "llm",
            records=run_dir / "records",
            metrics=run_dir / "metrics",
            figures=run_dir / "figures",
            report=run_dir / "report",
        )

    def ensure(self) -> None:
        for path in (
            self.run_dir,
            self.configs,
            self.provenance,
            self.evidence,
            self.llm,
            self.records,
            self.metrics,
            self.figures,
            self.report,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def _normalised_usage(usage: Mapping[str, Any] | None) -> tuple[int, int, int]:
    values = dict(usage or {})

    def numeric(*keys: str) -> int:
        for key in keys:
            value = values.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0, int(value))
        return 0

    input_tokens = numeric("prompt_tokens", "input_tokens")
    output_tokens = numeric("completion_tokens", "output_tokens")
    total_tokens = numeric("total_tokens") or input_tokens + output_tokens
    return input_tokens, output_tokens, total_tokens


def _compact_terminal_checkpoint(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "physical_query_id",
        "request_query_id",
        "prompt_hash",
        "provider_request_hash",
        "status",
        "terminal",
        "model_requested",
        "model_returned",
        "model_field_present",
        "provider_model_field_present",
        "model_returned_present",
        "model_matches_request",
        "historical_imported_response",
        "ordered_cell_ids",
        "max_tokens",
        "physical_schedule_index",
    )
    compact = {field: row.get(field) for field in fields}
    if str(row.get("status", "")) == "terminal_failure":
        compact.update(
            {
                "parse_status": row.get("parse_status"),
                "items": row.get("items"),
                "missing_cell_ids": row.get("missing_cell_ids"),
                "response_text": "",
            }
        )
    return compact


def _terminal_checkpoint_rows(
    path: Path,
    *,
    compact_for_resume: bool = False,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    request_hash_by_physical: dict[str, str] = {}
    for source_row in _iter_jsonl(path):
        # Raw response text remains durably in the checkpoint but is not kept
        # in memory during 100k-request resume/finalization scans.
        row = {key: value for key, value in source_row.items() if key != "response_text"}
        identifier = str(row.get("physical_query_id", ""))
        request_hash = str(row.get("provider_request_hash", ""))
        if not identifier or not request_hash:
            continue
        previous = request_hash_by_physical.setdefault(identifier, request_hash)
        if previous != request_hash:
            raise ValueError(f"checkpoint request drift for physical query {identifier}")
        if str(row.get("status", "")) in TERMINAL_EXECUTION_STATUSES:
            rows[identifier] = (
                _compact_terminal_checkpoint(row) if compact_for_resume else row
            )
    return rows


def _assert_terminal_checkpoint_matches_request(
    checkpoint: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    model_requested: str,
    verify_response: bool = True,
) -> None:
    """Verify that a terminal ledger row is self-consistent and plan-bound."""

    identity_fields = (
        "physical_query_id",
        "request_query_id",
        "prompt_hash",
        "provider_request_hash",
        "max_tokens",
        "physical_schedule_index",
    )
    for field in identity_fields:
        if str(checkpoint.get(field, "")) != str(request.get(field, "")):
            raise AssertionError(f"checkpoint {field} drift")
    checkpoint_cells = tuple(str(value) for value in checkpoint.get("ordered_cell_ids", ()))
    request_cells = tuple(str(value) for value in request.get("ordered_cell_ids", ()))
    if checkpoint_cells != request_cells:
        raise AssertionError("checkpoint ordered_cell_ids drift")
    if bool(checkpoint.get("historical_imported_response")):
        raise AssertionError("historical response import is forbidden")
    if str(checkpoint.get("model_requested", "")) != str(model_requested):
        raise AssertionError("checkpoint requested model drift")

    status = str(checkpoint.get("status", ""))
    if status == "terminal_failure":
        if (
            str(checkpoint.get("parse_status", "")) != "provider_failure"
            or checkpoint.get("items") not in ([], ())
            or str(checkpoint.get("response_text", ""))
            or tuple(str(value) for value in checkpoint.get("missing_cell_ids", ()))
            != request_cells
        ):
            raise AssertionError("terminal provider failure ledger is inconsistent")
        return
    if status != "completed":
        raise AssertionError(f"unsupported terminal checkpoint status: {status!r}")
    if (
        str(checkpoint.get("model_returned", "")) != str(model_requested)
        or not bool(checkpoint.get("model_field_present"))
        or not bool(checkpoint.get("provider_model_field_present"))
        or not bool(checkpoint.get("model_returned_present"))
        or not bool(checkpoint.get("model_matches_request"))
    ):
        raise AssertionError("completed response lacks exact provider model identity")

    if not verify_response:
        # Resume scans deliberately omit potentially large response bodies from
        # memory.  The final validator performs the full response reparse below.
        return

    parsed = parse_group_response(
        str(checkpoint.get("response_text", "")),
        str(request["request_query_id"]),
        request_cells,
    )
    expected = parsed.as_dict()
    for field in (
        "parse_status",
        "items",
        "missing_cell_ids",
        "unknown_cell_ids",
        "duplicate_cell_ids",
        "invalid_items",
    ):
        if checkpoint.get(field) != expected[field]:
            raise AssertionError(f"checkpoint parsed {field} disagrees with response_text")


def _csv_boolean(value: object, *, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise AssertionError(f"{field} is not a serialized boolean: {value!r}")


def _validate_finalized_ledgers(
    *,
    complementarity_path: Path,
    group_path: Path,
    cost_path: Path,
    population_ids: set[str],
    expected_group_memberships: Mapping[
        tuple[str, str, str, int, str], Mapping[str, object]
    ],
    physical_requests: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Recompute label-derived identities instead of trusting CSV row counts."""

    complementarity_by_cell: dict[str, dict[str, object]] = {}
    quadrants: Counter[str] = Counter()
    with complementarity_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "suite", "dataset", "cell_id", "clean_value", "baran_prediction",
            "baran_valid", "baran_correct", "llm_prediction", "llm_valid",
            "llm_correct", "outcome_quadrant",
        }
        if not required.issubset(set(reader.fieldnames or ())):
            raise AssertionError("complementarity ledger schema is incomplete")
        for row in reader:
            cell_id = str(row["cell_id"])
            if not cell_id or cell_id in complementarity_by_cell:
                raise AssertionError(f"duplicate or empty complementarity cell: {cell_id!r}")
            clean = normalize_for_match(row["clean_value"])
            baran_valid = _csv_boolean(row["baran_valid"], field="baran_valid")
            llm_valid = _csv_boolean(row["llm_valid"], field="llm_valid")
            baran_correct = _csv_boolean(row["baran_correct"], field="baran_correct")
            llm_correct = _csv_boolean(row["llm_correct"], field="llm_correct")
            expected_baran = bool(
                baran_valid and normalize_for_match(row["baran_prediction"]) == clean
            )
            expected_llm = bool(
                llm_valid and normalize_for_match(row["llm_prediction"]) == clean
            )
            if baran_correct != expected_baran or llm_correct != expected_llm:
                raise AssertionError(f"complementarity correctness drift for {cell_id}")
            quadrant = f"n_{int(baran_correct)}{int(llm_correct)}"
            if str(row["outcome_quadrant"]) != quadrant:
                raise AssertionError(f"complementarity quadrant drift for {cell_id}")
            quadrants[quadrant] += 1
            complementarity_by_cell[cell_id] = {
                "suite": str(row["suite"]),
                "dataset": str(row["dataset"]),
                "llm_prediction": str(row["llm_prediction"]),
                "llm_valid": llm_valid,
                "llm_correct": llm_correct,
            }
    if set(complementarity_by_cell) != population_ids:
        raise AssertionError("complementarity ledger does not cover the frozen population")

    observed_group_keys: set[tuple[str, str, str, int, str]] = set()
    with group_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "suite", "dataset", "source_view", "group_size", "cell_id",
            "clean_value", "member_position", "structured_group_id", "random_group_id",
            "singleton_prediction", "singleton_valid", "singleton_correct",
            "structured_prediction", "structured_valid", "structured_correct",
            "random_prediction", "random_valid", "random_correct",
            "structured_rescue", "structured_interference", "random_rescue",
            "random_interference",
        }
        if not required.issubset(set(reader.fieldnames or ())):
            raise AssertionError("group ledger schema is incomplete")
        for row in reader:
            try:
                size = int(row["group_size"])
                position = int(row["member_position"])
            except (TypeError, ValueError) as error:
                raise AssertionError("group ledger size/position is not integral") from error
            key = (
                str(row["suite"]), str(row["dataset"]), str(row["source_view"]),
                size, str(row["cell_id"]),
            )
            if key in observed_group_keys or key not in expected_group_memberships:
                raise AssertionError(f"unexpected or duplicate group cell incidence: {key}")
            expected_membership = expected_group_memberships[key]
            if (
                position != int(expected_membership["member_position"])
                or str(row["structured_group_id"])
                != str(expected_membership["structured_group_id"])
                or str(row["random_group_id"]) != str(expected_membership["random_group_id"])
            ):
                raise AssertionError(f"group membership provenance drift for {key}")
            clean = normalize_for_match(row["clean_value"])
            correctness: dict[str, bool] = {}
            for arm in ("singleton", "structured", "random"):
                valid = _csv_boolean(row[f"{arm}_valid"], field=f"{arm}_valid")
                correct = _csv_boolean(row[f"{arm}_correct"], field=f"{arm}_correct")
                expected_correct = bool(
                    valid and normalize_for_match(row[f"{arm}_prediction"]) == clean
                )
                if correct != expected_correct:
                    raise AssertionError(f"{arm} correctness drift for {key}")
                correctness[arm] = correct
            singleton = complementarity_by_cell.get(key[-1])
            if singleton is None or (
                singleton["suite"] != key[0]
                or singleton["dataset"] != key[1]
                or singleton["llm_prediction"] != str(row["singleton_prediction"])
                or singleton["llm_valid"]
                != _csv_boolean(row["singleton_valid"], field="singleton_valid")
                or singleton["llm_correct"] != correctness["singleton"]
            ):
                raise AssertionError(f"singleton cross-ledger drift for {key}")
            transitions = {
                "structured_rescue": correctness["structured"] and not correctness["singleton"],
                "structured_interference": correctness["singleton"] and not correctness["structured"],
                "random_rescue": correctness["random"] and not correctness["singleton"],
                "random_interference": correctness["singleton"] and not correctness["random"],
            }
            for field, expected in transitions.items():
                if _csv_boolean(row[field], field=field) != expected:
                    raise AssertionError(f"{field} identity drift for {key}")
            observed_group_keys.add(key)
    if observed_group_keys != set(expected_group_memberships):
        raise AssertionError("group ledger does not cover the frozen matched incidences")

    observed_physical: set[str] = set()
    with cost_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "physical_schedule_index", "physical_query_id", "provider_request_hash",
            "request_query_id", "model_requested", "status", "historical_imported_response",
        }
        if not required.issubset(set(reader.fieldnames or ())):
            raise AssertionError("API cost ledger schema is incomplete")
        for row in reader:
            identifier = str(row["physical_query_id"])
            request = physical_requests.get(identifier)
            if request is None or identifier in observed_physical:
                raise AssertionError(f"unexpected or duplicate API cost row: {identifier!r}")
            if (
                str(row["provider_request_hash"]) != str(request["provider_request_hash"])
                or str(row["request_query_id"]) != str(request["request_query_id"])
                or int(row["physical_schedule_index"])
                != int(request["physical_schedule_index"])
                or str(row["model_requested"]) != str(request["model_requested"])
                or str(row["status"]) not in TERMINAL_EXECUTION_STATUSES
                or _csv_boolean(
                    row["historical_imported_response"],
                    field="historical_imported_response",
                )
            ):
                raise AssertionError(f"API cost identity drift for {identifier}")
            observed_physical.add(identifier)
    if observed_physical != set(physical_requests):
        raise AssertionError("API cost ledger does not cover the physical schedule")

    return {
        "complementarity_cells": len(complementarity_by_cell),
        "group_cell_condition_rows": len(observed_group_keys),
        "physical_cost_rows": len(observed_physical),
        "quadrants": dict(sorted(quadrants.items())),
    }


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            yield value


class StreamingCheckpointExecutor:
    """Bounded concurrent executor with an fsync'd terminal ledger.

    ``DeepSeekGroupClient`` owns the six-attempt retry policy.  This layer
    guarantees that a successful response, a parse-invalid response, or an
    exhausted retryable provider failure is terminal within this run.  Model
    identity, authentication, configuration, and programming errors abort the
    run without marking all remaining requests as failures.
    """

    def __init__(
        self,
        client: DeepSeekGroupClient,
        checkpoint_path: str | Path,
        *,
        concurrency: int = 4,
    ) -> None:
        self.client = client
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.concurrency = max(1, int(concurrency))
        self._write_lock = threading.Lock()

    @staticmethod
    def _job_from_record(record: Mapping[str, Any]) -> GroupLLMJob:
        return GroupLLMJob(
            query_id=str(record["request_query_id"]),
            messages=tuple(record["messages"]),
            prompt_hash=str(record["prompt_hash"]),
            expected_cell_ids=tuple(record["ordered_cell_ids"]),
            max_tokens=int(record["max_tokens"]),
            metadata={
                "phase": "motivation_evidence",
                "model_requested": str(record["model_requested"]),
                "physical_query_id": str(record["physical_query_id"]),
                "require_complete_response": False,
            },
        )

    def _record_success(self, request: Mapping[str, Any], result: Any) -> dict[str, Any]:
        input_tokens, output_tokens, total_tokens = _normalised_usage(result.usage)
        return {
            "physical_query_id": str(request["physical_query_id"]),
            "request_query_id": str(request["request_query_id"]),
            "prompt_hash": str(request["prompt_hash"]),
            "provider_request_hash": str(request["provider_request_hash"]),
            "status": "completed",
            "terminal": True,
            "parse_status": result.parsed.parse_status,
            "items": [item.as_dict() for item in result.parsed.items],
            "missing_cell_ids": list(result.parsed.missing_cell_ids),
            "unknown_cell_ids": list(result.parsed.unknown_cell_ids),
            "duplicate_cell_ids": list(result.parsed.duplicate_cell_ids),
            "invalid_items": [dict(item) for item in result.parsed.invalid_items],
            "response_text": result.content,
            "model_requested": self.client.config.model,
            "model_returned": result.model,
            "model_field_present": bool(result.provider_model_field_present),
            "provider_model_field_present": bool(result.provider_model_field_present),
            "model_returned_present": bool(
                result.provider_model_field_present and result.model
            ),
            "model_matches_request": result.model == self.client.config.model,
            "usage": dict(result.usage),
            "observed_input_tokens": input_tokens,
            "observed_output_tokens": output_tokens,
            "observed_total_tokens": total_tokens,
            "latency_seconds": float(result.latency_seconds),
            "attempts": int(result.attempts),
            "usage_observed_attempts": int(result.usage_observed_attempts),
            "unknown_usage_attempts": int(result.unknown_usage_attempts),
            "response_id": str(result.response_id),
            "checkpoint_hit": False,
            "historical_imported_response": False,
            "ordered_cell_ids": list(request["ordered_cell_ids"]),
            "max_tokens": int(request["max_tokens"]),
            "physical_schedule_index": int(request.get("physical_schedule_index", -1)),
        }

    def _record_terminal_failure(
        self,
        request: Mapping[str, Any],
        error: RetryableGroupLLMError,
    ) -> dict[str, Any]:
        return {
            "physical_query_id": str(request["physical_query_id"]),
            "request_query_id": str(request["request_query_id"]),
            "prompt_hash": str(request["prompt_hash"]),
            "provider_request_hash": str(request["provider_request_hash"]),
            "status": "terminal_failure",
            "terminal": True,
            "parse_status": "provider_failure",
            "items": [],
            "missing_cell_ids": list(request["ordered_cell_ids"]),
            "unknown_cell_ids": [],
            "duplicate_cell_ids": [],
            "invalid_items": [],
            "response_text": "",
            "model_requested": self.client.config.model,
            "model_returned": "",
            "model_field_present": False,
            "provider_model_field_present": False,
            "model_returned_present": False,
            "model_matches_request": False,
            "usage": {},
            "observed_input_tokens": 0,
            "observed_output_tokens": 0,
            "observed_total_tokens": 0,
            "latency_seconds": 0.0,
            "attempts": self.client.config.max_retries + 1,
            "usage_observed_attempts": 0,
            "unknown_usage_attempts": self.client.config.max_retries + 1,
            "response_id": "",
            "checkpoint_hit": False,
            "historical_imported_response": False,
            "ordered_cell_ids": list(request["ordered_cell_ids"]),
            "max_tokens": int(request["max_tokens"]),
            "exception_class": type(error).__name__,
            "error": str(error)[:500],
            "physical_schedule_index": int(request.get("physical_schedule_index", -1)),
        }

    def execute(
        self,
        requests: Iterable[Mapping[str, Any]],
    ) -> dict[str, object]:
        terminal = _terminal_checkpoint_rows(
            self.checkpoint_path,
            compact_for_resume=True,
        )
        scheduled = 0
        resumed = 0
        executed = 0
        failures = 0

        def pending_requests() -> Iterator[dict[str, Any]]:
            nonlocal scheduled, resumed
            seen: set[str] = set()
            for raw in requests:
                request = dict(raw)
                scheduled += 1
                identifier = str(request.get("physical_query_id", ""))
                if not identifier or identifier in seen:
                    raise ValueError("physical schedule contains a missing or duplicate identity")
                seen.add(identifier)
                provider_hash = self.client.provider_request_hash(self._job_from_record(request))
                if provider_hash != str(request.get("provider_request_hash", "")):
                    raise ValueError(f"frozen provider request drift for {identifier}")
                previous = terminal.get(identifier)
                if previous is not None:
                    _assert_terminal_checkpoint_matches_request(
                        previous,
                        request,
                        model_requested=self.client.config.model,
                        verify_response=False,
                    )
                    resumed += 1
                    continue
                yield request

        iterator = iter(pending_requests())
        futures: dict[Future[Any], dict[str, Any]] = {}

        def submit_next(pool: ThreadPoolExecutor) -> bool:
            try:
                request = next(iterator)
            except StopIteration:
                return False
            future = pool.submit(self.client.chat, self._job_from_record(request))
            futures[future] = request
            return True

        def write_abort(request: Mapping[str, Any], error: BaseException) -> None:
            abort = {
                "physical_query_id": str(request["physical_query_id"]),
                "physical_schedule_index": int(request.get("physical_schedule_index", -1)),
                "provider_request_hash": str(request["provider_request_hash"]),
                "status": "model_identity_abort" if hasattr(error, "model_field_present") else "infrastructure_abort",
                "exception_class": type(error).__name__,
                "error": str(error)[:500],
                "model_requested": str(getattr(error, "model_requested", self.client.config.model)),
                "model_returned": str(getattr(error, "model_returned", "")),
                "model_field_present": bool(getattr(error, "model_field_present", False)),
                "provider_model_field_present": bool(
                    getattr(error, "model_field_present", False)
                ),
                "model_returned_present": bool(
                    getattr(error, "model_field_present", False)
                    and getattr(error, "model_returned", "")
                ),
            }
            _append_jsonl_fsync(
                self.checkpoint_path.with_name("infrastructure_abort.jsonl"),
                abort,
                self._write_lock,
            )

        def consume(future: Future[Any], request: Mapping[str, Any]) -> BaseException | None:
            nonlocal failures, executed
            try:
                result = future.result()
            except RetryableGroupLLMError as error:
                record = self._record_terminal_failure(request, error)
                failures += 1
            except PermanentGroupLLMError as error:
                write_abort(request, error)
                return error
            except Exception as error:
                write_abort(request, error)
                return error
            else:
                if (
                    result.model != self.client.config.model
                    or not bool(result.provider_model_field_present)
                ):
                    error = RuntimeError("provider model identity was not enforced by the client")
                    write_abort(request, error)
                    return error
                record = self._record_success(request, result)
            _append_jsonl_fsync(self.checkpoint_path, record, self._write_lock)
            terminal[str(request["physical_query_id"])] = _compact_terminal_checkpoint(record)
            executed += 1
            return None

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            # One in-flight preflight protects the expensive concurrent run
            # from fan-out on an invalid credential or provider model alias.
            if submit_next(pool):
                first_future = next(iter(futures))
                first_request = futures.pop(first_future)
                first_error = consume(first_future, first_request)
                if first_error is not None:
                    raise first_error
            for _ in range(self.concurrency):
                if not submit_next(pool):
                    break
            while futures:
                completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                abort_error: BaseException | None = None
                for future in completed:
                    request = futures.pop(future)
                    error = consume(future, request)
                    if abort_error is None and error is not None:
                        abort_error = error
                if abort_error is not None:
                    # Do not discard already-paid in-flight responses.  Stop
                    # submitting, drain every running future, and fsync every
                    # acceptable result before surfacing the hard abort.
                    while futures:
                        drained, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                        for future in drained:
                            request = futures.pop(future)
                            consume(future, request)
                    raise abort_error
                for _ in completed:
                    submit_next(pool)
        return {
            "scheduled_physical_calls": scheduled,
            "resumed_terminal_calls": resumed,
            "executed_physical_calls": executed,
            "terminal_provider_failures": failures,
            "checkpoint_path": str(self.checkpoint_path),
        }


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")
        self.count = 0

    def write(self, row: Mapping[str, object]) -> None:
        self.handle.write(_canonical_json(dict(row)).decode("utf-8") + "\n")
        self.count += 1

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()

    def __enter__(self) -> "_JsonlWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class _PhysicalStage:
    """Disk-backed request store; only byte offsets and hashes live in RAM."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("w+b")
        self.offsets: dict[str, tuple[int, int]] = {}
        self.request_hashes: dict[str, str] = {}
        self.sealed = False

    def add(self, row: Mapping[str, object]) -> None:
        if self.sealed:
            raise RuntimeError("cannot append to a sealed physical request stage")
        identifier = str(row["physical_query_id"])
        request_hash = str(row["provider_request_hash"])
        if identifier != physical_query_id(request_hash):
            raise ValueError("physical request identity/hash mismatch")
        encoded = _canonical_json(dict(row)) + b"\n"
        previous = self.offsets.get(identifier)
        if previous is not None:
            if self.request_hashes[identifier] != request_hash:
                raise ValueError("physical identity collision with request drift")
            return
        offset = self.handle.tell()
        self.handle.write(encoded)
        self.offsets[identifier] = (offset, len(encoded))
        self.request_hashes[identifier] = request_hash

    def seal(self) -> None:
        if not self.sealed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.sealed = True

    def get(self, identifier: str) -> dict[str, Any]:
        self.seal()
        offset, length = self.offsets[str(identifier)]
        self.handle.seek(offset)
        value = json.loads(self.handle.read(length).decode("utf-8"))
        if not isinstance(value, dict):  # pragma: no cover - defensive
            raise RuntimeError("physical stage contains a non-object record")
        return value

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()


@dataclass(slots=True)
class MotivationEvidenceRunner:
    """Fresh, standalone runner for both registered evidence experiments."""

    paths: EvidencePaths
    run_id: str
    config_path: Path
    llm_config_path: Path
    config: Mapping[str, Any]
    llm_config: Mapping[str, Any]
    resume: bool = False
    provider_token_cap: int | None = None
    allow_uncapped_provider_usage: bool = False

    @classmethod
    def create(
        cls,
        *,
        project_root: str | Path,
        data_root: str | Path,
        vendor_root: str | Path,
        runs_root: str | Path,
        run_id: str = DEFAULT_RUN_ID,
        config_path: str | Path | None = None,
        llm_config_path: str | Path | None = None,
        resume: bool = False,
        provider_token_cap: int | None = None,
        allow_uncapped_provider_usage: bool = False,
    ) -> "MotivationEvidenceRunner":
        project = Path(project_root).resolve()
        selected_config = Path(config_path or project / "configs" / DEFAULT_CONFIG_NAME).resolve()
        selected_llm = Path(llm_config_path or project / "configs" / DEFAULT_LLM_CONFIG_NAME).resolve()
        config = _load_object(selected_config)
        llm_config = _load_object(selected_llm)
        if str(config.get("mode")) != "full":
            raise ValueError("motivation evidence config must use mode='full'")
        if tuple((str(row["suite"]), str(row["dataset"])) for row in config["formal_datasets"]) != FORMAL_DATASETS:
            raise ValueError("formal dataset list differs from the frozen nine datasets")
        if tuple(config.get("primary_views", ())) != PRIMARY_VIEWS:
            raise ValueError("primary views must be pattern and semantic")
        if tuple(int(value) for value in config.get("group_sizes", ())) != PRIMARY_GROUP_SIZES:
            raise ValueError("primary group sizes must be 2, 4, and 8")
        if int(config.get("minimum_groups_per_column", 0)) != 3:
            raise ValueError("the frozen evidence revision requires g >= 3")
        if str(llm_config.get("model", "")) != "deepseek-v4-flash":
            raise ValueError("formal evidence requires model deepseek-v4-flash")
        if int(llm_config.get("max_retries", -1)) != 5:
            raise ValueError("formal evidence requires max_retries=5 (six total attempts)")
        if int(llm_config.get("concurrency", 0)) != 4:
            raise ValueError("formal evidence requires provider concurrency=4")
        if provider_token_cap is not None and int(provider_token_cap) <= 0:
            raise ValueError("provider_token_cap must be positive")
        return cls(
            paths=EvidencePaths.create(
                project_root=project,
                data_root=data_root,
                vendor_root=vendor_root,
                runs_root=runs_root,
                run_id=run_id,
            ),
            run_id=str(run_id),
            config_path=selected_config,
            llm_config_path=selected_llm,
            config=MappingProxyType(config),
            llm_config=MappingProxyType(llm_config),
            resume=bool(resume),
            provider_token_cap=(int(provider_token_cap) if provider_token_cap is not None else None),
            allow_uncapped_provider_usage=bool(allow_uncapped_provider_usage),
        )

    @property
    def manifest_path(self) -> Path:
        return self.paths.run_dir / "run_manifest.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.paths.llm / "query_checkpoint.jsonl"

    @property
    def physical_requests_path(self) -> Path:
        return self.paths.llm / "physical_requests.jsonl"

    def _bind_configs(self) -> None:
        self.paths.ensure()
        bound_experiment = self.paths.configs / "motivation_evidence.json"
        bound_llm = self.paths.configs / "deepseek_v4.json"
        for destination, value in (
            (bound_experiment, dict(self.config)),
            (bound_llm, dict(self.llm_config)),
        ):
            if destination.exists():
                if _load_object(destination) != value:
                    raise ValueError(f"bound configuration drift: {destination}")
            else:
                _atomic_json(destination, value)

    def _manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return _load_object(self.manifest_path)
        return {
            "schema_version": str(self.config["schema_version"]),
            "run_id": self.run_id,
            "mode": "full",
            "stages": {},
            "freshness": dict(self.config["freshness"]),
            "historical_input_paths_accepted": False,
        }

    def _mark_stage(self, stage: str, payload: Mapping[str, object]) -> dict[str, Any]:
        manifest = self._manifest()
        stages = dict(manifest.get("stages") or {})
        stages[str(stage)] = {"completed": True, **dict(payload)}
        manifest["stages"] = stages
        _atomic_json(self.manifest_path, manifest)
        return manifest

    def _dataset_specs(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self.config["formal_datasets"])

    def _baran_path(self, suite: str, dataset: str) -> Path:
        return self.paths.evidence / "baran" / f"{suite}__{dataset}.jsonl"

    def _baran_artifact_paths(self) -> tuple[Path, ...]:
        return tuple(
            self._baran_path(str(spec["suite"]), str(spec["dataset"]))
            for spec in self._dataset_specs()
        )

    def _necessary_code_paths(self) -> tuple[Path, ...]:
        package = Path(__file__).resolve().parent
        package_names = (
            "baran.py",
            "cell_features.py",
            "cli.py",
            "data.py",
            "group_context.py",
            "group_generator.py",
            "group_llm.py",
            "motivation_evidence.py",
            "motivation_reporting.py",
            "prompt_policy.py",
            "sampling.py",
            "statistics.py",
        )
        package_paths = tuple(package / name for name in package_names)
        # Baran imports the vendored Raha Python package.  Hash source only:
        # no models, checkpoints, datasets, notebooks, or historical runs.
        vendor_python = tuple(sorted((self.paths.vendor_root / "raha").rglob("*.py")))
        return package_paths + vendor_python

    def _fresh_baran_records(self, loaded: Any, cells: Sequence[Any]) -> list[dict[str, Any]]:
        from .baran import assert_online_baran_record_safe, run_baran

        path = self._baran_path(loaded.suite, loaded.name)
        if path.exists():
            if not self.resume:
                raise FileExistsError(
                    f"fresh Baran artifact already exists; pass --resume for this same run: {path}"
                )
            rows = read_jsonl(path)
            expected_ids = {str(cell.cell_id) for cell in cells}
            if {str(row.get("cell_id")) for row in rows} != expected_ids:
                raise ValueError(f"incomplete or foreign Baran checkpoint: {path}")
            for row in rows:
                assert_online_baran_record_safe(row)
            return rows
        settings = dict(self.config["baran"])
        rows = run_baran(
            loaded,
            cells,
            self.paths.vendor_root,
            labeling_budget=int(settings["labeling_budget"]),
            seed=int(settings["seed"]),
            workers=int(settings["workers"]),
            multiprocessing_start_method=str(settings["multiprocessing_start_method"]),
            extract_diagnostics=bool(settings.get("extract_diagnostics", True)),
        )
        write_jsonl(path, rows)
        return rows

    def _query_material(
        self,
        *,
        builder: GroupContextBuilder,
        client: DeepSeekGroupClient,
        suite: str,
        dataset: str,
        arm: str,
        source_view: str,
        ordered_cell_ids: Sequence[str],
        column: str,
        group_id: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        ordered = tuple(str(value) for value in ordered_cell_ids)
        size = len(ordered)
        schema = str(self.config["prompt"]["schema_version"])
        overhead = int(self.config["prompt"].get("call_overhead_tokens", 0))
        if size == 1:
            prompt_view = "singleton"
            request_query_id = compute_query_id(
                suite,
                dataset,
                prompt_view,
                ordered,
                arm="singleton",
                prompt_schema_version=schema,
            )
            material = builder.build_material(
                request_query_id,
                prompt_view,
                tuple(builder.cell_by_id[value] for value in ordered),
                prompt_schema_version=schema,
                call_overhead_tokens=overhead,
            )
        else:
            prompt_view = str(self.config["neutral_group_view"])
            request_query_id = compute_ordered_query_id(
                suite,
                dataset,
                ordered,
                group_view=prompt_view,
                prompt_schema_version=schema,
            )
            material = builder.build_ordered_material(
                request_query_id,
                ordered,
                group_view=prompt_view,
                prompt_schema_version=schema,
                call_overhead_tokens=overhead,
            )
        logical_id = compute_logical_query_id(
            suite=suite,
            dataset=dataset,
            arm=arm,
            source_view=source_view,
            group_size=size,
            ordered_cell_ids=ordered,
            prompt_schema_version=schema,
        )
        job = GroupLLMJob(
            query_id=request_query_id,
            messages=material.messages,
            prompt_hash=material.prompt_hash,
            expected_cell_ids=ordered,
            max_tokens=material.completion_token_ceiling,
            metadata={"model_requested": client.config.model},
        )
        provider_hash = client.provider_request_hash(job)
        physical_id = physical_query_id(provider_hash)
        logical: dict[str, object] = {
            "logical_query_id": logical_id,
            "physical_query_id": physical_id,
            "request_query_id": request_query_id,
            "provider_request_hash": provider_hash,
            "prompt_hash": material.prompt_hash,
            "suite": suite,
            "dataset": dataset,
            "arm": arm,
            "source_view": source_view,
            "prompt_group_view": prompt_view,
            "group_size": size,
            "column": column,
            "group_id": group_id,
            "ordered_cell_ids": list(ordered),
            "estimated_prompt_tokens": material.estimated_prompt_tokens,
            "completion_token_ceiling": material.completion_token_ceiling,
            "estimated_total_tokens": material.estimated_total_tokens,
        }
        physical: dict[str, object] = {
            "physical_query_id": physical_id,
            "request_query_id": request_query_id,
            "provider_request_hash": provider_hash,
            "prompt_hash": material.prompt_hash,
            "messages": messages_as_dicts(material.messages),
            "ordered_cell_ids": list(ordered),
            "model_requested": client.config.model,
            "max_tokens": material.completion_token_ceiling,
            "estimated_prompt_tokens": material.estimated_prompt_tokens,
            "estimated_completion_tokens": material.completion_token_ceiling,
            "estimated_total_tokens": material.estimated_total_tokens,
        }
        return logical, physical

    def plan(self) -> dict[str, object]:
        """Run fresh Baran, freeze all requests, and make zero provider calls."""

        previous_manifest = self._manifest() if self.manifest_path.exists() else None
        if previous_manifest is not None:
            planned = bool((previous_manifest.get("stages") or {}).get("plan", {}).get("completed"))
            if planned:
                if not self.resume:
                    raise FileExistsError("evidence plan already exists; use --resume for this run")
                summary_path = self.paths.evidence / "plan_summary.json"
                result = _load_object(summary_path)
                self.validate(require_execution=False, require_finalized=False)
                return result
            if not self.resume:
                raise FileExistsError("partial evidence run exists; use --resume for this same run")

        self._bind_configs()
        data_manifest = self.paths.data_root / "manifest.json"
        data_manifest_audit = validate_manifest(self.paths.data_root, data_manifest)
        derived_paths = (
            self.paths.evidence / "singleton_population.jsonl",
            self.paths.evidence / "leaf_orders.jsonl",
            self.paths.evidence / "structured_partition.jsonl",
            self.paths.evidence / "random_partition.jsonl",
            self.paths.evidence / "partition_coverage.csv",
            self.paths.evidence / "unmatched_cells.csv",
            self.paths.evidence / "logical_queries.jsonl",
            self.paths.provenance / "execution_schedule.jsonl",
            self.physical_requests_path,
            self.paths.evidence / "plan_summary.json",
        )
        for path in derived_paths:
            if path.exists():
                path.unlink()
        temporary_root = self.paths.evidence / ".plan_work"
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        temporary_root.mkdir(parents=True)

        client_config = GroupClientConfig.from_mapping(self.llm_config)
        client = DeepSeekGroupClient(client_config, api_key="")
        physical_stage = _PhysicalStage(temporary_root / "physical_stage.jsonl")
        bucket_writers: dict[tuple[object, ...], _JsonlWriter] = {}
        bucket_paths: dict[tuple[object, ...], Path] = {}
        logical_counts: Counter[tuple[str, str, int]] = Counter()
        cell_incidence_counts: Counter[tuple[str, str, int]] = Counter()
        population_counts: Counter[str] = Counter()
        coverage_rows: list[dict[str, object]] = []
        unmatched_rows: list[dict[str, object]] = []

        def bucket_writer(key: tuple[object, ...]) -> _JsonlWriter:
            writer = bucket_writers.get(key)
            if writer is None:
                path = temporary_root / "logical_buckets" / ("bucket_" + _digest(key) + ".jsonl")
                writer = _JsonlWriter(path)
                bucket_writers[key] = writer
                bucket_paths[key] = path
            return writer

        def add_query(
            *,
            builder: GroupContextBuilder,
            suite: str,
            dataset: str,
            arm: str,
            source_view: str,
            ordered_cell_ids: Sequence[str],
            column: str,
            group_id: str,
        ) -> None:
            logical, physical = self._query_material(
                builder=builder,
                client=client,
                suite=suite,
                dataset=dataset,
                arm=arm,
                source_view=source_view,
                ordered_cell_ids=ordered_cell_ids,
                column=column,
                group_id=group_id,
            )
            assert_label_blind_artifact(logical)
            assert_label_blind_artifact(physical)
            condition = (suite, dataset, arm, source_view, len(ordered_cell_ids))
            bucket_writer(condition).write(logical)
            physical_stage.add(physical)
            logical_counts[(arm, source_view, len(ordered_cell_ids))] += 1
            cell_incidence_counts[(arm, source_view, len(ordered_cell_ids))] += len(ordered_cell_ids)

        population_writer = _JsonlWriter(self.paths.evidence / "singleton_population.jsonl")
        leaf_writer = _JsonlWriter(self.paths.evidence / "leaf_orders.jsonl")
        structured_writer = _JsonlWriter(self.paths.evidence / "structured_partition.jsonl")
        random_writer = _JsonlWriter(self.paths.evidence / "random_partition.jsonl")
        try:
            for spec in self._dataset_specs():
                suite = str(spec["suite"])
                dataset_name = str(spec["dataset"])
                loaded = load_dataset(suite, dataset_name, self.paths.data_root)
                oracle_cells = loaded.oracle_cells(include_annotations=False)
                expected = int(spec["oracle_errors"])
                if len(oracle_cells) != expected or EXPECTED_ORACLE_ERRORS[(suite, dataset_name)] != expected:
                    raise AssertionError(
                        f"frozen population mismatch for {suite}/{dataset_name}: "
                        f"expected {expected}, observed {len(oracle_cells)}"
                    )
                safe_dataset = loaded.safe_view()
                safe_cells = tuple(safe_dataset.cells)
                base_family = str(spec["base_family"])
                for cell in safe_cells:
                    row = {**cell.as_dict(), "base_family": base_family, "row_cluster": f"{suite}:{dataset_name}:{cell.row}"}
                    assert_label_blind_artifact(row)
                    population_writer.write(row)
                    population_counts[f"{suite}/{dataset_name}"] += 1

                baran_rows = self._fresh_baran_records(loaded, oracle_cells)
                baran_by_cell = {str(row["cell_id"]): row for row in baran_rows}
                generator = GroupGenerator(
                    safe_dataset,
                    safe_cells,
                    baran_by_cell,
                    group_sizes=PRIMARY_GROUP_SIZES,
                    prompt_schema_version=str(self.config["prompt"]["schema_version"]),
                    similar_row_count=int(self.config["prompt"]["similar_row_count"]),
                )
                builder = GroupContextBuilder(
                    safe_dataset,
                    safe_cells,
                    known_error_cells=safe_cells,
                    similar_row_count=int(self.config["prompt"]["similar_row_count"]),
                    column_top_values=int(self.config["prompt"]["column_top_values"]),
                )
                for cell in safe_cells:
                    add_query(
                        builder=builder,
                        suite=suite,
                        dataset=dataset_name,
                        arm="singleton",
                        source_view="singleton",
                        ordered_cell_ids=(str(cell.cell_id),),
                        column=str(cell.column),
                        group_id="",
                    )

                cells_by_column: dict[str, list[str]] = defaultdict(list)
                for cell in safe_cells:
                    cells_by_column[str(cell.column)].append(str(cell.cell_id))
                for source_view in PRIMARY_VIEWS:
                    vectors = generator.pattern_by_id if source_view == "pattern" else generator.semantic_by_id
                    for column in sorted(cells_by_column):
                        member_ids = tuple(sorted(cells_by_column[column]))
                        leaf_order = stable_average_linkage_order(member_ids, vectors)
                        leaf_row = {
                            "suite": suite,
                            "dataset": dataset_name,
                            "source_view": source_view,
                            "column": column,
                            "ordered_cell_ids": list(leaf_order),
                            "leaf_order_hash": _digest(list(leaf_order)),
                        }
                        assert_label_blind_artifact(leaf_row)
                        leaf_writer.write(leaf_row)
                        for group_size in PRIMARY_GROUP_SIZES:
                            structured, coverage = build_structured_partition(
                                leaf_order,
                                group_size,
                                suite=suite,
                                dataset=dataset_name,
                                source_view=source_view,
                                column=column,
                                minimum_groups=int(self.config["minimum_groups_per_column"]),
                            )
                            coverage_rows.append(coverage)
                            eligible = {
                                cell_id for group in structured for cell_id in group.ordered_cell_ids
                            }
                            for cell_id in leaf_order:
                                if cell_id not in eligible:
                                    unmatched_rows.append(
                                        {
                                            "suite": suite,
                                            "dataset": dataset_name,
                                            "source_view": source_view,
                                            "group_size": group_size,
                                            "column": column,
                                            "cell_id": cell_id,
                                            "exclusion_reason": str(coverage["exclusion_reason"]),
                                        }
                                    )
                            if not structured:
                                continue
                            randomised = build_random_partition(
                                structured,
                                seed=int(self.config["random_partition_seed"]),
                            )
                            validate_matched_partitions(structured, randomised)
                            for group in structured:
                                structured_writer.write(group.as_dict())
                                add_query(
                                    builder=builder,
                                    suite=suite,
                                    dataset=dataset_name,
                                    arm="structured",
                                    source_view=source_view,
                                    ordered_cell_ids=group.ordered_cell_ids,
                                    column=column,
                                    group_id=group.group_id,
                                )
                            for group in randomised:
                                random_writer.write(group.as_dict())
                                add_query(
                                    builder=builder,
                                    suite=suite,
                                    dataset=dataset_name,
                                    arm="random",
                                    source_view=source_view,
                                    ordered_cell_ids=group.ordered_cell_ids,
                                    column=column,
                                    group_id=group.group_id,
                                )
        finally:
            population_writer.close()
            leaf_writer.close()
            structured_writer.close()
            random_writer.close()
            for writer in bucket_writers.values():
                writer.close()

        _atomic_csv(
            self.paths.evidence / "partition_coverage.csv",
            coverage_rows,
            (
                "suite", "dataset", "source_view", "group_size", "column",
                "raw_error_cells", "eligible_cells", "leftover_cells",
                "structured_groups", "random_groups", "coverage_rate",
                "eligibility_rule", "exclusion_reason",
            ),
        )
        _atomic_csv(
            self.paths.evidence / "unmatched_cells.csv",
            unmatched_rows,
            ("suite", "dataset", "source_view", "group_size", "column", "cell_id", "exclusion_reason"),
        )

        logical_writer = _JsonlWriter(self.paths.evidence / "logical_queries.jsonl")
        physical_writer = _JsonlWriter(self.physical_requests_path)
        schedule_writer = _JsonlWriter(self.paths.provenance / "execution_schedule.jsonl")
        physical_seen: set[str] = set()
        physical_prompt_tokens = 0
        physical_completion_tokens = 0
        physical_total_tokens = 0
        try:
            physical_stage.seal()
            streams = {key: _iter_jsonl(path) for key, path in bucket_paths.items()}
            for logical in round_robin_logical_schedule(
                streams,
                seed=int(self.config["execution_schedule_seed"]),
            ):
                logical_writer.write(logical)
                identifier = str(logical["physical_query_id"])
                if identifier in physical_seen:
                    continue
                physical_seen.add(identifier)
                request = physical_stage.get(identifier)
                physical_index = len(physical_seen) - 1
                request["physical_schedule_index"] = physical_index
                physical_writer.write(request)
                schedule_writer.write(
                    {
                        "physical_schedule_index": physical_index,
                        "physical_query_id": identifier,
                        "provider_request_hash": request["provider_request_hash"],
                        "first_logical_query_id": logical["logical_query_id"],
                        "first_logical_schedule_index": logical["logical_schedule_index"],
                        "dataset": logical["dataset"],
                        "arm": logical["arm"],
                        "source_view": logical["source_view"],
                        "group_size": logical["group_size"],
                    }
                )
                physical_prompt_tokens += int(request["estimated_prompt_tokens"])
                physical_completion_tokens += int(request["estimated_completion_tokens"])
                physical_total_tokens += int(request["estimated_total_tokens"])
        finally:
            logical_writer.close()
            physical_writer.close()
            schedule_writer.close()
            physical_stage.close()

        total_population = sum(population_counts.values())
        total_logical = sum(logical_counts.values())
        expected = dict(self.config["expected"])
        structured_by_size = {
            str(size): sum(
                count
                for (arm, _view, row_size), count in logical_counts.items()
                if arm == "structured" and row_size == size
            )
            for size in PRIMARY_GROUP_SIZES
        }
        structured_incidences_by_size = {
            str(size): sum(
                count
                for (arm, _view, row_size), count in cell_incidence_counts.items()
                if arm == "structured" and row_size == size
            )
            for size in PRIMARY_GROUP_SIZES
        }
        if total_population != int(expected["singleton_cells"]):
            raise AssertionError(f"singleton population mismatch: {total_population}")
        if total_logical != int(expected["logical_calls_before_exact_dedup"]):
            raise AssertionError(f"logical query count mismatch: {total_logical}")
        if structured_by_size != {str(key): int(value) for key, value in expected["structured_calls_by_size"].items()}:
            raise AssertionError(f"structured call counts mismatch: {structured_by_size}")
        if structured_incidences_by_size != {
            str(key): int(value) for key, value in expected["structured_cell_incidences_by_size"].items()
        }:
            raise AssertionError(f"structured cell incidences mismatch: {structured_incidences_by_size}")

        attempts = int(self.config["execution"]["attempts_per_physical_request"])
        artifact_hashes = {
            "singleton_population": sha256_file(self.paths.evidence / "singleton_population.jsonl"),
            "leaf_orders": sha256_file(self.paths.evidence / "leaf_orders.jsonl"),
            "structured_partition": sha256_file(self.paths.evidence / "structured_partition.jsonl"),
            "random_partition": sha256_file(self.paths.evidence / "random_partition.jsonl"),
            "logical_queries": sha256_file(self.paths.evidence / "logical_queries.jsonl"),
            "physical_requests": sha256_file(self.physical_requests_path),
            "execution_schedule": sha256_file(self.paths.provenance / "execution_schedule.jsonl"),
        }
        baran_artifact_hashes = _hash_file_set(
            self._baran_artifact_paths(),
            relative_to=self.paths.run_dir,
        )
        summary: dict[str, object] = {
            "run_id": self.run_id,
            "mode": "full",
            "provider_calls_made_during_plan": 0,
            "formal_dataset_count": len(self._dataset_specs()),
            "population_by_dataset": dict(sorted(population_counts.items())),
            "singleton_cells": total_population,
            "logical_calls_before_exact_dedup": total_logical,
            "logical_calls_by_arm_view_size": {
                f"{arm}:{view}:{size}": count
                for (arm, view, size), count in sorted(logical_counts.items())
            },
            "structured_calls_by_size": structured_by_size,
            "structured_cell_incidences_by_size": structured_incidences_by_size,
            "physical_calls_after_exact_dedup": len(physical_seen),
            "deduplicated_logical_calls": total_logical - len(physical_seen),
            "estimated_physical_prompt_tokens": physical_prompt_tokens,
            "estimated_physical_completion_tokens": physical_completion_tokens,
            "estimated_physical_total_tokens": physical_total_tokens,
            "retry_attempts_per_request": attempts,
            "retry_adjusted_token_estimate": physical_total_tokens * attempts,
            "model": client.config.model,
            "prompt_schema_version": str(self.config["prompt"]["schema_version"]),
            "random_partition_seed": int(self.config["random_partition_seed"]),
            "execution_schedule_seed": int(self.config["execution_schedule_seed"]),
            "minimum_groups_per_column": 3,
            "artifact_sha256": artifact_hashes,
            "baran_artifact_sha256": baran_artifact_hashes,
        }
        _atomic_json(self.paths.evidence / "plan_summary.json", summary)

        code_hashes = _hash_file_set(
            self._necessary_code_paths(),
            relative_to=self.paths.project_root,
        )
        _atomic_json(
            self.paths.provenance / "data_fingerprint.json",
            {
                "manifest_path": str(data_manifest.relative_to(self.paths.project_root)),
                "manifest_sha256": sha256_file(data_manifest),
                "manifest_audit": data_manifest_audit.as_dict(),
                "config_sha256": sha256_file(
                    self.paths.configs / "motivation_evidence.json"
                ),
                "llm_config_sha256": sha256_file(
                    self.paths.configs / "deepseek_v4.json"
                ),
                "code_sha256": code_hashes,
                "code_files_hashed": len(code_hashes),
                "vendor_raha_python_files_hashed": sum(
                    path.startswith("vendor/raha_source/raha/")
                    for path in code_hashes
                ),
                "excluded_from_hashing": list(self.config["hash_exclusions"]),
                "weights_hashed": False,
                "historical_runs_hashed": False,
            },
        )
        _atomic_json(self.paths.provenance / "freshness_audit.json", dict(self.config["freshness"]))
        _atomic_json(
            self.paths.provenance / "label_blind_plan_audit.json",
            {
                **dict(self.config["label_blind_plan_audit"]),
                "target_coordinates_are_protocol_inputs": True,
                "clean_repair_values_bound_during_plan": False,
                "baran_clean_access_isolated_to_simulated_label_budget": True,
            },
        )
        _atomic_json(
            self.paths.provenance / "protocol_revisions.json",
            {
                "eligibility": "g >= 3",
                "baran_label_boundary": (
                    "Baran may access clean values only inside its fresh simulated-label "
                    "budget; returned records are label-free and neither Baran correctness "
                    "nor target evaluation correctness is used for planning."
                ),
            },
        )
        self._mark_stage(
            "plan",
            {
                "plan_summary": str((self.paths.evidence / "plan_summary.json").relative_to(self.paths.run_dir)),
                "logical_calls": total_logical,
                "physical_calls": len(physical_seen),
            },
        )
        shutil.rmtree(temporary_root)
        self.validate(require_execution=False, require_finalized=False)
        return summary

    def run_queries(self) -> dict[str, object]:
        """Execute only the frozen physical schedule, with same-run resume."""

        self._bind_configs()
        manifest = self._manifest()
        if not bool((manifest.get("stages") or {}).get("plan", {}).get("completed")):
            raise RuntimeError("plan-motivation-evidence must complete before provider execution")
        if not self.physical_requests_path.is_file():
            raise FileNotFoundError("frozen physical request plan is missing")
        if not self.allow_uncapped_provider_usage and self.provider_token_cap is None:
            raise PermissionError("provider execution requires --token-cap or --no-token-cap")
        summary = _load_object(self.paths.evidence / "plan_summary.json")
        required_cap = int(
            summary.get("retry_adjusted_token_estimate", summary.get("retry_adjusted_token_cap", 0))
        )
        if self.provider_token_cap is not None and self.provider_token_cap < required_cap:
            raise ValueError(
                f"token cap {self.provider_token_cap} is below frozen retry-adjusted cap {required_cap}"
            )
        api_key_env = str(self.llm_config.get("api_key_env", "DEEPSEEK_API_KEY"))
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise PermissionError(f"required API credential variable is not set: {api_key_env}")
        client = DeepSeekGroupClient(GroupClientConfig.from_mapping(self.llm_config), api_key=api_key)
        pilot = self._run_provider_pilot(client)
        executor = StreamingCheckpointExecutor(
            client,
            self.checkpoint_path,
            concurrency=int(self.config["execution"]["concurrency"]),
        )
        result = executor.execute(_iter_jsonl(self.physical_requests_path))
        validation = self.validate(require_execution=True, require_finalized=False)
        terminal = _terminal_checkpoint_rows(
            self.checkpoint_path,
            compact_for_resume=True,
        )
        completed = [row for row in terminal.values() if row.get("status") == "completed"]
        failures = [row for row in terminal.values() if row.get("status") == "terminal_failure"]
        identity = {
            "model_requested": client.config.model,
            "provider_model_ids": sorted(
                {str(row.get("model_returned")) for row in completed if row.get("model_returned")}
            ),
            "completed_responses": len(completed),
            "terminal_provider_failures": len(failures),
            "missing_model_identity_responses": sum(
                not bool(row.get("provider_model_field_present")) for row in completed
            ),
            "mismatched_model_identity_responses": sum(
                str(row.get("model_returned")) != client.config.model for row in completed
            ),
        }
        _atomic_json(self.paths.provenance / "model_identity.json", identity)
        input_tokens = sum(int(row.get("observed_input_tokens", 0) or 0) for row in terminal.values())
        output_tokens = sum(int(row.get("observed_output_tokens", 0) or 0) for row in terminal.values())
        total_tokens = sum(int(row.get("observed_total_tokens", 0) or 0) for row in terminal.values())
        execution_summary = {
            **dict(result),
            "terminal_physical_calls": len(terminal),
            "observed_input_tokens": input_tokens,
            "observed_output_tokens": output_tokens,
            "observed_total_tokens": total_tokens,
            "model_identity": identity,
            "excluded_dataset_pilot": pilot,
            "validation": validation,
        }
        _atomic_json(self.paths.llm / "execution_summary.json", execution_summary)
        self._mark_stage("execution", execution_summary)
        return execution_summary

    def _run_provider_pilot(self, client: DeepSeekGroupClient) -> dict[str, object]:
        """Run two excluded-dataset requests in /tmp, verify resume, then delete them."""

        audit_path = self.paths.provenance / "excluded_dataset_pilot_audit.json"
        if audit_path.is_file():
            audit = _load_object(audit_path)
            if not bool(audit.get("passed")) or audit.get("model_returned") != client.config.model:
                raise RuntimeError("existing excluded-dataset pilot audit is invalid")
            return audit
        loaded = load_dataset("tableeg", "restaurant_rule_only", self.paths.data_root)
        safe_dataset = loaded.safe_view()
        cells_by_column: dict[str, list[SafeCell]] = defaultdict(list)
        for cell in safe_dataset.cells:
            cells_by_column[str(cell.column)].append(cell)
        pair = next(
            (
                tuple(sorted(cells, key=lambda value: str(value.cell_id))[:2])
                for _column, cells in sorted(cells_by_column.items())
                if len(cells) >= 2
            ),
            None,
        )
        if pair is None:  # pragma: no cover - frozen excluded dataset invariant
            raise RuntimeError("excluded pilot dataset has no same-column k=2 pair")
        builder = GroupContextBuilder(
            safe_dataset,
            safe_dataset.cells,
            known_error_cells=safe_dataset.cells,
            similar_row_count=int(self.config["prompt"]["similar_row_count"]),
            column_top_values=int(self.config["prompt"]["column_top_values"]),
        )
        requests: list[dict[str, object]] = []
        for index, (arm, ordered) in enumerate(
            (
                ("singleton", (str(pair[0].cell_id),)),
                ("structured", tuple(str(cell.cell_id) for cell in pair)),
            )
        ):
            _logical, physical = self._query_material(
                builder=builder,
                client=client,
                suite="tableeg",
                dataset="restaurant_rule_only",
                arm=arm,
                source_view="pilot",
                ordered_cell_ids=ordered,
                column=str(pair[0].column),
                group_id="excluded_dataset_pilot",
            )
            physical["physical_schedule_index"] = index
            requests.append(physical)
        temporary = Path(tempfile.mkdtemp(prefix="motivation_evidence_pilot_", dir="/tmp"))
        try:
            checkpoint = temporary / "pilot_checkpoint.jsonl"
            executor = StreamingCheckpointExecutor(client, checkpoint, concurrency=1)
            first = executor.execute(requests)
            second = executor.execute(requests)
            terminal = _terminal_checkpoint_rows(checkpoint)
            if len(terminal) != 2 or any(
                row.get("status") != "completed"
                or row.get("parse_status") != "ok"
                or row.get("model_returned") != client.config.model
                or not bool(row.get("provider_model_field_present"))
                for row in terminal.values()
            ):
                raise RuntimeError("excluded-dataset provider pilot failed identity/parse validation")
            if int(second["executed_physical_calls"]) != 0 or int(second["resumed_terminal_calls"]) != 2:
                raise RuntimeError("excluded-dataset pilot resume repeated a provider request")
            audit: dict[str, object] = {
                "passed": True,
                "suite": "tableeg",
                "dataset": "restaurant_rule_only",
                "physical_calls": 2,
                "singleton_calls": 1,
                "ordered_k2_calls": 1,
                "model_requested": client.config.model,
                "model_returned": client.config.model,
                "parse_statuses": sorted(str(row["parse_status"]) for row in terminal.values()),
                "resume_repeated_provider_calls": 0,
                "temporary_artifacts_deleted": True,
                "formal_results_included": False,
                "observed_total_tokens": sum(
                    int(row.get("observed_total_tokens", 0) or 0) for row in terminal.values()
                ),
                "first_execution": dict(first),
            }
            # Do not preserve paths to an already-deleted temporary directory.
            audit["first_execution"].pop("checkpoint_path", None)
            _atomic_json(audit_path, audit)
            return audit
        finally:
            shutil.rmtree(temporary)

    @staticmethod
    def _prediction_for_cell(
        checkpoint: Mapping[str, Any],
        cell: Mapping[str, Any],
    ) -> dict[str, object]:
        cell_id = str(cell["cell_id"])
        item = next(
            (
                value
                for value in checkpoint.get("items", [])
                if isinstance(value, Mapping) and str(value.get("cell_id")) == cell_id
            ),
            None,
        )
        decision = str(item.get("decision", "")) if isinstance(item, Mapping) else ""
        prediction = str(item.get("repair", "")) if isinstance(item, Mapping) else ""
        execution_status = str(checkpoint.get("status", "missing_execution"))
        response_parse_status = str(
            checkpoint.get("parse_status", "missing_execution")
        )
        if isinstance(item, Mapping):
            cell_parse_status = response_parse_status
        elif execution_status == "terminal_failure" or response_parse_status == "provider_failure":
            cell_parse_status = "provider_failure"
        else:
            invalid_cell_ids = {
                str(value.get("cell_id", ""))
                for value in checkpoint.get("invalid_items", ())
                if isinstance(value, Mapping)
            }
            duplicate_cell_ids = {
                str(value) for value in checkpoint.get("duplicate_cell_ids", ())
            }
            global_parse_failures = {
                "no_json_object",
                "query_id_mismatch",
                "invalid_repairs_array",
                "no_valid_items",
            }
            if (
                response_parse_status in global_parse_failures
                or cell_id in invalid_cell_ids
                or cell_id in duplicate_cell_ids
                or response_parse_status != "partial"
            ):
                cell_parse_status = "parse_failure"
            else:
                # A valid partial response may simply omit this target.  That
                # is distinct from an item that was present but malformed.
                cell_parse_status = "missing"
        valid = bool(
            execution_status == "completed"
            and isinstance(item, Mapping)
            and decision == "propose"
            and normalize_for_match(prediction)
            and normalize_for_match(prediction) != normalize_for_match(cell["dirty_value"])
        )
        return {
            # Preserve the raw parsed repair even when it is abstained, empty,
            # or unchanged; validity/correctness remain separate columns.
            "prediction": prediction,
            "valid": valid,
            "status": execution_status,
            "parse_status": cell_parse_status,
            "decision": decision,
            "observed_input_tokens": int(checkpoint.get("observed_input_tokens", 0) or 0),
            "observed_output_tokens": int(checkpoint.get("observed_output_tokens", 0) or 0),
            "observed_total_tokens": int(checkpoint.get("observed_total_tokens", 0) or 0),
            "latency_seconds": float(checkpoint.get("latency_seconds", 0.0) or 0.0),
            "attempts": int(checkpoint.get("attempts", 0) or 0),
            "usage_observed_attempts": int(checkpoint.get("usage_observed_attempts", 0) or 0),
            "unknown_usage_attempts": int(checkpoint.get("unknown_usage_attempts", 0) or 0),
        }

    def finalize(self) -> dict[str, object]:
        """Bind clean labels after execution and emit the two fixed cell ledgers."""

        self.validate(require_execution=True, require_finalized=False)
        checkpoints = _terminal_checkpoint_rows(self.checkpoint_path)
        population = {
            str(row["cell_id"]): row
            for row in _iter_jsonl(self.paths.evidence / "singleton_population.jsonl")
        }
        oracle: dict[str, Any] = {}
        base_family: dict[tuple[str, str], str] = {}
        for spec in self._dataset_specs():
            suite, dataset = str(spec["suite"]), str(spec["dataset"])
            base_family[(suite, dataset)] = str(spec["base_family"])
            loaded = load_dataset(suite, dataset, self.paths.data_root)
            for cell in loaded.oracle_cells(include_annotations=False):
                oracle[str(cell.cell_id)] = cell
        if set(oracle) != set(population):
            raise AssertionError("evaluation label universe differs from the frozen population")

        baran_by_cell: dict[str, dict[str, Any]] = {}
        for spec in self._dataset_specs():
            for row in _iter_jsonl(self._baran_path(str(spec["suite"]), str(spec["dataset"]))):
                baran_by_cell[str(row["cell_id"])] = row
        if set(baran_by_cell) != set(population):
            raise AssertionError("Baran universe differs from the frozen population")

        singleton_by_cell: dict[str, dict[str, Any]] = {}
        structured_by_key: dict[tuple[str, str, str, int, str], tuple[dict[str, Any], int]] = {}
        random_by_key: dict[tuple[str, str, str, int, str], tuple[dict[str, Any], int]] = {}
        logical_multiplicity: Counter[str] = Counter()
        for row in _iter_jsonl(self.paths.evidence / "logical_queries.jsonl"):
            logical_multiplicity[str(row["physical_query_id"])] += 1
            arm = str(row["arm"])
            ordered = tuple(str(value) for value in row["ordered_cell_ids"])
            if arm == "singleton":
                singleton_by_cell[ordered[0]] = row
                continue
            target = structured_by_key if arm == "structured" else random_by_key
            for position, cell_id in enumerate(ordered):
                key = (
                    str(row["suite"]), str(row["dataset"]), str(row["source_view"]),
                    int(row["group_size"]), cell_id,
                )
                if key in target:
                    raise AssertionError(f"duplicate {arm} membership for {key}")
                target[key] = (row, position)
        if set(singleton_by_cell) != set(population):
            raise AssertionError("singleton logical controls do not cover the frozen population")
        if set(structured_by_key) != set(random_by_key):
            raise AssertionError("structured/random paired cell conditions differ")

        complementarity_fields = (
            "suite", "dataset", "base_family", "cell_id", "row_cluster", "column",
            "dirty_value", "clean_value", "baran_prediction", "baran_valid",
            "baran_correct", "llm_prediction", "llm_valid", "llm_correct",
            "llm_status", "llm_parse_status", "llm_decision",
            "llm_observed_input_tokens", "llm_observed_output_tokens",
            "llm_observed_total_tokens", "outcome_quadrant",
        )
        complementarity_rows: list[dict[str, object]] = []
        for cell_id in sorted(population):
            safe = population[cell_id]
            clean_value = str(oracle[cell_id].clean_value)
            baran = baran_by_cell[cell_id]
            singleton = singleton_by_cell[cell_id]
            checkpoint = checkpoints[str(singleton["physical_query_id"])]
            llm = self._prediction_for_cell(checkpoint, safe)
            baran_prediction = str(baran.get("prediction", ""))
            baran_valid = bool(baran.get("valid_prediction", False))
            baran_correct = bool(
                baran_valid
                and normalize_for_match(baran_prediction) == normalize_for_match(clean_value)
            )
            llm_correct = bool(
                llm["valid"]
                and normalize_for_match(llm["prediction"]) == normalize_for_match(clean_value)
            )
            quadrant = f"n_{int(baran_correct)}{int(llm_correct)}"
            complementarity_rows.append(
                {
                    "suite": safe["suite"],
                    "dataset": safe["dataset"],
                    "base_family": safe["base_family"],
                    "cell_id": cell_id,
                    "row_cluster": safe["row_cluster"],
                    "column": safe["column"],
                    "dirty_value": safe["dirty_value"],
                    "clean_value": clean_value,
                    "baran_prediction": baran_prediction,
                    "baran_valid": baran_valid,
                    "baran_correct": baran_correct,
                    "llm_prediction": llm["prediction"],
                    "llm_valid": llm["valid"],
                    "llm_correct": llm_correct,
                    "llm_status": llm["status"],
                    "llm_parse_status": llm["parse_status"],
                    "llm_decision": llm["decision"],
                    "llm_observed_input_tokens": llm["observed_input_tokens"],
                    "llm_observed_output_tokens": llm["observed_output_tokens"],
                    "llm_observed_total_tokens": llm["observed_total_tokens"],
                    "outcome_quadrant": quadrant,
                }
            )
        complementarity_path = self.paths.records / "complementarity_cell_outcomes.csv"
        _atomic_csv(complementarity_path, complementarity_rows, complementarity_fields)

        group_fields = (
            "suite", "dataset", "base_family", "source_view", "group_size", "cell_id",
            "row_cluster", "column", "dirty_value", "clean_value", "member_position",
            "singleton_logical_query_id", "singleton_physical_query_id",
            "structured_logical_query_id", "structured_physical_query_id", "structured_group_id",
            "random_logical_query_id", "random_physical_query_id", "random_group_id",
            "singleton_prediction", "singleton_valid", "singleton_correct", "singleton_status",
            "singleton_parse_status", "singleton_decision", "structured_prediction",
            "structured_valid", "structured_correct", "structured_status",
            "structured_parse_status", "structured_decision", "random_prediction", "random_valid",
            "random_correct", "random_status", "random_parse_status", "random_decision",
            "structured_rescue", "structured_interference", "random_rescue", "random_interference",
            "singleton_query_observed_input_tokens", "singleton_query_observed_output_tokens",
            "singleton_query_observed_total_tokens", "structured_query_observed_input_tokens",
            "structured_query_observed_output_tokens", "structured_query_observed_total_tokens",
            "random_query_observed_input_tokens", "random_query_observed_output_tokens",
            "random_query_observed_total_tokens",
            "singleton_query_latency_seconds", "singleton_query_attempts",
            "singleton_query_usage_observed_attempts", "singleton_query_unknown_usage_attempts",
            "structured_query_latency_seconds", "structured_query_attempts",
            "structured_query_usage_observed_attempts", "structured_query_unknown_usage_attempts",
            "random_query_latency_seconds", "random_query_attempts",
            "random_query_usage_observed_attempts", "random_query_unknown_usage_attempts",
        )
        group_rows: list[dict[str, object]] = []
        for key in sorted(structured_by_key):
            suite, dataset, source_view, group_size, cell_id = key
            structured_logical, structured_position = structured_by_key[key]
            random_logical, random_position = random_by_key[key]
            if structured_position != random_position:
                raise AssertionError(f"member position mismatch for {key}")
            safe = population[cell_id]
            clean_value = str(oracle[cell_id].clean_value)
            singleton_logical = singleton_by_cell[cell_id]
            singleton = self._prediction_for_cell(
                checkpoints[str(singleton_logical["physical_query_id"])], safe
            )
            structured = self._prediction_for_cell(
                checkpoints[str(structured_logical["physical_query_id"])], safe
            )
            random_result = self._prediction_for_cell(
                checkpoints[str(random_logical["physical_query_id"])], safe
            )

            def correct(result: Mapping[str, object]) -> bool:
                return bool(
                    result["valid"]
                    and normalize_for_match(result["prediction"]) == normalize_for_match(clean_value)
                )

            singleton_correct = correct(singleton)
            structured_correct = correct(structured)
            random_correct = correct(random_result)
            group_rows.append(
                {
                    "suite": suite, "dataset": dataset,
                    "base_family": safe["base_family"], "source_view": source_view,
                    "group_size": group_size, "cell_id": cell_id,
                    "row_cluster": safe["row_cluster"], "column": safe["column"],
                    "dirty_value": safe["dirty_value"], "clean_value": clean_value,
                    "member_position": structured_position,
                    "singleton_logical_query_id": singleton_logical["logical_query_id"],
                    "singleton_physical_query_id": singleton_logical["physical_query_id"],
                    "structured_logical_query_id": structured_logical["logical_query_id"],
                    "structured_physical_query_id": structured_logical["physical_query_id"],
                    "structured_group_id": structured_logical["group_id"],
                    "random_logical_query_id": random_logical["logical_query_id"],
                    "random_physical_query_id": random_logical["physical_query_id"],
                    "random_group_id": random_logical["group_id"],
                    "singleton_prediction": singleton["prediction"], "singleton_valid": singleton["valid"],
                    "singleton_correct": singleton_correct, "singleton_status": singleton["status"],
                    "singleton_parse_status": singleton["parse_status"], "singleton_decision": singleton["decision"],
                    "structured_prediction": structured["prediction"], "structured_valid": structured["valid"],
                    "structured_correct": structured_correct, "structured_status": structured["status"],
                    "structured_parse_status": structured["parse_status"], "structured_decision": structured["decision"],
                    "random_prediction": random_result["prediction"], "random_valid": random_result["valid"],
                    "random_correct": random_correct, "random_status": random_result["status"],
                    "random_parse_status": random_result["parse_status"], "random_decision": random_result["decision"],
                    "structured_rescue": structured_correct and not singleton_correct,
                    "structured_interference": singleton_correct and not structured_correct,
                    "random_rescue": random_correct and not singleton_correct,
                    "random_interference": singleton_correct and not random_correct,
                    "singleton_query_observed_input_tokens": singleton["observed_input_tokens"],
                    "singleton_query_observed_output_tokens": singleton["observed_output_tokens"],
                    "singleton_query_observed_total_tokens": singleton["observed_total_tokens"],
                    "structured_query_observed_input_tokens": structured["observed_input_tokens"],
                    "structured_query_observed_output_tokens": structured["observed_output_tokens"],
                    "structured_query_observed_total_tokens": structured["observed_total_tokens"],
                    "random_query_observed_input_tokens": random_result["observed_input_tokens"],
                    "random_query_observed_output_tokens": random_result["observed_output_tokens"],
                    "random_query_observed_total_tokens": random_result["observed_total_tokens"],
                    "singleton_query_latency_seconds": singleton["latency_seconds"],
                    "singleton_query_attempts": singleton["attempts"],
                    "singleton_query_usage_observed_attempts": singleton["usage_observed_attempts"],
                    "singleton_query_unknown_usage_attempts": singleton["unknown_usage_attempts"],
                    "structured_query_latency_seconds": structured["latency_seconds"],
                    "structured_query_attempts": structured["attempts"],
                    "structured_query_usage_observed_attempts": structured["usage_observed_attempts"],
                    "structured_query_unknown_usage_attempts": structured["unknown_usage_attempts"],
                    "random_query_latency_seconds": random_result["latency_seconds"],
                    "random_query_attempts": random_result["attempts"],
                    "random_query_usage_observed_attempts": random_result["usage_observed_attempts"],
                    "random_query_unknown_usage_attempts": random_result["unknown_usage_attempts"],
                }
            )
        group_path = self.paths.records / "group_cell_outcomes.csv"
        _atomic_csv(group_path, group_rows, group_fields)

        planned_requests = {
            str(row["physical_query_id"]): {
                key: row[key]
                for key in (
                    "physical_schedule_index", "physical_query_id", "provider_request_hash",
                    "request_query_id", "model_requested", "estimated_prompt_tokens",
                    "estimated_completion_tokens", "estimated_total_tokens",
                )
            }
            for row in _iter_jsonl(self.physical_requests_path)
        }
        cost_fields = (
            "physical_schedule_index", "physical_query_id", "provider_request_hash",
            "request_query_id", "model_requested", "model_returned", "status", "parse_status",
            "attempts", "observed_input_tokens", "observed_output_tokens", "observed_total_tokens",
            "provider_model_field_present", "model_returned_present",
            "model_field_present", "model_matches_request", "latency_seconds",
            "usage_observed_attempts", "unknown_usage_attempts",
            "estimated_prompt_tokens", "estimated_completion_tokens", "estimated_total_tokens",
            "logical_query_mappings", "historical_imported_response",
        )
        cost_rows: list[dict[str, object]] = []
        for physical_id, request in sorted(
            planned_requests.items(), key=lambda item: int(item[1]["physical_schedule_index"])
        ):
            checkpoint = checkpoints[physical_id]
            cost_rows.append(
                {
                    "physical_schedule_index": request["physical_schedule_index"],
                    "physical_query_id": physical_id,
                    "provider_request_hash": request["provider_request_hash"],
                    "request_query_id": request["request_query_id"],
                    "model_requested": request["model_requested"],
                    "model_returned": checkpoint.get("model_returned", ""),
                    "status": checkpoint["status"], "parse_status": checkpoint["parse_status"],
                    "attempts": checkpoint.get("attempts", 0),
                    "model_field_present": checkpoint.get("model_field_present", False),
                    "provider_model_field_present": checkpoint.get("provider_model_field_present", False),
                    "model_returned_present": checkpoint.get("model_returned_present", False),
                    "model_matches_request": checkpoint.get("model_matches_request", False),
                    "latency_seconds": checkpoint.get("latency_seconds", 0.0),
                    "usage_observed_attempts": checkpoint.get("usage_observed_attempts", 0),
                    "unknown_usage_attempts": checkpoint.get("unknown_usage_attempts", 0),
                    "observed_input_tokens": checkpoint.get("observed_input_tokens", 0),
                    "observed_output_tokens": checkpoint.get("observed_output_tokens", 0),
                    "observed_total_tokens": checkpoint.get("observed_total_tokens", 0),
                    "estimated_prompt_tokens": request["estimated_prompt_tokens"],
                    "estimated_completion_tokens": request["estimated_completion_tokens"],
                    "estimated_total_tokens": request["estimated_total_tokens"],
                    "logical_query_mappings": logical_multiplicity[physical_id],
                    "historical_imported_response": False,
                }
            )
        cost_path = self.paths.metrics / "api_cost_audit.csv"
        _atomic_csv(cost_path, cost_rows, cost_fields)
        finalized_hashes = {
            "complementarity": sha256_file(complementarity_path),
            "group": sha256_file(group_path),
            "api_cost_audit": sha256_file(cost_path),
        }
        summary = {
            "complementarity_cells": len(complementarity_rows),
            "group_cell_condition_rows": len(group_rows),
            "physical_cost_rows": len(cost_rows),
            "quadrants": dict(sorted(Counter(row["outcome_quadrant"] for row in complementarity_rows).items())),
            "records": {
                "complementarity": str(complementarity_path.relative_to(self.paths.run_dir)),
                "group": str(group_path.relative_to(self.paths.run_dir)),
                "api_cost_audit": str(cost_path.relative_to(self.paths.run_dir)),
            },
            "artifact_sha256": finalized_hashes,
        }
        _atomic_json(self.paths.metrics / "finalization_summary.json", summary)
        self._mark_stage("finalize", summary)
        self.validate(require_execution=True, require_finalized=True)
        return summary

    def validate(
        self,
        *,
        require_execution: bool = True,
        require_finalized: bool = True,
    ) -> dict[str, object]:
        """Audit frozen identities, matched partitions, coverage, and stages."""

        self._bind_configs()
        required_plan = (
            self.paths.evidence / "plan_summary.json",
            self.paths.evidence / "singleton_population.jsonl",
            self.paths.evidence / "structured_partition.jsonl",
            self.paths.evidence / "random_partition.jsonl",
            self.paths.evidence / "logical_queries.jsonl",
            self.physical_requests_path,
            self.paths.provenance / "execution_schedule.jsonl",
            self.paths.provenance / "freshness_audit.json",
            self.paths.provenance / "label_blind_plan_audit.json",
            self.paths.provenance / "data_fingerprint.json",
        )
        missing = [str(path) for path in required_plan if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing motivation evidence artifacts: " + ", ".join(missing))
        summary = _load_object(self.paths.evidence / "plan_summary.json")
        data_manifest = self.paths.data_root / "manifest.json"
        data_audit = validate_manifest(self.paths.data_root, data_manifest)
        fingerprint = _load_object(self.paths.provenance / "data_fingerprint.json")
        if (
            str(fingerprint.get("manifest_sha256", "")) != sha256_file(data_manifest)
            or dict(fingerprint.get("manifest_audit") or {}) != data_audit.as_dict()
            or str(fingerprint.get("config_sha256", ""))
            != sha256_file(self.paths.configs / "motivation_evidence.json")
            or str(fingerprint.get("llm_config_sha256", ""))
            != sha256_file(self.paths.configs / "deepseek_v4.json")
        ):
            raise AssertionError("data/config fingerprint drift")
        _verify_file_set(
            dict(fingerprint.get("code_sha256") or {}),
            relative_to=self.paths.project_root,
            label="experiment code",
        )
        for name, expected_hash in dict(summary["artifact_sha256"]).items():
            path_by_name = {
                "singleton_population": self.paths.evidence / "singleton_population.jsonl",
                "leaf_orders": self.paths.evidence / "leaf_orders.jsonl",
                "structured_partition": self.paths.evidence / "structured_partition.jsonl",
                "random_partition": self.paths.evidence / "random_partition.jsonl",
                "logical_queries": self.paths.evidence / "logical_queries.jsonl",
                "physical_requests": self.physical_requests_path,
                "execution_schedule": self.paths.provenance / "execution_schedule.jsonl",
            }[str(name)]
            if sha256_file(path_by_name) != str(expected_hash):
                raise AssertionError(f"frozen artifact hash drift: {name}")
        baran_hashes = dict(summary.get("baran_artifact_sha256") or {})
        expected_baran_paths = {
            path.relative_to(self.paths.run_dir).as_posix()
            for path in self._baran_artifact_paths()
        }
        if set(baran_hashes) != expected_baran_paths:
            raise AssertionError("fresh Baran artifact set differs from the frozen plan")
        _verify_file_set(
            baran_hashes,
            relative_to=self.paths.run_dir,
            label="fresh Baran artifact",
        )

        population_ids: set[str] = set()
        population_meta: dict[str, tuple[str, str, str]] = {}
        population_by_dataset: Counter[str] = Counter()
        for row in _iter_jsonl(self.paths.evidence / "singleton_population.jsonl"):
            assert_label_blind_artifact(row)
            cell_id = str(row["cell_id"])
            if cell_id in population_ids:
                raise AssertionError(f"duplicate singleton population cell: {cell_id}")
            population_ids.add(cell_id)
            population_meta[cell_id] = (
                str(row["suite"]), str(row["dataset"]), str(row["column"])
            )
            population_by_dataset[f"{row['suite']}/{row['dataset']}"] += 1
        if len(population_ids) != int(self.config["expected"]["singleton_cells"]):
            raise AssertionError("singleton population is not the frozen 22,198 cells")
        expected_population = {
            f"{row['suite']}/{row['dataset']}": int(row["oracle_errors"])
            for row in self._dataset_specs()
        }
        if dict(population_by_dataset) != expected_population:
            raise AssertionError("per-dataset population counts differ from the frozen protocol")

        from .baran import assert_online_baran_record_safe

        baran_ids: set[str] = set()
        for path in self._baran_artifact_paths():
            for row in _iter_jsonl(path):
                assert_online_baran_record_safe(row)
                cell_id = str(row.get("cell_id", ""))
                if not cell_id or cell_id in baran_ids:
                    raise AssertionError(f"duplicate or empty fresh Baran cell ID: {cell_id!r}")
                baran_ids.add(cell_id)
        if baran_ids != population_ids:
            raise AssertionError("fresh Baran universe differs from the frozen population")

        structured_by_condition: dict[tuple[object, ...], list[PartitionGroup]] = defaultdict(list)
        random_by_condition: dict[tuple[object, ...], list[PartitionGroup]] = defaultdict(list)

        def load_partition(path: Path, target: dict[tuple[object, ...], list[PartitionGroup]]) -> None:
            for row in _iter_jsonl(path):
                assert_label_blind_artifact(row)
                group = PartitionGroup(
                    suite=str(row["suite"]), dataset=str(row["dataset"]),
                    source_view=str(row["source_view"]), group_size=int(row["group_size"]),
                    column=str(row["column"]), arm=str(row["arm"]),
                    group_index=int(row["group_index"]),
                    ordered_cell_ids=tuple(row["ordered_cell_ids"]),
                    structured_group_id=str(row.get("structured_group_id", "")),
                )
                target[group.condition_key].append(group)
                for cell_id in group.ordered_cell_ids:
                    if population_meta.get(cell_id) != (
                        group.suite, group.dataset, group.column
                    ):
                        raise AssertionError(
                            "partition cell escaped its frozen suite/dataset/column bucket"
                        )

        load_partition(self.paths.evidence / "structured_partition.jsonl", structured_by_condition)
        load_partition(self.paths.evidence / "random_partition.jsonl", random_by_condition)
        if set(structured_by_condition) != set(random_by_condition):
            raise AssertionError("partition condition sets differ")
        for condition in structured_by_condition:
            structured = sorted(structured_by_condition[condition], key=lambda value: value.group_index)
            randomised = sorted(random_by_condition[condition], key=lambda value: value.group_index)
            validate_matched_partitions(structured, randomised)
        partition_group_ids: dict[tuple[object, ...], str] = {}
        for arm, grouped in (
            ("structured", structured_by_condition),
            ("random", random_by_condition),
        ):
            for groups in grouped.values():
                for group in groups:
                    identity = (
                        arm,
                        group.suite,
                        group.dataset,
                        group.source_view,
                        group.group_size,
                        group.column,
                        group.ordered_cell_ids,
                    )
                    if identity in partition_group_ids:
                        raise AssertionError("duplicate exact partition group identity")
                    partition_group_ids[identity] = group.group_id
        expected_group_memberships: dict[
            tuple[str, str, str, int, str], dict[str, object]
        ] = {}
        for groups in structured_by_condition.values():
            for group in groups:
                for position, cell_id in enumerate(group.ordered_cell_ids):
                    key = (
                        group.suite,
                        group.dataset,
                        group.source_view,
                        group.group_size,
                        cell_id,
                    )
                    if key in expected_group_memberships:
                        raise AssertionError("structured partition repeats a cell condition")
                    expected_group_memberships[key] = {
                        "member_position": position,
                        "structured_group_id": group.group_id,
                        "random_group_id": "",
                    }
        for groups in random_by_condition.values():
            for group in groups:
                for position, cell_id in enumerate(group.ordered_cell_ids):
                    key = (
                        group.suite,
                        group.dataset,
                        group.source_view,
                        group.group_size,
                        cell_id,
                    )
                    membership = expected_group_memberships.get(key)
                    if membership is None or membership["random_group_id"]:
                        raise AssertionError("random partition cell pairing is ambiguous")
                    if int(membership["member_position"]) != position:
                        raise AssertionError("random partition changed member position")
                    membership["random_group_id"] = group.group_id
        if any(not row["random_group_id"] for row in expected_group_memberships.values()):
            raise AssertionError("random partition does not cover structured incidences")

        client = DeepSeekGroupClient(GroupClientConfig.from_mapping(self.llm_config), api_key="")
        physical_ids: set[str] = set()
        physical_hashes: dict[str, str] = {}
        physical_identity: dict[str, dict[str, Any]] = {}
        physical_order: list[str] = []
        for expected_index, request in enumerate(_iter_jsonl(self.physical_requests_path)):
            assert_label_blind_artifact(request)
            identifier = str(request["physical_query_id"])
            request_hash = str(request["provider_request_hash"])
            if identifier in physical_ids or identifier != physical_query_id(request_hash):
                raise AssertionError("invalid or duplicate physical request identity")
            if int(request["physical_schedule_index"]) != expected_index:
                raise AssertionError("physical request schedule indices are not contiguous")
            job = StreamingCheckpointExecutor._job_from_record(request)
            if client.provider_request_hash(job) != request_hash:
                raise AssertionError(f"provider request hash drift for {identifier}")
            if str(request.get("model_requested", "")) != client.config.model:
                raise AssertionError("physical request requested-model metadata drift")
            if compute_prompt_hash(
                request["messages"],
                int(request["max_tokens"]),
                prompt_schema_version=str(self.config["prompt"]["schema_version"]),
            ) != str(request["prompt_hash"]):
                raise AssertionError(f"prompt hash drift for {identifier}")
            assert_messages_safe(request["messages"])
            user_payload = json.loads(str(request["messages"][1]["content"]))
            if not isinstance(user_payload, Mapping):
                raise AssertionError("primary user Prompt is not a JSON object")
            payload_keys = set(_walk_keys(user_payload))
            if payload_keys.intersection({"arm", "source_view"}):
                raise AssertionError("arm/source_view provenance leaked into a primary prompt")
            ordered_request = tuple(str(value) for value in request["ordered_cell_ids"])
            prompt_group = user_payload.get("group")
            prompt_dataset = user_payload.get("dataset")
            targets = user_payload.get("targets")
            if not isinstance(prompt_group, Mapping) or not isinstance(prompt_dataset, Mapping):
                raise AssertionError("primary Prompt lacks group/dataset identity")
            if not isinstance(targets, list) or any(not isinstance(row, Mapping) for row in targets):
                raise AssertionError("primary Prompt targets are malformed")
            target_ids = tuple(str(row["cell_id"]) for row in targets)
            group_ids = tuple(str(value) for value in prompt_group.get("cell_ids", ()))
            expected_view = (
                "singleton" if len(ordered_request) == 1 else str(self.config["neutral_group_view"])
            )
            if (
                str(user_payload.get("query_id", "")) != str(request["request_query_id"])
                or target_ids != ordered_request
                or group_ids != ordered_request
                or int(prompt_group.get("size", -1)) != len(ordered_request)
                or str(prompt_group.get("view", "")) != expected_view
                or int(request["max_tokens"]) != completion_token_ceiling(len(ordered_request))
            ):
                raise AssertionError("primary Prompt/request ordered identity drift")
            physical_ids.add(identifier)
            physical_hashes[identifier] = request_hash
            compact_request = {
                field: request[field]
                for field in (
                    "physical_query_id",
                    "request_query_id",
                    "prompt_hash",
                    "provider_request_hash",
                    "max_tokens",
                    "physical_schedule_index",
                    "ordered_cell_ids",
                    "model_requested",
                )
            }
            physical_identity[identifier] = {
                "request_query_id": str(request["request_query_id"]),
                "prompt_hash": str(request["prompt_hash"]),
                "ordered_cell_ids": ordered_request,
                "suite": str(prompt_dataset.get("suite", "")),
                "dataset": str(prompt_dataset.get("name", "")),
                "prompt_group_view": expected_view,
                # Never retain 100k full Prompt bodies in the validator.  All
                # payload/message checks above are streaming; later stages
                # need only the compact frozen request identity.
                "request": compact_request,
            }
            physical_order.append(identifier)
        if len(physical_ids) != int(summary["physical_calls_after_exact_dedup"]):
            raise AssertionError("physical call count differs from the plan summary")
        schedule_ids: list[str] = []
        for expected_index, row in enumerate(
            _iter_jsonl(self.paths.provenance / "execution_schedule.jsonl")
        ):
            if int(row["physical_schedule_index"]) != expected_index:
                raise AssertionError("execution schedule indices are not contiguous")
            identifier = str(row["physical_query_id"])
            if physical_hashes.get(identifier) != str(row["provider_request_hash"]):
                raise AssertionError("execution schedule identity drift")
            schedule_ids.append(identifier)
        if schedule_ids != physical_order:
            raise AssertionError("physical requests do not follow the frozen execution schedule")

        logical_ids: set[str] = set()
        logical_count = 0
        for row in _iter_jsonl(self.paths.evidence / "logical_queries.jsonl"):
            assert_label_blind_artifact(row)
            logical_count += 1
            logical_id = str(row["logical_query_id"])
            if logical_id in logical_ids:
                raise AssertionError(f"duplicate logical query ID: {logical_id}")
            logical_ids.add(logical_id)
            physical_id = str(row["physical_query_id"])
            if physical_id not in physical_ids:
                raise AssertionError("logical query maps to an unknown physical request")
            if physical_hashes[physical_id] != str(row["provider_request_hash"]):
                raise AssertionError("logical/physical provider request hash mismatch")
            physical = physical_identity[physical_id]
            ordered = tuple(str(value) for value in row["ordered_cell_ids"])
            if (
                ordered != physical["ordered_cell_ids"]
                or str(row["request_query_id"]) != physical["request_query_id"]
                or str(row["prompt_hash"]) != physical["prompt_hash"]
                or str(row["suite"]) != physical["suite"]
                or str(row["dataset"]) != physical["dataset"]
            ):
                raise AssertionError("logical row maps to the wrong frozen physical request")
            arm = str(row["arm"])
            group_size = int(row["group_size"])
            if group_size != len(ordered):
                raise AssertionError("logical group_size differs from ordered targets")
            if arm == "singleton":
                if (
                    group_size != 1
                    or str(row["source_view"]) != "singleton"
                    or str(row.get("group_id", ""))
                ):
                    raise AssertionError("singleton logical provenance is malformed")
            elif arm in {"structured", "random"}:
                partition_identity = (
                    arm,
                    str(row["suite"]),
                    str(row["dataset"]),
                    str(row["source_view"]),
                    group_size,
                    str(row["column"]),
                    ordered,
                )
                if partition_group_ids.get(partition_identity) != str(row.get("group_id", "")):
                    raise AssertionError("logical group_id/membership differs from frozen partition")
            else:
                raise AssertionError(f"unsupported logical arm: {arm!r}")
            expected_logical_id = compute_logical_query_id(
                suite=str(row["suite"]), dataset=str(row["dataset"]), arm=str(row["arm"]),
                source_view=str(row["source_view"]), group_size=int(row["group_size"]),
                ordered_cell_ids=ordered,
                prompt_schema_version=str(self.config["prompt"]["schema_version"]),
            )
            if logical_id != expected_logical_id:
                raise AssertionError("logical query ID is not bound to ordered provenance")
            if any(
                population_meta.get(cell_id) != (
                    str(row["suite"]), str(row["dataset"]), str(row["column"])
                )
                for cell_id in ordered
            ):
                raise AssertionError("logical query contains a cell from another bucket")
            expected_request_id = (
                compute_query_id(
                    str(row["suite"]), str(row["dataset"]), "singleton", ordered,
                    arm="singleton",
                    prompt_schema_version=str(self.config["prompt"]["schema_version"]),
                )
                if int(row["group_size"]) == 1
                else compute_ordered_query_id(
                    str(row["suite"]), str(row["dataset"]), ordered,
                    group_view=str(self.config["neutral_group_view"]),
                    prompt_schema_version=str(self.config["prompt"]["schema_version"]),
                )
            )
            if str(row["request_query_id"]) != expected_request_id:
                raise AssertionError("request query ID is not bound to ordered targets")
        if logical_count != int(self.config["expected"]["logical_calls_before_exact_dedup"]):
            raise AssertionError("logical query total differs from the frozen 99,466 calls")

        execution_terminal = 0
        if require_execution:
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError("LLM checkpoint is missing")
            terminal_ids: set[str] = set()
            request_hash_by_physical: dict[str, str] = {}
            for row in _iter_jsonl(self.checkpoint_path):
                identifier = str(row.get("physical_query_id", ""))
                request_hash = str(row.get("provider_request_hash", ""))
                if not identifier or not request_hash:
                    raise AssertionError("checkpoint row lacks physical request identity")
                previous_hash = request_hash_by_physical.setdefault(identifier, request_hash)
                if previous_hash != request_hash:
                    raise AssertionError("checkpoint physical identity maps to multiple requests")
                if str(row.get("status", "")) not in TERMINAL_EXECUTION_STATUSES:
                    raise AssertionError("formal checkpoint contains a non-terminal record")
                if identifier in terminal_ids or identifier not in physical_identity:
                    raise AssertionError("checkpoint contains a duplicate or unknown physical query")
                _assert_terminal_checkpoint_matches_request(
                    row,
                    physical_identity[identifier]["request"],
                    model_requested=str(self.llm_config["model"]),
                )
                terminal_ids.add(identifier)
            if terminal_ids != physical_ids:
                missing_ids = physical_ids.difference(terminal_ids)
                raise AssertionError(f"execution is incomplete: {len(missing_ids)} physical calls missing")
            execution_terminal = len(terminal_ids)

        finalized_rows: dict[str, int] = {}
        if require_finalized:
            finalization_summary_path = self.paths.metrics / "finalization_summary.json"
            if not finalization_summary_path.is_file():
                raise FileNotFoundError(
                    f"finalized artifact is missing: {finalization_summary_path}"
                )
            finalization_summary = _load_object(finalization_summary_path)
            expected_outputs = {
                "complementarity": (
                    self.paths.records / "complementarity_cell_outcomes.csv",
                    int(self.config["expected"]["singleton_cells"]),
                ),
                "group": (
                    self.paths.records / "group_cell_outcomes.csv",
                    sum(int(value) for value in self.config["expected"]["structured_cell_incidences_by_size"].values()),
                ),
                "cost": (self.paths.metrics / "api_cost_audit.csv", len(physical_ids)),
            }
            expected_records = {
                "complementarity": str(
                    expected_outputs["complementarity"][0].relative_to(self.paths.run_dir)
                ),
                "group": str(expected_outputs["group"][0].relative_to(self.paths.run_dir)),
                "api_cost_audit": str(
                    expected_outputs["cost"][0].relative_to(self.paths.run_dir)
                ),
            }
            if dict(finalization_summary.get("records") or {}) != expected_records:
                raise AssertionError("finalization summary record paths drifted")
            expected_hashes = {
                "complementarity": sha256_file(expected_outputs["complementarity"][0]),
                "group": sha256_file(expected_outputs["group"][0]),
                "api_cost_audit": sha256_file(expected_outputs["cost"][0]),
            }
            if dict(finalization_summary.get("artifact_sha256") or {}) != expected_hashes:
                raise AssertionError("finalized ledger hash drift")
            for name, (path, expected_rows) in expected_outputs.items():
                with path.open("r", encoding="utf-8", newline="") as handle:
                    count = sum(1 for _ in csv.DictReader(handle))
                if count != expected_rows:
                    raise AssertionError(
                        f"{name} finalized row count mismatch: expected {expected_rows}, got {count}"
                    )
                finalized_rows[name] = count
            semantic_summary = _validate_finalized_ledgers(
                complementarity_path=expected_outputs["complementarity"][0],
                group_path=expected_outputs["group"][0],
                cost_path=expected_outputs["cost"][0],
                population_ids=population_ids,
                expected_group_memberships=expected_group_memberships,
                physical_requests={
                    identifier: value["request"]
                    for identifier, value in physical_identity.items()
                },
            )
            for field, value in semantic_summary.items():
                if finalization_summary.get(field) != value:
                    raise AssertionError(f"finalization summary {field} drift")

        return {
            "valid": True,
            "run_id": self.run_id,
            "population_cells": len(population_ids),
            "partition_conditions": len(structured_by_condition),
            "logical_queries": logical_count,
            "physical_queries": len(physical_ids),
            "terminal_execution_records": execution_terminal,
            "finalized_rows": finalized_rows,
            "historical_inputs_accepted": False,
            "weights_hashed": False,
        }

    def report_results(self) -> dict[str, object]:
        """Build statistics, PDF/SVG, and Markdown through the reporting boundary."""

        self.validate(require_execution=True, require_finalized=True)
        from .motivation_reporting import build_motivation_report

        result = build_motivation_report(
            self.paths.run_dir,
            bootstrap_replicates=int(self.config["bootstrap_replicates"]),
            bootstrap_seed=int(self.config["bootstrap_seed"]),
            confidence=float(self.config["confidence_level"]),
        )
        payload = dict(result) if isinstance(result, Mapping) else {"result": str(result)}
        self._mark_stage("report", payload)
        return payload


__all__ = [
    "DEFAULT_RUN_ID",
    "FORMAL_DATASETS",
    "PRIMARY_GROUP_SIZES",
    "PRIMARY_VIEWS",
    "EvidencePaths",
    "MotivationEvidenceRunner",
    "PartitionGroup",
    "StreamingCheckpointExecutor",
    "assert_label_blind_artifact",
    "build_random_partition",
    "build_structured_partition",
    "compute_logical_query_id",
    "compute_request_query_id",
    "deduplicate_physical_schedule",
    "physical_query_id",
    "round_robin_logical_schedule",
    "validate_matched_partitions",
]
