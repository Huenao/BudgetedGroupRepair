"""Resumable Router-v3 orchestration for No-Baran-Prompt Group Repair.

The runner keeps the online path physically separate from evaluation labels:
group generation, prompting, gating, selection, and verification receive only
``SafeDataset``/``SafeCell`` plus fresh label-free Baran records.  Clean values
are bound only while constructing calibration labels and final metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .baran import assert_online_baran_record_safe, run_baran
from .cell_features import CellFeatures
from .data import (
    EXPECTED_DATASET_COUNT,
    EXPECTED_ORACLE_ERROR_COUNT,
    LoadedDataset,
    SafeCell,
    load_dataset,
    normalize_for_match,
    read_jsonl,
    validate_manifest,
    write_jsonl,
)
from .group_context import (
    PROMPT_SCHEMA_VERSION,
    SYSTEM_PROMPT,
    canonical_messages,
    compute_prompt_hash,
    estimate_prompt_tokens,
)
from .group_gate import GroupUpliftGate, build_uplift_targets
from .group_generator import GroupGenerator, GroupQueryAction
from .group_llm import (
    DeepSeekGroupClient,
    GroupClientConfig,
    GroupLLMJob,
    ParsedRepairItem,
    run_group_llm_batch,
)
from .group_objective import GroupUpliftObjective, PairGain
from .group_optimizer import select_queries
from .metrics import (
    compare_methods,
    compute_aubc,
    summarize_records,
    verify_records,
)
from .protocol import (
    PROTOCOL_NAME,
    base_family,
    split_for_target,
    target_order as full_target_order,
)
from .prompt_policy import INFORMATION_POLICY, assert_messages_safe
from .sampling import SELECTED_DATASETS
from .public_fd import (
    PublicFD,
    build_fd_violation_components,
    fds_for_dataset,
    load_public_fds,
)
from .run_state import (
    RunState,
    build_run_binding,
    canonical_json_sha256,
    redacted_config,
    sha256_file,
    write_json,
)
from .statistics import exact_mcnemar, holm_adjust
from .verifier import GroupRepairVerifier, RankedRepairCandidate, VerifierConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_GATE_BACKENDS = ("lightgbm", "xgboost")
CATBOOST_GATE_BACKENDS = ("catboost",)
ROUTER_V3_REVISION = "router_v3_exact_size_conditioned"
ROUTER_V3_BUDGET_SWEEP_REVISION = (
    "router_v3_budget_sweep_exact_size_conditioned"
)
ROUTER_V3_CATBOOST_REVISION = "router_v3_catboost_exact_size_conditioned"
ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION = (
    "router_v4_lightgbm_isotonic_exact_size_conditioned"
)
ROUTER_V3_VARIANTS = ("1", "2", "4", "8", "all")
ROUTER_V3_SWEEP_VARIANTS = ("2", "4")
ROUTER_V3_SWEEP_BUDGETS = (0.01, 0.05, 0.1, 0.2, 0.5)
FROZEN_ROUTER_V3_IMPLEMENTATION_SHA256 = frozenset(
    {"979d3992fee2f99cd489009f468faa1e564994dfb5174778ea1302b9fe7784b5"}
)
FROZEN_ROUTER_V3_BUDGET_SWEEP_IMPLEMENTATION_SHA256 = frozenset(
    {"f0fcc9def342e0219f61f2383bb22908c177a43eb1b12ad8baac5d30c1bbd01c"}
)
FROZEN_ROUTER_V3_CATBOOST_IMPLEMENTATION_SHA256 = frozenset(
    {"1ac1d8e816e693bd3459ba69e57f5b633275a8ac2f2d2235ec5ddce193f4e54f"}
)


def _router_v3_implementation_binding_matches(
    revision: str,
    bound_implementation: str,
    current_implementation: str,
) -> bool:
    if bound_implementation == current_implementation:
        return True
    frozen_by_revision = {
        ROUTER_V3_REVISION: FROZEN_ROUTER_V3_IMPLEMENTATION_SHA256,
        ROUTER_V3_BUDGET_SWEEP_REVISION: (
            FROZEN_ROUTER_V3_BUDGET_SWEEP_IMPLEMENTATION_SHA256
        ),
        ROUTER_V3_CATBOOST_REVISION: (
            FROZEN_ROUTER_V3_CATBOOST_IMPLEMENTATION_SHA256
        ),
    }
    return bound_implementation in frozen_by_revision.get(revision, frozenset())


TEST_TARGETS = tuple(SELECTED_DATASETS)
TEST_TARGET_CELL_COUNT = 22_198
CALIBRATION_SINGLETON_CELL_COUNT = 5_543
INVALID_RESPONSE_POLICIES = (
    "fail_and_resume",
    "fallback_baran_after_client_retries",
)
REQUIRED_STAGES = (
    "input_validation",
    "baran",
    "groups",
    "calibration_plan",
    "model_preflight",
    "calibration_llm",
    "gate_selection",
    "router_diagnostics",
    "selected_llm",
    "final_records",
    "metrics",
    "audit",
)
BASELINE_REQUIRED_STAGES = (
    "input_validation",
    "baran",
    "groups",
    "response_reuse",
    "model_preflight",
    "selected_llm",
    "final_records",
    "metrics",
)

MODEL_FEATURE_COLUMNS = (
    "dirty_type",
    "dirty_format",
    "baran_type",
    "baran_format",
    "baran_changed",
    "baran_candidate_count",
    "baran_top_support",
    "baran_support_margin",
    "baran_source_agreement",
    "group_view",
    "group_size",
    "cohesion",
    "same_row",
    "same_column",
    "dirty_type_count",
    "baran_type_count",
    "baran_changed_share",
    "estimated_prompt_tokens",
    "completion_token_ceiling",
    "estimated_total_tokens",
    "shared_prefix_saving_ratio",
    "member_index",
    "member_position",
    "group_same_dirty_type",
    "group_same_baran_type",
)


def target_order() -> tuple[tuple[str, str], ...]:
    """Return only the nine frozen formal-test targets."""

    return TEST_TARGETS


def generation_order() -> tuple[tuple[str, str], ...]:
    """Return the five Source plus all nine TableEG datasets used by Router-v3."""

    return full_target_order()

LOGICAL_LEDGER_COLUMNS = (
    "target_suite",
    "target_dataset",
    "backend",
    "scenario",
    "group_size_variant",
    "budget_share",
    "query_id",
    "prompt_hash",
    "selected",
    "estimated_tokens",
    "actual_tokens_if_available",
    "logical_api_calls",
    "physical_api_calls",
    "covered_cells",
    "accepted_llm_cells",
)

class SafetyCapExceeded(RuntimeError):
    """Raised before an API phase whose estimate exceeds the fixed cap."""

    def __init__(self, phase: str, estimated_tokens: int, cap: int) -> None:
        self.phase = str(phase)
        self.estimated_tokens = int(estimated_tokens)
        self.cap = int(cap)
        super().__init__(
            f"{self.phase} estimated token total {self.estimated_tokens:,} "
            f"exceeds safety cap {self.cap:,}"
        )


@dataclass(frozen=True)
class ExperimentPaths:
    project_root: Path
    data_root: Path
    config_path: Path
    llm_config_path: Path
    vendor_root: Path
    runs_root: Path
    run_dir: Path

    @property
    def baran_dir(self) -> Path:
        return self.run_dir / "baran"

    @property
    def cell_features_dir(self) -> Path:
        return self.run_dir / "cell_features"

    @property
    def groups_dir(self) -> Path:
        return self.run_dir / "groups"

    @property
    def llm_dir(self) -> Path:
        return self.run_dir / "llm"

    @property
    def gates_dir(self) -> Path:
        return self.run_dir / "gates"

    @property
    def selections_dir(self) -> Path:
        return self.run_dir / "selections"

    @property
    def final_dir(self) -> Path:
        return self.run_dir / "final"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"


def load_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return value


def _stage_details(values: Mapping[str, object]) -> dict[str, object]:
    """Keep result status distinct from RunState.update_stage's status."""

    details = dict(values)
    if "status" in details:
        details["result_status"] = details.pop("status")
    return details


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("bgr_deepseek_v4_%Y%m%dT%H%M%SZ")


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    *,
    columns: Sequence[str] | None = None,
) -> None:
    materialized = [{str(key): _json_safe(value) for key, value in row.items()} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(materialized, columns=list(columns) if columns is not None else None)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def _hash_tree(root: Path, suffixes: Sequence[str]) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in set(suffixes)
        and not any(part in {"__pycache__", ".pytest_cache", ".venv", "runs", "data"} for part in path.parts)
    )
    for path in files:
        label = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _dataset_key(suite: str, dataset: str) -> str:
    return f"{suite}__{dataset}"


def _budget_label(value: float) -> str:
    return f"{int(round(float(value) * 100)):02d}pct"


def _action_from_dict(raw: Mapping[str, object]) -> GroupQueryAction:
    messages = raw.get("messages")
    features = raw.get("group_features")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("serialized action messages must be an array")
    if not isinstance(features, Mapping):
        raise ValueError("serialized action group_features must be an object")
    return GroupQueryAction(
        query_id=str(raw["query_id"]),
        suite=str(raw["suite"]),
        dataset=str(raw["dataset"]),
        arm=str(raw["arm"]),
        group_view=str(raw["group_view"]),
        cell_ids=tuple(str(value) for value in raw["cell_ids"]),  # type: ignore[arg-type]
        group_size=int(raw["group_size"]),
        prompt_schema_version=str(raw["prompt_schema_version"]),
        prompt_information_policy=str(raw["prompt_information_policy"]),
        messages=canonical_messages(messages),  # type: ignore[arg-type]
        prompt_hash=str(raw["prompt_hash"]),
        estimated_prompt_tokens=int(raw["estimated_prompt_tokens"]),
        completion_token_ceiling=int(raw["completion_token_ceiling"]),
        estimated_total_tokens=int(raw["estimated_total_tokens"]),
        group_features=dict(features),
    )


def _actual_tokens(record: Mapping[str, object] | None) -> int | None:
    if not record:
        return None
    observed_total = record.get("observed_total_tokens")
    observed_attempts = int(record.get("usage_observed_attempts", 0) or 0)
    if observed_total not in (None, "") and observed_attempts > 0:
        try:
            return int(observed_total)
        except (TypeError, ValueError):
            return None
    usage = record.get("usage")
    if not isinstance(usage, Mapping):
        return None
    direct = usage.get("total_tokens")
    if direct not in (None, ""):
        try:
            return int(direct)
        except (TypeError, ValueError):
            return None
    prompt = next(
        (usage.get(key) for key in ("prompt_tokens", "input_tokens") if usage.get(key) not in (None, "")),
        None,
    )
    completion = next(
        (usage.get(key) for key in ("completion_tokens", "output_tokens") if usage.get(key) not in (None, "")),
        None,
    )
    if prompt is None or completion is None:
        return None
    try:
        return int(prompt) + int(completion)
    except (TypeError, ValueError):
        return None


def _usage_value(record: Mapping[str, object], *keys: str) -> int | None:
    usage = record.get("usage")
    if not isinstance(usage, Mapping):
        return None
    for key in keys:
        if usage.get(key) not in (None, ""):
            try:
                return int(usage[key])
            except (TypeError, ValueError):
                return None
    return None


def _validate_api_cost_resolution(api_cost: pd.DataFrame) -> None:
    """Reject only provider failures that have no explicit operational fallback."""

    if "unresolved_operational_failures" not in api_cost.columns:
        raise ValueError(
            "api_cost_audit.csv is missing unresolved_operational_failures"
        )
    total = api_cost.loc[
        api_cost["phase"].astype(str) == "total_fresh_experiment"
    ]
    if len(total) != 1:
        raise ValueError("api_cost_audit.csv must contain one total_fresh_experiment row")
    if int(float(total.iloc[0]["unresolved_operational_failures"])) != 0:
        raise ValueError("api_cost_audit.csv contains unresolved operational failures")


def _prediction(record: Mapping[str, object], dirty_value: str) -> str:
    return str(record.get("prediction", dirty_value))


def _is_baran_correct(record: Mapping[str, object], clean_value: str) -> bool:
    return (
        str(record.get("parse_status", "")).startswith("ok")
        and normalize_for_match(record.get("prediction")) == normalize_for_match(clean_value)
    )


def _basic_item_valid(
    item: ParsedRepairItem | None,
    cell: SafeCell,
    baran_record: Mapping[str, object],
) -> bool:
    if item is None or item.decision != "propose":
        return False
    candidate = normalize_for_match(item.repair)
    return bool(
        candidate
        and candidate != normalize_for_match(cell.dirty_value)
        and candidate != normalize_for_match(_prediction(baran_record, cell.dirty_value))
    )


class ExperimentRunner:
    """Execute the No-Baran Router-v3 calibration and nine-target matrix."""

    def __init__(
        self,
        paths: ExperimentPaths,
        state: RunState,
        experiment_config: Mapping[str, object],
        llm_config: Mapping[str, object],
        *,
        provider_token_cap: int | None = None,
        allow_uncapped_provider_usage: bool = False,
    ) -> None:
        self.paths = paths
        self.state = state
        self.experiment_config = dict(experiment_config)
        self.llm_config = dict(llm_config)
        self.router_revision = str(
            self.experiment_config.get("router_revision", ROUTER_V3_REVISION)
        )
        manifest = state.manifest
        baran_source = str(manifest.get("baran_source_run") or "").strip()
        response_source = str(manifest.get("response_reuse_run") or "").strip()
        self.baran_source_run = Path(baran_source).resolve() if baran_source else None
        self.response_reuse_run = (
            Path(response_source).resolve() if response_source else None
        )
        calibration_source = str(
            manifest.get("calibration_source_run") or ""
        ).strip()
        self.calibration_source_run = (
            Path(calibration_source).resolve() if calibration_source else None
        )
        artifact_source = str(manifest.get("router_artifact_reuse_run", ""))
        self.router_artifact_reuse_run = (
            Path(artifact_source).resolve() if artifact_source else None
        )
        comparison_source = str(manifest.get("router_comparison_run", ""))
        self.router_comparison_run = (
            Path(comparison_source).resolve() if comparison_source else None
        )
        self.provider_token_cap = provider_token_cap
        self.allow_uncapped_provider_usage = bool(allow_uncapped_provider_usage)
        self.fd_registry = load_public_fds(paths.project_root / "configs" / "public_fds.json")
        self._datasets: dict[tuple[str, str], LoadedDataset] = {}
        self._baran: dict[tuple[str, str], list[dict[str, object]]] = {}

        if str(self.experiment_config.get("protocol")) != PROTOCOL_NAME:
            raise ValueError(f"experiment protocol must be {PROTOCOL_NAME!r}")
        if str(self.experiment_config.get("prompt_information_policy")) != INFORMATION_POLICY:
            raise ValueError(
                f"prompt_information_policy must be {INFORMATION_POLICY!r}"
            )
        if self.router_revision not in {
            ROUTER_V3_REVISION,
            ROUTER_V3_BUDGET_SWEEP_REVISION,
            ROUTER_V3_CATBOOST_REVISION,
            ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION,
        }:
            raise ValueError(f"unsupported router_revision: {self.router_revision!r}")
        backends = self._active_gate_backends()
        if self.is_router_v4_lightgbm_isotonic:
            if backends != ("lightgbm",):
                raise ValueError(
                    "Router-v4 isotonic gate_backends must be exactly lightgbm"
                )
            if self.calibration_source_run is None:
                raise ValueError(
                    "Router-v4 isotonic requires --calibration-source-run"
                )
        elif self.is_router_v3_budget_sweep:
            if backends != ("lightgbm",):
                raise ValueError(
                    "Router-v3 budget sweep gate_backends must be exactly lightgbm"
                )
            if self.router_artifact_reuse_run is None:
                raise ValueError(
                    "Router-v3 budget sweep requires --router-artifact-reuse-run"
                )
        elif self.is_router_v3_catboost:
            if backends != CATBOOST_GATE_BACKENDS:
                raise ValueError(
                    "Router-v3 CatBoost gate_backends must be exactly catboost"
                )
        elif len(backends) != 2 or set(backends) != set(EXPECTED_GATE_BACKENDS):
            raise ValueError("gate_backends must contain exactly lightgbm and xgboost")
        if str(self.llm_config.get("model")) != "deepseek-v4-flash":
            raise ValueError("formal BGR runs require model='deepseek-v4-flash'")
        if str(self.llm_config.get("prompt_schema_version")) != PROMPT_SCHEMA_VERSION:
            raise ValueError(
                f"prompt_schema_version must be {PROMPT_SCHEMA_VERSION!r}"
            )
        invalid_response_policy = str(
            self.experiment_config.get("invalid_response_policy", "fail_and_resume")
        )
        if invalid_response_policy not in INVALID_RESPONSE_POLICIES:
            raise ValueError(
                "invalid_response_policy must be one of "
                + ", ".join(INVALID_RESPONSE_POLICIES)
            )
        variants = self._router_training_variants()
        expected_variants = (
            ("1", "4")
            if self.is_router_v4_lightgbm_isotonic
            else ROUTER_V3_SWEEP_VARIANTS
            if self.is_router_v3_budget_sweep
            else ROUTER_V3_VARIANTS
        )
        if tuple(variants) != expected_variants:
            raise ValueError(
                "Router-v3 variant matrix does not match its frozen revision"
            )

    @property
    def is_router_v3_budget_sweep(self) -> bool:
        return self.router_revision == ROUTER_V3_BUDGET_SWEEP_REVISION

    @property
    def is_router_v3_catboost(self) -> bool:
        return self.router_revision == ROUTER_V3_CATBOOST_REVISION

    @property
    def is_router_v4_lightgbm_isotonic(self) -> bool:
        return self.router_revision == ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION

    @property
    def freezes_reused_terminal_failures(self) -> bool:
        return self.router_revision in {
            ROUTER_V3_BUDGET_SWEEP_REVISION,
            ROUTER_V3_CATBOOST_REVISION,
            ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION,
        }

    def _bgr_method_name(self, backend: str) -> str:
        return f"budgeted_group_{backend}"

    def _active_gate_backends(self) -> tuple[str, ...]:
        return tuple(
            str(value) for value in self.experiment_config.get("gate_backends", ())
        )

    def _router_budget_shares(self) -> tuple[float, ...]:
        raw = tuple(
            float(value)
            for value in self.experiment_config.get("budget_shares", ())
        )
        expected = (
            ROUTER_V3_SWEEP_BUDGETS
            if self.is_router_v3_budget_sweep
            else (0.2,)
        )
        if raw != expected:
            raise ValueError(
                f"Router-v3 budget shares must be exactly {list(expected)}"
            )
        return raw

    def _router_training_variants(self) -> dict[str, tuple[int, ...]]:
        raw = self.experiment_config.get("router_training_variants", {})
        if not isinstance(raw, Mapping):
            raise ValueError("router_training_variants must be an object")
        definitions = {
            "1": (1,),
            "2": (1, 2),
            "4": (1, 4),
            "8": (1, 8),
            "all": (1, 2, 4, 8),
        }
        names = (
            ROUTER_V3_SWEEP_VARIANTS
            if self.is_router_v3_budget_sweep
            else ROUTER_V3_VARIANTS
        )
        variants: dict[str, tuple[int, ...]] = {}
        for name in names:
            values = raw.get(name)
            if not isinstance(values, list):
                raise ValueError(f"missing Router-v3 training variant {name!r}")
            sizes = tuple(int(value) for value in values)
            if sizes != definitions[name]:
                raise ValueError(
                    f"Router-v3 variant {name!r} must use sizes {definitions[name]}"
                )
            variants[name] = sizes
        if set(raw) != set(names):
            raise ValueError("Router-v3 training variants contain unexpected keys")
        return variants

    @staticmethod
    def _filter_variant_pairs(
        frame: pd.DataFrame,
        allowed_sizes: Sequence[int],
        *,
        context: str,
    ) -> pd.DataFrame:
        """Filter before fit/predict and require every declared size to exist."""

        allowed = {int(value) for value in allowed_sizes}
        filtered = frame.loc[
            pd.to_numeric(frame["group_size"], errors="raise").isin(allowed)
        ].copy()
        observed = {
            int(value)
            for value in pd.to_numeric(filtered["group_size"], errors="raise")
        }
        if filtered.empty or observed != allowed:
            raise ValueError(
                f"Router-v3 {context} group sizes differ: "
                f"expected={sorted(allowed)}, observed={sorted(observed)}"
            )
        return filtered

    @classmethod
    def create(
        cls,
        *,
        project_root: str | Path,
        data_root: str | Path,
        config_path: str | Path,
        llm_config_path: str | Path,
        vendor_root: str | Path,
        runs_root: str | Path,
        run_dir: str | Path | None = None,
        run_id: str | None = None,
        resume: bool = False,
        baran_source_run: str | Path | None = None,
        response_reuse_run: str | Path | None = None,
        calibration_source_run: str | Path | None = None,
        router_artifact_reuse_run: str | Path | None = None,
        router_comparison_run: str | Path | None = None,
        provider_token_cap: int | None = None,
        allow_uncapped_provider_usage: bool = False,
    ) -> "ExperimentRunner":
        root = Path(project_root).resolve()
        data = Path(data_root).resolve()
        requested_config = Path(config_path).resolve()
        requested_llm_config = Path(llm_config_path).resolve()
        vendor = Path(vendor_root).resolve()
        runs = Path(runs_root).resolve()
        resolved_run = Path(run_dir).resolve() if run_dir is not None else runs / (run_id or _utc_run_id())
        if resume and (resolved_run / "run_manifest.json").is_file():
            config = resolved_run / "bound_experiment_config.json"
            llm_config_file = resolved_run / "bound_llm_config.json"
            if not config.is_file() or not llm_config_file.is_file():
                raise FileNotFoundError(
                    "resume requires run-local bound_experiment_config.json and "
                    "bound_llm_config.json"
                )
        else:
            config = requested_config
            llm_config_file = requested_llm_config
        existing_manifest = (
            load_json(resolved_run / "run_manifest.json")
            if resume and (resolved_run / "run_manifest.json").is_file()
            else {}
        )

        def optional_source(
            requested: str | Path | None,
            manifest_field: str,
        ) -> Path | None:
            value = requested
            if value is None:
                stored = str(existing_manifest.get(manifest_field) or "").strip()
                value = stored or None
            return Path(value).resolve() if value is not None else None

        baran_source = optional_source(baran_source_run, "baran_source_run")
        response_source = optional_source(response_reuse_run, "response_reuse_run")
        calibration_source = optional_source(
            calibration_source_run, "calibration_source_run"
        )
        artifact_source = (
            optional_source(
                router_artifact_reuse_run,
                "router_artifact_reuse_run",
            )
        )
        comparison_source = optional_source(
            router_comparison_run,
            "router_comparison_run",
        )
        for label, source in (
            ("Baran", baran_source),
            ("response reuse", response_source),
            ("calibration", calibration_source),
        ):
            if source is not None and not (source / "run_manifest.json").is_file():
                raise FileNotFoundError(
                    f"{label} source has no run_manifest.json: {source}"
                )
        if artifact_source is not None and not (
            artifact_source / "run_manifest.json"
        ).is_file():
            raise FileNotFoundError(
                "Router artifact reuse source has no run_manifest.json"
            )
        experiment_config = load_json(config)
        llm_config = load_json(llm_config_file)
        comparison_value = str(experiment_config.get("comparison_run", "")).strip()
        if comparison_source is None and comparison_value:
            comparison_path = Path(comparison_value)
            comparison_source = (
                comparison_path.resolve()
                if comparison_path.is_absolute()
                else (root / comparison_path).resolve()
            )
            if not (comparison_source / "run_manifest.json").is_file():
                raise FileNotFoundError(
                    "Router comparison source has no run_manifest.json"
                )

        manifest_audit = validate_manifest(data, require_portable=True)
        implementation_sha = _hash_tree(
            root / "src" / "budgeted_group_repair_no_baran", (".py",)
        )
        raha_sha = _hash_tree(vendor, (".py",))
        prompt_sha = canonical_json_sha256(
            {"schema": PROMPT_SCHEMA_VERSION, "system_prompt": SYSTEM_PROMPT}
        )
        binding = build_run_binding(
            run_id=resolved_run.name,
            protocol=str(experiment_config.get("protocol", "")),
            experiment_config_path=config,
            llm_config_path=llm_config_file,
            data_manifest_path=data / "manifest.json",
            implementation_sha256=implementation_sha,
            raha_code_sha256=raha_sha,
            model=str(llm_config.get("model", "")),
            prompt_schema_version=str(llm_config.get("prompt_schema_version", PROMPT_SCHEMA_VERSION)),
            prompt_schema_sha256=prompt_sha,
        )
        source_binding: dict[str, object] = {
            "baran_source_run": str(baran_source) if baran_source is not None else None,
            "baran_source_manifest_sha256": (
                sha256_file(baran_source / "run_manifest.json")
                if baran_source is not None
                else None
            ),
            "response_reuse_run": (
                str(response_source) if response_source is not None else None
            ),
            "response_reuse_manifest_sha256": (
                sha256_file(response_source / "run_manifest.json")
                if response_source is not None
                else None
            ),
        }
        if calibration_source is not None:
            source_binding.update(
                {
                    "calibration_source_run": str(calibration_source),
                    "calibration_source_manifest_sha256": sha256_file(
                        calibration_source / "run_manifest.json"
                    ),
                }
            )
        if artifact_source is not None:
            source_binding.update(
                {
                    "router_artifact_reuse_run": str(artifact_source),
                    "router_artifact_reuse_manifest_sha256": sha256_file(
                        artifact_source / "run_manifest.json"
                    ),
                }
            )
        if comparison_source is not None:
            source_binding.update(
                {
                    "router_comparison_run": str(comparison_source),
                    "router_comparison_manifest_sha256": sha256_file(
                        comparison_source / "run_manifest.json"
                    ),
                }
            )
        binding.update(source_binding)
        binding["binding_fingerprint"] = canonical_json_sha256(
            {
                key: value
                for key, value in binding.items()
                if key != "binding_fingerprint"
            }
        )
        metadata = {
            **binding,
            "experiment_config": experiment_config,
            "llm_config": redacted_config(llm_config),
            "data_root": str(data),
            "data_content_fingerprint": sha256_file(data / "manifest.json"),
            "raha_source_root": str(vendor),
            "manifest_audit": manifest_audit.as_dict(),
            "result_reuse": False,
            **source_binding,
        }
        if resume and (resolved_run / "run_manifest.json").is_file():
            existing = load_json(resolved_run / "run_manifest.json")
            provider_checkpoint = resolved_run / "llm" / "group_query_checkpoint.jsonl"
            checkpoint_rows = read_jsonl(provider_checkpoint)
            stages = existing.get("stages", {})
            gate_selection_complete = bool(
                isinstance(stages, Mapping)
                and isinstance(stages.get("gate_selection"), Mapping)
                and stages["gate_selection"].get("status") == "complete"  # type: ignore[index]
            )
            imported_only_v4 = bool(
                str(experiment_config.get("router_revision", ""))
                == ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION
                and checkpoint_rows
                and all(
                    bool(row.get("cache_hit"))
                    and bool(row.get("imported_response"))
                    for row in checkpoint_rows
                )
                and not gate_selection_complete
            )
            if (
                str(existing.get("implementation_sha256", ""))
                != str(metadata["implementation_sha256"])
                and (not checkpoint_rows or imported_only_v4)
            ):
                previous = str(existing.get("implementation_sha256", ""))
                existing["implementation_sha256"] = metadata["implementation_sha256"]
                existing["binding_fingerprint"] = metadata["binding_fingerprint"]
                history = existing.get("pre_provider_rebinds", [])
                rebinds = list(history) if isinstance(history, list) else []
                rebinds.append(
                    {
                        "reason": (
                            "router_v4_pre_selection_imported_cache_only_rebind"
                            if imported_only_v4
                            else "pre_provider_prompt_audit_false_positive_fix"
                        ),
                        "previous_implementation_sha256": previous,
                        "implementation_sha256": metadata["implementation_sha256"],
                        "provider_checkpoint_rows": len(checkpoint_rows),
                        "fresh_provider_checkpoint_rows": 0,
                    }
                )
                existing["pre_provider_rebinds"] = rebinds
                write_json(resolved_run / "run_manifest.json", existing)
        state = RunState.create(resolved_run, metadata, resume=resume)
        bound_config = resolved_run / "bound_experiment_config.json"
        bound_llm_config = resolved_run / "bound_llm_config.json"
        if not resume:
            shutil.copyfile(config, bound_config)
            shutil.copyfile(llm_config_file, bound_llm_config)
        for field, bound in (
            ("experiment_config_sha256", bound_config),
            ("llm_config_sha256", bound_llm_config),
        ):
            if sha256_file(bound) != str(state.manifest.get(field, "")):
                raise ValueError(f"run-local configuration snapshot drift: {field}")
        paths = ExperimentPaths(
            root,
            data,
            bound_config,
            bound_llm_config,
            vendor,
            runs,
            resolved_run,
        )
        return cls(
            paths,
            state,
            experiment_config,
            llm_config,
            provider_token_cap=provider_token_cap,
            allow_uncapped_provider_usage=allow_uncapped_provider_usage,
        )

    def _dataset(self, suite: str, dataset: str) -> LoadedDataset:
        key = (suite, dataset)
        if key not in self._datasets:
            self._datasets[key] = load_dataset(suite, dataset, self.paths.data_root)
        return self._datasets[key]

    def _baran_path(self, suite: str, dataset: str) -> Path:
        return self.paths.baran_dir / f"{_dataset_key(suite, dataset)}.jsonl"

    def _actions_path(self, suite: str, dataset: str) -> Path:
        return self.paths.groups_dir / "candidates" / f"{_dataset_key(suite, dataset)}.jsonl"

    def _membership_path(self, suite: str, dataset: str) -> Path:
        return self.paths.groups_dir / "memberships" / f"{_dataset_key(suite, dataset)}.csv"

    def _load_baran(self, suite: str, dataset: str) -> list[dict[str, object]]:
        key = (suite, dataset)
        if key not in self._baran:
            records = read_jsonl(self._baran_path(suite, dataset))
            for record in records:
                assert_online_baran_record_safe(record)
            self._baran[key] = records
        return self._baran[key]

    def _load_actions(self, suite: str, dataset: str) -> tuple[GroupQueryAction, ...]:
        # Query prompts dominate memory.  Keep action material dataset-local
        # instead of retaining all 14 candidate pools simultaneously.
        source = self._actions_path(suite, dataset)
        actions: list[GroupQueryAction] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                raw = json.loads(text)
                if not isinstance(raw, Mapping):
                    raise ValueError(f"{source}:{line_number}: action row must be an object")
                actions.append(_action_from_dict(raw))
        return tuple(actions)

    def _client(self) -> DeepSeekGroupClient:
        key_name = str(self.llm_config.get("api_key_env", "DEEPSEEK_API_KEY"))
        api_key = os.environ.get(key_name, "")
        if not api_key:
            raise RuntimeError(f"required environment variable is not set: {key_name}")
        return DeepSeekGroupClient(
            GroupClientConfig.from_mapping(self.llm_config),
            api_key=api_key,
        )

    def _provider_safety_debit(self) -> int:
        """Conservatively debit known usage plus estimated unknown attempts."""

        total = 0
        for row in read_jsonl(
            self.paths.llm_dir / "group_query_checkpoint.jsonl"
        ):
            if bool(row.get("cache_hit")):
                continue
            metadata = row.get("metadata")
            estimate = (
                int(metadata.get("estimated_total_tokens", 0) or 0)
                if isinstance(metadata, Mapping)
                else 0
            )
            attempts = max(1, int(row.get("attempts", 0) or 0))
            unknown = max(0, int(row.get("unknown_usage_attempts", 0) or 0))
            actual = _actual_tokens(row)
            if actual is None:
                total += estimate * attempts
            else:
                total += actual + estimate * unknown
        return total

    def _reserve_provider_safety(
        self,
        phase: str,
        estimated_tokens: Sequence[int],
    ) -> None:
        """Stop before a batch whose retry worst case can cross the run cap."""

        cap = self._effective_provider_cap(require=True)
        if cap is None:
            return
        attempts = int(self.llm_config.get("max_retries", 0)) + 1
        projected = self._provider_safety_debit() + attempts * sum(
            max(0, int(value)) for value in estimated_tokens
        )
        if projected > cap:
            raise SafetyCapExceeded(phase + " retry-adjusted batch", projected, cap)

    def _effective_provider_cap(self, *, require: bool) -> int | None:
        if self.allow_uncapped_provider_usage:
            return None
        raw_cap = (
            self.provider_token_cap
            if self.provider_token_cap is not None
            else self.experiment_config.get("max_estimated_tokens_safety_cap")
        )
        if raw_cap is None:
            if require:
                raise ValueError(
                    "provider execution requires --token-cap or explicit --no-token-cap"
                )
            return None
        cap = int(raw_cap)
        if cap <= 0:
            raise ValueError("provider token cap must be positive")
        return cap

    def validate_inputs(self) -> dict[str, object]:
        """Validate immutable inputs and materialize the run-local manifest."""

        audit = validate_manifest(self.paths.data_root, require_portable=True)
        if audit.dataset_count != EXPECTED_DATASET_COUNT or audit.oracle_error_count != EXPECTED_ORACLE_ERROR_COUNT:
            raise ValueError("input benchmark coverage is incomplete")
        source_manifest = self.paths.data_root / "manifest.json"
        destination = self.paths.run_dir / "input_data_manifest.json"
        if not destination.exists():
            destination.write_text(source_manifest.read_text(encoding="utf-8"), encoding="utf-8")
        if sha256_file(destination) != sha256_file(source_manifest):
            raise ValueError("run-local input manifest differs from the validated source manifest")
        self.state.update_stage("input_validation", "complete", **audit.as_dict())
        return audit.as_dict()

    def run_baran_stage(
        self,
        datasets: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, object]:
        """Run Baran locally or import a bound label-free 14-dataset ledger."""

        dataset_order = tuple(datasets) if datasets is not None else generation_order()
        expected_total = 0
        completed: list[str] = []
        fresh_datasets = 0
        imported_datasets = 0
        for suite, dataset in dataset_order:
            loaded = self._dataset(suite, dataset)
            expected = len(loaded.safe_cells())
            expected_total += expected
            path = self._baran_path(suite, dataset)
            records = read_jsonl(path) if path.is_file() else []
            if len(records) != expected:
                if self.baran_source_run is not None:
                    source = (
                        self.baran_source_run
                        / "baran"
                        / f"{_dataset_key(suite, dataset)}.jsonl"
                    )
                    if not source.is_file():
                        raise FileNotFoundError(
                            f"Baran source is missing {suite}/{dataset}"
                        )
                    records = read_jsonl(source)
                    imported_datasets += 1
                else:
                    records = run_baran(
                        loaded,
                        loaded.oracle_cells(include_annotations=False),
                        self.paths.vendor_root,
                        labeling_budget=int(
                            self.experiment_config.get("baran_labeling_budget", 20)
                        ),
                        seed=int(self.experiment_config.get("baran_seed", 16)),
                        workers=int(self.experiment_config.get("baran_workers", 4)),
                        multiprocessing_start_method=str(
                            self.experiment_config.get(
                                "baran_multiprocessing_start_method", "spawn"
                            )
                        ),
                    )
                    fresh_datasets += 1
                write_jsonl(path, records)
            if len(records) != expected or len({str(row.get("cell_id")) for row in records}) != expected:
                raise ValueError(f"invalid Baran coverage for {suite}/{dataset}")
            for record in records:
                assert_online_baran_record_safe(record)
            self._baran[(suite, dataset)] = [dict(row) for row in records]
            completed.append(f"{suite}/{dataset}")
            self.state.update_stage(
                "baran",
                "running",
                completed_datasets=completed,
                records=sum(len(self._baran[key]) for key in self._baran),
                fresh_datasets=fresh_datasets,
                imported_datasets=imported_datasets,
                source_run=(
                    str(self.baran_source_run)
                    if self.baran_source_run is not None
                    else None
                ),
            )
        frozen_total = (
            EXPECTED_ORACLE_ERROR_COUNT
            if dataset_order == generation_order()
            else TEST_TARGET_CELL_COUNT if dataset_order == target_order() else expected_total
        )
        if expected_total != frozen_total:
            raise ValueError(
                f"Baran expected-cell count is {expected_total}, not {frozen_total}"
            )
        self.state.update_stage(
            "baran",
            "complete",
            datasets=len(completed),
            records=expected_total,
            fresh=self.baran_source_run is None,
            imported=self.baran_source_run is not None,
            fresh_datasets=fresh_datasets,
            imported_datasets=imported_datasets,
            source_run=(
                str(self.baran_source_run)
                if self.baran_source_run is not None
                else None
            ),
        )
        return {
            "datasets": len(completed),
            "records": expected_total,
            "fresh": self.baran_source_run is None,
            "imported": self.baran_source_run is not None,
            "fresh_datasets": fresh_datasets,
            "imported_datasets": imported_datasets,
        }

    @staticmethod
    def _pair_feature_rows(
        actions: Sequence[GroupQueryAction],
        features: Sequence[CellFeatures],
        cells: Sequence[SafeCell],
        baran_records: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        feature_by_id = {feature.cell_id: feature for feature in features}
        cell_by_id = {str(cell.cell_id): cell for cell in cells}
        baran_by_id = {str(row.get("cell_id")): row for row in baran_records}
        singleton_cost = {
            action.cell_ids[0]: int(action.estimated_total_tokens)
            for action in actions
            if action.group_view == "singleton" and action.group_size == 1
        }
        rows: list[dict[str, object]] = []
        for action in actions:
            member_features = [feature_by_id[cell_id] for cell_id in action.cell_ids]
            singleton_sum = sum(singleton_cost[cell_id] for cell_id in action.cell_ids)
            saving = 1.0 - float(action.estimated_total_tokens) / float(singleton_sum)
            same_dirty_type = int(len({value.dirty_type for value in member_features}) == 1)
            same_baran_type = int(len({value.baran_type for value in member_features}) == 1)
            signature = hashlib.sha256(
                json.dumps(
                    {
                        "suite": action.suite,
                        "dataset": action.dataset,
                        "cell_ids": action.cell_ids,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for index, cell_id in enumerate(action.cell_ids):
                feature = feature_by_id[cell_id]
                cell = cell_by_id[cell_id]
                baran = baran_by_id[cell_id]
                group = action.group_features
                rows.append(
                    {
                        "suite": action.suite,
                        "dataset": action.dataset,
                        "base_family": base_family(action.dataset),
                        "cell_id": cell_id,
                        "row_id": cell.row_id,
                        "column": cell.column,
                        "query_id": action.query_id,
                        "group_signature": signature,
                        "cell_ids": json.dumps(list(action.cell_ids), ensure_ascii=False),
                        "prompt_hash": action.prompt_hash,
                        "dirty_type": feature.dirty_type,
                        "dirty_format": feature.dirty_format,
                        "baran_type": feature.baran_type,
                        "baran_format": feature.baran_format,
                        "baran_changed": int(
                            normalize_for_match(feature.baran_prediction)
                            != normalize_for_match(cell.dirty_value)
                        ),
                        "baran_candidate_count": float(baran.get("candidate_count", 0) or 0),
                        "baran_top_support": float(baran.get("top_support", 0.0) or 0.0),
                        "baran_support_margin": float(baran.get("support_margin", 0.0) or 0.0),
                        "baran_source_agreement": float(baran.get("source_agreement", 0.0) or 0.0),
                        "group_view": action.group_view,
                        "group_size": action.group_size,
                        "cohesion": float(group.get("cohesion", 0.0) or 0.0),
                        "same_row": int(group.get("same_row", 0) or 0),
                        "same_column": int(group.get("same_column", 0) or 0),
                        "dirty_type_count": int(group.get("dirty_type_count", 0) or 0),
                        "baran_type_count": int(group.get("baran_type_count", 0) or 0),
                        "baran_changed_share": float(group.get("baran_changed_share", 0.0) or 0.0),
                        "estimated_prompt_tokens": action.estimated_prompt_tokens,
                        "completion_token_ceiling": action.completion_token_ceiling,
                        "estimated_total_tokens": action.estimated_total_tokens,
                        "shared_prefix_saving_ratio": saving,
                        "member_index": index,
                        "member_position": index / max(1, action.group_size - 1),
                        "group_same_dirty_type": same_dirty_type,
                        "group_same_baran_type": same_baran_type,
                    }
                )
        return rows

    def generate_groups_stage(
        self,
        datasets: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, object]:
        """Build every fixed query action and its safe cell-query feature rows."""

        dataset_order = tuple(datasets) if datasets is not None else generation_order()
        registry = self.fd_registry
        query_total = 0
        incidence_total = 0
        completed: list[str] = []
        seen_query_ids: set[str] = set()
        for suite, dataset in dataset_order:
            loaded = self._dataset(suite, dataset)
            safe = loaded.safe_view()
            baran = self._load_baran(suite, dataset)
            rules = fds_for_dataset(registry, suite, dataset)
            components = build_fd_violation_components(
                safe.dirty, suite, dataset, safe.cells, rules
            )
            action_path = self._actions_path(suite, dataset)
            membership_path = self._membership_path(suite, dataset)
            feature_path = self.paths.cell_features_dir / f"{_dataset_key(suite, dataset)}.csv"
            audit_path = self.paths.groups_dir / "generation_audits" / f"{_dataset_key(suite, dataset)}.json"
            actions = (
                self._load_actions(suite, dataset)
                if action_path.is_file() and membership_path.is_file() and audit_path.is_file()
                else ()
            )
            if not actions:
                print(f"[groups] {suite}/{dataset}: generate", flush=True)
                generator = GroupGenerator(
                    safe,
                    safe.cells,
                    baran,
                    fd_components=components,
                    group_sizes=tuple(int(value) for value in self.experiment_config.get("group_sizes", (1, 2, 4, 8))),
                    prompt_schema_version=str(self.llm_config.get("prompt_schema_version", PROMPT_SCHEMA_VERSION)),
                    similar_row_count=int(
                        (self.llm_config.get("contexts") or {}).get("similar_row_count", 5)  # type: ignore[union-attr]
                    ),
                )
                result = generator.generate()
                actions = result.actions
                write_jsonl(action_path, (action.as_dict() for action in actions))
                _write_csv(feature_path, (feature.as_dict() for feature in generator.features))
                pair_rows = self._pair_feature_rows(actions, generator.features, safe.cells, baran)
                _write_csv(membership_path, pair_rows)
                write_json(
                    audit_path,
                    {
                        **dict(result.audit),
                        "suite": suite,
                        "dataset": dataset,
                        "fd_rule_count": len(rules),
                        "fd_component_count": len(components),
                        "pair_incidence_count": len(pair_rows),
                        "forbidden_fields_read": False,
                    },
                )
            membership = _read_csv(membership_path)
            for action in actions:
                assert_messages_safe(action.as_dict()["messages"])  # type: ignore[arg-type]
                recomputed_hash = compute_prompt_hash(
                    action.messages,
                    action.completion_token_ceiling,
                    prompt_schema_version=action.prompt_schema_version,
                )
                if recomputed_hash != action.prompt_hash:
                    raise ValueError(f"prompt hash drift for {action.query_id}")
                if action.query_id in seen_query_ids:
                    raise ValueError(f"duplicate query identity: {action.query_id}")
                seen_query_ids.add(action.query_id)
            singleton = [action for action in actions if action.group_size == 1]
            expected = len(safe.cells)
            if len(singleton) != expected or {action.cell_ids[0] for action in singleton} != {str(cell.cell_id) for cell in safe.cells}:
                raise ValueError(f"singleton coverage failure for {suite}/{dataset}")
            if len(membership) != sum(action.group_size for action in actions):
                raise ValueError(f"membership incidence mismatch for {suite}/{dataset}")
            query_total += len(actions)
            incidence_total += len(membership)
            completed.append(f"{suite}/{dataset}")
            self.state.update_stage(
                "groups",
                "running",
                completed_datasets=completed,
                queries=query_total,
                incidences=incidence_total,
            )
        self.state.update_stage(
            "groups",
            "complete",
            datasets=len(completed),
            queries=query_total,
            incidences=incidence_total,
            atoms_used=False,
        )
        write_json(
            self.paths.run_dir / "manifests" / "prompt_recursive_audit.json",
            {
                "ok": True,
                "queries": query_total,
                "prompt_information_policy": INFORMATION_POLICY,
                "prompt_schema_version": PROMPT_SCHEMA_VERSION,
                "baran_fields_found": 0,
                "oracle_fields_found": 0,
                "query_identity_duplicates": 0,
                "prompt_hash_mismatches": 0,
            },
        )
        return {"datasets": len(completed), "queries": query_total, "incidences": incidence_total}

    def import_reusable_no_baran_responses_stage(
        self,
        datasets: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, object]:
        """Optionally seed the checkpoint with request-identical responses."""

        dataset_order = tuple(datasets) if datasets is not None else generation_order()
        provenance_path = self.paths.run_dir / "provenance" / "response_reuse.json"
        checkpoint_path = self.paths.llm_dir / "group_query_checkpoint.jsonl"
        if provenance_path.is_file():
            return load_json(provenance_path)
        if checkpoint_path.is_file() and read_jsonl(checkpoint_path):
            raise RuntimeError(
                "response reuse must be frozen before any provider checkpoint exists"
            )
        if self.response_reuse_run is None:
            summary: dict[str, object] = {
                "source_run": None,
                "source_rows": 0,
                "imported_rows": 0,
                "imported_success_rows": 0,
                "imported_terminal_failure_rows": 0,
                "terminal_failures_frozen": self.freezes_reused_terminal_failures,
                "rejected": {},
                "matching_fields": [
                    "query_id",
                    "prompt_hash",
                    "provider_request_hash",
                    "model",
                    "prompt_schema_version",
                ],
                "physical_calls_saved_only": True,
            }
            write_json(provenance_path, summary)
            self.state.update_stage("response_reuse", "complete", **summary)
            return summary
        source_checkpoint = (
            self.response_reuse_run / "llm" / "shared" / "group_query_checkpoint.jsonl"
        )
        if not source_checkpoint.is_file():
            source_checkpoint = self.response_reuse_run / "llm" / "group_query_checkpoint.jsonl"
        if not source_checkpoint.is_file():
            raise FileNotFoundError("No-Baran reuse source has no checkpoint ledger")
        source_rows = read_jsonl(source_checkpoint)
        latest = {
            (str(row.get("query_id")), str(row.get("prompt_hash", ""))): row
            for row in source_rows
            if str(row.get("model", "")) == str(self.llm_config["model"])
            and row.get("model_matches_request", True) is not False
        }
        client = DeepSeekGroupClient(
            GroupClientConfig.from_mapping(self.llm_config),
            api_key="not-used-for-request-hashing",
        )
        imported: list[dict[str, object]] = []
        rejected = Counter()
        for suite, dataset in dataset_order:
            for action in self._load_actions(suite, dataset):
                source = latest.get((action.query_id, action.prompt_hash))
                if source is None:
                    continue
                if (
                    not self.freezes_reused_terminal_failures
                    and source.get("status") != "success"
                ):
                    continue
                if action.prompt_schema_version != PROMPT_SCHEMA_VERSION:
                    rejected["schema_mismatch"] += 1
                    continue
                if str(source.get("prompt_hash", "")) != action.prompt_hash:
                    rejected["prompt_hash_mismatch"] += 1
                    continue
                job = GroupLLMJob.from_action(action)
                request_hash = client.provider_request_hash(job)
                if str(source.get("provider_request_hash", "")) != request_hash:
                    rejected["provider_request_hash_mismatch"] += 1
                    continue
                metadata = source.get("metadata")
                copied = dict(source)
                copied["cache_hit"] = True
                copied["checkpoint_hit"] = False
                copied["imported_response"] = True
                copied["metadata"] = {
                    **(dict(metadata) if isinstance(metadata, Mapping) else {}),
                    "imported_from_run": self.response_reuse_run.name,
                    "imported_for_schema": PROMPT_SCHEMA_VERSION,
                }
                imported.append(copied)
        write_jsonl(checkpoint_path, imported)
        summary: dict[str, object] = {
            "source_run": str(self.response_reuse_run),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
            "source_rows": len(source_rows),
            "imported_rows": len(imported),
            "imported_success_rows": sum(
                row.get("status") == "success" for row in imported
            ),
            "imported_terminal_failure_rows": sum(
                row.get("status") != "success" for row in imported
            ),
            "terminal_failures_frozen": self.freezes_reused_terminal_failures,
            "rejected": dict(rejected),
            "matching_fields": [
                "query_id",
                "prompt_hash",
                "provider_request_hash",
                "model",
                "prompt_schema_version",
            ],
            "physical_calls_saved_only": True,
        }
        write_json(provenance_path, summary)
        self.state.update_stage("response_reuse", "complete", **summary)
        return summary


    @staticmethod
    def _calibration_sample(
        dataset: str,
        actions: Sequence[GroupQueryAction],
        *,
        seed: int,
        cap: int,
    ) -> tuple[GroupQueryAction, ...]:
        singletons = sorted((action for action in actions if action.group_size == 1), key=lambda value: value.query_id)
        non_singletons = [action for action in actions if action.group_size > 1]
        quartile: dict[str, int] = {}
        by_view_size: dict[tuple[str, int], list[GroupQueryAction]] = defaultdict(list)
        for action in non_singletons:
            by_view_size[(action.group_view, action.group_size)].append(action)
        for members in by_view_size.values():
            ordered_cohesion = sorted(
                members,
                key=lambda action: (
                    float(action.group_features.get("cohesion", 0.0)),
                    action.query_id,
                ),
            )
            count = len(ordered_cohesion)
            for rank, action in enumerate(ordered_cohesion):
                quartile[action.query_id] = min(3, int(rank * 4 / max(1, count)))
        strata: dict[tuple[str, int, int], list[GroupQueryAction]] = defaultdict(list)
        for action in non_singletons:
            strata[(action.group_view, action.group_size, quartile[action.query_id])].append(action)
        for key in strata:
            strata[key].sort(
                key=lambda action: hashlib.sha256(
                    f"{seed}|{dataset}|{action.query_id}".encode("utf-8")
                ).hexdigest()
            )
        selected: list[GroupQueryAction] = []
        positions = {key: 0 for key in strata}
        keys = sorted(strata)
        while len(selected) < cap:
            progressed = False
            for key in keys:
                position = positions[key]
                if position < len(strata[key]):
                    selected.append(strata[key][position])
                    positions[key] += 1
                    progressed = True
                    if len(selected) >= cap:
                        break
            if not progressed:
                break
        return tuple(singletons + selected)

    def plan_calibration_stage(self) -> dict[str, object]:
        """Select all singleton plus stratified non-singleton TableEG queries."""

        plan_path = self.paths.llm_dir / "calibration_queries.jsonl"
        planned: list[GroupQueryAction] = []
        per_dataset: dict[str, dict[str, int]] = {}
        cap = int(self.experiment_config.get("max_calibration_non_singleton_queries_per_dataset", 300))
        seed = int(self.experiment_config.get("seed", 42))
        for suite, dataset in generation_order():
            if suite != "tableeg":
                continue
            selected = self._calibration_sample(
                dataset, self._load_actions(suite, dataset), seed=seed, cap=cap
            )
            planned.extend(selected)
            per_dataset[dataset] = {
                "singletons": sum(action.group_size == 1 for action in selected),
                "non_singletons": sum(action.group_size > 1 for action in selected),
                "queries": len(selected),
                "estimated_tokens": sum(action.estimated_total_tokens for action in selected),
            }
        write_jsonl(
            plan_path,
            (
                {
                    "query_id": action.query_id,
                    "prompt_hash": action.prompt_hash,
                    "suite": action.suite,
                    "dataset": action.dataset,
                    "group_view": action.group_view,
                    "group_size": action.group_size,
                    "cell_ids": list(action.cell_ids),
                    "estimated_total_tokens": action.estimated_total_tokens,
                }
                for action in planned
            ),
        )
        estimated = sum(action.estimated_total_tokens for action in planned)
        raw_cap = self.experiment_config.get("max_estimated_tokens_safety_cap")
        safety_cap = None if raw_cap is None else int(raw_cap)
        singleton_count = sum(action.group_size == 1 for action in planned)
        if singleton_count != CALIBRATION_SINGLETON_CELL_COUNT:
            raise ValueError(
                "TableEG calibration singleton coverage must be exactly "
                f"{CALIBRATION_SINGLETON_CELL_COUNT:,}, found {singleton_count:,}"
            )
        summary: dict[str, object] = {
            "queries": len(planned),
            "singletons": singleton_count,
            "non_singletons": sum(action.group_size > 1 for action in planned),
            "estimated_tokens": estimated,
            "safety_cap": safety_cap,
            "per_dataset": per_dataset,
        }
        write_json(self.paths.llm_dir / "calibration_plan.json", summary)
        if safety_cap is not None and estimated > safety_cap:
            self.state.update_stage("calibration_plan", "safety_cap_exceeded", **summary)
            raise SafetyCapExceeded("offline calibration", estimated, safety_cap)
        self.state.update_stage("calibration_plan", "complete", **summary)
        return summary

    def plan_run(self) -> dict[str, object]:
        """Build local candidates and the exact offline-calibration cost plan."""

        self.validate_inputs()
        self.run_baran_stage()
        self.generate_groups_stage()
        reuse = self.import_reusable_no_baran_responses_stage()
        plan = self.plan_calibration_stage()
        return {
            "run_dir": str(self.paths.run_dir),
            "data": {
                "generation_datasets": len(generation_order()),
                "test_datasets": len(TEST_TARGETS),
                "test_oracle_cells": TEST_TARGET_CELL_COUNT,
                "calibration_datasets": 9,
                "calibration_singletons": CALIBRATION_SINGLETON_CELL_COUNT,
            },
            "calibration": plan,
            "response_reuse": reuse,
            "api_called": False,
        }

    def check_model(self) -> dict[str, object]:
        """Run one paid size-eight schema/model compatibility request."""

        receipt_path = self.paths.llm_dir / "model_preflight.json"
        if self.state.stage_completed("model_preflight") and receipt_path.is_file():
            return load_json(receipt_path)
        expected_ids = tuple(f"preflight:cell:{index}" for index in range(8))
        query_id = "bgr_preflight_size8"
        messages = canonical_messages(
            (
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object with query_id and repairs. "
                        "For each supplied cell_id return repair='ok', confidence=1.0, "
                        "decision='propose', evidence='schema preflight', affected_constraints=[]."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"query_id": query_id, "cell_ids": expected_ids},
                        sort_keys=True,
                    ),
                },
            )
        )
        max_tokens = 64 + 192 * len(expected_ids)
        estimated_total_tokens = estimate_prompt_tokens(messages) + max_tokens
        job = GroupLLMJob(
            query_id=query_id,
            messages=messages,
            prompt_hash=compute_prompt_hash(messages, max_tokens),
            expected_cell_ids=expected_ids,
            max_tokens=max_tokens,
            metadata={
                "phase": "model_preflight",
                "estimated_total_tokens": estimated_total_tokens,
                "require_complete_response": True,
            },
        )
        existing = self._response_index(phase="model_preflight").get(
            (job.query_id, job.prompt_hash)
        )
        self._reserve_provider_safety(
            "model preflight",
            ()
            if existing is not None and existing.get("status") == "success"
            else (estimated_total_tokens,),
        )
        returned = run_group_llm_batch(
            self._client(),
            (job,),
            self.paths.llm_dir,
            concurrency=1,
            retry_failed=True,
        )
        self._reserve_provider_safety("model preflight", ())
        if len(returned) != 1:
            raise RuntimeError("model preflight did not return exactly one ledger record")
        result = returned[0]
        configured_model = str(self.llm_config["model"])
        returned_model = str(result.get("model", ""))
        if result.get("status") != "success":
            raise RuntimeError(
                "size-eight model preflight failed: "
                + str(result.get("parse_status", "unknown"))
            )
        if returned_model != configured_model:
            raise RuntimeError(
                f"model drift: requested={configured_model!r}, returned={returned_model!r}"
            )
        raw_items = result.get("items", [])
        returned_ids = {
            str(item.get("cell_id", ""))
            for item in raw_items
            if isinstance(item, Mapping)
        } if isinstance(raw_items, list) else set()
        complete = (
            str(result.get("parse_status", "")) == "ok"
            and returned_ids == set(expected_ids)
            and len(raw_items) == len(expected_ids)
            and not result.get("missing_cell_ids")
            and not result.get("unknown_cell_ids")
            and not result.get("duplicate_cell_ids")
        )
        if not complete:
            raise RuntimeError(
                "size-eight model preflight failed schema: "
                + str(result.get("parse_status", "unknown"))
            )
        receipt = {
            "requested_model": configured_model,
            "returned_model": returned_model,
            "query_id": query_id,
            "group_size": len(expected_ids),
            "estimated_total_tokens": estimated_total_tokens,
            "status": str(result.get("status")),
            "parse_status": str(result.get("parse_status")),
            "returned_items": len(raw_items),
            "messages": [dict(message) for message in messages],
            "response_text": str(result.get("response_text", "")),
            "items": raw_items,
            "response_id": str(result.get("response_id", "")),
            "provider_request_hash": str(result.get("provider_request_hash", "")),
            "usage": dict(result.get("usage") or {}),
            "attempts": int(result.get("attempts", 0) or 0),
            "usage_observed_attempts": int(
                result.get("usage_observed_attempts", 0) or 0
            ),
            "unknown_usage_attempts": int(
                result.get("unknown_usage_attempts", 0) or 0
            ),
            "observed_total_tokens": int(
                result.get("observed_total_tokens", 0) or 0
            ),
            "latency_seconds": float(result.get("latency_seconds", 0.0) or 0.0),
            "cache_hit": bool(result.get("cache_hit")),
            "checkpoint_hit": bool(result.get("checkpoint_hit")),
            "prompt_hash": job.prompt_hash,
        }
        write_json(receipt_path, receipt)
        self.state.update_stage(
            "model_preflight",
            "complete",
            **_stage_details(receipt),
        )
        return receipt

    def _calibration_actions(self) -> tuple[GroupQueryAction, ...]:
        plan_rows = read_jsonl(self.paths.llm_dir / "calibration_queries.jsonl")
        requested = {str(row["query_id"]): str(row["prompt_hash"]) for row in plan_rows}
        actions: list[GroupQueryAction] = []
        for suite, dataset in generation_order():
            if suite != "tableeg":
                continue
            for action in self._load_actions(suite, dataset):
                if action.query_id in requested:
                    if requested[action.query_id] != action.prompt_hash:
                        raise ValueError(f"calibration prompt drift for {action.query_id}")
                    actions.append(action)
        if len(actions) != len(requested):
            raise ValueError("calibration plan references missing query actions")
        return tuple(sorted(actions, key=lambda action: (action.dataset, action.query_id)))

    @staticmethod
    def _clean_value_map(dataset: LoadedDataset, cells: Sequence[SafeCell]) -> dict[str, str]:
        """Bind clean values by coordinate without opening generation logs."""

        return {
            str(cell.cell_id): str(dataset.clean.iloc[int(cell.row), int(cell.col)])
            for cell in cells
        }

    def _execute_jobs(
        self,
        actions: Sequence[GroupQueryAction],
        *,
        phase: str,
        output_path: Path,
        retry_failed: bool = True,
    ) -> list[dict[str, object]]:
        client = self._client()
        chunk_size = max(1, int(self.experiment_config.get("llm_query_chunk_size", 100)))
        records: list[dict[str, object]] = []
        for start in range(0, len(actions), chunk_size):
            chunk = actions[start : start + chunk_size]
            response_index = self._response_index(phase=phase)
            pending_estimates = []
            for action in chunk:
                existing = response_index.get((action.query_id, action.prompt_hash))
                if existing is None or (
                    retry_failed and existing.get("status") != "success"
                ):
                    pending_estimates.append(action.estimated_total_tokens)
            self._reserve_provider_safety(phase, pending_estimates)
            print(
                f"[llm:{phase}] queries {start + 1}-{start + len(chunk)} / {len(actions)}",
                flush=True,
            )
            jobs = [
                GroupLLMJob.from_action(
                    action,
                    metadata={
                        "phase": phase,
                        "suite": action.suite,
                        "dataset": action.dataset,
                        "group_view": action.group_view,
                        "group_size": action.group_size,
                        "estimated_total_tokens": action.estimated_total_tokens,
                    },
                )
                for action in chunk
            ]
            returned = run_group_llm_batch(
                client,
                jobs,
                self.paths.llm_dir,
                concurrency=int(self.llm_config.get("concurrency", 4)),
                retry_failed=retry_failed,
            )
            self._reserve_provider_safety(phase, ())
            for row in returned:
                model = str(row.get("model", ""))
                if row.get("status") == "success" and model != str(self.llm_config["model"]):
                    raise RuntimeError(
                        f"model drift for query {row.get('query_id')}: "
                        f"requested={self.llm_config['model']!r}, returned={model!r}"
                    )
            records.extend(dict(row) for row in returned)
            write_jsonl(output_path, records)
        return records

    def _materialize_baran_fallbacks(
        self,
        actions: Sequence[GroupQueryAction],
        results: Sequence[Mapping[str, object]],
        *,
        phase: str,
    ) -> dict[str, object]:
        """Record unresolved response cells as explicit, auditable Baran fallbacks."""

        result_by_id = {str(row.get("query_id")): row for row in results}
        fallback_rows: list[dict[str, object]] = []
        unresolved_cells: list[str] = []
        provider_failed_queries = 0
        for action in actions:
            result = result_by_id.get(action.query_id, {})
            if result.get("status") != "success":
                provider_failed_queries += 1
            missing_raw = result.get("missing_cell_ids", ())
            missing = (
                tuple(str(value) for value in missing_raw)
                if isinstance(missing_raw, (list, tuple))
                else ()
            )
            if result.get("status") != "success" and not missing:
                missing = action.cell_ids
            missing = tuple(cell_id for cell_id in missing if cell_id in set(action.cell_ids))
            if not missing:
                continue
            baran_ids = {
                str(row.get("cell_id"))
                for row in self._load_baran(action.suite, action.dataset)
            }
            absent = sorted(set(missing) - baran_ids)
            unresolved_cells.extend(absent)
            fallback_rows.append(
                {
                    "phase": phase,
                    "query_id": action.query_id,
                    "prompt_hash": action.prompt_hash,
                    "suite": action.suite,
                    "dataset": action.dataset,
                    "group_view": action.group_view,
                    "group_size": action.group_size,
                    "cell_ids": list(missing),
                    "provider_status": str(result.get("status", "missing")),
                    "parse_status": str(result.get("parse_status", "missing")),
                    "fallback_source": "baran",
                    "fallback_reason": "llm_or_parse_failure",
                    "operational_status": "unresolved" if absent else "resolved",
                    "missing_baran_cell_ids": absent,
                }
            )
        write_jsonl(self.paths.llm_dir / f"{phase}_baran_fallbacks.jsonl", fallback_rows)
        return {
            "provider_failed_queries": provider_failed_queries,
            "operational_fallback_queries": len(fallback_rows),
            "operational_fallback_cells": sum(
                len(row["cell_ids"]) for row in fallback_rows
            ),
            "unresolved_operational_failures": len(unresolved_cells),
        }

    def _build_calibration_labels(
        self,
        actions: Sequence[GroupQueryAction],
        results: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        result_by_id = {str(row.get("query_id")): row for row in results}
        actions_by_dataset: dict[str, list[GroupQueryAction]] = defaultdict(list)
        for action in actions:
            actions_by_dataset[action.dataset].append(action)
        labels: list[dict[str, object]] = []
        for dataset, dataset_actions in sorted(actions_by_dataset.items()):
            loaded = self._dataset("tableeg", dataset)
            safe_cells = loaded.safe_cells()
            cell_by_id = {str(cell.cell_id): cell for cell in safe_cells}
            clean = self._clean_value_map(loaded, safe_cells)
            baran = {str(row["cell_id"]): row for row in self._load_baran("tableeg", dataset)}
            for action in dataset_actions:
                result = result_by_id.get(action.query_id, {})
                response_usable = (
                    result.get("status") == "success"
                    and result.get("model_matches_request", True) is not False
                )
                raw_items = result.get("items", []) if response_usable else []
                item_by_id: dict[str, ParsedRepairItem] = {}
                if isinstance(raw_items, list):
                    for raw in raw_items:
                        if not isinstance(raw, Mapping):
                            continue
                        try:
                            item = ParsedRepairItem(
                                cell_id=str(raw.get("cell_id", "")),
                                repair=str(raw.get("repair", "")),
                                confidence=float(raw.get("confidence", 0.0)),
                                decision=str(raw.get("decision", "")),
                                evidence=str(raw.get("evidence", "")),
                                affected_constraints=tuple(
                                    str(value) for value in raw.get("affected_constraints", [])
                                ),
                            )
                        except (TypeError, ValueError):
                            continue
                        item_by_id[item.cell_id] = item
                for cell_id in action.cell_ids:
                    item = item_by_id.get(cell_id)
                    executable = _basic_item_valid(item, cell_by_id[cell_id], baran[cell_id])
                    llm_correct = bool(
                        item is not None
                        and normalize_for_match(item.repair) == normalize_for_match(clean[cell_id])
                    )
                    baran_correct = _is_baran_correct(baran[cell_id], clean[cell_id])
                    targets = build_uplift_targets(
                        [baran_correct], [llm_correct], [executable]
                    )
                    labels.append(
                        {
                            "suite": "tableeg",
                            "dataset": dataset,
                            "base_family": base_family(dataset),
                            "cell_id": cell_id,
                            "query_id": action.query_id,
                            "group_view": action.group_view,
                            "group_size": action.group_size,
                            "baran_correct": int(baran_correct),
                            "llm_correct_in_query": int(llm_correct),
                            "executable_propose": int(executable),
                            "helpful": targets.helpful[0],
                            "harmful": targets.harmful[0],
                            "query_parse_status": str(result.get("parse_status", "missing")),
                            "item_present": int(item is not None),
                            "item_decision": item.decision if item is not None else "missing",
                        }
                    )
        return labels

    def run_calibration_stage(self) -> dict[str, object]:
        if self.state.stage_completed("calibration_llm"):
            execution = read_jsonl(
                self.paths.llm_dir / "calibration_execution.jsonl"
            )
            labels = _read_csv(
                self.paths.llm_dir / "calibration_pair_labels.csv"
            )
            return {
                "reused_from_parent": bool(
                    _stage_details(self.state.manifest.get("stages", {}).get(
                        "calibration_llm", {}
                    )).get("reused_from_parent", False)
                ) if isinstance(self.state.manifest.get("stages"), Mapping) else False,
                "queries": len(execution),
                "pair_labels": len(labels),
                "successful_queries": sum(
                    row.get("status") == "success" for row in execution
                ),
                "failed_queries": sum(
                    row.get("status") != "success" for row in execution
                ),
                "helpful_pairs": int(
                    pd.to_numeric(labels["helpful"], errors="raise").sum()
                ),
                "harmful_pairs": int(
                    pd.to_numeric(labels["harmful"], errors="raise").sum()
                ),
            }
        actions = self._calibration_actions()
        output = self.paths.llm_dir / "calibration_execution.jsonl"
        fallback_enabled = (
            str(self.experiment_config.get("invalid_response_policy"))
            == "fallback_baran_after_client_retries"
        )
        results = self._execute_jobs(
            actions,
            phase="offline_group_calibration",
            output_path=output,
            retry_failed=not fallback_enabled,
        )
        labels = self._build_calibration_labels(actions, results)
        _write_csv(self.paths.llm_dir / "calibration_pair_labels.csv", labels)
        if len(labels) != sum(action.group_size for action in actions):
            raise ValueError("calibration pair-label incidence count is incomplete")
        summary = {
            "queries": len(actions),
            "pair_labels": len(labels),
            "successful_queries": sum(row.get("status") == "success" for row in results),
            "failed_queries": sum(row.get("status") != "success" for row in results),
            "helpful_pairs": sum(int(row["helpful"]) for row in labels),
            "harmful_pairs": sum(int(row["harmful"]) for row in labels),
        }
        fallback_summary = self._materialize_baran_fallbacks(
            actions,
            results,
            phase="offline_group_calibration",
        )
        summary.update(fallback_summary)
        fallback_cells = {
            (str(row["query_id"]), str(cell_id))
            for row in read_jsonl(
                self.paths.llm_dir / "offline_group_calibration_baran_fallbacks.jsonl"
            )
            for cell_id in row.get("cell_ids", [])
        }
        non_neutral = [
            row
            for row in labels
            if (str(row["query_id"]), str(row["cell_id"])) in fallback_cells
            and any(int(row[field]) != 0 for field in ("executable_propose", "helpful", "harmful"))
        ]
        if non_neutral:
            raise ValueError("calibration fallback cells must have neutral executable-uplift labels")
        if int(summary["unresolved_operational_failures"]) > 0:
            self.state.update_stage("calibration_llm", "failed", **summary)
            raise RuntimeError("offline calibration contains Baran fallback cells without coverage")
        if int(summary["failed_queries"]) > 0 and not fallback_enabled:
            self.state.update_stage("calibration_llm", "failed", **summary)
            raise RuntimeError(
                f"offline calibration has {summary['failed_queries']} failed queries; "
                "resume the same run to retry them"
            )
        write_json(
            self.paths.run_dir / "provenance" / "calibration.json",
            {
                "mode": "fresh_or_checkpoint_resumed",
                "calibration_queries": len(actions),
                "calibration_pair_labels": len(labels),
                "calibration_queries_sha256": sha256_file(
                    self.paths.llm_dir / "calibration_queries.jsonl"
                ),
                "calibration_execution_sha256": sha256_file(
                    self.paths.llm_dir / "calibration_execution.jsonl"
                ),
                "calibration_pair_labels_sha256": sha256_file(
                    self.paths.llm_dir / "calibration_pair_labels.csv"
                ),
                "target_labels_or_responses_used_before_selection": False,
                "logical_cost_preserved": True,
            },
        )
        self.state.update_stage("calibration_llm", "complete", **summary)
        return summary

    def _all_pair_features(self) -> pd.DataFrame:
        frames = [
            _read_csv(self._membership_path(suite, dataset))
            for suite, dataset in generation_order()
        ]
        frame = pd.concat(frames, ignore_index=True)
        missing = set(MODEL_FEATURE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"pair feature table is missing model features: {sorted(missing)}")
        return frame

    def _scenario_specs(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "scenario": "size_conditioned",
                "group_size_variant": variant,
                "allowed_sizes": sizes,
                "budget_share": budget,
            }
            for variant, sizes in self._router_training_variants().items()
            for budget in self._router_budget_shares()
        )


    def _prediction_path(
        self,
        backend: str,
        variant: str,
        suite: str,
        dataset: str,
    ) -> Path:
        return (
            self.paths.gates_dir
            / backend
            / f"variant_{variant}"
            / f"{_dataset_key(suite, dataset)}.csv"
        )


    def _selection_path(
        self,
        backend: str,
        scenario: str,
        variant: str,
        budget_share: float,
        suite: str,
        dataset: str,
    ) -> Path:
        return (
            self.paths.selections_dir
            / backend
            / scenario
            / f"variant_{variant}"
            / _budget_label(budget_share)
            / f"{_dataset_key(suite, dataset)}.json"
        )

    def build_router_diagnostics_stage(
        self,
        all_pairs: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> dict[str, object]:
        """Evaluate size-conditioned LOFO routeability after selections freeze."""

        return self._build_router_v3_diagnostics(all_pairs, labels)


    def _build_router_v3_diagnostics(
        self,
        all_pairs: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> dict[str, object]:
        """Evaluate every size-conditioned Router-v3 model independently."""

        rows: list[dict[str, object]] = []
        for variant, allowed_sizes in self._router_training_variants().items():
            for dataset in sorted(
                value for suite, value in generation_order() if suite == "tableeg"
            ):
                train_safe, target_safe, _ = split_for_target(
                    all_pairs,
                    "tableeg",
                    dataset,
                    enforce_target_unlabeled=True,
                )
                train_safe = self._filter_variant_pairs(
                    train_safe,
                    allowed_sizes,
                    context=f"diagnostic train {variant}/tableeg/{dataset}",
                )
                target_safe = self._filter_variant_pairs(
                    target_safe,
                    allowed_sizes,
                    context=f"diagnostic test {variant}/tableeg/{dataset}",
                )
                train = train_safe.merge(
                    labels,
                    how="inner",
                    on=["cell_id", "query_id"],
                    validate="one_to_one",
                )
                target = target_safe.merge(
                    labels,
                    how="inner",
                    on=["cell_id", "query_id"],
                    validate="one_to_one",
                )
                if train.empty or target.empty:
                    raise ValueError(
                        f"empty Router-v3 diagnostic fold for {variant} tableeg/{dataset}"
                    )
                helpful = [int(value) for value in target["helpful"]]
                harmful = [int(value) for value in target["harmful"]]
                for backend in self._active_gate_backends():
                    gate = GroupUpliftGate(
                        backend,  # type: ignore[arg-type]
                        rho=float(self.experiment_config.get("harm_penalty_rho", 1.0)),
                        gamma=float(
                            self.experiment_config.get(
                                "uncertainty_penalty_gamma", 1.0
                            )
                        ),
                        random_state=int(self.experiment_config.get("seed", 42)),
                    ).fit(
                        train.loc[:, list(MODEL_FEATURE_COLUMNS)].to_dict("records"),
                        [bool(int(value)) for value in train["baran_correct"]],
                        [bool(int(value)) for value in train["llm_correct_in_query"]],
                        [bool(int(value)) for value in train["executable_propose"]],
                        [base_family(value) for value in train["dataset"].astype(str)],
                    )
                    predicted = gate.predict(
                        target.loc[:, list(MODEL_FEATURE_COLUMNS)].to_dict("records")
                    )
                    q_helpful = [float(value.q_helpful) for value in predicted]
                    q_harmful = [float(value.q_harmful) for value in predicted]
                    ranked = sorted(
                        range(len(predicted)),
                        key=lambda index: (
                            -float(predicted[index].conservative_uplift),
                            str(target.iloc[index]["query_id"]),
                            str(target.iloc[index]["cell_id"]),
                        ),
                    )
                    top_count = max(1, math.ceil(0.1 * len(ranked)))
                    top = ranked[:top_count]
                    helpful_prevalence = sum(helpful) / len(helpful)
                    harmful_prevalence = sum(harmful) / len(harmful)
                    rows.append(
                        {
                            "backend": backend,
                            "group_size_variant": variant,
                            "allowed_group_sizes": ",".join(
                                str(value) for value in allowed_sizes
                            ),
                            "target_suite": "tableeg",
                            "target_dataset": dataset,
                            "train_pairs": len(train),
                            "test_pairs": len(target),
                            "helpful_prevalence": helpful_prevalence,
                            "helpful_auprc": (
                                float(average_precision_score(helpful, q_helpful))
                                if len(set(helpful)) > 1
                                else helpful_prevalence
                            ),
                            "helpful_brier": float(
                                brier_score_loss(helpful, q_helpful)
                            ),
                            "harmful_prevalence": harmful_prevalence,
                            "harmful_auprc": (
                                float(average_precision_score(harmful, q_harmful))
                                if len(set(harmful)) > 1
                                else harmful_prevalence
                            ),
                            "harmful_brier": float(
                                brier_score_loss(harmful, q_harmful)
                            ),
                            "top_ranked_pairs": top_count,
                            "top_ranked_observed_uplift": sum(
                                helpful[index] - harmful[index] for index in top
                            )
                            / top_count,
                            "diagnostic_only": True,
                        }
                    )
        _write_csv(self.paths.metrics_dir / "routeability_by_dataset.csv", rows)
        grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[(str(row["backend"]), str(row["group_size_variant"]))].append(row)
        macro = {
            f"{backend}:{variant}": {
                field: statistics.fmean(float(row[field]) for row in values)
                for field in (
                    "helpful_auprc",
                    "helpful_brier",
                    "harmful_auprc",
                    "harmful_brier",
                    "top_ranked_observed_uplift",
                )
            }
            for (backend, variant), values in sorted(grouped.items())
        }
        summary: dict[str, object] = {
            "router_revision": self.router_revision,
            "diagnostic_only": True,
            "phase3_blocked": False,
            "folds": len(rows),
            "tableeg_datasets": 9,
            "variants": list(self._router_training_variants()),
            "macro": macro,
        }
        write_json(self.paths.metrics_dir / "routeability_summary.json", summary)
        self.state.update_stage("router_diagnostics", "complete", **summary)
        return summary

    @staticmethod
    def _singleton_reference_cost(pair_rows: pd.DataFrame) -> int:
        singletons = pair_rows.loc[
            (pd.to_numeric(pair_rows["group_size"], errors="raise") == 1)
            & pair_rows["group_view"].astype(str).eq("singleton")
        ].copy()
        if singletons["cell_id"].astype(str).duplicated().any():
            raise ValueError("singleton reference has duplicate cells")
        costs = pd.to_numeric(singletons["estimated_total_tokens"], errors="raise")
        if bool((costs <= 0).any()):
            raise ValueError("singleton reference costs must be positive")
        return int(costs.sum())

    def train_and_select_stage(self) -> dict[str, object]:
        """Fit size-conditioned family-holdout gates and select every slice."""

        return self._train_and_select_router_v3()


    def _reuse_router_v3_gate_artifacts(self) -> dict[str, object]:
        """Copy request-independent k=2/4 LightGBM predictions from frozen v3."""

        if not self.is_router_v3_budget_sweep:
            return {"reused": False}
        if self.router_artifact_reuse_run is None:
            raise ValueError("Router-v3 budget sweep has no gate artifact parent")
        provenance_path = (
            self.paths.run_dir / "provenance" / "router_artifact_reuse.json"
        )
        if provenance_path.is_file():
            summary = load_json(provenance_path)
            for row in summary.get("artifacts", []):
                if not isinstance(row, Mapping):
                    raise ValueError("gate artifact provenance row is invalid")
                for field, hash_field in (
                    ("prediction", "prediction_sha256"),
                    ("metadata", "metadata_sha256"),
                ):
                    path = self.paths.run_dir / str(row[field])
                    if not path.is_file() or sha256_file(path) != str(row[hash_field]):
                        raise ValueError("reused Router-v3 gate artifact drift")
            return summary

        parent = self.router_artifact_reuse_run
        parent_manifest = load_json(parent / "run_manifest.json")
        parent_experiment = parent_manifest.get("experiment_config", {})
        if (
            parent_manifest.get("status") != "complete"
            or not isinstance(parent_experiment, Mapping)
            or str(parent_experiment.get("router_revision", ""))
            != ROUTER_V3_REVISION
            or sha256_file(parent / "run_manifest.json")
            != str(
                self.state.manifest.get(
                    "router_artifact_reuse_manifest_sha256", ""
                )
            )
        ):
            raise ValueError("Router-v3 gate artifact parent is not the frozen complete run")

        rows: list[dict[str, object]] = []
        for suite, dataset in target_order():
            for variant, allowed_sizes in self._router_training_variants().items():
                for backend in self._active_gate_backends():
                    relative_prediction = (
                        Path("gates")
                        / backend
                        / f"variant_{variant}"
                        / f"{_dataset_key(suite, dataset)}.csv"
                    )
                    relative_metadata = relative_prediction.with_suffix(
                        ".metadata.json"
                    )
                    source_prediction = parent / relative_prediction
                    source_metadata = parent / relative_metadata
                    metadata = load_json(source_metadata)
                    predictions = _read_csv(source_prediction)
                    if (
                        str(metadata.get("target_suite", "")) != suite
                        or str(metadata.get("target_dataset", "")) != dataset
                        or str(metadata.get("group_size_variant", "")) != variant
                        or tuple(
                            int(value)
                            for value in metadata.get("train_group_sizes", [])
                        )
                        != allowed_sizes
                        or tuple(
                            int(value)
                            for value in metadata.get("test_group_sizes", [])
                        )
                        != allowed_sizes
                        or set(
                            pd.to_numeric(predictions["group_size"], errors="raise")
                        )
                        != set(allowed_sizes)
                    ):
                        raise ValueError(
                            f"Router-v3 parent gate metadata drift: {backend}/{variant}/{suite}/{dataset}"
                        )
                    destination_prediction = self.paths.run_dir / relative_prediction
                    destination_metadata = self.paths.run_dir / relative_metadata
                    destination_prediction.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source_prediction, destination_prediction)
                    shutil.copyfile(source_metadata, destination_metadata)
                    rows.append(
                        {
                            "backend": backend,
                            "variant": variant,
                            "suite": suite,
                            "dataset": dataset,
                            "prediction": relative_prediction.as_posix(),
                            "metadata": relative_metadata.as_posix(),
                            "prediction_sha256": sha256_file(source_prediction),
                            "metadata_sha256": sha256_file(source_metadata),
                        }
                    )
        expected = (
            len(TEST_TARGETS)
            * len(self._router_training_variants())
            * len(self._active_gate_backends())
        )
        if len(rows) != expected:
            raise ValueError("Router-v3 gate artifact reuse matrix is incomplete")
        summary: dict[str, object] = {
            "reused": True,
            "parent_run": str(parent),
            "parent_manifest_sha256": sha256_file(parent / "run_manifest.json"),
            "model_folds": len(rows),
            "prediction_files": len(rows),
            "metadata_files": len(rows),
            "artifacts": rows,
        }
        write_json(provenance_path, summary)
        return summary

    def _assert_parent_20pct_selection(
        self,
        *,
        backend: str,
        variant: str,
        suite: str,
        dataset: str,
        selected_ids: Sequence[str],
    ) -> None:
        if not self.is_router_v3_budget_sweep:
            return
        if self.router_artifact_reuse_run is None:
            raise ValueError("Router-v3 budget sweep has no selection parent")
        parent_path = (
            self.router_artifact_reuse_run
            / "selections"
            / backend
            / "size_conditioned"
            / f"variant_{variant}"
            / _budget_label(0.2)
            / f"{_dataset_key(suite, dataset)}.json"
        )
        parent_ids = tuple(
            str(value)
            for value in load_json(parent_path).get("selected_query_ids", [])
        )
        if tuple(selected_ids) != parent_ids:
            raise ValueError(
                f"20% selection differs from frozen Router-v3 parent: {backend}/{variant}/{suite}/{dataset}"
            )

    def _train_and_select_router_v3(self) -> dict[str, object]:
        """Fit or reuse one model per target/backend/k, then select each budget."""

        if not self.state.stage_completed("calibration_llm"):
            raise RuntimeError("router training requires a complete calibration ledger")
        planned = {
            (str(row["query_id"]), str(row["prompt_hash"]))
            for row in read_jsonl(self.paths.llm_dir / "calibration_queries.jsonl")
        }
        executed_rows = read_jsonl(self.paths.llm_dir / "calibration_execution.jsonl")
        executed = {
            (str(row.get("query_id", "")), str(row.get("prompt_hash", "")))
            for row in executed_rows
        }
        if len(executed) != len(executed_rows) or executed != planned:
            raise ValueError("calibration execution ledger coverage or uniqueness failed")
        if any(
            row.get("model_matches_request", True) is False
            or str(row.get("model", "")) != str(self.llm_config["model"])
            for row in executed_rows
            if row.get("status") == "success"
        ):
            raise ValueError("calibration execution ledger contains a model mismatch")

        gate_reuse = self._reuse_router_v3_gate_artifacts()
        all_pairs = self._all_pair_features()
        labels = _read_csv(self.paths.llm_dir / "calibration_pair_labels.csv")
        labels = labels.loc[
            :,
            [
                "cell_id",
                "query_id",
                "baran_correct",
                "llm_correct_in_query",
                "executable_propose",
                "helpful",
                "harmful",
            ],
        ].copy()
        if labels.duplicated(["cell_id", "query_id"]).any():
            raise ValueError("calibration labels contain duplicate cell-query pairs")

        variants = self._router_training_variants()
        budgets = self._router_budget_shares()
        backends = self._active_gate_backends()
        selection_rows: list[dict[str, object]] = []
        logical_rows: list[dict[str, object]] = []
        split_rows: list[dict[str, object]] = []
        prediction_parts: dict[str, list[pd.DataFrame]] = defaultdict(list)
        bgr_union_ids: set[str] = set()
        llm_only_ids: set[str] = set()
        total_prediction_rows = 0

        for suite, dataset in target_order():
            train_safe, test_safe, base_audit = split_for_target(
                all_pairs, suite, dataset, enforce_target_unlabeled=True
            )
            train_all = train_safe.merge(
                labels,
                how="inner",
                on=["cell_id", "query_id"],
                validate="one_to_one",
            )
            if train_all.empty:
                raise ValueError(f"no sampled calibration labels for target {suite}/{dataset}")
            actions = {
                action.query_id: action
                for action in self._load_actions(suite, dataset)
            }
            if len(test_safe) != sum(action.group_size for action in actions.values()):
                raise ValueError(f"test pair table is incomplete for {suite}/{dataset}")
            reference_cost = self._singleton_reference_cost(test_safe)
            dataset_singletons = {
                query_id
                for query_id, action in actions.items()
                if action.group_size == 1 and action.group_view == "singleton"
            }
            if len(dataset_singletons) != len(self._dataset(suite, dataset).safe_cells()):
                raise ValueError(f"LLM-only singleton coverage failed for {suite}/{dataset}")
            llm_only_ids.update(dataset_singletons)
            costs = {
                query_id: float(action.estimated_total_tokens)
                for query_id, action in actions.items()
            }
            print(
                f"[gate-v3] {suite}/{dataset}: train={len(train_all)} pairs, test={len(test_safe)} pairs",
                flush=True,
            )

            for variant, allowed_sizes in variants.items():
                allowed = set(allowed_sizes)
                train = self._filter_variant_pairs(
                    train_all,
                    allowed_sizes,
                    context=f"train {variant}/{suite}/{dataset}",
                )
                test = self._filter_variant_pairs(
                    test_safe,
                    allowed_sizes,
                    context=f"test {variant}/{suite}/{dataset}",
                )
                candidates = tuple(
                    sorted(
                        query_id
                        for query_id, action in actions.items()
                        if action.group_size in allowed
                    )
                )
                if len(test) != sum(actions[value].group_size for value in candidates):
                    raise ValueError(
                        f"Router-v3 pair coverage mismatch for {variant} {suite}/{dataset}"
                    )

                for backend in backends:
                    prediction_path = self._prediction_path(
                        backend, variant, suite, dataset
                    )
                    predictions = (
                        _read_csv(prediction_path)
                        if prediction_path.is_file()
                        else pd.DataFrame()
                    )
                    if len(predictions) != len(test):
                        gate = GroupUpliftGate(
                            backend,  # type: ignore[arg-type]
                            rho=float(self.experiment_config.get("harm_penalty_rho", 1.0)),
                            gamma=float(
                                self.experiment_config.get(
                                    "uncertainty_penalty_gamma", 1.0
                                )
                            ),
                            random_state=int(self.experiment_config.get("seed", 42)),
                        ).fit(
                            train.loc[:, list(MODEL_FEATURE_COLUMNS)].to_dict("records"),
                            [bool(int(value)) for value in train["baran_correct"]],
                            [bool(int(value)) for value in train["llm_correct_in_query"]],
                            [bool(int(value)) for value in train["executable_propose"]],
                            [base_family(value) for value in train["dataset"].astype(str)],
                        )
                        predicted = gate.predict(
                            test.loc[:, list(MODEL_FEATURE_COLUMNS)].to_dict("records")
                        )
                        predictions = test.loc[
                            :,
                            [
                                "suite",
                                "dataset",
                                "cell_id",
                                "query_id",
                                "group_signature",
                                "group_view",
                                "group_size",
                                "estimated_total_tokens",
                            ],
                        ].copy()
                        predictions["group_size_variant"] = variant
                        for field in (
                            "q_helpful",
                            "q_harmful",
                            "net_gain",
                            "sigma",
                            "conservative_uplift",
                        ):
                            predictions[field] = [
                                prediction.as_dict()[field] for prediction in predicted
                            ]
                        _write_csv(prediction_path, predictions.to_dict("records"))
                        write_json(
                            prediction_path.with_suffix(".metadata.json"),
                            {
                                "router_revision": self.router_revision,
                                "target_suite": suite,
                                "target_dataset": dataset,
                                "group_size_variant": variant,
                                "allowed_group_sizes": list(allowed_sizes),
                                "train_pair_rows": len(train),
                                "test_pair_rows": len(test),
                                "train_group_sizes": sorted(
                                    {int(value) for value in train["group_size"]}
                                ),
                                "test_group_sizes": sorted(
                                    {int(value) for value in test["group_size"]}
                                ),
                                "model": gate.metadata(),
                                "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
                                "target_labels_used": False,
                                "target_responses_used_before_selection": False,
                            },
                        )
                    if set(predictions["group_size_variant"].astype(str)) != {variant}:
                        raise ValueError(
                            f"prediction variant drift for {backend}/{variant}/{suite}/{dataset}"
                        )
                    if set(
                        pd.to_numeric(predictions["group_size"], errors="raise")
                    ) != allowed:
                        raise ValueError(
                            f"prediction size drift for {backend}/{variant}/{suite}/{dataset}"
                        )
                    total_prediction_rows += len(predictions)
                    prediction_parts[backend].append(predictions)
                    split_rows.append(
                        {
                            **base_audit.as_dict(),
                            "backend": backend,
                            "group_size_variant": variant,
                            "allowed_group_sizes": ",".join(
                                str(value) for value in allowed_sizes
                            ),
                            "train_test_row_overlap": base_audit.train_test_row_identity_overlap,
                            "train_pair_rows_after_sampling": len(train),
                            "test_pair_rows": len(test),
                            "target_group_label_used": False,
                            "target_response_used_before_selection": False,
                            "target_response_visible_before_selection": False,
                            "model_reused": bool(gate_reuse.get("reused")),
                        }
                    )
                    objective = GroupUpliftObjective(
                        [
                            PairGain(
                                str(row.cell_id),
                                str(row.query_id),
                                max(0.0, float(row.conservative_uplift)),
                            )
                            for row in predictions.itertuples(index=False)
                        ]
                    )
                    for budget_share in budgets:
                        budget = int(round(reference_cost * budget_share))
                        result = select_queries(
                            objective, costs, budget, candidates=candidates
                        )
                        if result.total_cost > budget + 1e-9:
                            raise AssertionError(
                                "Router-v3 selection exceeded its estimated-token budget"
                            )
                        selected_ids = tuple(result.selected_query_ids)
                        if any(
                            actions[query_id].group_size not in allowed
                            for query_id in selected_ids
                        ):
                            raise AssertionError(
                                "Router-v3 selected a disallowed group size"
                            )
                        if math.isclose(budget_share, 0.2, abs_tol=1e-12):
                            self._assert_parent_20pct_selection(
                                backend=backend,
                                variant=variant,
                                suite=suite,
                                dataset=dataset,
                                selected_ids=selected_ids,
                            )
                        bgr_union_ids.update(selected_ids)
                        covered = [
                            cell_id
                            for query_id in selected_ids
                            for cell_id in actions[query_id].cell_ids
                        ]
                        path = self._selection_path(
                            backend,
                            "size_conditioned",
                            variant,
                            budget_share,
                            suite,
                            dataset,
                        )
                        write_json(
                            path,
                            {
                                **result.as_dict(),
                                "router_revision": self.router_revision,
                                "suite": suite,
                                "dataset": dataset,
                                "backend": backend,
                                "scenario": "size_conditioned",
                                "group_size_variant": variant,
                                "training_group_sizes": list(allowed_sizes),
                                "allowed_group_sizes": list(allowed_sizes),
                                "budget_share": budget_share,
                                "budget_reference_tokens": reference_cost,
                                "selected_cell_incidence": len(covered),
                                "unique_covered_cells": len(set(covered)),
                            },
                        )
                        selection_rows.append(
                            {
                                "suite": suite,
                                "dataset": dataset,
                                "backend": backend,
                                "scenario": "size_conditioned",
                                "group_size_variant": variant,
                                "training_group_sizes": ",".join(
                                    str(value) for value in allowed_sizes
                                ),
                                "budget_share": budget_share,
                                "budget_reference_tokens": reference_cost,
                                "budget_estimated_tokens": budget,
                                "selected_groups": len(selected_ids),
                                "selected_estimated_tokens": int(result.total_cost),
                                "budget_slack_tokens": int(budget - result.total_cost),
                                "objective_value": result.objective_value,
                                "algorithm": result.algorithm,
                                "unique_covered_cells": len(set(covered)),
                                "covered_cell_incidences": len(covered),
                                "overlap_incidences": len(covered) - len(set(covered)),
                                "within_budget": True,
                            }
                        )
                        for query_id in selected_ids:
                            action = actions[query_id]
                            logical_rows.append(
                                {
                                    "target_suite": suite,
                                    "target_dataset": dataset,
                                    "backend": backend,
                                    "scenario": "size_conditioned",
                                    "group_size_variant": variant,
                                    "budget_share": budget_share,
                                    "query_id": query_id,
                                    "prompt_hash": action.prompt_hash,
                                    "selected": True,
                                    "estimated_tokens": action.estimated_total_tokens,
                                    "actual_tokens_if_available": "",
                                    "logical_api_calls": 1,
                                    "physical_api_calls": 0,
                                    "covered_cells": action.group_size,
                                    "accepted_llm_cells": 0,
                                }
                            )

            for query_id in sorted(dataset_singletons):
                action = actions[query_id]
                logical_rows.append(
                    {
                        "target_suite": suite,
                        "target_dataset": dataset,
                        "backend": "none",
                        "scenario": "llm_only_baseline",
                        "group_size_variant": "1",
                        "budget_share": 1.0,
                        "query_id": query_id,
                        "prompt_hash": action.prompt_hash,
                        "selected": True,
                        "estimated_tokens": action.estimated_total_tokens,
                        "actual_tokens_if_available": "",
                        "logical_api_calls": 1,
                        "physical_api_calls": 0,
                        "covered_cells": 1,
                        "accepted_llm_cells": 0,
                    }
                )

        _write_csv(self.paths.gates_dir / "split_audit.csv", split_rows)
        for backend in backends:
            _write_csv(
                self.paths.gates_dir / f"{backend}_pair_predictions.csv",
                pd.concat(prediction_parts[backend], ignore_index=True).to_dict(
                    "records"
                ),
            )
        _write_csv(self.paths.metrics_dir / "selection_audit.csv", selection_rows)

        union_ids = bgr_union_ids | llm_only_ids
        union_actions = {
            action.query_id: action
            for target_suite, target_dataset in target_order()
            for action in self._load_actions(target_suite, target_dataset)
            if action.query_id in union_ids
        }
        if set(union_actions) != union_ids:
            raise ValueError("Router-v3 union action identity coverage failed")
        response_index = self._response_index()
        cached_success_ids: set[str] = set()
        cached_failure_ids: set[str] = set()
        for query_id, action in union_actions.items():
            response = response_index.get((query_id, action.prompt_hash))
            if response is None or response.get("model_matches_request", True) is False:
                continue
            if response.get("status") == "success":
                cached_success_ids.add(query_id)
            else:
                cached_failure_ids.add(query_id)
        terminal_ids = (
            cached_success_ids | cached_failure_ids
            if self.freezes_reused_terminal_failures
            else cached_success_ids
        )
        online_ids = sorted(union_ids - terminal_ids)
        online_id_set = set(online_ids)
        online_estimate = sum(
            union_actions[query_id].estimated_total_tokens for query_id in online_ids
        )
        preflight_path = self.paths.llm_dir / "model_preflight.json"
        preflight_estimate = (
            int(load_json(preflight_path).get("estimated_total_tokens", 0) or 0)
            if preflight_path.is_file()
            else 1_722
        )
        calibration_estimate = int(
            load_json(self.paths.llm_dir / "calibration_plan.json").get(
                "estimated_tokens", 0
            )
            or 0
        )
        combined_estimate = online_estimate + preflight_estimate
        raw_cap = self.experiment_config.get("max_estimated_tokens_safety_cap")
        safety_cap = None if raw_cap is None else int(raw_cap)
        union_plan: dict[str, object] = {
            "router_revision": self.router_revision,
            "selection_slices": len(selection_rows),
            "model_folds": len(split_rows),
            "selected_union_queries": len(union_ids),
            "bgr_selected_union_queries": len(bgr_union_ids),
            "llm_only_singleton_queries": len(llm_only_ids),
            "cached_success_queries_in_union": len(cached_success_ids),
            "cached_terminal_failure_queries_in_union": len(cached_failure_ids),
            "cached_terminal_queries_in_union": len(terminal_ids),
            "online_physical_queries": len(online_ids),
            "offline_calibration_logical_estimated_tokens": calibration_estimate,
            "online_union_estimated_tokens": online_estimate,
            "model_preflight_estimated_tokens": preflight_estimate,
            "combined_physical_estimated_tokens": combined_estimate,
            "safety_cap": safety_cap,
            "query_ids": sorted(union_ids),
            "bgr_query_ids": sorted(bgr_union_ids),
            "llm_only_query_ids": sorted(llm_only_ids),
            "online_query_ids": online_ids,
            "cached_failure_query_ids": sorted(cached_failure_ids),
        }
        write_json(self.paths.llm_dir / "selected_union_plan.json", union_plan)
        if self.is_router_v3_budget_sweep:
            write_json(
                self.paths.llm_dir / "router_v3_budget_sweep_dry_plan.json",
                {
                    **union_plan,
                    "backends": list(backends),
                    "variants": list(variants),
                    "budget_shares": list(budgets),
                    "selection_summary": selection_rows,
                    "gate_artifact_reuse": gate_reuse,
                    "api_called": False,
                },
            )
        elif self.is_router_v3_catboost:
            write_json(
                self.paths.llm_dir / "router_v3_catboost_dry_plan.json",
                {
                    **union_plan,
                    "backends": list(backends),
                    "variants": list(variants),
                    "budget_shares": list(budgets),
                    "selection_summary": selection_rows,
                    "api_called": False,
                },
            )
        if safety_cap is not None and combined_estimate > safety_cap:
            self.state.update_stage(
                "gate_selection", "safety_cap_exceeded", **union_plan
            )
            raise SafetyCapExceeded(
                "Router-v3 missing selected union", combined_estimate, safety_cap
            )
        for row in logical_rows:
            response = response_index.get(
                (str(row["query_id"]), str(row["prompt_hash"]))
            )
            actual = _actual_tokens(response)
            row["actual_tokens_if_available"] = "" if actual is None else actual
            row["physical_api_calls"] = int(str(row["query_id"]) in online_id_set)
        _write_csv(
            self.paths.metrics_dir / "logical_budget_ledger.csv",
            logical_rows,
            columns=LOGICAL_LEDGER_COLUMNS,
        )
        self.build_router_diagnostics_stage(all_pairs, labels)
        stage_summary = {
            "router_revision": self.router_revision,
            "prediction_rows": total_prediction_rows,
            "selection_slices": len(selection_rows),
            "model_folds": len(split_rows),
            "gate_artifacts_reused": bool(gate_reuse.get("reused")),
            **{
                key: value
                for key, value in union_plan.items()
                if key
                not in {
                    "query_ids",
                    "bgr_query_ids",
                    "llm_only_query_ids",
                    "online_query_ids",
                    "cached_failure_query_ids",
                }
            },
        }
        self.state.update_stage("gate_selection", "complete", **stage_summary)
        return {
            "prediction_rows": total_prediction_rows,
            "selection_slices": len(selection_rows),
            "model_folds": len(split_rows),
            **union_plan,
        }

    def _response_index(
        self, *, phase: str | None = None
    ) -> dict[tuple[str, str], dict[str, object]]:
        index: dict[tuple[str, str], dict[str, object]] = {}
        for row in read_jsonl(self.paths.llm_dir / "group_query_checkpoint.jsonl"):
            query_id = str(row.get("query_id", ""))
            prompt_hash = str(row.get("prompt_hash", ""))
            metadata = row.get("metadata")
            source_phase = (
                str(metadata.get("phase", ""))
                if isinstance(metadata, Mapping)
                else ""
            )
            if phase is not None and not (
                source_phase == phase
                or (
                    source_phase == "offline_group_calibration"
                    and phase == "online_selected_union"
                )
                or (
                    source_phase
                    in {
                        "preliminary_singleton",
                        "preliminary_structured",
                        "preliminary_random",
                    }
                    and phase
                    in {"offline_group_calibration", "online_selected_union"}
                )
            ):
                continue
            if query_id and prompt_hash:
                index[(query_id, prompt_hash)] = dict(row)
        return index

    def run_selected_llm_stage(self) -> dict[str, object]:
        if not self.state.stage_completed("model_preflight"):
            raise RuntimeError("Router-v3 selected execution requires model preflight")
        plan = load_json(self.paths.llm_dir / "selected_union_plan.json")
        union_ids = {str(value) for value in plan.get("query_ids", [])}
        selected_actions: list[GroupQueryAction] = []
        for suite, dataset in target_order():
            selected_actions.extend(
                action
                for action in self._load_actions(suite, dataset)
                if action.query_id in union_ids
            )
        if {action.query_id for action in selected_actions} != union_ids:
            raise ValueError("selected union references missing query actions")
        actions = tuple(sorted(selected_actions, key=lambda value: value.query_id))
        results = self._execute_jobs(
            actions,
            phase="online_selected_union",
            output_path=self.paths.llm_dir / "selected_execution.jsonl",
            retry_failed=(
                str(self.experiment_config.get("invalid_response_policy"))
                != "fallback_baran_after_client_retries"
            ),
        )
        response_index = self._response_index()
        missing = [
            action.query_id
            for action in actions
            if (action.query_id, action.prompt_hash) not in response_index
        ]
        if missing:
            raise ValueError(f"selected query ledger is missing {len(missing)} records")
        summary = {
            "union_queries": len(actions),
            "successful_queries": sum(row.get("status") == "success" for row in results),
            "failed_queries": sum(row.get("status") != "success" for row in results),
            "checkpoint_hits": sum(bool(row.get("checkpoint_hit")) for row in results),
            "cache_hits": sum(bool(row.get("cache_hit")) for row in results),
        }
        fallback_enabled = (
            str(self.experiment_config.get("invalid_response_policy"))
            == "fallback_baran_after_client_retries"
        )
        fallback_summary = self._materialize_baran_fallbacks(
            actions,
            results,
            phase="online_selected_union",
        )
        summary.update(fallback_summary)
        if int(summary["unresolved_operational_failures"]) > 0:
            self.state.update_stage("selected_llm", "failed", **summary)
            raise RuntimeError("selected union contains Baran fallback cells without coverage")
        if int(summary["failed_queries"]) > 0 and not fallback_enabled:
            self.state.update_stage("selected_llm", "failed", **summary)
            raise RuntimeError(
                f"selected union has {summary['failed_queries']} failed queries; "
                "resume the same run to retry them"
            )
        self.state.update_stage("selected_llm", "complete", **summary)
        return summary

    @staticmethod
    def _compact_baran_record(
        record: Mapping[str, object],
        clean_value: str,
    ) -> dict[str, object]:
        prediction = record.get("prediction")
        parse_status = str(record.get("parse_status", "no_prediction"))
        correct = bool(
            parse_status.startswith("ok")
            and prediction is not None
            and normalize_for_match(prediction) == normalize_for_match(clean_value)
        )
        return {
            "cell_id": str(record["cell_id"]),
            "suite": str(record["suite"]),
            "dataset": str(record["dataset"]),
            "method": "baran",
            "scenario": "baseline",
            "backend": "none",
            "budget_share": None,
            "group_size_variant": "all",
            "prediction": prediction,
            "clean_value": clean_value,
            "parse_status": parse_status,
            "valid_prediction": parse_status.startswith("ok"),
            "correct_repair": correct,
            "final_source": "baran",
        }

    def build_final_records_stage(self) -> dict[str, object]:
        """Build both baselines and every Router-v3 BGR method slice."""

        return self._build_final_records_router_v3()


    @staticmethod
    def _compact_llm_only_record(
        action: GroupQueryAction,
        response: Mapping[str, object],
        cell: SafeCell,
        clean_value: str,
    ) -> dict[str, object]:
        """Materialize a standalone singleton LLM result without Baran fallback."""

        response_usable = (
            response.get("status") == "success"
            and response.get("model_matches_request", True) is not False
        )
        raw_items = response.get("items", []) if response_usable else []
        matches = [
            raw
            for raw in raw_items
            if isinstance(raw, Mapping)
            and str(raw.get("cell_id", "")) == str(cell.cell_id)
        ] if isinstance(raw_items, list) else []
        item = matches[0] if len(matches) == 1 else None
        decision = str(item.get("decision", "")) if item is not None else ""
        repair = str(item.get("repair", "")) if item is not None else ""
        valid = bool(
            response_usable
            and item is not None
            and decision == "propose"
            and repair.strip()
            and normalize_for_match(repair) != normalize_for_match(cell.dirty_value)
        )
        if valid:
            parse_status = "ok_llm_only"
        elif not response_usable:
            parse_status = (
                "model_mismatch"
                if response.get("model_matches_request", True) is False
                else "provider_failure"
            )
        elif item is None:
            parse_status = "missing_or_duplicate_item"
        elif decision == "abstain":
            parse_status = "abstain"
        elif not repair.strip():
            parse_status = "empty_repair"
        elif normalize_for_match(repair) == normalize_for_match(cell.dirty_value):
            parse_status = "unchanged_dirty"
        else:
            parse_status = "invalid_item"
        prediction = repair if valid else None
        correct = bool(
            valid
            and normalize_for_match(prediction) == normalize_for_match(clean_value)
        )
        return {
            "cell_id": str(cell.cell_id),
            "suite": cell.suite,
            "dataset": cell.dataset,
            "method": "llm_only",
            "scenario": "baseline",
            "backend": "none",
            "budget_share": None,
            "group_size_variant": "1",
            "prediction": prediction,
            "clean_value": clean_value,
            "parse_status": parse_status,
            "valid_prediction": valid,
            "correct_repair": correct,
            "final_source": "llm" if valid else "no_repair",
            "accepted_llm": valid,
            "selected_query_id": action.query_id,
            "llm_decision": decision or "missing",
            "baran_fallback_used": False,
        }

    def _full_singleton_actions(self) -> tuple[GroupQueryAction, ...]:
        actions: list[GroupQueryAction] = []
        for suite, dataset in target_order():
            singletons = [
                action
                for action in self._load_actions(suite, dataset)
                if action.group_view == "singleton" and action.group_size == 1
            ]
            expected = len(self._dataset(suite, dataset).safe_cells())
            if len(singletons) != expected:
                raise ValueError(
                    f"singleton coverage failure for {suite}/{dataset}: "
                    f"expected={expected}, observed={len(singletons)}"
                )
            actions.extend(singletons)
        if len(actions) != TEST_TARGET_CELL_COUNT:
            raise ValueError(
                "formal singleton baseline must contain exactly "
                f"{TEST_TARGET_CELL_COUNT:,} queries"
            )
        if len({action.query_id for action in actions}) != len(actions):
            raise ValueError("formal singleton baseline contains duplicate query IDs")
        return tuple(sorted(actions, key=lambda value: value.query_id))

    def plan_full_baselines_stage(self) -> dict[str, object]:
        """Plan the formal nine-dataset baselines without training a Router."""

        self.validate_inputs()
        baran = self.run_baran_stage(target_order())
        groups = self.generate_groups_stage(target_order())
        reuse = self.import_reusable_no_baran_responses_stage(target_order())
        actions = self._full_singleton_actions()
        response_index = self._response_index()
        cached_success = {
            action.query_id
            for action in actions
            if (response := response_index.get((action.query_id, action.prompt_hash)))
            is not None
            and response.get("status") == "success"
        }
        cached_failure = {
            action.query_id
            for action in actions
            if (response := response_index.get((action.query_id, action.prompt_hash)))
            is not None
            and response.get("status") != "success"
        }
        terminal_ids = cached_success | cached_failure
        online_actions = [
            action for action in actions if action.query_id not in terminal_ids
        ]
        online_estimate = sum(
            int(action.estimated_total_tokens) for action in online_actions
        )
        plan: dict[str, object] = {
            "run_kind": "full_baselines",
            "router_trained": False,
            "test_datasets": len(TEST_TARGETS),
            "test_oracle_cells": TEST_TARGET_CELL_COUNT,
            "selected_union_queries": len(actions),
            "bgr_selected_union_queries": 0,
            "llm_only_singleton_queries": len(actions),
            "cached_success_queries_in_union": len(cached_success),
            "cached_terminal_failure_queries_in_union": len(cached_failure),
            "cached_terminal_queries_in_union": len(terminal_ids),
            "online_physical_queries": len(online_actions),
            "online_union_estimated_tokens": online_estimate,
            "query_ids": [action.query_id for action in actions],
            "bgr_query_ids": [],
            "llm_only_query_ids": [action.query_id for action in actions],
            "online_query_ids": [action.query_id for action in online_actions],
            "cached_failure_query_ids": sorted(cached_failure),
        }
        write_json(self.paths.llm_dir / "selected_union_plan.json", plan)
        return {
            "baran": baran,
            "groups": groups,
            "response_reuse": reuse,
            "selected_union": plan,
            "api_called": False,
        }

    def build_baseline_records_stage(self) -> dict[str, object]:
        """Materialize only Baran and pure No-Baran singleton LLM records."""

        response_index = self._response_index()
        records: list[dict[str, object]] = []
        for suite, dataset in target_order():
            loaded = self._dataset(suite, dataset)
            cells = tuple(loaded.safe_view().cells)
            clean = self._clean_value_map(loaded, cells)
            baran = {
                str(row["cell_id"]): row
                for row in self._load_baran(suite, dataset)
            }
            singleton_by_cell = {
                action.cell_ids[0]: action
                for action in self._load_actions(suite, dataset)
                if action.group_view == "singleton" and action.group_size == 1
            }
            for cell in sorted(cells, key=lambda value: str(value.cell_id)):
                cell_id = str(cell.cell_id)
                action = singleton_by_cell[cell_id]
                response = response_index.get((action.query_id, action.prompt_hash))
                if response is None:
                    raise ValueError(
                        f"singleton response ledger is missing {action.query_id}"
                    )
                records.append(self._compact_baran_record(baran[cell_id], clean[cell_id]))
                records.append(
                    self._compact_llm_only_record(
                        action,
                        response,
                        cell,
                        clean[cell_id],
                    )
                )
        expected = TEST_TARGET_CELL_COUNT * 2
        if len(records) != expected:
            raise ValueError(
                f"baseline cell ledger contains {len(records)} rows, expected {expected}"
            )
        audit = verify_records(records)
        if not bool(audit.get("ok")):
            raise ValueError(f"baseline record audit failed: {audit}")
        if any(
            row.get("baran_fallback_used") is not False
            for row in records
            if str(row.get("method")) == "llm_only"
        ):
            raise ValueError("LLM-only baseline contains a Baran fallback")
        write_jsonl(self.paths.final_dir / "all_methods.jsonl", records)
        write_json(self.paths.metrics_dir / "record_audit.json", audit)
        summary = {
            "records": len(records),
            "method_slices": 2,
            "baran_records": TEST_TARGET_CELL_COUNT,
            "llm_only_records": TEST_TARGET_CELL_COUNT,
            "llm_only_baran_fallbacks": 0,
            "coverage_ok": True,
        }
        self.state.update_stage("final_records", "complete", **summary)
        return summary

    def build_baseline_metrics_stage(self) -> dict[str, object]:
        records = read_jsonl(self.paths.final_dir / "all_methods.jsonl")
        summaries = summarize_records(records, strict=True)
        _write_csv(self.paths.metrics_dir / "method_metrics.csv", summaries)
        summary = {
            "method_metric_rows": len(summaries),
            "methods": ["baran", "llm_only"],
            "datasets": len(TEST_TARGETS),
            "cells_per_method": TEST_TARGET_CELL_COUNT,
        }
        self.state.update_stage("metrics", "complete", **summary)
        return summary

    def run_full_baselines(
        self,
        *,
        baseline_dir: str | Path,
        output_dir: str | Path,
        bootstrap_replicates: int = 2_000,
        bootstrap_seed: int = 45,
        confidence: float = 0.95,
    ) -> dict[str, object]:
        """Run, resume, and analyze the two complete formal baselines."""

        plan = self.plan_full_baselines_stage()
        preflight = self.check_model()
        selected = self.run_selected_llm_stage()
        final = self.build_baseline_records_stage()
        metrics = self.build_baseline_metrics_stage()
        self.state.complete(
            required_stages=BASELINE_REQUIRED_STAGES,
            completed_matrix={
                "run_kind": "full_baselines",
                "datasets": len(TEST_TARGETS),
                "baselines": ["baran_only", "llm_only"],
                "method_slices": 2,
                "cell_records": TEST_TARGET_CELL_COUNT * 2,
                "router_trained": False,
            },
        )
        validation = validate_run(self.paths.run_dir, require_complete=True)
        from .full_complementarity import build_full_complementarity

        analysis = build_full_complementarity(
            self.paths.run_dir,
            baseline_dir=baseline_dir,
            output_dir=output_dir,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        )
        return {
            "run_dir": str(self.paths.run_dir),
            "plan": plan,
            "model_preflight": preflight,
            "selected_llm": selected,
            "final": final,
            "metrics": metrics,
            "validation": validation,
            "complementarity": analysis,
        }

    def _build_final_records_router_v3(self) -> dict[str, object]:
        """Build Baran, standalone LLM, and ten size-conditioned BGR slices."""

        response_index = self._response_index()
        final_path = self.paths.final_dir / "all_methods.jsonl"
        records: list[dict[str, object]] = []
        specs = self._scenario_specs()
        for suite, dataset in target_order():
            dataset_record_start = len(records)
            loaded = self._dataset(suite, dataset)
            safe = loaded.safe_view()
            cells = tuple(safe.cells)
            cell_by_id = {str(cell.cell_id): cell for cell in cells}
            clean = self._clean_value_map(loaded, cells)
            baran = {
                str(row["cell_id"]): row
                for row in self._load_baran(suite, dataset)
            }
            rules = fds_for_dataset(self.fd_registry, suite, dataset)
            verifier_raw = self.experiment_config.get("verifier", {})
            verifier_config = VerifierConfig(
                **dict(verifier_raw) if isinstance(verifier_raw, Mapping) else {}
            )
            verifier = GroupRepairVerifier(
                safe.dirty,
                cells,
                rules,
                verifier_config,
            )
            actions = {
                action.query_id: action
                for action in self._load_actions(suite, dataset)
            }
            singleton_by_cell = {
                action.cell_ids[0]: action
                for action in actions.values()
                if action.group_size == 1 and action.group_view == "singleton"
            }
            if set(singleton_by_cell) != set(cell_by_id):
                raise ValueError(f"LLM-only action coverage failed for {suite}/{dataset}")

            for cell_id in sorted(cell_by_id):
                records.append(
                    self._compact_baran_record(baran[cell_id], clean[cell_id])
                )
            for cell_id in sorted(cell_by_id):
                action = singleton_by_cell[cell_id]
                response = response_index.get(
                    (action.query_id, action.prompt_hash), {}
                )
                records.append(
                    self._compact_llm_only_record(
                        action,
                        response,
                        cell_by_id[cell_id],
                        clean[cell_id],
                    )
                )

            for backend in self._active_gate_backends():
                for spec in specs:
                    scenario = str(spec["scenario"])
                    variant = str(spec["group_size_variant"])
                    budget_share = float(spec["budget_share"])
                    prediction_frame = _read_csv(
                        self._prediction_path(backend, variant, suite, dataset)
                    )
                    uplift = {
                        (str(row.cell_id), str(row.query_id)): float(
                            row.conservative_uplift
                        )
                        for row in prediction_frame.itertuples(index=False)
                    }
                    selection = load_json(
                        self._selection_path(
                            backend,
                            scenario,
                            variant,
                            budget_share,
                            suite,
                            dataset,
                        )
                    )
                    selected_ids = tuple(
                        str(value)
                        for value in selection.get("selected_query_ids", [])
                    )
                    selected_by_cell: dict[str, list[str]] = defaultdict(list)
                    for query_id in selected_ids:
                        for cell_id in actions[query_id].cell_ids:
                            selected_by_cell[cell_id].append(query_id)
                    verification_cache: dict[
                        tuple[str, str, str, str], object
                    ] = {}
                    for cell_id in sorted(cell_by_id):
                        candidates: list[RankedRepairCandidate] = []
                        for query_id in selected_by_cell.get(cell_id, []):
                            action = actions[query_id]
                            response = response_index.get(
                                (query_id, action.prompt_hash), {}
                            )
                            response_usable = (
                                response.get("status") == "success"
                                and response.get("model_matches_request", True)
                                is not False
                            )
                            raw_items = (
                                response.get("items", []) if response_usable else []
                            )
                            item = next(
                                (
                                    raw
                                    for raw in raw_items
                                    if isinstance(raw, Mapping)
                                    and str(raw.get("cell_id")) == cell_id
                                ),
                                None,
                            ) if isinstance(raw_items, list) else None
                            if isinstance(item, Mapping):
                                item = {**dict(item), "parse_status": "ok_item"}
                            pair_key = (cell_id, query_id)
                            if pair_key not in uplift:
                                raise ValueError(
                                    f"missing Router-v3 uplift for {backend}/{variant}/{pair_key}"
                                )
                            candidates.append(
                                RankedRepairCandidate(
                                    query_id=query_id,
                                    item=item or {"parse_status": "missing_item"},
                                    conservative_uplift=uplift[pair_key],
                                    cost=action.estimated_total_tokens,
                                    group_size=action.group_size,
                                )
                            )
                        ordered_candidates = sorted(
                            candidates,
                            key=lambda candidate: (
                                -candidate.conservative_uplift,
                                candidate.cost,
                                candidate.group_size,
                                candidate.query_id,
                            ),
                        )
                        attempted_query_ids: list[str] = []
                        rejected_reasons: list[str] = []
                        decision = None
                        for candidate in ordered_candidates:
                            attempted_query_ids.append(candidate.query_id)
                            cache_key = (
                                variant,
                                _budget_label(budget_share),
                                cell_id,
                                candidate.query_id,
                            )
                            if cache_key not in verification_cache:
                                verification_cache[cache_key] = verifier.verify(
                                    cell_by_id[cell_id],
                                    baran[cell_id],
                                    candidate.item,
                                    candidate.conservative_uplift,
                                    query_id=candidate.query_id,
                                )
                            candidate_decision = verification_cache[cache_key]
                            if bool(getattr(candidate_decision, "accept_llm")):
                                decision = candidate_decision
                                break
                            rejected_reasons.append(
                                str(getattr(candidate_decision, "reason"))
                            )
                        if decision is None:
                            decision = verifier.arbitrate(
                                cell_by_id[cell_id], baran[cell_id], ()
                            ).decision
                            verification_reason = (
                                "all_candidates_rejected"
                                if ordered_candidates
                                else "no_candidate"
                            )
                        else:
                            verification_reason = str(getattr(decision, "reason"))
                        parse_status = (
                            "ok_llm"
                            if decision.accept_llm
                            else str(
                                baran[cell_id].get("parse_status", "no_prediction")
                            )
                        )
                        prediction = decision.final_prediction
                        correct = bool(
                            parse_status.startswith("ok")
                            and normalize_for_match(prediction)
                            == normalize_for_match(clean[cell_id])
                        )
                        records.append(
                            {
                                "cell_id": cell_id,
                                "suite": suite,
                                "dataset": dataset,
                                "method": self._bgr_method_name(backend),
                                "scenario": scenario,
                                "backend": backend,
                                "budget_share": budget_share,
                                "group_size_variant": variant,
                                "prediction": prediction,
                                "clean_value": clean[cell_id],
                                "parse_status": parse_status,
                                "valid_prediction": parse_status.startswith("ok"),
                                "correct_repair": correct,
                                "final_source": decision.final_source,
                                "accepted_llm": decision.accept_llm,
                                "selected_query_id": decision.query_id,
                                "verification_reason": verification_reason,
                                "verification_score": decision.score,
                                "conservative_uplift": decision.conservative_uplift,
                                "selected_queries_covering_cell": len(candidates),
                                "attempted_query_count": len(attempted_query_ids),
                                "rejected_candidate_count": len(rejected_reasons),
                            }
                        )
            write_jsonl(
                self.paths.final_dir
                / "per_dataset"
                / f"{_dataset_key(suite, dataset)}.jsonl",
                records[dataset_record_start:],
            )
            print(
                f"[final-v3] {suite}/{dataset}: cumulative records={len(records)}",
                flush=True,
            )

        write_jsonl(final_path, records)
        expected = {
            (suite, dataset): {
                str(cell.cell_id)
                for cell in self._dataset(suite, dataset).safe_cells()
            }
            for suite, dataset in target_order()
        }
        audit = verify_records(records, expected_cell_ids=expected)
        method_slices = 2 + len(self._active_gate_backends()) * len(specs)
        expected_records = TEST_TARGET_CELL_COUNT * method_slices
        expected_dataset_slices = len(TEST_TARGETS) * method_slices
        if (
            not bool(audit.get("ok"))
            or int(audit.get("records", 0)) != expected_records
            or int(audit.get("unique_records", 0)) != expected_records
            or int(audit.get("slices", 0)) != expected_dataset_slices
        ):
            raise ValueError("Router-v3 final record matrix audit failed")
        write_json(self.paths.metrics_dir / "record_audit.json", audit)
        summary = {
            "router_revision": self.router_revision,
            "records": len(records),
            "slices": int(audit["slices"]),
            "unique_records": int(audit["unique_records"]),
            "accepted_llm_records": sum(
                bool(record.get("accepted_llm")) for record in records
            ),
            "coverage_ok": True,
        }
        self.state.update_stage("final_records", "complete", **summary)
        return summary

    @staticmethod
    def _batch_interference_rows(labels: pd.DataFrame) -> list[dict[str, object]]:
        singleton = labels.loc[pd.to_numeric(labels["group_size"], errors="raise") == 1]
        singleton_correct = {
            str(row.cell_id): bool(int(row.llm_correct_in_query))
            for row in singleton.itertuples(index=False)
        }
        rows: list[dict[str, object]] = []
        grouped = labels.loc[pd.to_numeric(labels["group_size"], errors="raise") > 1].groupby(
            ["group_size", "group_view"], dropna=False
        )
        for (group_size, group_view), frame in grouped:
            paired = [
                (
                    singleton_correct[str(row.cell_id)],
                    bool(int(row.llm_correct_in_query)),
                )
                for row in frame.itertuples(index=False)
                if str(row.cell_id) in singleton_correct
            ]
            if not paired:
                continue
            singleton_rate = sum(value[0] for value in paired) / len(paired)
            group_rate = sum(value[1] for value in paired) / len(paired)
            rows.append(
                {
                    "group_size": int(group_size),
                    "group_view": str(group_view),
                    "paired_items": len(paired),
                    "singleton_correct_rate": singleton_rate,
                    "group_correct_rate": group_rate,
                    "batch_interference": singleton_rate - group_rate,
                }
            )
        return rows

    def _api_cost_rows(self) -> list[dict[str, object]]:
        calibration = read_jsonl(self.paths.llm_dir / "calibration_execution.jsonl")
        selected = read_jsonl(self.paths.llm_dir / "selected_execution.jsonl")
        calibration_fallbacks = read_jsonl(
            self.paths.llm_dir
            / "offline_group_calibration_baran_fallbacks.jsonl"
        )
        selected_fallbacks = read_jsonl(
            self.paths.llm_dir / "online_selected_union_baran_fallbacks.jsonl"
        )
        preflight = load_json(self.paths.llm_dir / "model_preflight.json")
        checkpoint = read_jsonl(
            self.paths.llm_dir / "group_query_checkpoint.jsonl"
        )
        calibration_cached_identities = {
            (str(row.get("query_id")), str(row.get("prompt_hash")))
            for row in calibration
            if row.get("status") == "success"
        }
        online = [
            row
            for row in selected
            if (
                str(row.get("query_id")),
                str(row.get("prompt_hash")),
            )
            not in calibration_cached_identities
        ]

        def key(row: Mapping[str, object]) -> tuple[str, str]:
            return str(row.get("query_id")), str(row.get("prompt_hash"))

        def phase_of(row: Mapping[str, object]) -> str:
            metadata = row.get("metadata")
            return (
                str(metadata.get("phase", ""))
                if isinstance(metadata, Mapping)
                else ""
            )

        def physical_ledger_rows(
            phase: str,
            rows: Sequence[Mapping[str, object]],
        ) -> list[Mapping[str, object]]:
            requested = {key(row) for row in rows}
            return [
                row
                for row in checkpoint
                if key(row) in requested
                and phase_of(row) == phase
                and not bool(row.get("cache_hit"))
            ]

        def summarize(
            phase: str,
            rows: Sequence[Mapping[str, object]],
            ledger_rows: Sequence[Mapping[str, object]],
            fallback_rows: Sequence[Mapping[str, object]],
        ) -> dict[str, object]:
            unique: dict[tuple[str, str], Mapping[str, object]] = {}
            for row in rows:
                unique[key(row)] = row
            failed_keys = {
                identity
                for identity, row in unique.items()
                if row.get("status") != "success"
            }
            resolved_keys = {
                key(row)
                for row in fallback_rows
                if str(row.get("operational_status")) == "resolved"
                and str(row.get("fallback_source")) == "baran"
            }
            resolved_keys.update(
                identity
                for identity, row in unique.items()
                if bool(row.get("operationally_resolved"))
                and str(row.get("operational_resolution")) == "baran_fallback"
            )
            unresolved_keys = failed_keys - resolved_keys
            prompt_values = [
                _usage_value(row, "prompt_tokens", "input_tokens")
                for row in ledger_rows
            ]
            completion_values = [
                _usage_value(row, "completion_tokens", "output_tokens")
                for row in ledger_rows
            ]
            total_values = [_actual_tokens(row) for row in ledger_rows]
            return {
                "phase": phase,
                "records": len(unique),
                "physical_requests": len(ledger_rows),
                "attempts": sum(
                    int(row.get("attempts", 0) or 0) for row in ledger_rows
                ),
                "prompt_tokens": sum(value or 0 for value in prompt_values),
                "completion_tokens": sum(value or 0 for value in completion_values),
                "total_tokens": sum(value or 0 for value in total_values),
                "cache_hits": sum(
                    bool(row.get("cache_hit")) or bool(row.get("checkpoint_hit"))
                    for row in unique.values()
                ),
                "failed_records": len(failed_keys),
                "provider_failed_records": len(failed_keys),
                "historical_failed_records": sum(
                    row.get("status") != "success" for row in ledger_rows
                ),
                "operational_fallback_records": len(fallback_rows),
                "operational_fallback_cells": sum(
                    len(row.get("cell_ids", []))
                    for row in fallback_rows
                    if isinstance(row.get("cell_ids"), list)
                ),
                "unresolved_operational_failures": len(unresolved_keys),
                "unknown_usage_records": sum(value is None for value in total_values),
                "unknown_usage_attempts": sum(
                    int(row.get("unknown_usage_attempts", 0) or 0)
                    for row in ledger_rows
                ),
                "ledger_source": "group_query_checkpoint.jsonl",
            }

        offline = summarize(
            "offline_group_calibration",
            calibration,
            physical_ledger_rows("offline_group_calibration", calibration),
            calibration_fallbacks,
        )
        online_row = summarize(
            "online_selected_union",
            online,
            physical_ledger_rows("online_selected_union", online),
            selected_fallbacks,
        )
        preflight_counts = summarize(
            "model_preflight",
            (preflight,),
            physical_ledger_rows("model_preflight", (preflight,)),
            (),
        )
        preflight_total = int(preflight_counts["total_tokens"])
        calibration_plan = load_json(self.paths.llm_dir / "calibration_plan.json")
        logical = _read_csv(self.paths.metrics_dir / "logical_budget_ledger.csv")
        logical_estimated = int(
            pd.to_numeric(logical["estimated_tokens"], errors="raise").sum()
        )
        logical_calls = int(pd.to_numeric(logical["logical_api_calls"], errors="raise").sum())
        total = {
            "phase": "total_fresh_experiment",
            "scope": "total",
            **{
                key: int(offline[key]) + int(online_row[key]) + int(preflight_counts[key])
                for key in (
                    "records",
                    "physical_requests",
                    "attempts",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cache_hits",
                    "failed_records",
                    "provider_failed_records",
                    "historical_failed_records",
                    "operational_fallback_records",
                    "operational_fallback_cells",
                    "unresolved_operational_failures",
                    "unknown_usage_records",
                    "unknown_usage_attempts",
                )
            },
        }
        offline["calibration_estimated_tokens"] = int(
            calibration_plan.get("estimated_tokens", 0)
        )
        offline["calibration_provider_tokens"] = int(offline["total_tokens"])
        online_row["online_logical_estimated_tokens"] = logical_estimated
        total.update(
            {
                "calibration_estimated_tokens": int(
                    calibration_plan.get("estimated_tokens", 0)
                ),
                "calibration_provider_tokens": int(offline["total_tokens"]),
                "online_logical_estimated_tokens": logical_estimated,
                "provider_input_tokens": int(total["prompt_tokens"]),
                "provider_output_tokens": int(total["completion_tokens"]),
                "provider_total_tokens": int(total["total_tokens"]),
                "logical_api_calls": logical_calls,
                "physical_api_calls": int(total["physical_requests"]),
                "unknown_token_attempts": int(total["unknown_usage_attempts"]),
                "model_preflight_provider_tokens": int(preflight_total or 0),
                "ledger_source": "group_query_checkpoint.jsonl",
            }
        )
        return [offline, online_row, total]

    def build_metrics_stage(self) -> dict[str, object]:
        """Build the Router-v3 metric, comparison, and statistical artifacts."""

        return self._build_metrics_router_v3()


    def _catboost_comparison_records(self) -> list[dict[str, object]]:
        """Load only frozen 20% LightGBM/XGBoost slices for CatBoost comparison."""

        if not self.is_router_v3_catboost:
            return []
        if self.router_comparison_run is None:
            return []
        source = self.router_comparison_run
        source_manifest_path = source / "run_manifest.json"
        source_records_path = source / "final" / "all_methods.jsonl"
        source_manifest = load_json(source_manifest_path)
        source_experiment = source_manifest.get("experiment_config", {})
        current_manifest = self.state.manifest
        if (
            source_manifest.get("status") != "complete"
            or not isinstance(source_experiment, Mapping)
            or str(source_experiment.get("router_revision", ""))
            != ROUTER_V3_REVISION
            or str(source_manifest.get("model", ""))
            != str(current_manifest.get("model", ""))
            or str(source_manifest.get("prompt_schema_sha256", ""))
            != str(current_manifest.get("prompt_schema_sha256", ""))
            or str(source_manifest.get("data_content_fingerprint", ""))
            != str(current_manifest.get("data_content_fingerprint", ""))
        ):
            raise ValueError("CatBoost comparison run identity differs")
        if not source_records_path.is_file():
            raise FileNotFoundError("CatBoost comparison run has no final records")
        records = [
            row
            for row in read_jsonl(source_records_path)
            if str(row.get("method", ""))
            in {"budgeted_group_lightgbm", "budgeted_group_xgboost"}
            and str(row.get("scenario", "")) == "size_conditioned"
            and str(row.get("group_size_variant", "")) in ROUTER_V3_VARIANTS
            and math.isclose(
                float(row.get("budget_share") or 0.0), 0.2, abs_tol=1e-12
            )
        ]
        expected = (
            TEST_TARGET_CELL_COUNT
            * len(EXPECTED_GATE_BACKENDS)
            * len(ROUTER_V3_VARIANTS)
        )
        if len(records) != expected:
            raise ValueError("CatBoost comparison record matrix is incomplete")
        provenance = {
            "source_run": str(source),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "source_records": str(source_records_path),
            "source_records_sha256": sha256_file(source_records_path),
            "comparison_backends": list(EXPECTED_GATE_BACKENDS),
            "comparison_variants": list(ROUTER_V3_VARIANTS),
            "comparison_budget_share": 0.2,
            "comparison_records": len(records),
            "copied_into_cell_ledger": False,
        }
        write_json(
            self.paths.run_dir / "provenance" / "comparison_reuse.json",
            provenance,
        )
        return records

    def _router_v3_paired_statistics(
        self,
        records: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Compute paired row-cluster intervals for all Router-v3 BGR slices."""

        comparison_records = self._catboost_comparison_records()
        comparison_series = (
            [
                (self._bgr_method_name(backend), backend, variant, budget)
                for backend in EXPECTED_GATE_BACKENDS
                for variant in self._router_training_variants()
                for budget in self._router_budget_shares()
            ]
            if comparison_records
            else []
        )
        series_order = [
            ("baran", "none", "all", None),
            ("llm_only", "none", "1", None),
            *[
                (self._bgr_method_name(backend), backend, variant, budget)
                for backend in self._active_gate_backends()
                for variant in self._router_training_variants()
                for budget in self._router_budget_shares()
            ],
            *comparison_series,
        ]
        series_index = {key: index for index, key in enumerate(series_order)}
        by_dataset: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
        for record in [*records, *comparison_records]:
            by_dataset[(str(record["suite"]), str(record["dataset"]))].append(record)

        output: list[dict[str, object]] = []
        p_value_groups: dict[
            tuple[str, str, str, float], dict[str, float]
        ] = defaultdict(dict)
        replicate_count = int(self.experiment_config.get("bootstrap_replicates", 2_000))
        base_seed = int(self.experiment_config.get("bootstrap_seed", 45))
        for suite, dataset in target_order():
            dataset_records = by_dataset[(suite, dataset)]
            cells = self._dataset(suite, dataset).safe_cells()
            row_by_cell = {str(cell.cell_id): str(cell.row_id) for cell in cells}
            cluster_keys = tuple(sorted(set(row_by_cell.values())))
            cluster_index = {value: index for index, value in enumerate(cluster_keys)}
            cell_counts = np.zeros(len(cluster_keys), dtype=float)
            for cell_id, row_id in row_by_cell.items():
                del cell_id
                cell_counts[cluster_index[row_id]] += 1.0
            valid_counts = np.zeros((len(cluster_keys), len(series_order)), dtype=float)
            correct_counts = np.zeros_like(valid_counts)
            correctness_by_series: dict[
                tuple[str, str, str, float | None], dict[str, bool]
            ] = {
                key: {} for key in series_order
            }
            observed_cells: dict[
                tuple[str, str, str, float | None], set[str]
            ] = {
                key: set() for key in series_order
            }
            for record in dataset_records:
                method = str(record.get("method", ""))
                backend = str(record.get("backend", "none"))
                variant = str(record.get("group_size_variant", "all"))
                budget = (
                    None
                    if method in {"baran", "llm_only"}
                    else round(float(record.get("budget_share") or 0.0), 12)
                )
                key = (method, backend, variant, budget)
                if key not in series_index:
                    raise ValueError(f"unexpected Router-v3 record series: {key}")
                cell_id = str(record["cell_id"])
                if cell_id in observed_cells[key]:
                    raise ValueError(f"duplicate Router-v3 paired record: {key}/{cell_id}")
                observed_cells[key].add(cell_id)
                row_index = cluster_index[row_by_cell[cell_id]]
                column_index = series_index[key]
                valid = str(record.get("parse_status", "")).startswith("ok")
                correct = bool(record.get("correct_repair"))
                valid_counts[row_index, column_index] += int(valid)
                correct_counts[row_index, column_index] += int(correct)
                correctness_by_series[key][cell_id] = correct
            expected_cells = set(row_by_cell)
            if any(values != expected_cells for values in observed_cells.values()):
                raise ValueError(f"paired record coverage differs for {suite}/{dataset}")

            cluster_count = len(cluster_keys)
            seed = int(
                hashlib.sha256(
                    f"{base_seed}|{suite}|{dataset}".encode("utf-8")
                ).hexdigest()[:16],
                16,
            )
            rng = np.random.default_rng(seed)
            boot_f1 = np.zeros((replicate_count, len(series_order)), dtype=float)
            boot_recall = np.zeros_like(boot_f1)
            probabilities = np.full(cluster_count, 1.0 / cluster_count)
            offset = 0
            while offset < replicate_count:
                batch = min(100, replicate_count - offset)
                weights = rng.multinomial(cluster_count, probabilities, size=batch)
                total = weights @ cell_counts
                valid = weights @ valid_counts
                correct = weights @ correct_counts
                precision = np.divide(
                    correct,
                    valid,
                    out=np.zeros_like(correct),
                    where=valid > 0,
                )
                recall = np.divide(
                    correct,
                    total[:, None],
                    out=np.zeros_like(correct),
                    where=total[:, None] > 0,
                )
                f1 = np.divide(
                    2.0 * precision * recall,
                    precision + recall,
                    out=np.zeros_like(correct),
                    where=(precision + recall) > 0,
                )
                boot_f1[offset : offset + batch] = f1
                boot_recall[offset : offset + batch] = recall
                offset += batch

            total_cells = len(expected_cells)
            observed_valid = valid_counts.sum(axis=0)
            observed_correct = correct_counts.sum(axis=0)
            observed_precision = np.divide(
                observed_correct,
                observed_valid,
                out=np.zeros_like(observed_correct),
                where=observed_valid > 0,
            )
            observed_recall = observed_correct / total_cells
            observed_f1 = np.divide(
                2.0 * observed_precision * observed_recall,
                observed_precision + observed_recall,
                out=np.zeros_like(observed_correct),
                where=(observed_precision + observed_recall) > 0,
            )
            for backend in self._active_gate_backends():
                for variant in self._router_training_variants():
                    for budget in self._router_budget_shares():
                        method_key = (
                            self._bgr_method_name(backend),
                            backend,
                            variant,
                            budget,
                        )
                        method_index = series_index[method_key]
                        comparators = [
                            ("baran", ("baran", "none", "all", None)),
                            ("llm_only", ("llm_only", "none", "1", None)),
                        ]
                        if self.is_router_v3_catboost:
                            comparators.extend(
                                (
                                    f"budgeted_group_{comparison_backend}",
                                    (
                                        f"budgeted_group_{comparison_backend}",
                                        comparison_backend,
                                        variant,
                                        budget,
                                    ),
                                )
                                for comparison_backend in EXPECTED_GATE_BACKENDS
                            )
                        for baseline_name, baseline_key in comparators:
                            baseline_index = series_index[baseline_key]
                            delta_f1 = (
                                boot_f1[:, method_index]
                                - boot_f1[:, baseline_index]
                            )
                            delta_recall = (
                                boot_recall[:, method_index]
                                - boot_recall[:, baseline_index]
                            )
                            method_correct = correctness_by_series[method_key]
                            baseline_correct = correctness_by_series[baseline_key]
                            n10 = sum(
                                baseline_correct[cell_id]
                                and not method_correct[cell_id]
                                for cell_id in expected_cells
                            )
                            n01 = sum(
                                method_correct[cell_id]
                                and not baseline_correct[cell_id]
                                for cell_id in expected_cells
                            )
                            bootstrap_p_value = min(
                                1.0,
                                2.0
                                * min(
                                    (
                                        float(
                                            np.count_nonzero(delta_f1 <= 0.0)
                                        )
                                        + 1.0
                                    )
                                    / (replicate_count + 1.0),
                                    (
                                        float(
                                            np.count_nonzero(delta_f1 >= 0.0)
                                        )
                                        + 1.0
                                    )
                                    / (replicate_count + 1.0),
                                ),
                            )
                            mcnemar_p_value = exact_mcnemar(n10, n01)
                            comparison_id = f"{suite}/{dataset}"
                            p_value_groups[
                                (baseline_name, backend, variant, budget)
                            ][comparison_id] = bootstrap_p_value
                            output.append(
                                {
                                    "suite": suite,
                                    "dataset": dataset,
                                    "backend": backend,
                                    "group_size_variant": variant,
                                    "budget_share": budget,
                                    "baseline": baseline_name,
                                    "n_eval_cells": total_cells,
                                    "method_f1": float(observed_f1[method_index]),
                                    "baseline_f1": float(
                                        observed_f1[baseline_index]
                                    ),
                                    "delta_f1": float(
                                        observed_f1[method_index]
                                        - observed_f1[baseline_index]
                                    ),
                                    "delta_f1_ci_low": float(
                                        np.quantile(delta_f1, 0.025)
                                    ),
                                    "delta_f1_ci_high": float(
                                        np.quantile(delta_f1, 0.975)
                                    ),
                                    "method_correct_repairs": int(
                                        observed_correct[method_index]
                                    ),
                                    "baseline_correct_repairs": int(
                                        observed_correct[baseline_index]
                                    ),
                                    "delta_correct_repairs": int(
                                        observed_correct[method_index]
                                        - observed_correct[baseline_index]
                                    ),
                                    "delta_correction_accuracy_ci_low": float(
                                        np.quantile(delta_recall, 0.025)
                                    ),
                                    "delta_correction_accuracy_ci_high": float(
                                        np.quantile(delta_recall, 0.975)
                                    ),
                                    "mcnemar_n10": int(n10),
                                    "mcnemar_n01": int(n01),
                                    "mcnemar_p_value": mcnemar_p_value,
                                    "p_value": bootstrap_p_value,
                                    "bootstrap_p_value": bootstrap_p_value,
                                    "holm_adjusted_p_value": 1.0,
                                    "bootstrap_replicates": replicate_count,
                                    "bootstrap_seed": seed,
                                    "cluster_unit": "dirty_row",
                                }
                            )
        adjusted_lookup: dict[tuple[str, str, str, float, str], float] = {}
        for group, values in p_value_groups.items():
            for dataset_key, adjusted in holm_adjust(values).items():
                adjusted_lookup[(*group, dataset_key)] = adjusted
        for row in output:
            row["holm_adjusted_p_value"] = adjusted_lookup[
                (
                    str(row["baseline"]),
                    str(row["backend"]),
                    str(row["group_size_variant"]),
                    float(row["budget_share"]),
                    f"{row['suite']}/{row['dataset']}",
                )
            ]
        return output

    def _build_metrics_router_v3(self) -> dict[str, object]:
        records = read_jsonl(self.paths.final_dir / "all_methods.jsonl")
        summaries = summarize_records(records, strict=True)
        comparisons_baran = compare_methods(records, baseline="baran", strict=True)
        backends = self._active_gate_backends()
        variants = self._router_training_variants()
        budgets = self._router_budget_shares()
        bgr_methods = [self._bgr_method_name(value) for value in backends]
        comparisons_llm = compare_methods(
            records,
            baseline="llm_only",
            method=bgr_methods,
            strict=True,
        )
        _write_csv(self.paths.metrics_dir / "method_metrics.csv", summaries)
        _write_csv(
            self.paths.metrics_dir / "comparison_vs_baran.csv",
            comparisons_baran,
        )
        _write_csv(
            self.paths.metrics_dir / "comparison_vs_llm_only.csv",
            comparisons_llm,
        )
        primary = [
            row
            for row in comparisons_baran
            if str(row.get("method")) in set(bgr_methods)
            and str(row.get("scenario")) == "size_conditioned"
            and math.isclose(
                float(row.get("budget_share") or 0.0), 0.2, abs_tol=1e-12
            )
        ]
        _write_csv(self.paths.metrics_dir / "primary_vs_baran.csv", primary)
        size_rows = [
            row
            for row in summaries
            if str(row.get("method")) in set(bgr_methods)
            and str(row.get("scenario")) == "size_conditioned"
        ]
        _write_csv(self.paths.metrics_dir / "size_ablation.csv", size_rows)
        budget_rows: list[dict[str, object]] = []
        aubc_rows: list[dict[str, object]] = []
        if self.is_router_v3_budget_sweep:
            budget_rows = size_rows
            aubc_rows = compute_aubc(
                [
                    row
                    for row in summaries
                    if str(row.get("method")) == "baran"
                    or str(row.get("method")) in set(bgr_methods)
                ],
                baseline_method="baran",
                metric="f1",
                max_budget=0.5,
            )
            _write_csv(self.paths.metrics_dir / "budget_curves.csv", budget_rows)
            _write_csv(self.paths.metrics_dir / "aubc.csv", aubc_rows)

        logical_path = self.paths.metrics_dir / "logical_budget_ledger.csv"
        logical = _read_csv(logical_path)
        responses = self._response_index()
        plan = load_json(self.paths.llm_dir / "selected_union_plan.json")
        online_ids = {str(value) for value in plan.get("online_query_ids", [])}
        accepted_counts: Counter[tuple[str, str, str, str, str, float, str]] = Counter()
        for record in records:
            if not bool(record.get("accepted_llm")) or not record.get("selected_query_id"):
                continue
            logical_scenario = str(record.get("scenario"))
            logical_budget = float(record.get("budget_share") or 0.0)
            if str(record.get("method")) == "llm_only":
                logical_scenario = "llm_only_baseline"
                logical_budget = 1.0
            accepted_counts[
                (
                    str(record.get("suite")),
                    str(record.get("dataset")),
                    str(record.get("backend")),
                    logical_scenario,
                    str(record.get("group_size_variant")),
                    logical_budget,
                    str(record.get("selected_query_id")),
                )
            ] += 1
        charged_physical: set[str] = set()
        for index, row in logical.iterrows():
            response = responses.get((str(row["query_id"]), str(row["prompt_hash"])))
            actual = _actual_tokens(response)
            logical.at[index, "actual_tokens_if_available"] = "" if actual is None else actual
            query_id = str(row["query_id"])
            physical = query_id in online_ids and query_id not in charged_physical
            logical.at[index, "physical_api_calls"] = int(physical)
            if physical:
                charged_physical.add(query_id)
            logical.at[index, "accepted_llm_cells"] = accepted_counts[
                (
                    str(row["target_suite"]),
                    str(row["target_dataset"]),
                    str(row["backend"]),
                    str(row["scenario"]),
                    str(row["group_size_variant"]),
                    float(row["budget_share"]),
                    query_id,
                )
            ]
        _write_csv(
            logical_path,
            logical.to_dict("records"),
            columns=LOGICAL_LEDGER_COLUMNS,
        )

        cost_index: dict[tuple[str, str, str, float, str, str], dict[str, object]] = {}
        for key, frame in logical.groupby(
            [
                "backend",
                "scenario",
                "group_size_variant",
                "budget_share",
                "target_suite",
                "target_dataset",
            ],
            dropna=False,
        ):
            actual_values = pd.to_numeric(
                frame["actual_tokens_if_available"], errors="coerce"
            )
            cost_index[
                (
                    str(key[0]),
                    str(key[1]),
                    str(key[2]),
                    float(key[3]),
                    str(key[4]),
                    str(key[5]),
                )
            ] = {
                "logical_calls": int(len(frame)),
                "logical_estimated_tokens": int(
                    pd.to_numeric(frame["estimated_tokens"], errors="raise").sum()
                ),
                "logical_provider_tokens_observed": int(actual_values.dropna().sum()),
                "logical_unknown_usage_queries": int(actual_values.isna().sum()),
                "physical_calls_charged": int(
                    pd.to_numeric(frame["physical_api_calls"], errors="raise").sum()
                ),
            }

        dataset_summaries = [row for row in summaries if row["scope"] == "dataset"]
        upgraded_counts: Counter[
            tuple[str, str, str, str, str, float | None]
        ] = Counter()
        for record in records:
            if not bool(record.get("accepted_llm")):
                continue
            method = str(record.get("method"))
            upgraded_counts[
                (
                    str(record.get("suite")),
                    str(record.get("dataset")),
                    method,
                    str(record.get("backend")),
                    str(record.get("group_size_variant")),
                    (
                        None
                        if method == "llm_only"
                        else round(float(record.get("budget_share") or 0.0), 12)
                    ),
                )
            ] += 1
        baseline_lookup = {
            (str(row["method"]), str(row["suite"]), str(row["dataset"])): row
            for row in dataset_summaries
            if str(row["method"]) in {"baran", "llm_only"}
        }
        detailed_rows: list[dict[str, object]] = []
        for row in dataset_summaries:
            method = str(row["method"])
            suite = str(row["suite"])
            dataset = str(row["dataset"])
            detail = dict(row)
            baran_row = baseline_lookup[("baran", suite, dataset)]
            llm_row = baseline_lookup[("llm_only", suite, dataset)]
            detail["delta_f1_vs_baran"] = float(row["f1"]) - float(baran_row["f1"])
            detail["delta_f1_vs_llm_only"] = float(row["f1"]) - float(llm_row["f1"])
            detail["delta_correct_repairs_vs_baran"] = int(row["correct_repairs"]) - int(
                baran_row["correct_repairs"]
            )
            detail["delta_correct_repairs_vs_llm_only"] = int(
                row["correct_repairs"]
            ) - int(llm_row["correct_repairs"])
            detail["llm_upgraded_cells"] = upgraded_counts[
                (
                    suite,
                    dataset,
                    method,
                    str(row["backend"]),
                    str(row["group_size_variant"]),
                    (
                        None
                        if method in {"baran", "llm_only"}
                        else round(float(row.get("budget_share") or 0.0), 12)
                    ),
                )
            ]
            if method == "baran":
                costs = {
                    "logical_calls": 0,
                    "logical_estimated_tokens": 0,
                    "logical_provider_tokens_observed": 0,
                    "logical_unknown_usage_queries": 0,
                    "physical_calls_charged": 0,
                }
            elif method == "llm_only":
                costs = cost_index[("none", "llm_only_baseline", "1", 1.0, suite, dataset)]
            else:
                costs = cost_index[
                    (
                        str(row["backend"]),
                        "size_conditioned",
                        str(row["group_size_variant"]),
                        float(row["budget_share"]),
                        suite,
                        dataset,
                    )
                ]
            detail.update(costs)
            detailed_rows.append(detail)
        _write_csv(
            self.paths.metrics_dir / "per_dataset_method_comparison.csv",
            detailed_rows,
        )

        matrix_rows: list[dict[str, object]] = []
        for suite, dataset in target_order():
            values = {
                (
                    str(row["method"]),
                    str(row["backend"]),
                    str(row["group_size_variant"]),
                    (
                        None
                        if str(row["method"]) in {"baran", "llm_only"}
                        else round(float(row.get("budget_share") or 0.0), 12)
                    ),
                ): row
                for row in dataset_summaries
                if str(row["suite"]) == suite and str(row["dataset"]) == dataset
            }
            matrix: dict[str, object] = {
                "suite": suite,
                "dataset": dataset,
                "baran_only_f1": float(values[("baran", "none", "all", None)]["f1"]),
                "llm_only_f1": float(values[("llm_only", "none", "1", None)]["f1"]),
                "llm_only_valid_llm_cells": int(
                    values[("llm_only", "none", "1", None)]["valid_predictions"]
                ),
            }
            for backend in backends:
                for variant in variants:
                    for budget in budgets:
                        value = values[
                            (
                                self._bgr_method_name(backend),
                                backend,
                                variant,
                                budget,
                            )
                        ]
                        if self.is_router_v3_budget_sweep:
                            prefix = (
                                f"bgr_{backend}_k{variant}_{_budget_label(budget)}"
                            )
                            matrix[f"{prefix}_f1"] = float(value["f1"])
                            matrix[f"{prefix}_llm_cells"] = upgraded_counts[
                                (
                                    suite,
                                    dataset,
                                    self._bgr_method_name(backend),
                                    backend,
                                    variant,
                                    budget,
                                )
                            ]
                        else:
                            matrix[f"bgr_{backend}_k{variant}_f1"] = float(
                                value["f1"]
                            )
                            if self.is_router_v3_catboost:
                                matrix[f"bgr_{backend}_k{variant}_llm_cells"] = (
                                    upgraded_counts[
                                        (
                                            suite,
                                            dataset,
                                            self._bgr_method_name(backend),
                                            backend,
                                            variant,
                                            budget,
                                        )
                                    ]
                                )
            matrix_rows.append(matrix)
        _write_csv(
            self.paths.metrics_dir / "per_dataset_f1_matrix.csv",
            matrix_rows,
        )

        router_comparison_rows: list[dict[str, object]] = []
        if self.is_router_v3_catboost and self.router_comparison_run is not None:
            comparison_records = self._catboost_comparison_records()
            comparison_summaries = [
                row
                for row in summarize_records(comparison_records, strict=True)
                if str(row.get("scope", "")) == "dataset"
            ]
            current_index = {
                (
                    str(row["suite"]),
                    str(row["dataset"]),
                    str(row["group_size_variant"]),
                ): row
                for row in dataset_summaries
                if str(row.get("method", "")) == "budgeted_group_catboost"
            }
            comparison_index = {
                (
                    str(row["suite"]),
                    str(row["dataset"]),
                    str(row["backend"]),
                    str(row["group_size_variant"]),
                ): row
                for row in comparison_summaries
            }
            comparison_upgraded = Counter(
                (
                    str(row["suite"]),
                    str(row["dataset"]),
                    str(row["backend"]),
                    str(row["group_size_variant"]),
                )
                for row in comparison_records
                if bool(row.get("accepted_llm"))
            )
            for suite, dataset in target_order():
                for variant in variants:
                    current = current_index[(suite, dataset, variant)]
                    for comparison_backend in EXPECTED_GATE_BACKENDS:
                        comparison = comparison_index[
                            (suite, dataset, comparison_backend, variant)
                        ]
                        router_comparison_rows.append(
                            {
                                "suite": suite,
                                "dataset": dataset,
                                "group_size_variant": variant,
                                "comparison_backend": comparison_backend,
                                "catboost_f1": float(current["f1"]),
                                "comparison_f1": float(comparison["f1"]),
                                "delta_f1": float(current["f1"])
                                - float(comparison["f1"]),
                                "catboost_correct_repairs": int(
                                    current["correct_repairs"]
                                ),
                                "comparison_correct_repairs": int(
                                    comparison["correct_repairs"]
                                ),
                                "catboost_llm_upgraded_cells": upgraded_counts[
                                    (
                                        suite,
                                        dataset,
                                        "budgeted_group_catboost",
                                        "catboost",
                                        variant,
                                        0.2,
                                    )
                                ],
                                "comparison_llm_upgraded_cells": comparison_upgraded[
                                    (
                                        suite,
                                        dataset,
                                        comparison_backend,
                                        variant,
                                    )
                                ],
                            }
                        )
            _write_csv(
                self.paths.metrics_dir / "per_dataset_router_comparison.csv",
                router_comparison_rows,
            )

        paired = self._router_v3_paired_statistics(records)
        _write_csv(self.paths.metrics_dir / "paired_statistics.csv", paired)

        selection = _read_csv(self.paths.metrics_dir / "selection_audit.csv")
        selected_union = {
            str(value) for value in plan.get("query_ids", [])
        }
        selected_sizes: dict[str, int] = {}
        for target_suite, target_dataset in target_order():
            for action in self._load_actions(target_suite, target_dataset):
                if action.query_id in selected_union:
                    selected_sizes[action.query_id] = action.group_size
        group_rows: list[dict[str, object]] = []
        for row in selection.to_dict("records"):
            selection_doc = load_json(
                self._selection_path(
                    str(row["backend"]),
                    str(row["scenario"]),
                    str(row["group_size_variant"]),
                    float(row["budget_share"]),
                    str(row["suite"]),
                    str(row["dataset"]),
                )
            )
            sizes = [
                selected_sizes[str(query_id)]
                for query_id in selection_doc.get("selected_query_ids", [])
            ]
            group_rows.append(
                {
                    **row,
                    "mean_group_size": statistics.fmean(sizes) if sizes else 0.0,
                    "median_group_size": statistics.median(sizes) if sizes else 0.0,
                    "maximum_group_size": max(sizes, default=0),
                    "singleton_selected": sum(size == 1 for size in sizes),
                    "non_singleton_selected": sum(size > 1 for size in sizes),
                }
            )
        _write_csv(self.paths.metrics_dir / "group_metrics.csv", group_rows)
        labels = _read_csv(self.paths.llm_dir / "calibration_pair_labels.csv")
        _write_csv(
            self.paths.metrics_dir / "batch_interference.csv",
            self._batch_interference_rows(labels),
        )
        _write_csv(self.paths.metrics_dir / "api_cost_audit.csv", self._api_cost_rows())

        summary = {
            "router_revision": self.router_revision,
            "method_metric_rows": len(summaries),
            "primary_comparison_rows": len(primary),
            "per_dataset_rows": len(detailed_rows),
            "f1_matrix_rows": len(matrix_rows),
            "paired_statistics_rows": len(paired),
            "router_comparison_rows": len(router_comparison_rows),
            "budget_curve_rows": len(budget_rows),
            "aubc_rows": len(aubc_rows),
        }
        self.state.update_stage("metrics", "complete", **summary)
        return summary

    def build_audit_stage(self) -> dict[str, object]:
        split = _read_csv(self.paths.gates_dir / "split_audit.csv")
        overlap_columns = (
            "train_test_cell_overlap",
            "train_test_base_family_overlap",
            "train_test_row_identity_overlap",
            "train_test_query_overlap",
            "train_test_group_signature_overlap",
        )
        overlap_failures = {
            column: int(pd.to_numeric(split[column], errors="raise").sum())
            for column in overlap_columns
        }
        label_flags = {
            "target_group_label_used": bool(
                split["target_group_label_used"].astype(str).str.lower().isin({"1", "true", "yes"}).any()
            ),
            "target_response_used_before_selection": bool(
                split["target_response_used_before_selection"].astype(str).str.lower().isin({"1", "true", "yes"}).any()
            ),
        }
        source_import_violations: list[str] = []
        forbidden_module = "at" + "oms"
        forbidden_symbol = "Probabilistic" + "Coverage"
        import ast

        for path in sorted(
            (self.paths.project_root / "src" / "budgeted_group_repair_no_baran").glob("*.py")
        ):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [str(node.module or ""), *(alias.name for alias in node.names)]
                else:
                    continue
                if any(forbidden_module in name.split(".") or name == forbidden_symbol for name in names):
                    source_import_violations.append(f"{path.name}:{getattr(node, 'lineno', 0)}")

        prompt_violations: list[str] = []
        for suite, dataset in generation_order():
            for action in self._load_actions(suite, dataset):
                try:
                    assert_messages_safe(action.as_dict()["messages"])  # type: ignore[arg-type]
                except ValueError:
                    prompt_violations.append(action.query_id)
                    break
        selection = _read_csv(self.paths.metrics_dir / "selection_audit.csv")
        budget_failures = int(
            (
                pd.to_numeric(selection["selected_estimated_tokens"], errors="raise")
                > pd.to_numeric(selection["budget_estimated_tokens"], errors="raise")
            ).sum()
        )
        record_audit = load_json(self.paths.metrics_dir / "record_audit.json")
        leakage = {
            **overlap_failures,
            **label_flags,
            "prompt_forbidden_field_queries": prompt_violations,
            "forbidden_imports": source_import_violations,
            "atoms_used": False,
        }
        ok = bool(
            not any(overlap_failures.values())
            and not any(label_flags.values())
            and not prompt_violations
            and not source_import_violations
            and budget_failures == 0
            and record_audit.get("ok") is True
        )
        leakage["ok"] = ok
        write_json(self.paths.metrics_dir / "leakage_audit.json", leakage)
        audit = {
            "ok": ok,
            "datasets": len(TEST_TARGETS),
            "oracle_cells": TEST_TARGET_CELL_COUNT,
            "split_rows": len(split),
            "selection_rows": len(selection),
            "budget_failures": budget_failures,
            "record_audit_ok": bool(record_audit.get("ok")),
            "leakage": leakage,
        }
        write_json(self.paths.metrics_dir / "formal_run_audit.json", audit)
        if not ok:
            raise ValueError("formal run audit failed")
        self.state.update_stage("audit", "complete", **audit)
        return audit

    def run_all(self) -> dict[str, object]:
        """Execute and independently validate the entire Router-v3 matrix."""

        plan = self.plan_run()
        preflight = self.check_model()
        calibration = self.run_calibration_stage()
        selection = self.train_and_select_stage()
        selected = self.run_selected_llm_stage()
        final = self.build_final_records_stage()
        metrics = self.build_metrics_stage()
        audit = self.build_audit_stage()
        validate_run(self.paths.run_dir, require_complete=False)
        completed_matrix = {
            "datasets": len(TEST_TARGETS),
            "baselines": ["baran_only", "llm_only"],
            "backends": len(self._active_gate_backends()),
            "budget_shares": list(self._router_budget_shares()),
            "group_size_variants": list(self._router_training_variants()),
            "method_slices": 2
            + len(self._active_gate_backends()) * len(self._scenario_specs()),
            "cell_records": TEST_TARGET_CELL_COUNT
            * (
                2
                + len(self._active_gate_backends())
                * len(self._scenario_specs())
            ),
            "selection_slices": len(TEST_TARGETS)
            * len(self._active_gate_backends())
            * len(self._scenario_specs()),
        }
        self.state.complete(
            required_stages=REQUIRED_STAGES,
            completed_matrix=completed_matrix,
        )
        validation = validate_run(self.paths.run_dir, require_complete=True)
        return {
            "run_dir": str(self.paths.run_dir),
            "plan": plan,
            "model_preflight": preflight,
            "calibration": calibration,
            "selection": {
                key: value
                for key, value in selection.items()
                if key not in {"query_ids", "online_query_ids"}
            },
            "selected_llm": selected,
            "final": final,
            "metrics": metrics,
            "audit": audit,
            "validation": validation,
        }



_RECOVERY_LINKED_ROOTS = frozenset({"baran", "cell_features", "groups"})


@dataclass(frozen=True, slots=True)
class _RecoveryProof:
    source_run: Path
    linked_roots: frozenset[str]
    source_snapshot: Mapping[str, tuple[int, str]]


def _strict_snapshot_path(value: object) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("derived run source snapshot escapes its parent")
    return relative


def _lexical_symlink_target(path: Path) -> Path:
    target = Path(os.readlink(path))
    if not target.is_absolute():
        target = path.parent / target
    return Path(os.path.abspath(target))


def _validate_recovery_provenance(
    root: Path,
    manifest: Mapping[str, object],
    *,
    _run_store: Path | None = None,
    _stack: tuple[Path, ...] = (),
) -> _RecoveryProof | None:
    """Validate a derived snapshot against its declared transitive lineage."""

    root = root.resolve()
    run_store = (_run_store or root.parent).resolve()
    if root.parent != run_store:
        raise ValueError("recovery run is not a direct child of the run store")
    if root in _stack:
        raise ValueError("derived run recovery lineage contains a cycle")
    if str(manifest.get("run_kind", "fresh")) != "derived_recovery":
        return None

    recovery_relative = _strict_snapshot_path(manifest.get("recovery_manifest", ""))
    recovery_path = (root / recovery_relative).resolve()
    try:
        recovery_path.relative_to(root)
    except ValueError as error:
        raise ValueError("derived run recovery manifest is not run-local") from error
    if not recovery_path.is_file() or sha256_file(recovery_path) != str(
        manifest.get("recovery_manifest_sha256", "")
    ):
        raise ValueError("derived run recovery manifest is missing or changed")
    recovery = load_json(recovery_path)
    if str(recovery.get("policy")) != "fallback_baran_after_client_retries":
        raise ValueError("derived run recovery policy is not the declared Baran fallback")
    if (
        str(recovery.get("derived_run_id", "")) != root.name
        or Path(str(recovery.get("derived_run", ""))).resolve() != root
    ):
        raise ValueError("derived run recovery manifest has inconsistent child identity")

    source_run_value = str(recovery.get("source_run", ""))
    if not source_run_value or not Path(source_run_value).is_absolute():
        raise ValueError("derived run recovery manifest has no absolute source_run")
    source_run = Path(source_run_value).resolve()
    if source_run == root or source_run.parent != run_store:
        raise ValueError("derived run source is not a distinct sibling run")
    if str(recovery.get("source_run_id", "")) != source_run.name:
        raise ValueError("derived run recovery manifest has inconsistent source identity")
    source_manifest = source_run / "run_manifest.json"
    if (
        not source_manifest.is_file()
        or sha256_file(source_manifest)
        != str(recovery.get("source_manifest_sha256", ""))
    ):
        raise ValueError("derived run source manifest is missing or changed")
    derived_from = manifest.get("derived_from")
    if (
        not isinstance(derived_from, Mapping)
        or str(derived_from.get("run_id", "")) != source_run.name
        or Path(str(derived_from.get("run_dir", ""))).resolve() != source_run
        or str(derived_from.get("manifest_sha256", ""))
        != str(recovery.get("source_manifest_sha256", ""))
    ):
        raise ValueError("derived run manifest disagrees with its recovery source")
    source_manifest_payload = load_json(source_manifest)
    parent_proof = _validate_recovery_provenance(
        source_run,
        source_manifest_payload,
        _run_store=run_store,
        _stack=(*_stack, root),
    )

    linked_raw = recovery.get("linked_immutable_directories")
    if not isinstance(linked_raw, list):
        raise ValueError("derived run recovery manifest has no linked directory declaration")
    linked_roots = frozenset(str(value) for value in linked_raw)
    if not linked_roots or not linked_roots.issubset(_RECOVERY_LINKED_ROOTS):
        raise ValueError("derived run declares an unsupported linked directory")
    for name in linked_roots:
        link = root / name
        if (
            not link.is_symlink()
            or _lexical_symlink_target(link) != source_run / name
        ):
            raise ValueError("derived run immutable link is not bound to its direct parent")

    source_rows = recovery.get("source_files")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("derived run recovery manifest has no source file snapshot")
    snapshot: dict[str, tuple[int, str]] = {}
    for raw in source_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("derived run source snapshot contains a non-object row")
        relative = _strict_snapshot_path(raw.get("path", ""))
        relative_key = relative.as_posix()
        if relative_key in snapshot:
            raise ValueError("derived run source snapshot contains duplicate paths")
        declared = (int(raw.get("size", -1)), str(raw.get("sha256", "")))
        logical_source = source_run / relative
        resolved_source = logical_source.resolve()
        try:
            resolved_source.relative_to(source_run)
        except ValueError:
            if parent_proof is None:
                raise ValueError(
                    "derived run source snapshot is not backed by a verified ancestor"
                )
            top = relative.parts[0]
            expected = parent_proof.source_snapshot.get(relative_key)
            expected_source = (parent_proof.source_run / relative).resolve()
            if (
                top not in parent_proof.linked_roots
                or not (source_run / top).is_symlink()
                or _lexical_symlink_target(source_run / top)
                != parent_proof.source_run / top
                or expected != declared
                or resolved_source != expected_source
            ):
                raise ValueError(
                    "derived run source snapshot is not backed by a verified ancestor"
                )
        if (
            not logical_source.is_file()
            or logical_source.stat().st_size != declared[0]
            or sha256_file(logical_source) != declared[1]
        ):
            raise ValueError(f"derived run source artifact changed: {logical_source}")
        snapshot[relative_key] = declared
    return _RecoveryProof(
        source_run=source_run,
        linked_roots=linked_roots,
        source_snapshot=snapshot,
    )


def _validate_router_v3_run(
    root: Path,
    manifest: Mapping[str, object],
    *,
    require_complete: bool,
) -> dict[str, object]:
    """Independently validate the exact Router-v3 method and selection matrix."""

    for field, bound_name in (
        ("experiment_config_sha256", "bound_experiment_config.json"),
        ("llm_config_sha256", "bound_llm_config.json"),
    ):
        bound = root / bound_name
        if not bound.is_file() or str(manifest.get(field, "")) != sha256_file(bound):
            raise ValueError(f"run fingerprint drift: {field}")
    config = load_json(root / "bound_experiment_config.json")
    revision = str(config.get("router_revision", ""))
    if revision not in {
        ROUTER_V3_REVISION,
        ROUTER_V3_BUDGET_SWEEP_REVISION,
        ROUTER_V3_CATBOOST_REVISION,
    }:
        raise ValueError("bound experiment is not Router-v3")
    current_implementation = _hash_tree(
        PROJECT_ROOT / "src" / "budgeted_group_repair_no_baran", (".py",)
    )
    bound_implementation = str(manifest.get("implementation_sha256", ""))
    if not _router_v3_implementation_binding_matches(
        revision, bound_implementation, current_implementation
    ):
        raise ValueError("run fingerprint drift: implementation_sha256")
    sweep = revision == ROUTER_V3_BUDGET_SWEEP_REVISION
    catboost_run = revision == ROUTER_V3_CATBOOST_REVISION
    catboost_comparison = bool(
        catboost_run and str(manifest.get("router_comparison_run") or "").strip()
    )
    expected_backends = (
        CATBOOST_GATE_BACKENDS
        if catboost_run
        else (("lightgbm",) if sweep else EXPECTED_GATE_BACKENDS)
    )
    expected_budgets = ROUTER_V3_SWEEP_BUDGETS if sweep else (0.2,)
    expected_variants = (
        {"2": (1, 2), "4": (1, 4)}
        if sweep
        else {
            "1": (1,),
            "2": (1, 2),
            "4": (1, 4),
            "8": (1, 8),
            "all": (1, 2, 4, 8),
        }
    )
    observed_variants = config.get("router_training_variants")
    if not isinstance(observed_variants, Mapping) or {
        str(key): tuple(int(value) for value in values)
        for key, values in observed_variants.items()
        if isinstance(values, list)
    } != expected_variants:
        raise ValueError("Router-v3 variant declaration drift")
    if tuple(float(value) for value in config.get("budget_shares", [])) != expected_budgets:
        raise ValueError("Router-v3 budget matrix drift")
    if tuple(str(value) for value in config.get("gate_backends", [])) != expected_backends:
        raise ValueError("Router-v3 backend matrix drift")
    if int(config.get("baran_labeling_budget", -1)) != 20:
        raise ValueError("Router-v3 Baran labeling budget drift")
    if str(manifest.get("model", "")) != "deepseek-v4-flash":
        raise ValueError("run fingerprint model mismatch")
    if str(manifest.get("prompt_schema_version", "")) != PROMPT_SCHEMA_VERSION:
        raise ValueError("run fingerprint prompt schema mismatch")
    expected_prompt_sha = canonical_json_sha256(
        {"schema": PROMPT_SCHEMA_VERSION, "system_prompt": SYSTEM_PROMPT}
    )
    if str(manifest.get("prompt_schema_sha256", "")) != expected_prompt_sha:
        raise ValueError("run fingerprint prompt content mismatch")

    for path_field, hash_field in (
        ("baran_source_run", "baran_source_manifest_sha256"),
        ("response_reuse_run", "response_reuse_manifest_sha256"),
    ):
        raw_source = str(manifest.get(path_field) or "").strip()
        if not raw_source:
            continue
        source = Path(raw_source).resolve()
        source_manifest = source / "run_manifest.json"
        if (
            not source_manifest.is_file()
            or sha256_file(source_manifest) != str(manifest.get(hash_field, ""))
        ):
            raise ValueError(f"run fingerprint source drift: {path_field}")
    if sweep:
        source = Path(str(manifest.get("router_artifact_reuse_run", ""))).resolve()
        source_manifest = source / "run_manifest.json"
        if (
            not source_manifest.is_file()
            or sha256_file(source_manifest)
            != str(manifest.get("router_artifact_reuse_manifest_sha256", ""))
        ):
            raise ValueError("run fingerprint source drift: router_artifact_reuse_run")
    if catboost_comparison:
        source = Path(str(manifest.get("router_comparison_run", ""))).resolve()
        source_manifest = source / "run_manifest.json"
        if (
            not source_manifest.is_file()
            or sha256_file(source_manifest)
            != str(manifest.get("router_comparison_manifest_sha256", ""))
        ):
            raise ValueError("run fingerprint source drift: router_comparison_run")
    completion_recovery = manifest.get("completion_recovery")
    if completion_recovery is not None:
        if not isinstance(completion_recovery, Mapping):
            raise ValueError("Router-v3 completion_recovery must be an object")
        addendum_relative = _strict_snapshot_path(
            completion_recovery.get("addendum", "")
        )
        addendum_path = (root / addendum_relative).resolve()
        try:
            addendum_path.relative_to(root)
        except ValueError as error:
            raise ValueError("Router-v3 completion addendum is not run-local") from error
        if (
            not addendum_path.is_file()
            or sha256_file(addendum_path)
            != str(completion_recovery.get("sha256", ""))
        ):
            raise ValueError("Router-v3 completion addendum changed")
    response_reuse = load_json(root / "provenance" / "response_reuse.json")
    if set(response_reuse.get("matching_fields", [])) != {
            "query_id",
            "prompt_hash",
            "provider_request_hash",
            "model",
            "prompt_schema_version",
        }:
        raise ValueError("Router-v3 strict response-reuse provenance failed")
    response_source = str(response_reuse.get("source_run") or "").strip()
    if response_source:
        reuse_checkpoint = Path(
            str(response_reuse.get("source_checkpoint", ""))
        ).resolve()
        if (
            not reuse_checkpoint.is_file()
            or sha256_file(reuse_checkpoint)
            != str(response_reuse.get("source_checkpoint_sha256", ""))
        ):
            raise ValueError("Router-v3 strict response-reuse provenance failed")
    elif int(response_reuse.get("imported_rows", -1)) != 0:
        raise ValueError("fresh Router-v3 run declares imported responses")
    if catboost_run and response_reuse.get("terminal_failures_frozen") is not True:
        raise ValueError("Router-v3 CatBoost terminal failure reuse is not frozen")
    calibration_plan = read_jsonl(root / "llm" / "calibration_queries.jsonl")
    calibration_execution = read_jsonl(
        root / "llm" / "calibration_execution.jsonl"
    )
    calibration_labels = _read_csv(
        root / "llm" / "calibration_pair_labels.csv"
    )
    planned_calibration = {
        (str(row.get("query_id", "")), str(row.get("prompt_hash", "")))
        for row in calibration_plan
    }
    executed_calibration = {
        (str(row.get("query_id", "")), str(row.get("prompt_hash", "")))
        for row in calibration_execution
    }
    if (
        len(calibration_plan) != 8_197
        or len(planned_calibration) != len(calibration_plan)
        or len(calibration_execution) != len(calibration_plan)
        or executed_calibration != planned_calibration
        or len(calibration_labels) != 16_451
        or calibration_labels.duplicated(["cell_id", "query_id"]).any()
    ):
        raise ValueError("Router-v3 calibration artifacts are incomplete")

    data_root_value = manifest.get("data_root")
    if not isinstance(data_root_value, str) or not data_root_value:
        raise ValueError("run manifest has no bound data_root")
    data_root = Path(data_root_value).resolve()
    validate_manifest(data_root, require_portable=True)
    expected_cells = {
        (suite, dataset): {
            str(cell.cell_id)
            for cell in load_dataset(suite, dataset, data_root).safe_cells()
        }
        for suite, dataset in target_order()
    }
    if sum(len(values) for values in expected_cells.values()) != TEST_TARGET_CELL_COUNT:
        raise ValueError("Router-v3 target cell universe is not exactly 22,198")

    def budget_value(value: object) -> float | None:
        return None if value in {None, ""} else round(float(value), 12)

    def slice_key(row: Mapping[str, object]) -> tuple[object, ...]:
        return (
            str(row.get("method", "")),
            str(row.get("scenario", "")),
            str(row.get("backend", "")),
            budget_value(row.get("budget_share")),
            str(row.get("group_size_variant", "")),
            str(row.get("suite", "")),
            str(row.get("dataset", "")),
        )

    expected_slices: set[tuple[object, ...]] = set()
    for suite, dataset in target_order():
        expected_slices.add(("baran", "baseline", "none", None, "all", suite, dataset))
        expected_slices.add(("llm_only", "baseline", "none", None, "1", suite, dataset))
        for backend in expected_backends:
            for variant in expected_variants:
                for budget in expected_budgets:
                    expected_slices.add(
                        (
                            f"budgeted_group_{backend}",
                            "size_conditioned",
                            backend,
                            budget,
                            variant,
                            suite,
                            dataset,
                        )
                    )
    final_records = read_jsonl(root / "final" / "all_methods.jsonl")
    if {slice_key(row) for row in final_records} != expected_slices:
        raise ValueError("Router-v3 final method matrix differs")
    if any(
        str(row.get("final_source", "")) == "baran"
        or bool(row.get("baran_fallback_used"))
        for row in final_records
        if str(row.get("method", "")) == "llm_only"
    ):
        raise ValueError("LLM-only read or used a Baran fallback")
    independent_record_audit = verify_records(
        final_records,
        expected_cell_ids=expected_cells,
    )
    method_slice_count = 2 + (
        len(expected_backends) * len(expected_variants) * len(expected_budgets)
    )
    expected_record_count = TEST_TARGET_CELL_COUNT * method_slice_count
    expected_dataset_slices = len(TEST_TARGETS) * method_slice_count
    if (
        independent_record_audit.get("ok") is not True
        or int(independent_record_audit.get("records", 0)) != expected_record_count
        or int(independent_record_audit.get("unique_records", 0)) != expected_record_count
        or int(independent_record_audit.get("slices", 0)) != expected_dataset_slices
    ):
        raise ValueError("Router-v3 independent cell-ledger audit failed")

    method_metrics = _read_csv(root / "metrics" / "method_metrics.csv")
    independent_metrics = summarize_records(final_records, strict=True)
    key_fields = (
        "method",
        "scenario",
        "backend",
        "budget_share",
        "group_size_variant",
        "scope",
        "suite",
        "dataset",
    )

    def metric_key(row: Mapping[str, object]) -> tuple[object, ...]:
        return tuple(
            budget_value(row.get(field))
            if field == "budget_share"
            else str(row.get(field, ""))
            for field in key_fields
        )

    actual_metric_index = {
        metric_key(row): row for row in method_metrics.to_dict("records")
    }
    expected_metric_index = {metric_key(row): row for row in independent_metrics}
    if (
        len(actual_metric_index) != len(method_metrics)
        or len(method_metrics) != (len(TEST_TARGETS) + 2) * method_slice_count
        or set(actual_metric_index) != set(expected_metric_index)
    ):
        raise ValueError("Router-v3 method_metrics matrix differs")
    for key, expected in expected_metric_index.items():
        actual = actual_metric_index[key]
        for field in (
            "true_error_cells",
            "predicted_repairs",
            "correct_repairs",
            "precision",
            "recall",
            "correction_accuracy",
            "f1",
        ):
            if not math.isclose(
                float(actual[field]),
                float(expected[field]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"Router-v3 metric mismatch: {field}/{key}")

    detailed = _read_csv(root / "metrics" / "per_dataset_method_comparison.csv")
    f1_matrix = _read_csv(root / "metrics" / "per_dataset_f1_matrix.csv")
    paired = _read_csv(root / "metrics" / "paired_statistics.csv")
    expected_detailed = expected_dataset_slices
    comparator_count = 4 if catboost_comparison else 2
    expected_paired = (
        len(TEST_TARGETS)
        * len(expected_backends)
        * len(expected_variants)
        * len(expected_budgets)
        * comparator_count
    )
    if (
        len(detailed) != expected_detailed
        or len(f1_matrix) != 9
        or len(paired) != expected_paired
    ):
        raise ValueError("Router-v3 per-dataset report inputs are incomplete")
    required_matrix_columns = {
        "suite",
        "dataset",
        "baran_only_f1",
        "llm_only_f1",
    }
    if sweep or catboost_run:
        required_matrix_columns.add("llm_only_valid_llm_cells")
        required_matrix_columns.update(
            (
                f"bgr_{backend}_k{variant}_{_budget_label(budget)}_{suffix}"
                if sweep
                else f"bgr_{backend}_k{variant}_{suffix}"
            )
            for backend in expected_backends
            for variant in expected_variants
            for budget in expected_budgets
            for suffix in ("f1", "llm_cells")
        )
    else:
        required_matrix_columns.update(
            f"bgr_{backend}_k{variant}_f1"
            for backend in expected_backends
            for variant in expected_variants
        )
    if not required_matrix_columns.issubset(f1_matrix.columns):
        raise ValueError("Router-v3 F1 matrix columns are incomplete")
    if set(pd.to_numeric(paired["bootstrap_replicates"], errors="raise")) != {2_000}:
        raise ValueError("Router-v3 paired bootstrap replicate count drift")
    if set(paired["cluster_unit"].astype(str)) != {"dirty_row"}:
        raise ValueError("Router-v3 bootstrap cluster unit drift")
    if bool(
        (
            (pd.to_numeric(paired["holm_adjusted_p_value"], errors="raise") < 0)
            | (pd.to_numeric(paired["holm_adjusted_p_value"], errors="raise") > 1)
        ).any()
    ):
        raise ValueError("Router-v3 adjusted p-values are invalid")
    if sweep and {
        round(float(value), 12) for value in paired["budget_share"]
    } != set(expected_budgets):
        raise ValueError("Router-v3 paired statistics budget identity drift")

    selection = _read_csv(root / "metrics" / "selection_audit.csv")
    expected_selection_keys = {
        (suite, dataset, backend, variant, budget)
        for suite, dataset in target_order()
        for backend in expected_backends
        for variant in expected_variants
        for budget in expected_budgets
    }
    selection_keys = {
        (
            str(row.suite),
            str(row.dataset),
            str(row.backend),
            str(row.group_size_variant),
            round(float(row.budget_share), 12),
        )
        for row in selection.itertuples(index=False)
    }
    if len(selection) != len(expected_selection_keys) or selection_keys != expected_selection_keys:
        raise ValueError("Router-v3 selection matrix is incomplete")
    actions_by_dataset: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    singleton_references: dict[tuple[str, str], int] = {}
    for suite, dataset in target_order():
        rows = read_jsonl(
            root / "groups" / "candidates" / f"{_dataset_key(suite, dataset)}.jsonl"
        )
        actions = {str(row.get("query_id", "")): row for row in rows}
        if len(actions) != len(rows) or not actions:
            raise ValueError("Router-v3 candidate ledger is empty or duplicated")
        actions_by_dataset[(suite, dataset)] = actions
        singleton_references[(suite, dataset)] = sum(
            int(row.get("estimated_total_tokens", 0) or 0)
            for row in rows
            if int(row.get("group_size", 0) or 0) == 1
            and str(row.get("group_view", "")) == "singleton"
        )
    for row in selection.itertuples(index=False):
        suite, dataset = str(row.suite), str(row.dataset)
        backend, variant = str(row.backend), str(row.group_size_variant)
        allowed = set(expected_variants[variant])
        reference = singleton_references[(suite, dataset)]
        share = round(float(row.budget_share), 12)
        budget = int(round(reference * share))
        if (
            str(row.scenario) != "size_conditioned"
            or share not in set(expected_budgets)
            or int(row.budget_reference_tokens) != reference
            or int(row.budget_estimated_tokens) != budget
            or int(row.selected_estimated_tokens) > budget
        ):
            raise ValueError("Router-v3 selection budget declaration failed")
        document = load_json(
            root
            / "selections"
            / backend
            / "size_conditioned"
            / f"variant_{variant}"
            / _budget_label(share)
            / f"{_dataset_key(suite, dataset)}.json"
        )
        selected_ids = [str(value) for value in document.get("selected_query_ids", [])]
        actions = actions_by_dataset[(suite, dataset)]
        if len(selected_ids) != len(set(selected_ids)) or any(
            query_id not in actions for query_id in selected_ids
        ):
            raise ValueError("Router-v3 selection contains invalid query identities")
        if any(
            int(actions[query_id].get("group_size", 0) or 0) not in allowed
            for query_id in selected_ids
        ):
            raise ValueError("Router-v3 selection contains a disallowed group size")
        recomputed_cost = sum(
            int(actions[query_id].get("estimated_total_tokens", 0) or 0)
            for query_id in selected_ids
        )
        if (
            recomputed_cost != int(row.selected_estimated_tokens)
            or recomputed_cost != int(float(document.get("total_cost", -1)))
            or recomputed_cost > budget
            or tuple(int(value) for value in document.get("training_group_sizes", []))
            != expected_variants[variant]
        ):
            raise ValueError("Router-v3 selection cost or training sizes drift")
        if sweep and math.isclose(share, 0.2, abs_tol=1e-12):
            parent = Path(str(manifest["router_artifact_reuse_run"]))
            parent_selection = load_json(
                parent
                / "selections"
                / backend
                / "size_conditioned"
                / f"variant_{variant}"
                / _budget_label(0.2)
                / f"{_dataset_key(suite, dataset)}.json"
            )
            if selected_ids != [
                str(value)
                for value in parent_selection.get("selected_query_ids", [])
            ]:
                raise ValueError("Router-v3 20% selection differs from parent")

    split = _read_csv(root / "gates" / "split_audit.csv")
    split_keys = {
        (
            str(row.target_suite),
            str(row.target_dataset),
            str(row.backend),
            str(row.group_size_variant),
        )
        for row in split.itertuples(index=False)
    }
    expected_split_keys = {
        (suite, dataset, backend, variant)
        for suite, dataset in target_order()
        for backend in expected_backends
        for variant in expected_variants
    }
    if len(split) != len(expected_split_keys) or split_keys != expected_split_keys:
        raise ValueError("Router-v3 split audit matrix is incomplete")
    for column in (
        "train_test_cell_overlap",
        "train_test_base_family_overlap",
        "train_test_row_identity_overlap",
        "train_test_query_overlap",
        "train_test_group_signature_overlap",
        "validation_cells",
    ):
        if bool((pd.to_numeric(split[column], errors="raise") != 0).any()):
            raise ValueError(f"Router-v3 leakage field is non-zero: {column}")
    for column in (
        "target_in_train",
        "target_group_label_used",
        "target_response_used_before_selection",
        "target_response_visible_before_selection",
    ):
        if not split[column].astype(str).str.strip().str.lower().isin(
            {"0", "false", "no"}
        ).all():
            raise ValueError(f"Router-v3 leakage flag is not false: {column}")
    for row in split.itertuples(index=False):
        expected_sizes = expected_variants[str(row.group_size_variant)]
        declared_sizes = tuple(
            int(value) for value in str(row.allowed_group_sizes).split(",")
        )
        if declared_sizes != expected_sizes:
            raise ValueError("Router-v3 split group-size declaration drift")
        metadata_path = (
            root
            / "gates"
            / str(row.backend)
            / f"variant_{row.group_size_variant}"
            / f"{_dataset_key(str(row.target_suite), str(row.target_dataset))}.metadata.json"
        )
        prediction_path = metadata_path.with_suffix("").with_suffix(".csv")
        metadata = load_json(metadata_path)
        predictions = _read_csv(prediction_path)
        if (
            tuple(int(value) for value in metadata.get("train_group_sizes", []))
            != expected_sizes
            or tuple(int(value) for value in metadata.get("test_group_sizes", []))
            != expected_sizes
            or str(metadata.get("group_size_variant", ""))
            != str(row.group_size_variant)
            or int(metadata.get("train_pair_rows", -1))
            != int(row.train_pair_rows_after_sampling)
            or int(metadata.get("test_pair_rows", -1)) != int(row.test_pair_rows)
            or len(predictions) != int(row.test_pair_rows)
            or set(pd.to_numeric(predictions["group_size"], errors="raise"))
            != set(expected_sizes)
        ):
            raise ValueError("Router-v3 isolated model/prediction metadata drift")
        if catboost_run:
            model_metadata = metadata.get("model", {})
            full_metadata = (
                model_metadata.get("full", {})
                if isinstance(model_metadata, Mapping)
                else {}
            )
            encoder_metadata = (
                full_metadata.get("encoder", {})
                if isinstance(full_metadata, Mapping)
                else {}
            )
            helpful_metadata = (
                full_metadata.get("helpful_head", {})
                if isinstance(full_metadata, Mapping)
                else {}
            )
            parameters = (
                helpful_metadata.get("parameters", {})
                if isinstance(helpful_metadata, Mapping)
                else {}
            )
            expected_parameters = {
                "iterations": 200,
                "learning_rate": 0.05,
                "depth": 6,
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "l2_leaf_reg": 3.0,
                "random_seed": 42,
                "task_type": "CPU",
                "thread_count": 1,
                "allow_writing_files": False,
                "verbose": False,
            }
            if (
                not isinstance(model_metadata, Mapping)
                or str(model_metadata.get("backend", "")) != "catboost"
                or str(model_metadata.get("backend_version", "")) != "1.2.10"
                or not isinstance(encoder_metadata, Mapping)
                or encoder_metadata.get("kind") != "catboost_native_categorical"
                or not encoder_metadata.get("categorical_feature_indices")
                or not isinstance(parameters, Mapping)
                or any(parameters.get(key) != value for key, value in expected_parameters.items())
            ):
                raise ValueError("Router-v3 CatBoost model metadata drift")

    routeability = _read_csv(root / "metrics" / "routeability_by_dataset.csv")
    if len(routeability) != len(expected_split_keys):
        raise ValueError("Router-v3 diagnostic model-fold matrix is incomplete")

    if sweep:
        artifact_reuse = load_json(
            root / "provenance" / "router_artifact_reuse.json"
        )
        parent = Path(str(manifest["router_artifact_reuse_run"])).resolve()
        artifact_rows = artifact_reuse.get("artifacts", [])
        if (
            artifact_reuse.get("reused") is not True
            or Path(str(artifact_reuse.get("parent_run", ""))).resolve() != parent
            or int(artifact_reuse.get("model_folds", -1))
            != len(expected_split_keys)
            or not isinstance(artifact_rows, list)
            or len(artifact_rows) != len(expected_split_keys)
        ):
            raise ValueError("Router-v3 gate artifact reuse provenance failed")
        for artifact in artifact_rows:
            if not isinstance(artifact, Mapping):
                raise ValueError("Router-v3 gate artifact provenance row is invalid")
            for field, hash_field in (
                ("prediction", "prediction_sha256"),
                ("metadata", "metadata_sha256"),
            ):
                relative = Path(str(artifact.get(field, "")))
                local = root / relative
                source = parent / relative
                declared = str(artifact.get(hash_field, ""))
                if (
                    not local.is_file()
                    or not source.is_file()
                    or sha256_file(local) != declared
                    or sha256_file(source) != declared
                ):
                    raise ValueError("Router-v3 reused gate artifact changed")

        budget_curves = _read_csv(root / "metrics" / "budget_curves.csv")
        aubc = _read_csv(root / "metrics" / "aubc.csv")
        if (
            len(budget_curves)
            != (len(TEST_TARGETS) + 2)
            * len(expected_backends)
            * len(expected_variants)
            * len(expected_budgets)
            or len(aubc)
            != (len(TEST_TARGETS) + 2)
            * len(expected_backends)
            * len(expected_variants)
            or {
                round(float(value), 12)
                for value in budget_curves["budget_share"]
            }
            != set(expected_budgets)
        ):
            raise ValueError("Router-v3 budget curves or AUBC matrix is incomplete")

        parent_records = read_jsonl(parent / "final" / "all_methods.jsonl")

        def frozen_20pct_index(
            rows: Sequence[Mapping[str, object]],
        ) -> dict[tuple[str, str, str, str], tuple[object, ...]]:
            indexed: dict[
                tuple[str, str, str, str], tuple[object, ...]
            ] = {}
            for record in rows:
                if (
                    str(record.get("method", ""))
                    != "budgeted_group_lightgbm"
                    or str(record.get("group_size_variant", ""))
                    not in expected_variants
                    or not math.isclose(
                        float(record.get("budget_share") or 0.0),
                        0.2,
                        abs_tol=1e-12,
                    )
                ):
                    continue
                key = (
                    str(record["suite"]),
                    str(record["dataset"]),
                    str(record["group_size_variant"]),
                    str(record["cell_id"]),
                )
                indexed[key] = (
                    record.get("prediction"),
                    bool(record.get("correct_repair")),
                    bool(record.get("accepted_llm")),
                    record.get("selected_query_id"),
                    record.get("final_source"),
                )
            return indexed

        if frozen_20pct_index(final_records) != frozen_20pct_index(parent_records):
            raise ValueError(
                "Router-v3 20% F1 or LLM-upgraded cells differ from parent"
            )

    if catboost_comparison:
        comparison_reuse = load_json(
            root / "provenance" / "comparison_reuse.json"
        )
        comparison = Path(str(manifest["router_comparison_run"])).resolve()
        comparison_manifest = comparison / "run_manifest.json"
        comparison_records_path = comparison / "final" / "all_methods.jsonl"
        if (
            Path(str(comparison_reuse.get("source_run", ""))).resolve()
            != comparison
            or not comparison_manifest.is_file()
            or sha256_file(comparison_manifest)
            != str(comparison_reuse.get("source_manifest_sha256", ""))
            or not comparison_records_path.is_file()
            or sha256_file(comparison_records_path)
            != str(comparison_reuse.get("source_records_sha256", ""))
            or int(comparison_reuse.get("comparison_records", -1))
            != TEST_TARGET_CELL_COUNT * len(EXPECTED_GATE_BACKENDS) * len(ROUTER_V3_VARIANTS)
        ):
            raise ValueError("Router-v3 CatBoost comparison provenance failed")
        if set(paired["baseline"].astype(str)) != {
            "baran",
            "llm_only",
            "budgeted_group_lightgbm",
            "budgeted_group_xgboost",
        }:
            raise ValueError("Router-v3 CatBoost paired comparator matrix differs")
        source_records = [
            row
            for row in read_jsonl(comparison_records_path)
            if str(row.get("method", ""))
            in {"budgeted_group_lightgbm", "budgeted_group_xgboost"}
            and str(row.get("scenario", "")) == "size_conditioned"
            and str(row.get("group_size_variant", "")) in expected_variants
            and math.isclose(
                float(row.get("budget_share") or 0.0), 0.2, abs_tol=1e-12
            )
        ]
        comparison_table = _read_csv(
            root / "metrics" / "per_dataset_router_comparison.csv"
        )
        expected_comparison_keys = {
            (suite, dataset, variant, backend)
            for suite, dataset in target_order()
            for variant in expected_variants
            for backend in EXPECTED_GATE_BACKENDS
        }
        actual_comparison_keys = {
            (
                str(row.suite),
                str(row.dataset),
                str(row.group_size_variant),
                str(row.comparison_backend),
            )
            for row in comparison_table.itertuples(index=False)
        }
        if (
            len(comparison_table) != len(expected_comparison_keys)
            or actual_comparison_keys != expected_comparison_keys
        ):
            raise ValueError("Router-v3 CatBoost comparison table matrix differs")
        current_dataset_metrics = {
            (
                str(row["suite"]),
                str(row["dataset"]),
                str(row["group_size_variant"]),
            ): row
            for row in independent_metrics
            if str(row.get("scope", "")) == "dataset"
            and str(row.get("method", "")) == "budgeted_group_catboost"
        }
        source_dataset_metrics = {
            (
                str(row["suite"]),
                str(row["dataset"]),
                str(row["backend"]),
                str(row["group_size_variant"]),
            ): row
            for row in summarize_records(source_records, strict=True)
            if str(row.get("scope", "")) == "dataset"
        }
        for row in comparison_table.itertuples(index=False):
            current = current_dataset_metrics[
                (str(row.suite), str(row.dataset), str(row.group_size_variant))
            ]
            source = source_dataset_metrics[
                (
                    str(row.suite),
                    str(row.dataset),
                    str(row.comparison_backend),
                    str(row.group_size_variant),
                )
            ]
            if not (
                math.isclose(float(row.catboost_f1), float(current["f1"]), abs_tol=1e-12)
                and math.isclose(float(row.comparison_f1), float(source["f1"]), abs_tol=1e-12)
                and math.isclose(
                    float(row.delta_f1),
                    float(current["f1"]) - float(source["f1"]),
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("Router-v3 CatBoost comparison metric drift")
        if (root / "catboost_info").exists():
            raise ValueError("Router-v3 CatBoost wrote an undeclared training directory")

    plan = load_json(root / "llm" / "selected_union_plan.json")
    llm_only_ids = {str(value) for value in plan.get("llm_only_query_ids", [])}
    union_ids = {str(value) for value in plan.get("query_ids", [])}
    online_ids = {str(value) for value in plan.get("online_query_ids", [])}
    if len(llm_only_ids) != TEST_TARGET_CELL_COUNT or not llm_only_ids.issubset(union_ids):
        raise ValueError("Router-v3 LLM-only singleton union is incomplete")
    selected_execution = read_jsonl(root / "llm" / "selected_execution.jsonl")
    execution_identities = {
        (str(row.get("query_id", "")), str(row.get("prompt_hash", "")))
        for row in selected_execution
    }
    if len(execution_identities) != len(selected_execution) or {
        value[0] for value in execution_identities
    } != union_ids:
        raise ValueError("Router-v3 selected execution identity coverage failed")
    logical = _read_csv(root / "metrics" / "logical_budget_ledger.csv")
    llm_logical = logical.loc[logical["scenario"].astype(str) == "llm_only_baseline"]
    if len(llm_logical) != TEST_TARGET_CELL_COUNT:
        raise ValueError("Router-v3 LLM-only logical ledger is incomplete")
    physical = pd.to_numeric(logical["physical_api_calls"], errors="raise")
    if int(physical.sum()) != len(online_ids):
        raise ValueError("Router-v3 logical/physical cost de-duplication failed")
    charged = logical.loc[physical == 1, "query_id"].astype(str)
    if charged.duplicated().any() or set(charged) != online_ids:
        raise ValueError("Router-v3 physical query charging is not unique")

    record_audit = load_json(root / "metrics" / "record_audit.json")
    formal_audit = load_json(root / "metrics" / "formal_run_audit.json")
    leakage_audit = load_json(root / "metrics" / "leakage_audit.json")
    if any(
        audit.get("ok") is not True
        for audit in (record_audit, formal_audit, leakage_audit)
    ):
        raise ValueError("Router-v3 record, leakage, or formal audit failed")
    for field in ("records", "unique_records", "slices"):
        if int(record_audit.get(field, -1)) != int(independent_record_audit[field]):
            raise ValueError(f"Router-v3 record audit mismatch: {field}")

    api_cost = _read_csv(root / "metrics" / "api_cost_audit.csv")
    if set(api_cost["phase"].astype(str)) != {
        "offline_group_calibration",
        "online_selected_union",
        "total_fresh_experiment",
    }:
        raise ValueError("Router-v3 API cost phases are incomplete")
    _validate_api_cost_resolution(api_cost)
    audit_runner = object.__new__(ExperimentRunner)
    audit_runner.paths = ExperimentPaths(
        root,
        data_root,
        root / "bound_experiment_config.json",
        root / "bound_llm_config.json",
        root,
        root.parent,
        root,
    )
    recomputed_cost = {
        str(row["phase"]): row for row in audit_runner._api_cost_rows()
    }
    reported_cost = {
        str(row["phase"]): row for row in api_cost.to_dict("records")
    }
    for phase, expected in recomputed_cost.items():
        for field in (
            "records",
            "physical_requests",
            "attempts",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cache_hits",
            "failed_records",
            "unresolved_operational_failures",
        ):
            if int(float(reported_cost[phase].get(field, -1))) != int(expected[field]):
                raise ValueError(f"Router-v3 API cost mismatch: {phase}/{field}")

    stages = manifest.get("stages", {})
    if not isinstance(stages, Mapping):
        raise ValueError("run manifest stages must be an object")
    missing_stages = [
        stage
        for stage in REQUIRED_STAGES
        if not isinstance(stages.get(stage), Mapping)
        or stages[stage].get("status") != "complete"  # type: ignore[index]
    ]
    if missing_stages:
        raise ValueError(f"Router-v3 has incomplete stages: {missing_stages}")
    if require_complete and manifest.get("status") != "complete":
        raise ValueError("Router-v3 run is not marked complete")
    return {
        "ok": True,
        "router_revision": revision,
        "run_dir": str(root),
        "status": str(manifest.get("status", "")),
        "method_metric_rows": len(method_metrics),
        "per_dataset_rows": len(detailed),
        "f1_matrix_rows": len(f1_matrix),
        "paired_statistics_rows": len(paired),
        "selection_rows": len(selection),
        "split_rows": len(split),
        "record_count": expected_record_count,
        "independently_recomputed": True,
    }


def _validate_full_baseline_run(
    root: Path,
    manifest: Mapping[str, object],
    *,
    require_complete: bool,
) -> dict[str, object]:
    """Validate a completed baseline-only run without Router artifacts."""

    if require_complete and manifest.get("status") != "complete":
        raise ValueError("full baseline run is not marked complete")
    current_implementation = _hash_tree(
        PROJECT_ROOT / "src" / "budgeted_group_repair_no_baran", (".py",)
    )
    if str(manifest.get("implementation_sha256", "")) != current_implementation:
        raise ValueError("baseline run fingerprint drift: implementation_sha256")
    for field, bound_name in (
        ("experiment_config_sha256", "bound_experiment_config.json"),
        ("llm_config_sha256", "bound_llm_config.json"),
    ):
        bound = root / bound_name
        if not bound.is_file() or str(manifest.get(field, "")) != sha256_file(bound):
            raise ValueError(f"baseline run fingerprint drift: {field}")
    for path_field, hash_field in (
        ("baran_source_run", "baran_source_manifest_sha256"),
        ("response_reuse_run", "response_reuse_manifest_sha256"),
    ):
        raw_source = str(manifest.get(path_field) or "").strip()
        if not raw_source:
            continue
        source_manifest = Path(raw_source).resolve() / "run_manifest.json"
        if (
            not source_manifest.is_file()
            or sha256_file(source_manifest) != str(manifest.get(hash_field, ""))
        ):
            raise ValueError(f"baseline run source drift: {path_field}")

    records = read_jsonl(root / "final" / "all_methods.jsonl")
    baran = [row for row in records if str(row.get("method")) == "baran"]
    llm_only = [
        row for row in records if str(row.get("method")) == "llm_only"
    ]
    if len(records) != TEST_TARGET_CELL_COUNT * 2:
        raise ValueError("baseline run cell ledger size differs")
    from .full_complementarity import pair_baseline_records

    paired = pair_baseline_records(baran, llm_only)
    summaries = summarize_records(records, strict=True)
    reported = _read_csv(root / "metrics" / "method_metrics.csv")
    if len(reported) != len(summaries):
        raise ValueError("baseline method metric row count differs")
    reported_index = {
        (
            str(row.get("scope", "")),
            str(row.get("suite", "")),
            str(row.get("dataset", "")),
            str(row.get("method", "")),
        ): row
        for row in reported.to_dict("records")
    }
    expected_index = {
        (
            str(row.get("scope", "")),
            str(row.get("suite", "")),
            str(row.get("dataset", "")),
            str(row.get("method", "")),
        ): row
        for row in summaries
    }
    if set(reported_index) != set(expected_index):
        raise ValueError("baseline method metric identity differs")
    for key, expected in expected_index.items():
        actual = reported_index[key]
        for field in ("predicted_repairs", "correct_repairs", "precision", "recall", "f1"):
            if not math.isclose(
                float(actual[field]),
                float(expected[field]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"baseline metric mismatch: {field}/{key}")
    plan = load_json(root / "llm" / "selected_union_plan.json")
    if (
        plan.get("run_kind") != "full_baselines"
        or int(plan.get("llm_only_singleton_queries", -1))
        != TEST_TARGET_CELL_COUNT
        or int(plan.get("bgr_selected_union_queries", -1)) != 0
    ):
        raise ValueError("baseline singleton plan differs")
    stages = manifest.get("stages", {})
    if not isinstance(stages, Mapping) or any(
        not isinstance(stages.get(stage), Mapping)
        or stages[stage].get("status") != "complete"  # type: ignore[index]
        for stage in BASELINE_REQUIRED_STAGES
    ):
        raise ValueError("baseline run has incomplete stages")
    return {
        "ok": True,
        "run_kind": "full_baselines",
        "run_dir": str(root),
        "status": str(manifest.get("status", "")),
        "datasets": len(TEST_TARGETS),
        "cells_per_method": TEST_TARGET_CELL_COUNT,
        "record_count": len(records),
        "paired_cells": len(paired),
        "method_metric_rows": len(summaries),
        "independently_recomputed": True,
    }


def validate_run(
    run_dir: str | Path,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    """Independently validate a Router-v3 run from its frozen artifacts."""

    root = Path(run_dir).resolve()
    manifest = load_json(root / "run_manifest.json")
    bound_config_path = root / "bound_experiment_config.json"
    if bound_config_path.is_file():
        bound_revision = str(
            load_json(bound_config_path).get("router_revision", "")
        )
        if bound_revision == ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION:
            from .router_v4 import validate_router_v4_run

            return validate_router_v4_run(
                root,
                manifest,
                require_complete=require_complete,
            )
    completed_matrix = manifest.get("completed_matrix", {})
    if (
        isinstance(completed_matrix, Mapping)
        and completed_matrix.get("run_kind") == "full_baselines"
    ):
        return _validate_full_baseline_run(
            root,
            manifest,
            require_complete=require_complete,
        )
    return _validate_router_v3_run(
        root,
        manifest,
        require_complete=require_complete,
    )




def finalize_existing_run(run_dir: str | Path) -> dict[str, object]:
    """Validate and finalize a fully materialized run without any model calls."""

    root = Path(run_dir).resolve()
    manifest_path = root / "run_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") == "complete":
        return {
            "run_dir": str(root),
            "already_complete": True,
            "validation": validate_run(root, require_complete=True),
        }

    validation_before = validate_run(root, require_complete=False)
    frozen_files = {
        "pre_completion_manifest": manifest_path,
        "final_cell_ledger": root / "final" / "all_methods.jsonl",
        "method_metrics": root / "metrics" / "method_metrics.csv",
        "primary_vs_baran": root / "metrics" / "primary_vs_baran.csv",
        "size_ablation": root / "metrics" / "size_ablation.csv",
    }
    recovery_relative = str(manifest.get("recovery_manifest", ""))
    if recovery_relative:
        frozen_files["recovery_manifest"] = root / _strict_snapshot_path(
            recovery_relative
        )
    frozen_hashes = {
        label: sha256_file(path) for label, path in frozen_files.items()
    }
    addendum = {
        "schema_version": "bgr-completion-recovery-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": "validation_only_recovery_after_transitive_lineage_fix",
        "no_model_calls": True,
        "run_id": root.name,
        "validator_file": str(Path(__file__).resolve()),
        "validator_file_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_hashes": frozen_hashes,
        "pre_completion_validation": validation_before,
    }
    addendum_path = root / "provenance" / "completion_validation_addendum.json"
    write_json(addendum_path, addendum)
    state = RunState(root, manifest_path)
    experiment = manifest.get("experiment_config", {})
    v3_backends = (
        [str(value) for value in experiment.get("gate_backends", [])]
        if isinstance(experiment, Mapping)
        else []
    )
    v3_variants = (
        list(experiment.get("router_training_variants", {}))
        if isinstance(experiment, Mapping)
        and isinstance(experiment.get("router_training_variants"), Mapping)
        else []
    )
    v3_budgets = (
        [float(value) for value in experiment.get("budget_shares", [])]
        if isinstance(experiment, Mapping)
        else []
    )
    v3_method_slices = 2 + len(v3_backends) * len(v3_variants) * len(v3_budgets)
    completed_matrix = {
        "datasets": len(TEST_TARGETS),
        "baselines": ["baran_only", "llm_only"],
        "backends": len(v3_backends),
        "budget_shares": v3_budgets,
        "group_size_variants": v3_variants,
        "method_slices": v3_method_slices,
        "cell_records": TEST_TARGET_CELL_COUNT * v3_method_slices,
        "selection_slices": len(TEST_TARGETS)
        * len(v3_backends)
        * len(v3_variants)
        * len(v3_budgets),
    }
    state.complete(
        required_stages=REQUIRED_STAGES,
        completed_matrix=completed_matrix,
        completion_recovery={
            "addendum": str(addendum_path.relative_to(root)),
            "sha256": sha256_file(addendum_path),
            "no_model_calls": True,
        },
    )
    validation_after = validate_run(root, require_complete=True)
    return {
        "run_dir": str(root),
        "already_complete": False,
        "completion_addendum": str(addendum_path),
        "validation": validation_after,
    }


__all__ = [
    "BASELINE_REQUIRED_STAGES",
    "CATBOOST_GATE_BACKENDS",
    "EXPECTED_GATE_BACKENDS",
    "ExperimentPaths",
    "ExperimentRunner",
    "MODEL_FEATURE_COLUMNS",
    "PROJECT_ROOT",
    "REQUIRED_STAGES",
    "ROUTER_V3_BUDGET_SWEEP_REVISION",
    "ROUTER_V3_CATBOOST_REVISION",
    "ROUTER_V3_REVISION",
    "ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION",
    "SafetyCapExceeded",
    "_validate_api_cost_resolution",
    "finalize_existing_run",
    "load_json",
    "validate_run",
]
