"""Resumable orchestration for the no-Baran preliminary experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .baran_reference import import_baran_reference, load_baran_records
from .costing import build_cost_audit
from .data import (
    EXPECTED_ORACLE_ERRORS,
    load_dataset,
    read_jsonl,
    sha256_file,
    validate_manifest,
    write_jsonl,
)
from .evaluation import (
    bind_oracle_correctness,
    complementarity_metrics,
    grouping_metrics,
    materialize_arm_results,
)
from .group_context import canonical_messages
from .group_generator import GroupGenerator, GroupQueryAction
from .group_llm import (
    DeepSeekGroupClient,
    GroupClientConfig,
    GroupLLMJob,
    run_group_llm_batch,
)
from .partitioning import build_matched_random_groups, select_primary_structured_groups
from .prompt_policy import INFORMATION_POLICY, PROMPT_SCHEMA_VERSION, assert_messages_safe
from .public_fd import build_fd_violation_components, fds_for_dataset, load_public_fds
from .run_state import (
    RunState,
    canonical_json_sha256,
    read_json,
    redacted_config,
    utc_now,
    write_json,
)
from .sampling import (
    EXPECTED_SELECTED_ORACLE_ERRORS,
    SELECTED_DATASETS,
    build_sample_manifest,
)


def _hash_tree(root: Path, suffixes: Sequence[str]) -> str:
    digest = hashlib.sha256()
    allowed = set(suffixes)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in allowed
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


def _action_from_dict(raw: Mapping[str, Any]) -> GroupQueryAction:
    return GroupQueryAction(
        query_id=str(raw["query_id"]),
        suite=str(raw["suite"]),
        dataset=str(raw["dataset"]),
        arm=str(raw["arm"]),
        group_view=str(raw["group_view"]),
        cell_ids=tuple(str(value) for value in raw["cell_ids"]),
        group_size=int(raw["group_size"]),
        prompt_schema_version=str(raw["prompt_schema_version"]),
        prompt_information_policy=str(raw["prompt_information_policy"]),
        messages=canonical_messages(raw["messages"]),
        prompt_hash=str(raw["prompt_hash"]),
        estimated_prompt_tokens=int(raw["estimated_prompt_tokens"]),
        completion_token_ceiling=int(raw["completion_token_ceiling"]),
        estimated_total_tokens=int(raw["estimated_total_tokens"]),
        group_features=dict(raw.get("group_features") or {}),
    )


def _read_actions(path: Path) -> tuple[GroupQueryAction, ...]:
    return tuple(_action_from_dict(row) for row in read_jsonl(path))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_csv(path, index=False)


def _markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        values: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append("NA" if math.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


@dataclass(frozen=True, slots=True)
class ExperimentPaths:
    project_root: Path
    run_dir: Path

    @property
    def manifests(self) -> Path:
        return self.run_dir / "manifests"

    @property
    def queries(self) -> Path:
        return self.run_dir / "queries"

    @property
    def llm(self) -> Path:
        return self.run_dir / "llm"

    @property
    def records(self) -> Path:
        return self.run_dir / "records"

    @property
    def metrics(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def report(self) -> Path:
        return self.run_dir / "report"


class SafetyCapExceeded(RuntimeError):
    pass


class ExperimentRunner:
    def __init__(
        self,
        *,
        project_root: str | Path,
        run_id: str,
        source_run: str | Path,
        resume: bool = False,
        experiment_config: str | Path | None = None,
    ) -> None:
        root = Path(project_root).resolve()
        self.paths = ExperimentPaths(root, root / "runs" / str(run_id))
        self.run_id = str(run_id)
        self.source_run = Path(source_run).resolve()
        self.resume = bool(resume)
        self.data_root = root / "data"
        frozen_config = self.paths.run_dir / "configs" / "experiment.json"
        frozen_llm_config = self.paths.run_dir / "configs" / "llm_redacted.json"
        if self.resume and frozen_config.is_file():
            self.config_path = frozen_config
        elif experiment_config is not None:
            self.config_path = Path(experiment_config).expanduser().resolve()
        else:
            self.config_path = root / "configs" / "experiment.json"
        self.llm_config_path = (
            frozen_llm_config
            if self.resume and frozen_llm_config.is_file()
            else root / "configs" / "deepseek_v4.json"
        )
        self.fd_path = root / "configs" / "public_fds.json"
        self.data_manifest_path = self.data_root / "manifest.json"
        self.config = read_json(self.config_path)
        self.llm_config = read_json(self.llm_config_path)
        if self.config.get("prompt_information_policy") != INFORMATION_POLICY:
            raise ValueError("experiment config uses the wrong information policy")
        if self.llm_config.get("prompt_schema_version") != PROMPT_SCHEMA_VERSION:
            raise ValueError("LLM config uses the wrong prompt schema version")

    def _binding_metadata(self) -> dict[str, Any]:
        metadata = {
                "run_id": self.run_id,
                "protocol": self.config["protocol"],
                "experiment_config": self.config,
                "experiment_config_sha256": sha256_file(self.config_path),
                "llm_config": self.llm_config,
                "llm_config_sha256": sha256_file(self.llm_config_path),
                "data_root": str(self.data_root),
                "input_data_manifest_sha256": sha256_file(self.data_manifest_path),
                "implementation_sha256": _hash_tree(self.paths.project_root / "src", (".py",)),
                "raha_code_sha256": _hash_tree(self.paths.project_root / "vendor" / "raha_source", (".py",)),
                "model": self.llm_config["model"],
                "prompt_schema_version": PROMPT_SCHEMA_VERSION,
                "prompt_schema_sha256": canonical_json_sha256(
                    {"version": PROMPT_SCHEMA_VERSION, "policy": INFORMATION_POLICY}
                ),
            }
        metadata["binding_fingerprint"] = canonical_json_sha256(metadata)
        return metadata

    def _allow_pre_provider_rebind(self) -> None:
        """Refresh implementation identity only before any provider checkpoint exists."""

        manifest_path = self.paths.run_dir / "run_manifest.json"
        if not self.resume or not manifest_path.is_file():
            return
        checkpoint = self.paths.llm / "shared" / "group_query_checkpoint.jsonl"
        if read_jsonl(checkpoint):
            return
        existing = read_json(manifest_path)
        requested = redacted_config(self._binding_metadata())
        immutable = set(requested).difference(
            {"implementation_sha256", "binding_fingerprint"}
        )
        if any(existing.get(key) != requested.get(key) for key in immutable):
            return
        if existing.get("implementation_sha256") == requested["implementation_sha256"]:
            return
        previous = str(existing.get("implementation_sha256", ""))
        existing.update(requested)
        history = list(existing.get("pre_provider_binding_amendments") or [])
        history.append(
            {
                "updated_at": utc_now(),
                "reason": "implementation completed after dry-run and before provider execution",
                "previous_implementation_sha256": previous,
                "new_implementation_sha256": requested["implementation_sha256"],
                "provider_checkpoint_records_at_update": 0,
            }
        )
        existing["pre_provider_binding_amendments"] = history
        write_json(manifest_path, existing)

    def _state(self, *, create: bool = False) -> RunState:
        manifest_path = self.paths.run_dir / "run_manifest.json"
        if create:
            metadata = self._binding_metadata()
            return RunState.create(self.paths.run_dir, metadata, resume=self.resume)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"run has not been planned: {self.paths.run_dir}")
        return RunState(self.paths.run_dir, manifest_path)

    def plan_run(self) -> dict[str, Any]:
        audit = validate_manifest(self.data_root, self.data_manifest_path)
        if audit.dataset_count != 14 or audit.oracle_error_count != 23_957:
            raise ValueError("the local CARE snapshot is incomplete")
        self._allow_pre_provider_rebind()
        state = self._state(create=True)
        for path in (
            self.paths.manifests,
            self.paths.queries,
            self.paths.llm,
            self.paths.records,
            self.paths.metrics,
            self.paths.report,
            self.paths.run_dir / "configs",
        ):
            path.mkdir(parents=True, exist_ok=True)
        frozen_config_path = self.paths.run_dir / "configs" / "experiment.json"
        frozen_llm_path = self.paths.run_dir / "configs" / "llm_redacted.json"
        if self.config_path.resolve() != frozen_config_path.resolve():
            shutil.copy2(self.config_path, frozen_config_path)
        if self.llm_config_path.resolve() != frozen_llm_path.resolve():
            shutil.copy2(self.llm_config_path, frozen_llm_path)
        shutil.copy2(self.data_manifest_path, self.paths.run_dir / "input_data_manifest.json")
        reference_dir = self.paths.run_dir / "baran_reference"
        reference = import_baran_reference(
            self.source_run,
            reference_dir,
            selected_datasets=SELECTED_DATASETS,
            data_manifest_path=self.data_manifest_path,
            expected_seed=int(self.config["baran_seed"]),
            expected_labeling_budget=int(self.config["baran_labeling_budget"]),
            expected_raha_code_sha256=_hash_tree(
                self.paths.project_root / "vendor" / "raha_source", (".py",)
            ),
        )
        shutil.copy2(
            reference_dir / "baran_reference_manifest.json",
            self.paths.manifests / "baran_reference_manifest.json",
        )
        loaded = {key: load_dataset(*key, self.data_root) for key in SELECTED_DATASETS}
        selected_count = sum(len(dataset.safe_cells()) for dataset in loaded.values())
        if selected_count != EXPECTED_SELECTED_ORACLE_ERRORS:
            raise ValueError(
                f"selected dataset error count mismatch: {selected_count} != {EXPECTED_SELECTED_ORACLE_ERRORS}"
            )
        if int(reference.manifest["record_count"]) != selected_count:
            raise ValueError("Baran reference does not cover all selected oracle error cells")
        safe_by_key = {key: dataset.safe_view() for key, dataset in loaded.items()}
        sample_records = build_sample_manifest(
            {key: safe.cells for key, safe in safe_by_key.items()},
            mode=str(self.config["sample_mode"]),
            sample_n_per_dataset=int(self.config["sample_n_per_dataset"]),
            seed=int(self.config["sample_seed"]),
        )
        write_jsonl(
            self.paths.manifests / "sample_manifest.jsonl",
            [record.as_dict() for record in sample_records],
        )
        records_by_key: dict[tuple[str, str], list[Any]] = {key: [] for key in SELECTED_DATASETS}
        for record in sample_records:
            records_by_key[(record.suite, record.dataset)].append(record)
        fd_registry = load_public_fds(self.fd_path)
        singleton_actions: list[GroupQueryAction] = []
        all_candidate_actions: list[GroupQueryAction] = []
        structured_actions: list[GroupQueryAction] = []
        random_actions: list[GroupQueryAction] = []
        generation_audits: dict[str, Any] = {}
        partition_audits: dict[str, Any] = {}
        for key in SELECTED_DATASETS:
            suite, dataset_name = key
            safe = safe_by_key[key]
            safe_map = {str(cell.cell_id): cell for cell in safe.cells}
            sample_cells = tuple(safe_map[record.cell_id] for record in records_by_key[key])
            baran_rows = load_baran_records(reference_dir, suite, dataset_name)
            baran_by_cell = {str(row["cell_id"]): row for row in baran_rows}
            missing_baran = sorted(str(cell.cell_id) for cell in sample_cells if str(cell.cell_id) not in baran_by_cell)
            if missing_baran:
                raise ValueError(f"Baran reference misses sampled cells for {suite}/{dataset_name}")
            fds = fds_for_dataset(fd_registry, suite, dataset_name)
            components = build_fd_violation_components(
                safe.dirty, suite, dataset_name, sample_cells, fds
            )
            generator = GroupGenerator(
                safe,
                sample_cells,
                baran_by_cell,
                fd_components=components,
                group_sizes=(1, 2, 4, 8),
                prompt_schema_version=PROMPT_SCHEMA_VERSION,
                similar_row_count=int(self.llm_config["contexts"]["similar_row_count"]),
            )
            generated = generator.generate()
            primary, primary_audit = select_primary_structured_groups(
                generated.actions,
                group_size=int(self.config["primary_group_size"]),
                view_priority=tuple(self.config["group_views"]),
                seed=int(self.config["structured_partition_seed"]),
            )
            dataset_singletons = tuple(
                action for action in generated.actions if action.arm == "singleton"
            )
            random_partition, random_audit = build_matched_random_groups(
                primary,
                dataset=safe,
                cells=sample_cells,
                singleton_actions={action.cell_ids[0]: action for action in dataset_singletons},
                fd_components=components,
                seed=int(self.config["random_partition_seed"]),
                prompt_schema_version=PROMPT_SCHEMA_VERSION,
                similar_row_count=int(self.llm_config["contexts"]["similar_row_count"]),
            )
            singleton_actions.extend(dataset_singletons)
            all_candidate_actions.extend(generated.actions)
            structured_actions.extend(primary)
            random_actions.extend(random_partition)
            label = f"{suite}/{dataset_name}"
            generation_audits[label] = dict(generated.audit)
            partition_audits[label] = {"structured": primary_audit, "random": random_audit}
        for action in (*singleton_actions, *structured_actions, *random_actions):
            assert_messages_safe(action.messages)
        write_jsonl(self.paths.queries / "singleton_actions.jsonl", [action.as_dict() for action in singleton_actions])
        write_jsonl(self.paths.queries / "all_candidate_actions.jsonl", [action.as_dict() for action in all_candidate_actions])
        write_jsonl(self.paths.queries / "structured_group_actions.jsonl", [action.as_dict() for action in structured_actions])
        write_jsonl(self.paths.queries / "random_group_actions.jsonl", [action.as_dict() for action in random_actions])
        write_json(self.paths.manifests / "partition_matching_audit.json", partition_audits)
        write_json(
            self.paths.queries / "primary_structured_partition.json",
            {
                "query_ids": [action.query_id for action in structured_actions],
                "cell_ids": sorted({cell_id for action in structured_actions for cell_id in action.cell_ids}),
                "audits": partition_audits,
            },
        )
        write_json(
            self.paths.queries / "matched_random_partition.json",
            {
                "query_ids": [action.query_id for action in random_actions],
                "cell_ids": sorted({cell_id for action in random_actions for cell_id in action.cell_ids}),
                "audits": partition_audits,
            },
        )
        estimated_by_arm = {
            "singleton": sum(action.estimated_total_tokens for action in singleton_actions),
            "structured": sum(action.estimated_total_tokens for action in structured_actions),
            "random": sum(action.estimated_total_tokens for action in random_actions),
        }
        attempts = int(self.llm_config["max_retries"]) + 1
        plan = {
            "protocol": self.config["protocol"],
            "prompt_information_policy": INFORMATION_POLICY,
            "sample_cells": len(sample_records),
            "selected_oracle_cells": selected_count,
            "calls_by_arm": {
                "model_preflight": 1,
                "singleton": len(singleton_actions),
                "structured": len(structured_actions),
                "random": len(random_actions),
            },
            "logical_calls_total": 1
            + len(singleton_actions)
            + len(structured_actions)
            + len(random_actions),
            "maximum_unique_provider_requests": len(singleton_actions)
            + len(structured_actions)
            + len(random_actions),
            "estimated_tokens_by_arm": estimated_by_arm,
            "estimated_tokens_total": sum(estimated_by_arm.values()),
            "conservative_retry_reservation_tokens": attempts * sum(estimated_by_arm.values()),
            "formal_token_cap": self.config.get("max_provider_tokens_safety_cap"),
            "generation_audits": generation_audits,
            "baran_reference_fingerprint": reference.manifest["reference_fingerprint"],
            "planned_at": utc_now(),
        }
        write_json(self.paths.manifests / "selected_datasets.json", {
            "datasets": [f"{suite}/{dataset}" for suite, dataset in SELECTED_DATASETS],
            "oracle_error_count": selected_count,
        })
        write_json(self.paths.manifests / "prompt_policy_audit.json", {
            "baran_fields_found": 0,
            "oracle_fields_found": 0,
            "query_count": len(singleton_actions) + len(structured_actions) + len(random_actions),
            "serialized_messages_recursively_scanned": True,
            "information_policy": INFORMATION_POLICY,
        })
        write_json(self.paths.run_dir / "dry_run_plan.json", plan)
        write_json(self.paths.metrics / "decision_gates.json", {
            "complementarity_supported": None,
            "grouping_supported": None,
            "routeability_supported": None,
            "phase3_allowed": False,
        })
        state.update_stage("plan", "complete", **{key: value for key, value in plan.items() if key != "generation_audits"})
        return plan

    def _client(self) -> DeepSeekGroupClient:
        env_name = str(self.llm_config["api_key_env"])
        api_key = os.environ.get(env_name, "")
        if not api_key:
            raise RuntimeError(f"required environment variable is not set: {env_name}")
        return DeepSeekGroupClient(GroupClientConfig.from_mapping(self.llm_config), api_key=api_key)

    def write_cost_audit(self) -> dict[str, Any]:
        estimates: dict[str, int] = {}
        for name in (
            "singleton_actions.jsonl",
            "structured_group_actions.jsonl",
            "random_group_actions.jsonl",
            "all_candidate_actions.jsonl",
        ):
            path = self.paths.queries / name
            if path.is_file():
                estimates.update(
                    {
                        action.query_id: action.estimated_total_tokens
                        for action in _read_actions(path)
                    }
                )
        return build_cost_audit(
            self.paths.run_dir,
            estimated_tokens_by_query=estimates,
        )

    def freeze_token_cap(self, token_cap: int | None) -> int | None:
        self.assert_binding_current()
        cap = int(token_cap) if token_cap is not None else None
        if cap is not None and cap <= 0:
            raise SafetyCapExceeded("provider token safety cap must be positive")
        configured = self.config.get("max_provider_tokens_safety_cap")
        if configured is not None and (cap is None or int(configured) != cap):
            raise SafetyCapExceeded("CLI token cap differs from the frozen experiment config")
        manifest = self._state().manifest
        requested_policy = (
            {"mode": "unlimited"}
            if cap is None
            else {"mode": "capped", "tokens": cap}
        )
        existing = manifest.get("provider_token_policy")
        if existing is not None and existing != requested_policy:
            raise SafetyCapExceeded(
                "provider token policy is already frozen for this run; use a new run ID to change it"
            )
        if existing is None:
            manifest["provider_token_policy"] = requested_policy
            manifest["provider_token_policy_frozen_at"] = utc_now()
            write_json(self.paths.run_dir / "run_manifest.json", manifest)
        return cap

    def assert_binding_current(self) -> None:
        manifest = self._state().manifest
        checks = {
            "implementation_sha256": _hash_tree(self.paths.project_root / "src", (".py",)),
            "raha_code_sha256": _hash_tree(
                self.paths.project_root / "vendor" / "raha_source", (".py",)
            ),
            "experiment_config_sha256": sha256_file(self.config_path),
            "llm_config_sha256": sha256_file(self.llm_config_path),
            "input_data_manifest_sha256": sha256_file(self.data_manifest_path),
        }
        drift = [
            field
            for field, value in checks.items()
            if str(manifest.get(field, "")) != str(value)
        ]
        if drift:
            raise ValueError("run binding drift: " + ", ".join(drift))

    def _existing_conservative_tokens(self) -> int:
        estimates: dict[str, int] = {}
        for name in (
            "singleton_actions.jsonl",
            "structured_group_actions.jsonl",
            "random_group_actions.jsonl",
            "all_candidate_actions.jsonl",
        ):
            path = self.paths.queries / name
            if path.is_file():
                estimates.update(
                    {
                        action.query_id: action.estimated_total_tokens
                        for action in _read_actions(path)
                    }
                )
        total = 0
        checkpoint = self.paths.llm / "shared" / "group_query_checkpoint.jsonl"
        for record in read_jsonl(checkpoint):
            if bool(record.get("cache_hit")) or bool(record.get("checkpoint_hit")):
                continue
            query_id = str(record.get("query_id", ""))
            estimate = int(estimates.get(query_id, 0))
            total += int(record.get("observed_total_tokens", 0) or 0)
            total += int(record.get("unknown_usage_attempts", 0) or 0) * estimate
        return total

    def response_reusable(self, action: GroupQueryAction, target_phase: str) -> bool:
        def compatible(source: str) -> bool:
            if source == target_phase:
                return True
            if source == "model_preflight" and target_phase == "preliminary_singleton":
                return True
            return source in {
                "preliminary_singleton",
                "preliminary_structured",
                "preliminary_random",
            } and target_phase == "bgr_selected_union"

        checkpoint = self.paths.llm / "shared" / "group_query_checkpoint.jsonl"
        for record in reversed(read_jsonl(checkpoint)):
            metadata = record.get("metadata")
            source = str(metadata.get("phase", "")) if isinstance(metadata, Mapping) else ""
            if (
                str(record.get("query_id", "")) == action.query_id
                and str(record.get("prompt_hash", "")) == action.prompt_hash
                and record.get("status") == "success"
                and record.get("model_matches_request", True)
                and str(record.get("model_returned", record.get("model", "")))
                == str(self.llm_config["model"])
                and compatible(source)
            ):
                return True
        return False

    def _execute(
        self,
        arm: str,
        token_cap: int | None,
        *,
        actions_override: Sequence[GroupQueryAction] | None = None,
        response_filename: str | None = None,
    ) -> list[dict[str, Any]]:
        file_by_arm = {
            "singleton": "singleton_actions.jsonl",
            "structured": "structured_group_actions.jsonl",
            "random": "random_group_actions.jsonl",
        }
        actions = (
            tuple(actions_override)
            if actions_override is not None
            else _read_actions(self.paths.queries / file_by_arm[arm])
        )
        phase = f"preliminary_{arm}"
        pending_actions = [
            action for action in actions if not self.response_reusable(action, phase)
        ]
        estimated = sum(action.estimated_total_tokens for action in pending_actions)
        reservation = estimated * (int(self.llm_config["max_retries"]) + 1)
        already_consumed = self._existing_conservative_tokens()
        if token_cap is not None and int(token_cap) < already_consumed + reservation:
            raise SafetyCapExceeded(
                f"token cap {token_cap} is below consumed-plus-reserved tokens "
                f"{already_consumed + reservation} for arm {arm}"
            )
        self.freeze_token_cap(token_cap)
        if not actions:
            records: list[dict[str, Any]] = []
            write_jsonl(
                self.paths.llm / (response_filename or f"{arm}_responses.jsonl"),
                records,
            )
            return records
        client = self._client()
        jobs = [
            GroupLLMJob.from_action(
                action,
                metadata={
                    "phase": phase,
                    "arm": arm,
                    "model_requested": str(self.llm_config["model"]),
                    "prompt_information_policy": INFORMATION_POLICY,
                    "require_complete_response": False,
                },
            )
            for action in actions
        ]
        records = run_group_llm_batch(
            client,
            jobs,
            self.paths.llm / "shared",
            concurrency=int(self.llm_config["concurrency"]),
        )
        total_after = self._existing_conservative_tokens()
        consumed = max(0, total_after - already_consumed)
        if token_cap is not None and total_after > int(token_cap):
            raise SafetyCapExceeded(
                f"conservative run usage {total_after} exceeded token cap {token_cap}"
            )
        write_jsonl(
            self.paths.llm / (response_filename or f"{arm}_responses.jsonl"), records
        )
        self.write_cost_audit()
        mismatches = [
            str(record.get("query_id", ""))
            for record in records
            if record.get("model_matches_request") is False
        ]
        if mismatches:
            raise RuntimeError(
                f"provider returned a non-frozen model for {len(mismatches)} {arm} queries"
            )
        self._state().update_stage(f"llm_{arm}", "complete", queries=len(records), conservative_tokens=consumed)
        return records

    def check_model(self, token_cap: int | None) -> dict[str, Any]:
        actions = _read_actions(self.paths.queries / "singleton_actions.jsonl")
        if not actions:
            raise RuntimeError("model preflight needs at least one singleton query")
        action = actions[0]
        if token_cap is not None and int(token_cap) < action.estimated_total_tokens * (int(self.llm_config["max_retries"]) + 1):
            raise SafetyCapExceeded("model preflight cap is too small")
        self.freeze_token_cap(token_cap)
        client = self._client()
        job = GroupLLMJob.from_action(
            action,
            metadata={
                "phase": "model_preflight",
                "model_requested": str(self.llm_config["model"]),
                "require_complete_response": False,
            },
        )
        record = run_group_llm_batch(client, [job], self.paths.llm / "shared")[0]
        if str(record.get("model_returned")) != str(self.llm_config["model"]):
            raise RuntimeError("provider returned a different model from the frozen request")
        if record.get("status") != "success":
            raise RuntimeError("model preflight did not return a reusable structured response")
        write_json(self.paths.llm / "model_preflight.json", record)
        self.write_cost_audit()
        self._state().update_stage("model_preflight", "complete", model=record.get("model_returned"))
        return record

    def _oracle_maps(self) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        clean: dict[str, str] = {}
        dirty: dict[str, str] = {}
        rows: dict[str, str] = {}
        for key in SELECTED_DATASETS:
            for cell in load_dataset(*key, self.data_root).oracle_cells(include_annotations=False):
                clean[str(cell.cell_id)] = cell.clean_value
                dirty[str(cell.cell_id)] = cell.dirty_value
                rows[str(cell.cell_id)] = cell.row_id
        return clean, dirty, rows

    def run_experiment1(self, token_cap: int | None) -> dict[str, Any]:
        if not self._state().stage_completed("model_preflight"):
            raise RuntimeError("experiment one requires a successful model compatibility preflight")
        actions = _read_actions(self.paths.queries / "singleton_actions.jsonl")
        responses = self._execute("singleton", token_cap)
        self.write_cost_audit()
        clean, dirty, row_ids = self._oracle_maps()
        raw = materialize_arm_results(actions, responses, dirty_by_cell=dirty)
        evaluated = bind_oracle_correctness(raw, clean)
        write_jsonl(self.paths.records / "experiment1_cells.jsonl", evaluated)
        baran_predictions: dict[str, str] = {}
        for suite, dataset in SELECTED_DATASETS:
            for record in load_baran_records(self.paths.run_dir / "baran_reference", suite, dataset):
                baran_predictions[str(record["cell_id"])] = str(record.get("prediction", record.get("dirty_value", "")))
        by_dataset, summary = complementarity_metrics(
            evaluated,
            baran_prediction_by_cell=baran_predictions,
            clean_by_cell=clean,
            row_id_by_cell=row_ids,
            bootstrap_replicates=int(self.config["bootstrap_replicates"]),
            bootstrap_seed=int(self.config["bootstrap_seed"]),
            confidence=float(self.config["confidence_level"]),
        )
        _write_csv(self.paths.metrics / "experiment1_by_dataset.csv", by_dataset)
        write_json(self.paths.metrics / "experiment1_summary.json", summary)
        columns = ["dataset", "N", "baran_accuracy", "singleton_accuracy", "n11", "n10", "n01", "n00", "oracle_upper_bound", "upper_bound_minus_best"]
        (self.paths.report / "experiment1.md").write_text(
            "# 实验一：Baran 与 no-Baran singleton LLM 互补性\n\n" + _markdown_table(by_dataset, columns) + "\n",
            encoding="utf-8",
        )
        gain = float(summary["macro"]["upper_bound_minus_best"])
        supported = gain >= float(self.config["minimum_oracle_gain_vs_best_standalone"])
        gates = read_json(self.paths.metrics / "decision_gates.json")
        gates["complementarity_supported"] = supported
        gates["phase3_allowed"] = all(gates.get(key) is True for key in ("complementarity_supported", "grouping_supported", "routeability_supported"))
        write_json(self.paths.metrics / "decision_gates.json", gates)
        self._state().update_stage("experiment1", "complete", datasets=len(by_dataset), complementarity_supported=supported)
        return {"by_dataset": by_dataset, "summary": summary, "supported": supported}

    def run_experiment2(self, token_cap: int | None) -> dict[str, Any]:
        if not self._state().stage_completed("experiment1"):
            raise RuntimeError("experiment two requires the frozen experiment-one singleton ledger")
        singleton_actions = _read_actions(self.paths.queries / "singleton_actions.jsonl")
        singleton_responses = read_jsonl(self.paths.llm / "singleton_responses.jsonl")
        structured_actions = _read_actions(self.paths.queries / "structured_group_actions.jsonl")
        random_actions = _read_actions(self.paths.queries / "random_group_actions.jsonl")
        structured_responses = self._execute("structured", token_cap)
        random_responses = self._execute("random", token_cap)
        clean, dirty, _ = self._oracle_maps()
        evaluation_cells = {cell_id for action in structured_actions for cell_id in action.cell_ids}
        singleton_subset = tuple(action for action in singleton_actions if action.cell_ids[0] in evaluation_cells)
        singleton_raw = materialize_arm_results(singleton_subset, singleton_responses, dirty_by_cell=dirty)
        structured_raw = materialize_arm_results(structured_actions, structured_responses, dirty_by_cell=dirty)
        random_raw = materialize_arm_results(random_actions, random_responses, dirty_by_cell=dirty)
        singleton_eval = bind_oracle_correctness(singleton_raw, clean)
        structured_eval = bind_oracle_correctness(structured_raw, clean)
        random_eval = bind_oracle_correctness(random_raw, clean)
        combined = (*singleton_eval, *structured_eval, *random_eval)
        write_jsonl(self.paths.records / "experiment2_primary_cells.jsonl", combined)
        by_dataset, summary = grouping_metrics(
            singleton_eval,
            structured_eval,
            random_eval,
            bootstrap_replicates=int(self.config["bootstrap_replicates"]),
            bootstrap_seed=int(self.config["bootstrap_seed"]),
            confidence=float(self.config["confidence_level"]),
            noninferiority_margin=float(self.config["noninferiority_margin_absolute"]),
            minimum_token_saving=float(self.config["minimum_practical_token_saving_ratio"]),
            maximum_parse_validity_drop=float(
                self.config["maximum_parse_validity_drop_absolute"]
            ),
            maximum_missing_item_rate_increase=float(
                self.config["maximum_missing_item_rate_increase_absolute"]
            ),
        )
        _write_csv(self.paths.metrics / "experiment2_by_dataset.csv", by_dataset)
        write_json(self.paths.metrics / "experiment2_summary.json", summary)
        columns = ["dataset", "N_cells", "N_structured_groups", "singleton_accuracy", "structured_accuracy", "delta_accuracy", "random_accuracy", "structured_minus_random", "token_per_cell_saving", "decision"]
        (self.paths.report / "experiment2.md").write_text(
            "# 实验二：singleton、结构化 group 与 matched random group\n\n" + _markdown_table(by_dataset, columns) + "\n",
            encoding="utf-8",
        )
        supported = summary["decision"] in {
            "A_quality_superiority",
            "B_noninferior_more_efficient",
        }
        gates = read_json(self.paths.metrics / "decision_gates.json")
        gates["grouping_supported"] = supported
        gates["phase3_allowed"] = all(gates.get(key) is True for key in ("complementarity_supported", "grouping_supported", "routeability_supported"))
        write_json(self.paths.metrics / "decision_gates.json", gates)
        self._state().update_stage("experiment2", "complete", datasets=len(by_dataset), grouping_supported=supported)
        return {
            "by_dataset": by_dataset,
            "summary": summary,
            "supported": supported,
        }

    def validate_run(self, *, require_experiments: bool = False) -> dict[str, Any]:
        plan = read_json(self.paths.run_dir / "dry_run_plan.json")
        sample = read_jsonl(self.paths.manifests / "sample_manifest.jsonl")
        prompt_audit = read_json(self.paths.manifests / "prompt_policy_audit.json")
        partition_audit = read_json(
            self.paths.manifests / "partition_matching_audit.json"
        )
        errors: list[str] = []
        run_manifest = self._state().manifest
        binding_checks = {
            "implementation_sha256": _hash_tree(self.paths.project_root / "src", (".py",)),
            "raha_code_sha256": _hash_tree(
                self.paths.project_root / "vendor" / "raha_source", (".py",)
            ),
            "experiment_config_sha256": sha256_file(self.config_path),
            "llm_config_sha256": sha256_file(self.llm_config_path),
            "input_data_manifest_sha256": sha256_file(self.data_manifest_path),
        }
        for field, current in binding_checks.items():
            if str(run_manifest.get(field, "")) != str(current):
                errors.append(f"run binding drift: {field}")
        if len(sample) != int(plan.get("sample_cells", -1)):
            errors.append("sample count differs from dry-run plan")
        if prompt_audit.get("baran_fields_found") != 0 or prompt_audit.get("oracle_fields_found") != 0:
            errors.append("prompt leakage audit failed")
        actions = []
        for name in ("singleton_actions.jsonl", "structured_group_actions.jsonl", "random_group_actions.jsonl"):
            actions.extend(_read_actions(self.paths.queries / name))
        if len({action.query_id for action in actions}) != len(actions):
            errors.append("query IDs are not unique")
        for action in actions:
            try:
                assert_messages_safe(action.messages)
            except Exception as error:
                errors.append(f"unsafe prompt {action.query_id}: {error}")
        structured_population = {
            cell_id
            for action in _read_actions(self.paths.queries / "structured_group_actions.jsonl")
            for cell_id in action.cell_ids
        }
        random_population = {
            cell_id
            for action in _read_actions(self.paths.queries / "random_group_actions.jsonl")
            for cell_id in action.cell_ids
        }
        if structured_population != random_population:
            errors.append("structured and random primary populations differ")
        if require_experiments:
            for stage in ("experiment1", "experiment2"):
                if not self._state().stage_completed(stage):
                    errors.append(f"missing completed stage: {stage}")
            action_files = {
                "singleton": "singleton_actions.jsonl",
                "structured": "structured_group_actions.jsonl",
                "random": "random_group_actions.jsonl",
            }
            for arm, name in action_files.items():
                expected = {
                    action.query_id: action
                    for action in _read_actions(self.paths.queries / name)
                }
                response_path = self.paths.llm / f"{arm}_responses.jsonl"
                responses = read_jsonl(response_path)
                by_query = {str(row.get("query_id", "")): row for row in responses}
                if len(by_query) != len(responses) or set(by_query) != set(expected):
                    errors.append(f"{arm} response ledger coverage or uniqueness failed")
                    continue
                for query_id, response in by_query.items():
                    if str(response.get("prompt_hash", "")) != expected[query_id].prompt_hash:
                        errors.append(f"{arm} prompt hash mismatch: {query_id}")
                    if response.get("model_matches_request") is False:
                        errors.append(f"{arm} provider model mismatch: {query_id}")
            experiment1_path = self.paths.records / "experiment1_cells.jsonl"
            experiment2_path = self.paths.records / "experiment2_primary_cells.jsonl"
            exp1 = read_jsonl(experiment1_path)
            exp2 = read_jsonl(experiment2_path)
            if len(exp1) != int(plan["sample_cells"]):
                errors.append("experiment-one cell record coverage failed")
            if len(exp2) != 3 * len(structured_population):
                errors.append("experiment-two cell record coverage failed")
            if len({(str(row.get("arm")), str(row.get("cell_id"))) for row in exp2}) != len(exp2):
                errors.append("experiment-two arm/cell identities are not unique")
            if not errors:
                clean, _, row_ids = self._oracle_maps()
                rebound_exp1 = bind_oracle_correctness(exp1, clean)
                if any(
                    bool(before.get("correct")) != bool(after.get("correct"))
                    for before, after in zip(exp1, rebound_exp1)
                ):
                    errors.append("experiment-one correctness cannot be independently reproduced")
                baran_predictions: dict[str, str] = {}
                for suite, dataset in SELECTED_DATASETS:
                    for record in load_baran_records(
                        self.paths.run_dir / "baran_reference", suite, dataset
                    ):
                        baran_predictions[str(record["cell_id"])] = str(
                            record.get("prediction", record.get("dirty_value", ""))
                        )
                by_dataset, recomputed_exp1 = complementarity_metrics(
                    rebound_exp1,
                    baran_prediction_by_cell=baran_predictions,
                    clean_by_cell=clean,
                    row_id_by_cell=row_ids,
                    bootstrap_replicates=int(self.config["bootstrap_replicates"]),
                    bootstrap_seed=int(self.config["bootstrap_seed"]),
                    confidence=float(self.config["confidence_level"]),
                )
                if canonical_json_sha256(recomputed_exp1) != canonical_json_sha256(
                    read_json(self.paths.metrics / "experiment1_summary.json")
                ):
                    errors.append("experiment-one summary does not reproduce")
                if len(by_dataset) != 9:
                    errors.append("experiment-one per-dataset coverage is not nine")
                exp2_by_arm = {
                    arm: [row for row in exp2 if str(row.get("arm")) == arm]
                    for arm in ("singleton", "structured", "random")
                }
                rebound_exp2 = {
                    arm: bind_oracle_correctness(rows_for_arm, clean)
                    for arm, rows_for_arm in exp2_by_arm.items()
                }
                by_dataset2, recomputed_exp2 = grouping_metrics(
                    rebound_exp2["singleton"],
                    rebound_exp2["structured"],
                    rebound_exp2["random"],
                    bootstrap_replicates=int(self.config["bootstrap_replicates"]),
                    bootstrap_seed=int(self.config["bootstrap_seed"]),
                    confidence=float(self.config["confidence_level"]),
                    noninferiority_margin=float(
                        self.config["noninferiority_margin_absolute"]
                    ),
                    minimum_token_saving=float(
                        self.config["minimum_practical_token_saving_ratio"]
                    ),
                    maximum_parse_validity_drop=float(
                        self.config["maximum_parse_validity_drop_absolute"]
                    ),
                    maximum_missing_item_rate_increase=float(
                        self.config["maximum_missing_item_rate_increase_absolute"]
                    ),
                )
                if canonical_json_sha256(recomputed_exp2) != canonical_json_sha256(
                    read_json(self.paths.metrics / "experiment2_summary.json")
                ):
                    errors.append("experiment-two summary does not reproduce")
                expected_group_datasets = {
                    str(dataset_key).split("/", 1)[-1]
                    for dataset_key, dataset_audit in partition_audit.items()
                    if int(dataset_audit.get("structured", {}).get("covered_cell_count", 0))
                    > 0
                }
                observed_group_datasets = {
                    str(row.get("dataset", "")) for row in by_dataset2
                }
                if observed_group_datasets != expected_group_datasets:
                    errors.append(
                        "experiment-two per-dataset coverage differs from the frozen "
                        "positive-coverage partition"
                    )
            self.write_cost_audit()
            if not (self.paths.metrics / "api_cost_audit.csv").is_file():
                errors.append("API cost audit is missing")
        result = {
            "ok": not errors,
            "errors": errors,
            "sample_cells": len(sample),
            "queries": len(actions),
            "prompt_policy": INFORMATION_POLICY,
        }
        write_json(self.paths.metrics / "record_audit.json", result)
        if errors:
            raise ValueError("run validation failed: " + "; ".join(errors))
        return result

    def finalize_run(self) -> dict[str, Any]:
        audit = self.validate_run(require_experiments=True)
        from .reporting import build_report

        report = build_report(self.paths.run_dir, deliver=True)
        self._state().complete(required_stages=("plan", "experiment1", "experiment2"), validation=audit)
        return {"validation": audit, "report": report}


__all__ = ["ExperimentPaths", "ExperimentRunner", "SafetyCapExceeded"]
