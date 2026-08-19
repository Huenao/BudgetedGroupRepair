"""Leakage-safe Router-v4 LightGBM isotonic-calibration experiment."""

from __future__ import annotations

import hashlib
import math
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .data import load_dataset, read_jsonl, validate_manifest, write_jsonl
from .group_gate import GroupUpliftGate
from .group_objective import GroupUpliftObjective, PairGain
from .group_optimizer import select_queries
from .metrics import summarize_records, verify_records
from .protocol import base_family, split_for_target
from .router_v3 import (
    CALIBRATION_SINGLETON_CELL_COUNT,
    LOGICAL_LEDGER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    PROJECT_ROOT,
    REQUIRED_STAGES,
    ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION,
    TEST_TARGET_CELL_COUNT,
    ExperimentRunner,
    SafetyCapExceeded,
    _actual_tokens,
    _budget_label,
    _dataset_key,
    _hash_tree,
    _read_csv,
    _write_csv,
    generation_order,
    load_json,
    target_order,
)
from .run_state import canonical_json_sha256, sha256_file, write_json
from .statistics import holm_adjust


HISTORICAL_COMPARATOR_HASHES = {
    "run_manifest.json": "18f15a04c8a119296aea9367d2f52f0d20effd3e94a6169c3955889ccc356435",
    "bound_experiment_config.json": "2ea0fbec5a014260b8bfd8c440378f0bbbd875966037d499c0df854ae1411894",
    "llm/calibration_pair_labels.csv": "5179427ffb84f2f4a02f9b00d2dc422ba6ae6e685fb3a694d00f0e8812c06358",
    "metrics/method_metrics.csv": "b20a8bdd22fa9989f1463da054c36c5578fb16718048949e5e49e520d30e09dc",
    "final/all_methods.jsonl": "bb6977606600a239eb64a8cd8984b19d5129d59e0f1518e28dd5bda6ce93ebde",
}
EXPECTED_CALIBRATION_QUERIES = 8_197
EXPECTED_CALIBRATION_PAIR_LABELS = 16_451
FROZEN_ROUTER_V4_IMPLEMENTATION_SHA256 = frozenset(
    {"7159d9d20fb2197670ee53f1a59b604c8919135d6dc8eea3b49011c7d866f67f"}
)


def _probability_diagnostics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    max_bins: int = 10,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("calibration diagnostics require equally sized nonempty arrays")
    if not bool(np.isfinite(p).all()) or bool(((p < 0.0) | (p > 1.0)).any()):
        raise ValueError("calibration probabilities must be finite and in [0, 1]")
    prevalence = float(y.mean())
    brier = float(brier_score_loss(y, p))
    baseline_brier = prevalence * (1.0 - prevalence)
    auprc = (
        float(average_precision_score(y, p))
        if len(set(int(value) for value in y)) > 1
        else prevalence
    )
    order = np.lexsort((np.arange(len(p)), p))
    chunks = np.array_split(order, min(max_bins, len(order)))
    bins: list[dict[str, float | int]] = []
    for index, chunk in enumerate(chunks, start=1):
        if len(chunk) == 0:
            continue
        bin_p = p[chunk]
        bin_y = y[chunk]
        bins.append(
            {
                "bin": index,
                "count": int(len(chunk)),
                "score_min": float(bin_p.min()),
                "score_max": float(bin_p.max()),
                "mean_predicted_probability": float(bin_p.mean()),
                "observed_positive_rate": float(bin_y.mean()),
                "absolute_gap": float(abs(bin_p.mean() - bin_y.mean())),
            }
        )
    ece = float(
        sum(int(row["count"]) * float(row["absolute_gap"]) for row in bins)
        / len(y)
    )
    mce = max(float(row["absolute_gap"]) for row in bins)
    return (
        {
            "rows": int(len(y)),
            "prevalence": prevalence,
            "average_precision": auprc,
            "brier": brier,
            "prevalence_constant_brier": baseline_brier,
            "brier_skill": (
                float(1.0 - brier / baseline_brier)
                if baseline_brier > 0.0
                else 0.0
            ),
            "ece": ece,
            "maximum_calibration_error": mce,
            "mean_predicted_probability": float(p.mean()),
            "observed_positive_rate": prevalence,
            "actual_bins": len(bins),
        },
        bins,
    )


class RouterV4ExperimentRunner(ExperimentRunner):
    """Run the frozen k=1/k=4 LightGBM isotonic experiment."""

    def _router_training_variants(self) -> dict[str, tuple[int, ...]]:
        raw = self.experiment_config.get("router_training_variants", {})
        expected = {"1": [1], "4": [1, 4]}
        if raw != expected:
            raise ValueError(f"Router-v4 training variants must be exactly {expected}")
        return {"1": (1,), "4": (1, 4)}

    def _bgr_method_name(self, backend: str) -> str:
        if backend != "lightgbm":
            raise ValueError("Router-v4 supports only LightGBM")
        return "budgeted_group_lightgbm_isotonic"

    def _validate_calibration_configuration(self) -> None:
        expected = {
            "enabled": True,
            "method": "isotonic",
            "source": "train_family_out_of_fold",
            "heads": ["helpful", "harmful"],
            "out_of_bounds": "clip",
            "class_rebalancing": False,
            "calibrated_replica_uncertainty": True,
        }
        if self.experiment_config.get("probability_calibration") != expected:
            raise ValueError("Router-v4 probability_calibration block drift")

    def generate_groups_stage(
        self,
        datasets: Sequence[tuple[str, str]] | None = None,
    ) -> dict[str, object]:
        """Strictly reuse request-identical candidate/features, never gate scores."""

        dataset_order = tuple(datasets) if datasets is not None else generation_order()
        if self.calibration_source_run is None:
            raise ValueError("Router-v4 group reuse requires the calibration source")
        source = self.calibration_source_run
        provenance_path = self.paths.run_dir / "provenance" / "group_artifact_reuse.json"
        frozen_reuse = load_json(source / "provenance" / "reuse_manifest.json")
        identity_rows = frozen_reuse.get("identity_artifacts", [])
        identity_hashes = {
            str(row["artifact"]): str(row["current_sha256"])
            for row in identity_rows
            if isinstance(row, Mapping) and row.get("matches") is True
        }
        source_config = load_json(source / "bound_experiment_config.json")
        group_keys = (
            "group_views",
            "group_sizes",
            "exact_non_singleton_group_sizes",
            "max_group_size",
            "all_singletons",
            "prompt_information_policy",
        )
        if any(source_config.get(key) != self.experiment_config.get(key) for key in group_keys):
            raise ValueError("Router-v4 candidate-generation configuration differs")
        source_llm = load_json(source / "bound_llm_config.json")
        for key in ("model", "prompt_schema_version", "contexts"):
            if source_llm.get(key) != self.llm_config.get(key):
                raise ValueError(f"Router-v4 group source LLM configuration differs: {key}")

        artifacts: list[dict[str, object]] = []
        for suite, dataset in dataset_order:
            key = _dataset_key(suite, dataset)
            relative_paths = (
                Path("cell_features") / f"{key}.csv",
                Path("groups") / "candidates" / f"{key}.jsonl",
                Path("groups") / "memberships" / f"{key}.csv",
            )
            for relative in relative_paths:
                expected = identity_hashes.get(relative.as_posix())
                source_path = source / relative
                if expected is None or sha256_file(source_path) != expected:
                    raise ValueError(f"Router-v4 frozen group source drift: {relative}")
                destination = self.paths.run_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.is_file() or sha256_file(destination) != expected:
                    shutil.copyfile(source_path, destination)
                if sha256_file(destination) != expected:
                    raise ValueError(f"Router-v4 copied group artifact drift: {relative}")
                artifacts.append(
                    {"artifact": relative.as_posix(), "sha256": expected}
                )
            audit_relative = (
                Path("groups") / "generation_audits" / f"{key}.json"
            )
            audit_source = source / audit_relative
            audit_destination = self.paths.run_dir / audit_relative
            audit_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(audit_source, audit_destination)
            artifacts.append(
                {
                    "artifact": audit_relative.as_posix(),
                    "sha256": sha256_file(audit_destination),
                }
            )
        summary: dict[str, object] = {
            "mode": "strict_request_identical_group_artifact_reuse",
            "source_run": str(source),
            "source_manifest_sha256": sha256_file(source / "run_manifest.json"),
            "source_reuse_manifest_sha256": sha256_file(
                source / "provenance" / "reuse_manifest.json"
            ),
            "datasets": len(dataset_order),
            "artifacts": artifacts,
            "historical_gate_predictions_reused": False,
            "target_labels_or_responses_used": False,
        }
        write_json(provenance_path, summary)
        result = super().generate_groups_stage(dataset_order)
        result["reused_from_frozen_source"] = True
        result["reused_artifacts"] = len(artifacts)
        return result

    def _import_calibration_source(self) -> dict[str, object]:
        self._validate_calibration_configuration()
        if self.calibration_source_run is None:
            raise ValueError("Router-v4 requires a calibration source run")
        provenance_path = self.paths.run_dir / "provenance" / "calibration.json"
        if provenance_path.is_file():
            provenance = load_json(provenance_path)
            for relative, hash_field in (
                ("llm/calibration_queries.jsonl", "calibration_queries_sha256"),
                ("llm/calibration_execution.jsonl", "calibration_execution_sha256"),
                ("llm/calibration_pair_labels.csv", "calibration_pair_labels_sha256"),
            ):
                destination = self.paths.run_dir / relative
                if sha256_file(destination) != str(provenance[hash_field]):
                    raise ValueError(f"imported calibration artifact drift: {relative}")
            return provenance

        source = self.calibration_source_run
        source_manifest_path = source / "run_manifest.json"
        if sha256_file(source_manifest_path) != str(
            self.state.manifest.get("calibration_source_manifest_sha256", "")
        ):
            raise ValueError("calibration source manifest drift")
        source_manifest = load_json(source_manifest_path)
        if (
            source_manifest.get("status") != "complete"
            or str(source_manifest.get("model", "")) != str(self.llm_config["model"])
            or str(source_manifest.get("prompt_schema_sha256", ""))
            != str(self.state.manifest.get("prompt_schema_sha256", ""))
            or str(source_manifest.get("data_content_fingerprint", ""))
            != str(self.state.manifest.get("data_content_fingerprint", ""))
        ):
            raise ValueError("calibration source identity differs from Router-v4 run")

        planned = read_jsonl(self.paths.llm_dir / "calibration_queries.jsonl")
        source_planned = read_jsonl(source / "llm" / "calibration_queries.jsonl")
        if len(planned) != EXPECTED_CALIBRATION_QUERIES or planned != source_planned:
            raise ValueError("calibration query plan is not request-identical")
        source_execution = read_jsonl(source / "llm" / "calibration_execution.jsonl")
        source_labels = _read_csv(source / "llm" / "calibration_pair_labels.csv")
        planned_ids = {
            (str(row["query_id"]), str(row["prompt_hash"])) for row in planned
        }
        executed_ids = {
            (str(row.get("query_id", "")), str(row.get("prompt_hash", "")))
            for row in source_execution
        }
        label_ids = set(
            zip(
                source_labels["cell_id"].astype(str),
                source_labels["query_id"].astype(str),
            )
        )
        if (
            len(source_execution) != EXPECTED_CALIBRATION_QUERIES
            or len(executed_ids) != len(source_execution)
            or executed_ids != planned_ids
            or len(source_labels) != EXPECTED_CALIBRATION_PAIR_LABELS
            or len(label_ids) != len(source_labels)
        ):
            raise ValueError("calibration source coverage or uniqueness differs")
        if any(
            row.get("model_matches_request", True) is False
            or str(row.get("model", "")) != str(self.llm_config["model"])
            for row in source_execution
            if row.get("status") == "success"
        ):
            raise ValueError("calibration source execution model differs")

        execution_path = self.paths.llm_dir / "calibration_execution.jsonl"
        labels_path = self.paths.llm_dir / "calibration_pair_labels.csv"
        execution_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "llm" / "calibration_execution.jsonl", execution_path)
        shutil.copyfile(source / "llm" / "calibration_pair_labels.csv", labels_path)
        fallback_source = (
            source / "llm" / "offline_group_calibration_baran_fallbacks.jsonl"
        )
        fallback_destination = (
            self.paths.llm_dir / "offline_group_calibration_baran_fallbacks.jsonl"
        )
        if fallback_source.is_file():
            shutil.copyfile(fallback_source, fallback_destination)
        else:
            write_jsonl(fallback_destination, ())
        provenance: dict[str, object] = {
            "mode": "strict_frozen_source_import",
            "source_run": str(source),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "calibration_queries": len(planned),
            "calibration_executions": len(source_execution),
            "calibration_pair_labels": len(source_labels),
            "calibration_queries_sha256": sha256_file(
                self.paths.llm_dir / "calibration_queries.jsonl"
            ),
            "calibration_execution_sha256": sha256_file(execution_path),
            "calibration_pair_labels_sha256": sha256_file(labels_path),
            "request_identical": True,
            "target_labels_or_responses_used_before_selection": False,
            "logical_cost_preserved": True,
        }
        write_json(provenance_path, provenance)
        self.state.update_stage(
            "calibration_llm",
            "complete",
            queries=len(source_execution),
            pair_labels=len(source_labels),
            reused_from_parent=True,
            parent_run=str(source),
        )
        return provenance

    def plan_run(self) -> dict[str, object]:
        self.validate_inputs()
        self.run_baran_stage()
        self.generate_groups_stage()
        reuse = self.import_reusable_no_baran_responses_stage()
        plan = self.plan_calibration_stage()
        calibration = self._import_calibration_source()
        return {
            "run_dir": str(self.paths.run_dir),
            "data": {
                "generation_datasets": len(generation_order()),
                "test_datasets": len(target_order()),
                "test_oracle_cells": TEST_TARGET_CELL_COUNT,
                "calibration_datasets": 9,
                "calibration_singletons": CALIBRATION_SINGLETON_CELL_COUNT,
            },
            "calibration": plan,
            "calibration_source": calibration,
            "response_reuse": reuse,
            "api_called": False,
        }

    def run_calibration_stage(self) -> dict[str, object]:
        if not self.state.stage_completed("calibration_llm"):
            return self._import_calibration_source()
        return {
            "queries": len(read_jsonl(self.paths.llm_dir / "calibration_execution.jsonl")),
            "pair_labels": len(_read_csv(self.paths.llm_dir / "calibration_pair_labels.csv")),
            "reused_from_parent": True,
        }

    def _validated_calibration_labels(self) -> pd.DataFrame:
        planned_rows = read_jsonl(self.paths.llm_dir / "calibration_queries.jsonl")
        executed_rows = read_jsonl(self.paths.llm_dir / "calibration_execution.jsonl")
        planned = {
            (str(row["query_id"]), str(row["prompt_hash"])) for row in planned_rows
        }
        executed = {
            (str(row.get("query_id", "")), str(row.get("prompt_hash", "")))
            for row in executed_rows
        }
        if (
            len(planned_rows) != EXPECTED_CALIBRATION_QUERIES
            or len(executed_rows) != EXPECTED_CALIBRATION_QUERIES
            or len(planned) != len(planned_rows)
            or len(executed) != len(executed_rows)
            or executed != planned
        ):
            raise ValueError("calibration execution ledger coverage or uniqueness failed")
        labels = _read_csv(self.paths.llm_dir / "calibration_pair_labels.csv")
        required = {
            "cell_id",
            "query_id",
            "dataset",
            "baran_correct",
            "llm_correct_in_query",
            "executable_propose",
            "helpful",
            "harmful",
        }
        if required - set(labels.columns):
            raise ValueError("calibration labels lack required columns")
        if (
            len(labels) != EXPECTED_CALIBRATION_PAIR_LABELS
            or labels.duplicated(["cell_id", "query_id"]).any()
        ):
            raise ValueError("calibration labels coverage or uniqueness failed")
        return labels.loc[
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

    def _gate_input_hashes(self) -> dict[str, object]:
        membership = {
            f"groups/memberships/{_dataset_key(suite, dataset)}.csv": sha256_file(
                self._membership_path(suite, dataset)
            )
            for suite, dataset in generation_order()
        }
        return {
            "bound_experiment_config_sha256": sha256_file(
                self.paths.run_dir / "bound_experiment_config.json"
            ),
            "calibration_pair_labels_sha256": sha256_file(
                self.paths.llm_dir / "calibration_pair_labels.csv"
            ),
            "model_feature_columns_sha256": canonical_json_sha256(
                list(MODEL_FEATURE_COLUMNS)
            ),
            "membership_sha256": membership,
        }

    def _fit_and_freeze_gates(
        self,
        all_pairs: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        split_rows: list[dict[str, object]] = []
        calibration_rows: list[dict[str, object]] = []
        reliability_rows: list[dict[str, object]] = []
        artifacts: list[dict[str, object]] = []
        input_hashes = self._gate_input_hashes()
        for suite, dataset in target_order():
            train_safe, test_safe, audit = split_for_target(
                all_pairs, suite, dataset, enforce_target_unlabeled=True
            )
            train_all = train_safe.merge(
                labels,
                how="inner",
                on=["cell_id", "query_id"],
                validate="one_to_one",
            )
            if train_all.empty:
                raise ValueError(f"no calibration labels for {suite}/{dataset}")
            actions = {
                action.query_id: action
                for action in self._load_actions(suite, dataset)
            }
            if len(test_safe) != sum(action.group_size for action in actions.values()):
                raise ValueError(f"target pair coverage differs for {suite}/{dataset}")
            for variant, allowed_sizes in self._router_training_variants().items():
                train = self._filter_variant_pairs(
                    train_all,
                    allowed_sizes,
                    context=f"v4 train {variant}/{suite}/{dataset}",
                )
                test = self._filter_variant_pairs(
                    test_safe,
                    allowed_sizes,
                    context=f"v4 test {variant}/{suite}/{dataset}",
                )
                families = [base_family(value) for value in train["dataset"].astype(str)]
                gate = GroupUpliftGate(
                    "lightgbm",
                    rho=float(self.experiment_config["harm_penalty_rho"]),
                    gamma=float(self.experiment_config["uncertainty_penalty_gamma"]),
                    random_state=int(self.experiment_config["seed"]),
                    probability_calibration="isotonic",
                ).fit(
                    train.loc[:, list(MODEL_FEATURE_COLUMNS)].to_dict("records"),
                    [bool(int(value)) for value in train["baran_correct"]],
                    [bool(int(value)) for value in train["llm_correct_in_query"]],
                    [bool(int(value)) for value in train["executable_propose"]],
                    families,
                )
                audited = gate.predict_audited(
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
                for field in audited[0].as_dict():
                    predictions[field] = [value.as_dict()[field] for value in audited]
                if bool((pd.to_numeric(predictions["conservative_uplift"]) < 0).any()):
                    raise ValueError("calibrated conservative uplift must be non-negative")
                prediction_path = self._prediction_path(
                    "lightgbm", variant, suite, dataset
                )
                _write_csv(prediction_path, predictions.to_dict("records"))

                oof = gate.calibration_oof_predictions()
                if len(oof) != len(train) or {value.row_index for value in oof} != set(
                    range(len(train))
                ):
                    raise ValueError("OOF calibration does not uniquely cover train rows")
                oof_families = {value.family for value in oof}
                if oof_families != set(families):
                    raise ValueError("OOF calibration family coverage differs")
                for head in ("helpful", "harmful"):
                    y = [int(getattr(value, head)) for value in oof]
                    raw_p = [
                        float(getattr(value, f"raw_q_{head}_oof")) for value in oof
                    ]
                    calibrated_p = [
                        float(getattr(value, f"q_{head}_oof")) for value in oof
                    ]
                    raw_metrics, raw_bins = _probability_diagnostics(y, raw_p)
                    calibrated_metrics, calibrated_bins = _probability_diagnostics(
                        y, calibrated_p
                    )
                    row: dict[str, object] = {
                        "target_suite": suite,
                        "target_dataset": dataset,
                        "backend": "lightgbm",
                        "group_size_variant": variant,
                        "head": head,
                        "oof_rows": len(oof),
                        "oof_families": len(oof_families),
                    }
                    for prefix, values in (
                        ("raw", raw_metrics),
                        ("calibrated", calibrated_metrics),
                    ):
                        row.update({f"{prefix}_{key}": value for key, value in values.items()})
                    calibration_rows.append(row)
                    for scale, bins in (("raw", raw_bins), ("calibrated", calibrated_bins)):
                        reliability_rows.extend(
                            {
                                "target_suite": suite,
                                "target_dataset": dataset,
                                "backend": "lightgbm",
                                "group_size_variant": variant,
                                "head": head,
                                "probability_scale": scale,
                                **bin_row,
                            }
                            for bin_row in bins
                        )

                metadata_path = prediction_path.with_suffix(".metadata.json")
                metadata: dict[str, object] = {
                    "router_revision": ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION,
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
                    "input_hashes": input_hashes,
                    "target_labels_used": False,
                    "target_responses_used_before_selection": False,
                    "oof_unique_coverage": True,
                    "oof_replica_excluded_own_family": True,
                }
                write_json(metadata_path, metadata)
                split_rows.append(
                    {
                        **audit.as_dict(),
                        "backend": "lightgbm",
                        "group_size_variant": variant,
                        "allowed_group_sizes": ",".join(str(value) for value in allowed_sizes),
                        "train_test_row_overlap": audit.train_test_row_identity_overlap,
                        "train_pair_rows_after_sampling": len(train),
                        "test_pair_rows": len(test),
                        "target_group_label_used": False,
                        "target_response_used_before_selection": False,
                        "target_response_visible_before_selection": False,
                        "probability_calibration": "isotonic",
                    }
                )
                artifacts.append(
                    {
                        "backend": "lightgbm",
                        "variant": variant,
                        "suite": suite,
                        "dataset": dataset,
                        "prediction": prediction_path.relative_to(
                            self.paths.run_dir
                        ).as_posix(),
                        "prediction_sha256": sha256_file(prediction_path),
                        "metadata": metadata_path.relative_to(
                            self.paths.run_dir
                        ).as_posix(),
                        "metadata_sha256": sha256_file(metadata_path),
                    }
                )
                print(
                    f"[gate-v4] frozen {suite}/{dataset} k={variant}: "
                    f"train={len(train)}, test={len(test)}",
                    flush=True,
                )

        if len(artifacts) != 18 or len(split_rows) != 18 or len(calibration_rows) != 36:
            raise ValueError("Router-v4 gate matrix is incomplete")
        split_path = self.paths.gates_dir / "split_audit.csv"
        calibration_path = self.paths.gates_dir / "calibration_audit.csv"
        calibration_metrics_path = self.paths.metrics_dir / "calibration_by_dataset.csv"
        reliability_path = self.paths.metrics_dir / "calibration_reliability_bins.csv"
        _write_csv(split_path, split_rows)
        _write_csv(calibration_path, calibration_rows)
        _write_csv(calibration_metrics_path, calibration_rows)
        _write_csv(reliability_path, reliability_rows)
        for path in (split_path, calibration_path):
            artifacts.append(
                {
                    "artifact": path.relative_to(self.paths.run_dir).as_posix(),
                    "sha256": sha256_file(path),
                }
            )
        manifest: dict[str, object] = {
            "schema_version": "router-v4-frozen-score-ledger-v1",
            "router_revision": self.router_revision,
            "gate_folds": 18,
            "selection_slices_at_freeze": 0,
            "optimizer_input_column": "conservative_uplift",
            "scores_nonnegative": True,
            "target_labels_or_responses_used_before_selection": False,
            "input_hashes": input_hashes,
            "artifacts": artifacts,
        }
        manifest["content_sha256"] = canonical_json_sha256(manifest)
        write_json(self.paths.gates_dir / "frozen_score_ledger_manifest.json", manifest)
        return split_rows, calibration_rows

    def _verify_frozen_gate_manifest(self) -> dict[str, object]:
        path = self.paths.gates_dir / "frozen_score_ledger_manifest.json"
        manifest = load_json(path)
        stored = str(manifest.pop("content_sha256", ""))
        if canonical_json_sha256(manifest) != stored:
            raise ValueError("frozen score-ledger manifest content drift")
        manifest["content_sha256"] = stored
        for row in manifest.get("artifacts", []):
            if not isinstance(row, Mapping):
                raise ValueError("frozen score-ledger artifact row is invalid")
            relative = str(row.get("prediction") or row.get("artifact") or "")
            expected = str(row.get("prediction_sha256") or row.get("sha256") or "")
            if sha256_file(self.paths.run_dir / relative) != expected:
                raise ValueError(f"frozen score-ledger artifact drift: {relative}")
            metadata = str(row.get("metadata") or "")
            if metadata and sha256_file(self.paths.run_dir / metadata) != str(
                row.get("metadata_sha256", "")
            ):
                raise ValueError(f"frozen gate metadata drift: {metadata}")
        return manifest

    def train_and_select_stage(self) -> dict[str, object]:
        """Freeze all calibrated scores before any one of the 18 selections."""

        if not self.state.stage_completed("calibration_llm"):
            raise RuntimeError("Router-v4 training requires imported calibration data")
        labels = self._validated_calibration_labels()
        all_pairs = self._all_pair_features()
        freeze_path = self.paths.gates_dir / "frozen_score_ledger_manifest.json"
        selection_audit_path = self.paths.metrics_dir / "selection_audit.csv"
        if freeze_path.is_file():
            freeze = self._verify_frozen_gate_manifest()
            if selection_audit_path.is_file():
                expected = 18
                if len(_read_csv(selection_audit_path)) != expected:
                    raise ValueError("existing Router-v4 selection matrix is incomplete")
        else:
            self._fit_and_freeze_gates(all_pairs, labels)
            freeze = self._verify_frozen_gate_manifest()
        if int(freeze.get("gate_folds", -1)) != 18:
            raise ValueError("Router-v4 score ledger must freeze exactly 18 folds")

        variants = self._router_training_variants()
        budget_share = self._router_budget_shares()[0]
        selection_rows: list[dict[str, object]] = []
        logical_rows: list[dict[str, object]] = []
        prediction_parts: list[pd.DataFrame] = []
        bgr_union_ids: set[str] = set()
        llm_only_ids: set[str] = set()
        freeze_hash = sha256_file(freeze_path)
        for suite, dataset in target_order():
            _, test_safe, _ = split_for_target(
                all_pairs, suite, dataset, enforce_target_unlabeled=True
            )
            actions = {
                action.query_id: action
                for action in self._load_actions(suite, dataset)
            }
            reference_cost = self._singleton_reference_cost(test_safe)
            dataset_singletons = {
                query_id
                for query_id, action in actions.items()
                if action.group_size == 1 and action.group_view == "singleton"
            }
            if len(dataset_singletons) != len(self._dataset(suite, dataset).safe_cells()):
                raise ValueError(f"LLM-only singleton coverage differs for {suite}/{dataset}")
            llm_only_ids.update(dataset_singletons)
            costs = {
                query_id: float(action.estimated_total_tokens)
                for query_id, action in actions.items()
            }
            for variant, allowed_sizes in variants.items():
                allowed = set(allowed_sizes)
                predictions = _read_csv(
                    self._prediction_path("lightgbm", variant, suite, dataset)
                )
                test = self._filter_variant_pairs(
                    test_safe,
                    allowed_sizes,
                    context=f"v4 selection {variant}/{suite}/{dataset}",
                )
                if len(predictions) != len(test):
                    raise ValueError("frozen prediction/test pair coverage differs")
                if set(pd.to_numeric(predictions["group_size"], errors="raise")) != allowed:
                    raise ValueError("frozen prediction group sizes differ")
                if set(predictions["probability_calibration"].astype(str)) != {"isotonic"}:
                    raise ValueError("frozen predictions are not isotonic calibrated")
                prediction_parts.append(predictions)
                candidates = tuple(
                    sorted(
                        query_id
                        for query_id, action in actions.items()
                        if action.group_size in allowed
                    )
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
                budget = int(round(reference_cost * budget_share))
                result = select_queries(objective, costs, budget, candidates=candidates)
                if result.total_cost > budget + 1e-9:
                    raise AssertionError("Router-v4 selection exceeded token budget")
                selected_ids = tuple(result.selected_query_ids)
                if any(actions[value].group_size not in allowed for value in selected_ids):
                    raise AssertionError("Router-v4 selected a disallowed group size")
                bgr_union_ids.update(selected_ids)
                covered = [
                    cell_id
                    for query_id in selected_ids
                    for cell_id in actions[query_id].cell_ids
                ]
                selection_path = self._selection_path(
                    "lightgbm",
                    "size_conditioned",
                    variant,
                    budget_share,
                    suite,
                    dataset,
                )
                write_json(
                    selection_path,
                    {
                        **result.as_dict(),
                        "router_revision": self.router_revision,
                        "suite": suite,
                        "dataset": dataset,
                        "backend": "lightgbm",
                        "scenario": "size_conditioned",
                        "group_size_variant": variant,
                        "training_group_sizes": list(allowed_sizes),
                        "allowed_group_sizes": list(allowed_sizes),
                        "budget_share": budget_share,
                        "budget_reference_tokens": reference_cost,
                        "selected_cell_incidence": len(covered),
                        "unique_covered_cells": len(set(covered)),
                        "score_ledger_manifest_sha256": freeze_hash,
                        "optimizer_input_column": "conservative_uplift",
                    },
                )
                selection_rows.append(
                    {
                        "suite": suite,
                        "dataset": dataset,
                        "backend": "lightgbm",
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
                        "score_ledger_manifest_sha256": freeze_hash,
                    }
                )
                for query_id in selected_ids:
                    action = actions[query_id]
                    logical_rows.append(
                        {
                            "target_suite": suite,
                            "target_dataset": dataset,
                            "backend": "lightgbm",
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

        if len(selection_rows) != 18:
            raise ValueError("Router-v4 must produce exactly 18 selections")
        _write_csv(selection_audit_path, selection_rows)
        _write_csv(
            self.paths.gates_dir / "lightgbm_pair_predictions.csv",
            pd.concat(prediction_parts, ignore_index=True).to_dict("records"),
        )
        union_ids = bgr_union_ids | llm_only_ids
        union_actions = {
            action.query_id: action
            for suite, dataset in target_order()
            for action in self._load_actions(suite, dataset)
            if action.query_id in union_ids
        }
        if set(union_actions) != union_ids:
            raise ValueError("Router-v4 selected union action coverage differs")
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
        terminal_ids = cached_success_ids | cached_failure_ids
        online_ids = sorted(union_ids - terminal_ids)
        online_id_set = set(online_ids)
        online_estimate = sum(
            union_actions[query_id].estimated_total_tokens for query_id in online_ids
        )
        preflight_estimate = 1_722
        preflight_path = self.paths.llm_dir / "model_preflight.json"
        if preflight_path.is_file():
            preflight_estimate = int(
                load_json(preflight_path).get("estimated_total_tokens", 0) or 0
            )
        calibration_estimate = int(
            load_json(self.paths.llm_dir / "calibration_plan.json").get(
                "estimated_tokens", 0
            )
            or 0
        )
        retry_attempt_ceiling = int(self.llm_config.get("max_retries", 0)) + 1
        online_scope: list[dict[str, object]] = []
        for suite, dataset in target_order():
            scoped = [
                union_actions[query_id]
                for query_id in online_ids
                if union_actions[query_id].suite == suite
                and union_actions[query_id].dataset == dataset
            ]
            online_scope.append(
                {
                    "suite": suite,
                    "dataset": dataset,
                    "physical_queries": len(scoped),
                    "cell_incidences": sum(action.group_size for action in scoped),
                    "estimated_tokens": sum(
                        action.estimated_total_tokens for action in scoped
                    ),
                    "group_sizes": {
                        str(size): sum(action.group_size == size for action in scoped)
                        for size in (1, 2, 4, 8)
                    },
                }
            )
        union_plan: dict[str, object] = {
            "router_revision": self.router_revision,
            "selection_slices": len(selection_rows),
            "model_folds": 18,
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
            "combined_physical_estimated_tokens": online_estimate + preflight_estimate,
            "retry_attempt_ceiling": retry_attempt_ceiling,
            "conservative_retry_adjusted_token_ceiling": retry_attempt_ceiling
            * (online_estimate + preflight_estimate),
            "online_external_scope_by_dataset": online_scope,
            "score_ledger_manifest_sha256": freeze_hash,
            "query_ids": sorted(union_ids),
            "bgr_query_ids": sorted(bgr_union_ids),
            "llm_only_query_ids": sorted(llm_only_ids),
            "online_query_ids": online_ids,
            "cached_failure_query_ids": sorted(cached_failure_ids),
        }
        write_json(self.paths.llm_dir / "selected_union_plan.json", union_plan)
        write_json(
            self.paths.llm_dir / "router_v4_isotonic_dry_plan.json",
            {
                **union_plan,
                "backends": ["lightgbm"],
                "variants": list(variants),
                "budget_shares": [budget_share],
                "selection_summary": selection_rows,
                "api_called": False,
            },
        )
        for row in logical_rows:
            response = response_index.get((str(row["query_id"]), str(row["prompt_hash"])))
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
        }
        self.state.update_stage("gate_selection", "complete", **stage_summary)
        return {
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
        }

    def build_router_diagnostics_stage(
        self,
        all_pairs: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> dict[str, object]:
        """Bind held-out TableEG labels only after the score ledger is frozen."""

        freeze = self._verify_frozen_gate_manifest()
        selection_path = self.paths.metrics_dir / "selection_audit.csv"
        if not selection_path.is_file() or len(_read_csv(selection_path)) != 18:
            raise RuntimeError("routeability diagnostics require 18 frozen selections")
        rows: list[dict[str, object]] = []
        tableeg_targets = sorted(
            dataset for suite, dataset in generation_order() if suite == "tableeg"
        )
        for variant, allowed_sizes in self._router_training_variants().items():
            for dataset in tableeg_targets:
                train_safe, target_safe, _ = split_for_target(
                    all_pairs,
                    "tableeg",
                    dataset,
                    enforce_target_unlabeled=True,
                )
                train_safe = self._filter_variant_pairs(
                    train_safe,
                    allowed_sizes,
                    context=f"v4 diagnostic train {variant}/tableeg/{dataset}",
                )
                target_safe = self._filter_variant_pairs(
                    target_safe,
                    allowed_sizes,
                    context=f"v4 diagnostic target {variant}/tableeg/{dataset}",
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
                        f"empty v4 diagnostic fold for {variant}/tableeg/{dataset}"
                    )
                gate = GroupUpliftGate(
                    "lightgbm",
                    rho=float(self.experiment_config["harm_penalty_rho"]),
                    gamma=float(self.experiment_config["uncertainty_penalty_gamma"]),
                    random_state=int(self.experiment_config["seed"]),
                    probability_calibration="isotonic",
                ).fit(
                    train.loc[:, list(MODEL_FEATURE_COLUMNS)].to_dict("records"),
                    [bool(int(value)) for value in train["baran_correct"]],
                    [bool(int(value)) for value in train["llm_correct_in_query"]],
                    [bool(int(value)) for value in train["executable_propose"]],
                    [base_family(value) for value in train["dataset"].astype(str)],
                )
                predicted = gate.predict_audited(
                    target.loc[:, list(MODEL_FEATURE_COLUMNS)].to_dict("records")
                )
                row: dict[str, object] = {
                    "backend": "lightgbm",
                    "group_size_variant": variant,
                    "allowed_group_sizes": ",".join(
                        str(value) for value in allowed_sizes
                    ),
                    "target_suite": "tableeg",
                    "target_dataset": dataset,
                    "train_pairs": len(train),
                    "test_pairs": len(target),
                    "score_ledger_manifest_sha256": str(freeze["content_sha256"]),
                    "selection_frozen_before_target_labels": True,
                    "diagnostic_only": True,
                }
                for head in ("helpful", "harmful"):
                    y = [int(value) for value in target[head]]
                    raw = [float(getattr(value, f"raw_q_{head}")) for value in predicted]
                    calibrated = [float(getattr(value, f"q_{head}")) for value in predicted]
                    raw_metrics, _ = _probability_diagnostics(y, raw)
                    calibrated_metrics, _ = _probability_diagnostics(y, calibrated)
                    for field in ("average_precision", "brier", "ece"):
                        row[f"raw_{head}_{field}"] = raw_metrics[field]
                        row[f"calibrated_{head}_{field}"] = calibrated_metrics[field]
                ranked_raw = sorted(
                    range(len(predicted)),
                    key=lambda index: (
                        -float(predicted[index].raw_net_gain),
                        str(target.iloc[index]["query_id"]),
                        str(target.iloc[index]["cell_id"]),
                    ),
                )
                ranked_calibrated = sorted(
                    range(len(predicted)),
                    key=lambda index: (
                        -float(predicted[index].conservative_uplift),
                        str(target.iloc[index]["query_id"]),
                        str(target.iloc[index]["cell_id"]),
                    ),
                )
                top_count = max(1, math.ceil(0.1 * len(predicted)))
                observed = [
                    int(target.iloc[index]["helpful"])
                    - int(target.iloc[index]["harmful"])
                    for index in range(len(target))
                ]
                row["top_ranked_pairs"] = top_count
                row["raw_top_ranked_observed_uplift"] = statistics.fmean(
                    observed[index] for index in ranked_raw[:top_count]
                )
                row["calibrated_top_ranked_observed_uplift"] = statistics.fmean(
                    observed[index] for index in ranked_calibrated[:top_count]
                )
                rows.append(row)
        if len(rows) != 18:
            raise ValueError("Router-v4 routeability diagnostics must contain 18 folds")
        _write_csv(self.paths.metrics_dir / "routeability_by_dataset.csv", rows)
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["group_size_variant"])].append(row)
        macro = {
            variant: {
                field: statistics.fmean(float(row[field]) for row in values)
                for field in (
                    "raw_helpful_average_precision",
                    "calibrated_helpful_average_precision",
                    "raw_helpful_brier",
                    "calibrated_helpful_brier",
                    "raw_harmful_average_precision",
                    "calibrated_harmful_average_precision",
                    "raw_harmful_brier",
                    "calibrated_harmful_brier",
                    "raw_top_ranked_observed_uplift",
                    "calibrated_top_ranked_observed_uplift",
                )
            }
            for variant, values in sorted(grouped.items())
        }
        summary: dict[str, object] = {
            "router_revision": self.router_revision,
            "diagnostic_only": True,
            "folds": len(rows),
            "tableeg_datasets": 9,
            "variants": list(self._router_training_variants()),
            "selection_frozen_before_target_labels": True,
            "target_labels_or_responses_used_before_selection": False,
            "macro": macro,
        }
        write_json(self.paths.metrics_dir / "routeability_summary.json", summary)
        self.state.update_stage("router_diagnostics", "complete", **summary)
        return summary

    def run_selected_llm_stage(self) -> dict[str, object]:
        if not bool(getattr(self, "cache_only", False)):
            if not self.state.stage_completed("model_preflight"):
                self.check_model()
            return super().run_selected_llm_stage()
        plan = load_json(self.paths.llm_dir / "selected_union_plan.json")
        missing = int(plan.get("online_physical_queries", -1))
        if missing != 0:
            raise RuntimeError(
                "--cache-only requires dry plan online_physical_queries == 0; "
                f"observed {missing}"
            )
        union_ids = {str(value) for value in plan.get("query_ids", [])}
        actions = tuple(
            sorted(
                (
                    action
                    for suite, dataset in target_order()
                    for action in self._load_actions(suite, dataset)
                    if action.query_id in union_ids
                ),
                key=lambda value: value.query_id,
            )
        )
        if {action.query_id for action in actions} != union_ids:
            raise ValueError("cache-only selected union references missing actions")
        response_index = self._response_index()
        results: list[dict[str, object]] = []
        for action in actions:
            response = response_index.get((action.query_id, action.prompt_hash))
            if response is None:
                raise RuntimeError("cache-only selected response disappeared after dry plan")
            copied = dict(response)
            copied["cache_hit"] = True
            copied["checkpoint_hit"] = True
            results.append(copied)
        write_jsonl(self.paths.llm_dir / "selected_execution.jsonl", results)
        if not self.state.stage_completed("model_preflight"):
            receipt = {
                "mode": "cache_only",
                "provider_calls": 0,
                "requested_model": str(self.llm_config["model"]),
                "schema_covered_by_request_identical_cache": True,
                "estimated_total_tokens": 0,
            }
            write_json(self.paths.llm_dir / "model_preflight.json", receipt)
            self.state.update_stage("model_preflight", "complete", **receipt)
        fallback_summary = self._materialize_baran_fallbacks(
            actions, results, phase="online_selected_union"
        )
        summary = {
            "union_queries": len(actions),
            "successful_queries": sum(row.get("status") == "success" for row in results),
            "failed_queries": sum(row.get("status") != "success" for row in results),
            "checkpoint_hits": len(results),
            "cache_hits": len(results),
            "physical_api_calls": 0,
            "cache_only": True,
            **fallback_summary,
        }
        if int(summary["unresolved_operational_failures"]) > 0:
            raise RuntimeError("cache-only response fallbacks have missing Baran coverage")
        self.state.update_stage("selected_llm", "complete", **summary)
        return summary

    def _historical_raw_records(self) -> list[dict[str, object]]:
        if self.router_comparison_run is None:
            raise ValueError("Router-v4 requires --router-comparison-run")
        source = self.router_comparison_run
        for supplied in (
            self.calibration_source_run,
            self.baran_source_run,
            self.response_reuse_run,
        ):
            if supplied is None or supplied.resolve() != source.resolve():
                raise ValueError("all four Router-v4 source runs must be the frozen historical run")
        observed = {
            relative: sha256_file(source / relative)
            for relative in HISTORICAL_COMPARATOR_HASHES
        }
        if observed != HISTORICAL_COMPARATOR_HASHES:
            raise ValueError("historical raw comparator content hash drift")
        addendum = source / "provenance" / "completion_validation_addendum.json"
        reuse = source / "provenance" / "reuse_manifest.json"
        if not addendum.is_file() or not reuse.is_file():
            raise FileNotFoundError("historical comparator provenance is incomplete")
        records = [
            dict(row)
            for row in read_jsonl(source / "final" / "all_methods.jsonl")
            if str(row.get("method", "")) == "budgeted_group_lightgbm"
            and str(row.get("scenario", "")) == "size_conditioned"
            and str(row.get("group_size_variant", "")) in {"1", "4"}
            and math.isclose(float(row.get("budget_share") or 0.0), 0.2, abs_tol=1e-12)
        ]
        if len(records) != 2 * TEST_TARGET_CELL_COUNT:
            raise ValueError("historical raw comparator final matrix is incomplete")
        summaries = [
            row
            for row in summarize_records(records, strict=True)
            if str(row["scope"]) == "dataset"
        ]
        expected = {
            ("source", "beers", "1"): (0.8709, 0.8709, 0.8709),
            ("source", "flights", "1"): (0.9984, 0.9984, 0.9984),
            ("source", "hospital", "1"): (0.9421, 0.9273, 0.9347),
            ("source", "movies_1", "1"): (0.9042, 0.7609, 0.8264),
            ("source", "rayyan", "1"): (0.8236, 0.5222, 0.6391),
            ("tableeg", "company", "1"): (0.6392, 0.5699, 0.6026),
            ("tableeg", "marketing", "1"): (0.6334, 0.6141, 0.6236),
            ("tableeg", "restaurant_20", "1"): (0.6466, 0.1238, 0.2078),
            ("tableeg", "soccer", "1"): (0.9060, 0.8800, 0.8928),
            ("source", "beers", "4"): (0.8744, 0.8744, 0.8744),
            ("source", "flights", "4"): (0.9986, 0.9986, 0.9986),
            ("source", "hospital", "4"): (0.9360, 0.9194, 0.9277),
            ("source", "movies_1", "4"): (0.8944, 0.7670, 0.8258),
            ("source", "rayyan", "4"): (0.7923, 0.5190, 0.6272),
            ("tableeg", "company", "4"): (0.6540, 0.6014, 0.6266),
            ("tableeg", "marketing", "4"): (0.6419, 0.6224, 0.6320),
            ("tableeg", "restaurant_20", "4"): (0.6352, 0.1667, 0.2641),
            ("tableeg", "soccer", "4"): (0.9126, 0.9039, 0.9082),
        }
        if len(summaries) != 18:
            raise ValueError("historical comparator has an unexpected dataset matrix")
        for row in summaries:
            key = (
                str(row["suite"]),
                str(row["dataset"]),
                str(row["group_size_variant"]),
            )
            actual = tuple(round(float(row[field]), 4) for field in ("precision", "recall", "f1"))
            if expected.get(key) != actual:
                raise ValueError(f"historical raw metric drift: {key} {actual}")
        write_json(
            self.paths.run_dir / "provenance" / "historical_raw_comparator.json",
            {
                "source_run": str(source),
                "frozen_hashes": observed,
                "completion_addendum_sha256": sha256_file(addendum),
                "calibration_reuse_manifest_sha256": sha256_file(reuse),
                "raw_records": len(records),
                "dataset_variant_slices": len(summaries),
                "independently_recomputed": True,
                "copied_into_final_ledger": False,
                "comparison_only": True,
            },
        )
        return records

    @staticmethod
    def _record_metric_arrays(
        records: Sequence[Mapping[str, object]],
        cell_ids: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        by_id = {str(row["cell_id"]): row for row in records}
        if set(by_id) != set(cell_ids) or len(by_id) != len(records):
            raise ValueError("paired metric record coverage differs")
        valid = np.asarray(
            [str(by_id[cell_id].get("parse_status", "")).startswith("ok") for cell_id in cell_ids],
            dtype=float,
        )
        correct = np.asarray(
            [bool(by_id[cell_id].get("correct_repair")) for cell_id in cell_ids],
            dtype=float,
        )
        return valid, correct

    def _raw_calibrated_bootstrap(
        self,
        raw_records: Sequence[Mapping[str, object]],
        calibrated_records: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        p_groups: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        replicates = int(self.experiment_config["bootstrap_replicates"])
        base_seed = int(self.experiment_config["bootstrap_seed"])
        for suite, dataset in target_order():
            cells = self._dataset(suite, dataset).safe_cells()
            cell_ids = [str(cell.cell_id) for cell in cells]
            row_ids = [str(cell.row_id) for cell in cells]
            clusters = sorted(set(row_ids))
            cluster_index = {value: index for index, value in enumerate(clusters)}
            membership = np.asarray([cluster_index[value] for value in row_ids], dtype=int)
            cell_count = np.bincount(membership, minlength=len(clusters)).astype(float)
            for variant in ("1", "4"):
                raw_slice = [
                    row
                    for row in raw_records
                    if str(row["suite"]) == suite
                    and str(row["dataset"]) == dataset
                    and str(row["group_size_variant"]) == variant
                ]
                calibrated_slice = [
                    row
                    for row in calibrated_records
                    if str(row["suite"]) == suite
                    and str(row["dataset"]) == dataset
                    and str(row["group_size_variant"]) == variant
                ]
                raw_valid, raw_correct = self._record_metric_arrays(raw_slice, cell_ids)
                cal_valid, cal_correct = self._record_metric_arrays(calibrated_slice, cell_ids)
                raw_cluster_valid = np.bincount(
                    membership, weights=raw_valid, minlength=len(clusters)
                )
                raw_cluster_correct = np.bincount(
                    membership, weights=raw_correct, minlength=len(clusters)
                )
                cal_cluster_valid = np.bincount(
                    membership, weights=cal_valid, minlength=len(clusters)
                )
                cal_cluster_correct = np.bincount(
                    membership, weights=cal_correct, minlength=len(clusters)
                )
                seed = int(
                    hashlib.sha256(
                        f"{base_seed}|{suite}|{dataset}|{variant}|raw-calibrated".encode()
                    ).hexdigest()[:16],
                    16,
                )
                rng = np.random.default_rng(seed)
                deltas = {
                    "precision": np.zeros(replicates),
                    "recall": np.zeros(replicates),
                    "f1": np.zeros(replicates),
                }
                offset = 0
                probabilities = np.full(len(clusters), 1.0 / len(clusters))
                while offset < replicates:
                    batch = min(100, replicates - offset)
                    weights = rng.multinomial(len(clusters), probabilities, size=batch)
                    total = weights @ cell_count
                    metrics: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
                    for name, valid_by_cluster, correct_by_cluster in (
                        ("raw", raw_cluster_valid, raw_cluster_correct),
                        ("calibrated", cal_cluster_valid, cal_cluster_correct),
                    ):
                        valid = weights @ valid_by_cluster
                        correct = weights @ correct_by_cluster
                        precision = np.divide(correct, valid, out=np.zeros_like(correct), where=valid > 0)
                        recall = np.divide(correct, total, out=np.zeros_like(correct), where=total > 0)
                        f1 = np.divide(
                            2.0 * precision * recall,
                            precision + recall,
                            out=np.zeros_like(correct),
                            where=(precision + recall) > 0,
                        )
                        metrics[name] = (precision, recall, f1)
                    for metric_index, metric in enumerate(("precision", "recall", "f1")):
                        deltas[metric][offset : offset + batch] = (
                            metrics["calibrated"][metric_index]
                            - metrics["raw"][metric_index]
                        )
                    offset += batch
                for metric, values in deltas.items():
                    p_value = min(
                        1.0,
                        2.0
                        * min(
                            (float(np.count_nonzero(values <= 0.0)) + 1.0) / (replicates + 1.0),
                            (float(np.count_nonzero(values >= 0.0)) + 1.0) / (replicates + 1.0),
                        ),
                    )
                    dataset_key = f"{suite}/{dataset}"
                    p_groups[(variant, metric)][dataset_key] = p_value
                    output.append(
                        {
                            "suite": suite,
                            "dataset": dataset,
                            "group_size_variant": variant,
                            "metric": metric,
                            "delta_ci_low": float(np.quantile(values, 0.025)),
                            "delta_ci_high": float(np.quantile(values, 0.975)),
                            "p_value": p_value,
                            "holm_adjusted_p_value": 1.0,
                            "bootstrap_replicates": replicates,
                            "bootstrap_seed": seed,
                            "cluster_unit": "dirty_row",
                        }
                    )
        adjusted = {
            (variant, metric, dataset): value
            for (variant, metric), values in p_groups.items()
            for dataset, value in holm_adjust(values).items()
        }
        for row in output:
            row["holm_adjusted_p_value"] = adjusted[
                (
                    str(row["group_size_variant"]),
                    str(row["metric"]),
                    f"{row['suite']}/{row['dataset']}",
                )
            ]
        return output

    def _calibration_effect_rows(
        self,
        raw_records: Sequence[Mapping[str, object]],
        calibrated_records: Sequence[Mapping[str, object]],
        paired: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        raw_metrics = {
            (str(row["suite"]), str(row["dataset"]), str(row["group_size_variant"])): row
            for row in summarize_records(raw_records, strict=True)
            if str(row["scope"]) == "dataset"
        }
        calibrated_metrics = {
            (str(row["suite"]), str(row["dataset"]), str(row["group_size_variant"])): row
            for row in summarize_records(calibrated_records, strict=True)
            if str(row["scope"]) == "dataset"
        }
        paired_index = {
            (
                str(row["suite"]),
                str(row["dataset"]),
                str(row["group_size_variant"]),
                str(row["metric"]),
            ): row
            for row in paired
        }
        rows: list[dict[str, object]] = []
        source = self.router_comparison_run
        assert source is not None
        for suite, dataset in target_order():
            for variant in ("1", "4"):
                key = (suite, dataset, variant)
                raw = raw_metrics[key]
                calibrated = calibrated_metrics[key]
                raw_selection = load_json(
                    source
                    / "selections"
                    / "lightgbm"
                    / "size_conditioned"
                    / f"variant_{variant}"
                    / _budget_label(0.2)
                    / f"{_dataset_key(suite, dataset)}.json"
                )
                calibrated_selection = load_json(
                    self._selection_path(
                        "lightgbm",
                        "size_conditioned",
                        variant,
                        0.2,
                        suite,
                        dataset,
                    )
                )
                raw_ids = {str(value) for value in raw_selection["selected_query_ids"]}
                calibrated_ids = {
                    str(value) for value in calibrated_selection["selected_query_ids"]
                }
                intersection = raw_ids & calibrated_ids
                union = raw_ids | calibrated_ids
                raw_slice = [
                    row
                    for row in raw_records
                    if str(row["suite"]) == suite
                    and str(row["dataset"]) == dataset
                    and str(row["group_size_variant"]) == variant
                ]
                calibrated_slice = [
                    row
                    for row in calibrated_records
                    if str(row["suite"]) == suite
                    and str(row["dataset"]) == dataset
                    and str(row["group_size_variant"]) == variant
                ]
                result: dict[str, object] = {
                    "suite": suite,
                    "dataset": dataset,
                    "group_size_variant": variant,
                    "budget_share": 0.2,
                    "raw_correct_repairs": int(raw["correct_repairs"]),
                    "calibrated_correct_repairs": int(calibrated["correct_repairs"]),
                    "raw_predicted_repairs": int(raw["predicted_repairs"]),
                    "calibrated_predicted_repairs": int(calibrated["predicted_repairs"]),
                    "raw_selected_queries": len(raw_ids),
                    "calibrated_selected_queries": len(calibrated_ids),
                    "selection_intersection": len(intersection),
                    "selection_union": len(union),
                    "selection_jaccard": len(intersection) / len(union) if union else 1.0,
                    "raw_estimated_tokens": int(raw_selection["total_cost"]),
                    "calibrated_estimated_tokens": int(calibrated_selection["total_cost"]),
                    "raw_accepted_llm_cells": sum(bool(row.get("accepted_llm")) for row in raw_slice),
                    "calibrated_accepted_llm_cells": sum(
                        bool(row.get("accepted_llm")) for row in calibrated_slice
                    ),
                }
                for metric in ("precision", "recall", "f1"):
                    raw_value = float(raw[metric])
                    calibrated_value = float(calibrated[metric])
                    stat = paired_index[(suite, dataset, variant, metric)]
                    result[f"raw_{metric}"] = raw_value
                    result[f"calibrated_{metric}"] = calibrated_value
                    result[f"delta_{metric}"] = calibrated_value - raw_value
                    result[f"delta_{metric}_ci_low"] = stat["delta_ci_low"]
                    result[f"delta_{metric}_ci_high"] = stat["delta_ci_high"]
                    result[f"{metric}_p_value"] = stat["p_value"]
                    result[f"{metric}_holm_adjusted_p_value"] = stat[
                        "holm_adjusted_p_value"
                    ]
                rows.append(result)
        if len(rows) != 18:
            raise ValueError("raw/calibrated comparison must have 18 rows")
        return rows

    def _calibration_conclusion(
        self,
        effects: Sequence[Mapping[str, object]],
        paired: Sequence[Mapping[str, object]],
        raw_records: Sequence[Mapping[str, object]],
        calibrated_records: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        calibration = _read_csv(self.paths.metrics_dir / "calibration_by_dataset.csv")
        calibration_by_variant: dict[str, object] = {}
        calibration_success: dict[str, bool] = {}
        for variant in ("1", "4"):
            head_summary: dict[str, object] = {}
            for head in ("helpful", "harmful"):
                frame = calibration.loc[
                    calibration["group_size_variant"].astype(str).eq(variant)
                    & calibration["head"].astype(str).eq(head)
                ]
                values = {
                    field: float(pd.to_numeric(frame[field], errors="raise").mean())
                    for field in (
                        "raw_brier",
                        "calibrated_brier",
                        "raw_ece",
                        "calibrated_ece",
                        "raw_average_precision",
                        "calibrated_average_precision",
                    )
                }
                values["brier_improved"] = values["calibrated_brier"] < values["raw_brier"]
                values["ece_improved"] = values["calibrated_ece"] < values["raw_ece"]
                head_summary[head] = values
            success = all(
                bool(head_summary[head][field])  # type: ignore[index]
                for head in ("helpful", "harmful")
                for field in ("brier_improved", "ece_improved")
            )
            calibration_success[variant] = success
            calibration_by_variant[variant] = {
                "heads": head_summary,
                "success": success,
            }

        combined = [*raw_records, *calibrated_records]
        aggregates = {
            (
                str(row["method"]),
                str(row["group_size_variant"]),
                str(row["scope"]),
            ): row
            for row in summarize_records(combined, strict=True)
            if str(row["scope"]) in {"micro", "macro"}
        }
        decision_by_variant: dict[str, object] = {}
        robustness_by_variant: dict[str, object] = {}
        effect_by_variant = {
            variant: [row for row in effects if str(row["group_size_variant"]) == variant]
            for variant in ("1", "4")
        }
        for variant in ("1", "4"):
            raw_micro = aggregates[("budgeted_group_lightgbm", variant, "micro")]
            raw_macro = aggregates[("budgeted_group_lightgbm", variant, "macro")]
            cal_micro = aggregates[(self._bgr_method_name("lightgbm"), variant, "micro")]
            cal_macro = aggregates[(self._bgr_method_name("lightgbm"), variant, "macro")]
            deltas = [float(row["delta_f1"]) for row in effect_by_variant[variant]]
            wins = sum(value > 1e-12 for value in deltas)
            losses = sum(value < -1e-12 for value in deltas)
            ties = len(deltas) - wins - losses
            decision_success = (
                float(cal_micro["f1"]) > float(raw_micro["f1"])
                and float(cal_macro["f1"]) > float(raw_macro["f1"])
                and wins > losses
            )
            decision_by_variant[variant] = {
                "raw_micro": {field: float(raw_micro[field]) for field in ("precision", "recall", "f1")},
                "calibrated_micro": {field: float(cal_micro[field]) for field in ("precision", "recall", "f1")},
                "raw_dataset_macro": {field: float(raw_macro[field]) for field in ("precision", "recall", "f1")},
                "calibrated_dataset_macro": {field: float(cal_macro[field]) for field in ("precision", "recall", "f1")},
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "worst_dataset_delta_f1": min(deltas),
                "success": decision_success,
            }
            negative = [
                row
                for row in paired
                if str(row["group_size_variant"]) == variant
                and str(row["metric"]) == "f1"
                and float(row["holm_adjusted_p_value"]) < 0.05
                and float(
                    next(
                        value["delta_f1"]
                        for value in effect_by_variant[variant]
                        if str(value["suite"]) == str(row["suite"])
                        and str(value["dataset"]) == str(row["dataset"])
                    )
                )
                < 0.0
            ]
            robustness_by_variant[variant] = {
                "negative_holm_significant_datasets": [
                    f"{row['suite']}/{row['dataset']}" for row in negative
                ],
                "worst_dataset_delta_f1": min(deltas),
                "success": not negative,
            }
        complete_success = {
            variant: bool(calibration_success[variant])
            and bool(decision_by_variant[variant]["success"])  # type: ignore[index]
            and bool(robustness_by_variant[variant]["success"])  # type: ignore[index]
            for variant in ("1", "4")
        }
        useful = any(complete_success.values()) and all(
            bool(robustness_by_variant[variant]["success"])  # type: ignore[index]
            for variant in ("1", "4")
        )
        if useful:
            conclusion = "有用"
        elif any(calibration_success.values()) or any(
            bool(decision_by_variant[variant]["success"])  # type: ignore[index]
            for variant in ("1", "4")
        ):
            conclusion = "部分有用"
        else:
            conclusion = "无证据支持"
        improved_probability_no_f1: list[str] = []
        for effect in effects:
            key_frame = calibration.loc[
                calibration["target_suite"].astype(str).eq(str(effect["suite"]))
                & calibration["target_dataset"].astype(str).eq(str(effect["dataset"]))
                & calibration["group_size_variant"].astype(str).eq(
                    str(effect["group_size_variant"])
                )
            ]
            probability_improved = len(key_frame) == 2 and all(
                bool(
                    (pd.to_numeric(key_frame[f"calibrated_{field}"]) < pd.to_numeric(key_frame[f"raw_{field}"])).all()
                )
                for field in ("brier", "ece")
            )
            if probability_improved and float(effect["delta_f1"]) <= 0.0:
                improved_probability_no_f1.append(
                    f"{effect['suite']}/{effect['dataset']}:k={effect['group_size_variant']}"
                )
        return {
            "decision_rule": (
                "at least one k satisfies calibration, decision, and robustness; "
                "the other k has no Holm-significant negative dataset"
            ),
            "calibration_by_variant": calibration_by_variant,
            "decision_by_variant": decision_by_variant,
            "robustness_by_variant": robustness_by_variant,
            "complete_success_by_variant": complete_success,
            "calibration_improved_but_final_f1_not_improved": improved_probability_no_f1,
            "usefulness_conclusion": conclusion,
        }

    def build_metrics_stage(self) -> dict[str, object]:
        standard = self._build_metrics_router_v3()
        all_records = read_jsonl(self.paths.final_dir / "all_methods.jsonl")
        calibrated_records = [
            row
            for row in all_records
            if str(row.get("method", "")) == self._bgr_method_name("lightgbm")
        ]
        if len(all_records) != 4 * TEST_TARGET_CELL_COUNT or len(calibrated_records) != 2 * TEST_TARGET_CELL_COUNT:
            raise ValueError("Router-v4 final record matrix must contain 88,792 rows")
        raw_records = self._historical_raw_records()
        paired = self._raw_calibrated_bootstrap(raw_records, calibrated_records)
        effects = self._calibration_effect_rows(
            raw_records, calibrated_records, paired
        )
        _write_csv(self.paths.metrics_dir / "calibration_paired_statistics.csv", paired)
        _write_csv(self.paths.metrics_dir / "calibration_effect_by_dataset.csv", effects)
        conclusion = self._calibration_conclusion(
            effects, paired, raw_records, calibrated_records
        )
        conclusion.update(
            {
                "router_revision": self.router_revision,
                "historical_comparator_hashes": HISTORICAL_COMPARATOR_HASHES,
                "dataset_variant_comparisons": len(effects),
                "paired_statistic_rows": len(paired),
                "bootstrap_replicates": int(self.experiment_config["bootstrap_replicates"]),
                "bootstrap_seed": int(self.experiment_config["bootstrap_seed"]),
            }
        )
        write_json(self.paths.metrics_dir / "calibration_summary.json", conclusion)
        summary = {
            **standard,
            "calibration_effect_rows": len(effects),
            "calibration_paired_statistics_rows": len(paired),
            "usefulness_conclusion": conclusion["usefulness_conclusion"],
        }
        self.state.update_stage("metrics", "complete", **summary)
        return summary


def validate_router_v4_run(
    root: str | Path,
    manifest: Mapping[str, object] | None = None,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    """Independently validate the frozen Router-v4 matrix and provenance."""

    run_dir = Path(root).resolve()
    run_manifest = dict(manifest or load_json(run_dir / "run_manifest.json"))
    for field, relative in (
        ("experiment_config_sha256", "bound_experiment_config.json"),
        ("llm_config_sha256", "bound_llm_config.json"),
    ):
        if sha256_file(run_dir / relative) != str(run_manifest.get(field, "")):
            raise ValueError(f"Router-v4 run fingerprint drift: {field}")
    config = load_json(run_dir / "bound_experiment_config.json")
    if (
        str(config.get("router_revision", ""))
        != ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION
        or config.get("gate_backends") != ["lightgbm"]
        or config.get("router_training_variants") != {"1": [1], "4": [1, 4]}
        or config.get("budget_shares") != [0.2]
    ):
        raise ValueError("Router-v4 bound experiment matrix drift")
    current_implementation = _hash_tree(
        PROJECT_ROOT / "src" / "budgeted_group_repair_no_baran", (".py",)
    )
    bound_implementation = str(run_manifest.get("implementation_sha256", ""))
    if (
        bound_implementation != current_implementation
        and bound_implementation not in FROZEN_ROUTER_V4_IMPLEMENTATION_SHA256
    ):
        raise ValueError("Router-v4 implementation hash drift")
    data_root = Path(str(run_manifest.get("data_root", ""))).resolve()
    validate_manifest(data_root, require_portable=True)
    expected_cells = {
        (suite, dataset): {
            str(cell.cell_id)
            for cell in load_dataset(suite, dataset, data_root).safe_cells()
        }
        for suite, dataset in target_order()
    }
    if sum(map(len, expected_cells.values())) != TEST_TARGET_CELL_COUNT:
        raise ValueError("Router-v4 target cell universe drift")

    for path_field, hash_field in (
        ("baran_source_run", "baran_source_manifest_sha256"),
        ("response_reuse_run", "response_reuse_manifest_sha256"),
        ("calibration_source_run", "calibration_source_manifest_sha256"),
        ("router_comparison_run", "router_comparison_manifest_sha256"),
    ):
        source = Path(str(run_manifest.get(path_field, ""))).resolve()
        if not source.is_dir() or sha256_file(source / "run_manifest.json") != str(
            run_manifest.get(hash_field, "")
        ):
            raise ValueError(f"Router-v4 source binding drift: {path_field}")
    response_reuse = load_json(run_dir / "provenance" / "response_reuse.json")
    if set(response_reuse.get("matching_fields", [])) != {
        "query_id",
        "prompt_hash",
        "provider_request_hash",
        "model",
        "prompt_schema_version",
    }:
        raise ValueError("Router-v4 response reuse is not request-identical")
    calibration_provenance = load_json(run_dir / "provenance" / "calibration.json")
    if (
        calibration_provenance.get("mode") != "strict_frozen_source_import"
        or int(calibration_provenance.get("calibration_queries", -1))
        != EXPECTED_CALIBRATION_QUERIES
        or int(calibration_provenance.get("calibration_executions", -1))
        != EXPECTED_CALIBRATION_QUERIES
        or int(calibration_provenance.get("calibration_pair_labels", -1))
        != EXPECTED_CALIBRATION_PAIR_LABELS
        or calibration_provenance.get("target_labels_or_responses_used_before_selection")
        is not False
    ):
        raise ValueError("Router-v4 calibration import provenance differs")
    for relative, field in (
        ("llm/calibration_queries.jsonl", "calibration_queries_sha256"),
        ("llm/calibration_execution.jsonl", "calibration_execution_sha256"),
        ("llm/calibration_pair_labels.csv", "calibration_pair_labels_sha256"),
    ):
        if sha256_file(run_dir / relative) != str(calibration_provenance[field]):
            raise ValueError(f"Router-v4 calibration artifact drift: {relative}")

    freeze_path = run_dir / "gates" / "frozen_score_ledger_manifest.json"
    freeze = load_json(freeze_path)
    stored_content_hash = str(freeze.pop("content_sha256", ""))
    if canonical_json_sha256(freeze) != stored_content_hash:
        raise ValueError("Router-v4 frozen score-ledger content drift")
    freeze["content_sha256"] = stored_content_hash
    if (
        int(freeze.get("gate_folds", -1)) != 18
        or int(freeze.get("selection_slices_at_freeze", -1)) != 0
        or freeze.get("target_labels_or_responses_used_before_selection") is not False
        or freeze.get("optimizer_input_column") != "conservative_uplift"
    ):
        raise ValueError("Router-v4 score freeze declaration differs")
    gate_artifacts = freeze.get("artifacts", [])
    if not isinstance(gate_artifacts, list) or len(gate_artifacts) != 20:
        raise ValueError("Router-v4 frozen gate artifact matrix differs")
    for row in gate_artifacts:
        if not isinstance(row, Mapping):
            raise ValueError("Router-v4 frozen gate artifact row is invalid")
        relative = str(row.get("prediction") or row.get("artifact") or "")
        expected_hash = str(row.get("prediction_sha256") or row.get("sha256") or "")
        if sha256_file(run_dir / relative) != expected_hash:
            raise ValueError(f"Router-v4 frozen gate drift: {relative}")
        metadata_relative = str(row.get("metadata") or "")
        if metadata_relative and sha256_file(run_dir / metadata_relative) != str(
            row.get("metadata_sha256", "")
        ):
            raise ValueError(f"Router-v4 gate metadata drift: {metadata_relative}")
    split = _read_csv(run_dir / "gates" / "split_audit.csv")
    calibration = _read_csv(run_dir / "metrics" / "calibration_by_dataset.csv")
    reliability = _read_csv(run_dir / "metrics" / "calibration_reliability_bins.csv")
    selection = _read_csv(run_dir / "metrics" / "selection_audit.csv")
    routeability = _read_csv(run_dir / "metrics" / "routeability_by_dataset.csv")
    if (
        len(split) != 18
        or len(calibration) != 36
        or reliability.empty
        or len(selection) != 18
        or len(routeability) != 18
        or bool(
            (
                pd.to_numeric(selection["selected_estimated_tokens"], errors="raise")
                > pd.to_numeric(selection["budget_estimated_tokens"], errors="raise")
            ).any()
        )
    ):
        raise ValueError("Router-v4 gate/selection/diagnostic acceptance counts failed")
    freeze_file_hash = sha256_file(freeze_path)
    for suite, dataset in target_order():
        for variant, sizes in {"1": {1}, "4": {1, 4}}.items():
            prediction_path = (
                run_dir
                / "gates"
                / "lightgbm"
                / f"variant_{variant}"
                / f"{_dataset_key(suite, dataset)}.csv"
            )
            predictions = _read_csv(prediction_path)
            required_columns = {
                "raw_q_helpful",
                "raw_q_harmful",
                "q_helpful",
                "q_harmful",
                "raw_net_gain",
                "net_gain",
                "sigma",
                "conservative_uplift",
                "probability_calibration",
            }
            if (
                required_columns - set(predictions.columns)
                or set(pd.to_numeric(predictions["group_size"], errors="raise")) != sizes
                or set(predictions["probability_calibration"].astype(str)) != {"isotonic"}
                or bool((pd.to_numeric(predictions["conservative_uplift"]) < 0).any())
            ):
                raise ValueError("Router-v4 prediction schema or values differ")
            selection_doc = load_json(
                run_dir
                / "selections"
                / "lightgbm"
                / "size_conditioned"
                / f"variant_{variant}"
                / "20pct"
                / f"{_dataset_key(suite, dataset)}.json"
            )
            if (
                str(selection_doc.get("score_ledger_manifest_sha256", ""))
                != freeze_file_hash
                or float(selection_doc.get("total_cost", math.inf))
                > float(selection_doc.get("budget", -1))
            ):
                raise ValueError("Router-v4 selection is not bound to frozen scores")

    final_records = read_jsonl(run_dir / "final" / "all_methods.jsonl")
    expected_methods = {
        ("baran", "all"),
        ("llm_only", "1"),
        ("budgeted_group_lightgbm_isotonic", "1"),
        ("budgeted_group_lightgbm_isotonic", "4"),
    }
    observed_methods = {
        (str(row.get("method", "")), str(row.get("group_size_variant", "")))
        for row in final_records
    }
    record_audit = verify_records(final_records, expected_cell_ids=expected_cells)
    if (
        len(final_records) != 88_792
        or observed_methods != expected_methods
        or record_audit.get("ok") is not True
        or int(record_audit.get("slices", -1)) != 36
    ):
        raise ValueError("Router-v4 final cell ledger differs")
    independent_metrics = summarize_records(final_records, strict=True)
    recorded_metrics = _read_csv(run_dir / "metrics" / "method_metrics.csv")
    if len(independent_metrics) != len(recorded_metrics):
        raise ValueError("Router-v4 method metrics row count differs")
    effects = _read_csv(run_dir / "metrics" / "calibration_effect_by_dataset.csv")
    paired = _read_csv(run_dir / "metrics" / "calibration_paired_statistics.csv")
    summary = load_json(run_dir / "metrics" / "calibration_summary.json")
    comparator = load_json(
        run_dir / "provenance" / "historical_raw_comparator.json"
    )
    if (
        len(effects) != 18
        or len(paired) != 54
        or int(summary.get("dataset_variant_comparisons", -1)) != 18
        or comparator.get("frozen_hashes") != HISTORICAL_COMPARATOR_HASHES
        or int(comparator.get("raw_records", -1)) != 44_396
    ):
        raise ValueError("Router-v4 raw/calibrated comparison artifacts differ")
    stages = run_manifest.get("stages", {})
    if require_complete:
        if run_manifest.get("status") != "complete":
            raise ValueError("Router-v4 run is not marked complete")
        if not isinstance(stages, Mapping) or any(
            not isinstance(stages.get(stage), Mapping)
            or stages[stage].get("status") != "complete"  # type: ignore[index]
            for stage in REQUIRED_STAGES
        ):
            raise ValueError("Router-v4 run has incomplete required stages")
    return {
        "ok": True,
        "run_dir": str(run_dir),
        "status": str(run_manifest.get("status", "")),
        "router_revision": ROUTER_V4_LIGHTGBM_ISOTONIC_REVISION,
        "gate_folds": len(split),
        "selection_slices": len(selection),
        "routeability_folds": len(routeability),
        "dataset_slices": int(record_audit["slices"]),
        "final_records": len(final_records),
        "calibration_comparisons": len(effects),
        "paired_statistics": len(paired),
        "usefulness_conclusion": summary.get("usefulness_conclusion"),
        "independently_recomputed": True,
    }


__all__ = [
    "FROZEN_ROUTER_V4_IMPLEMENTATION_SHA256",
    "HISTORICAL_COMPARATOR_HASHES",
    "RouterV4ExperimentRunner",
    "validate_router_v4_run",
]
