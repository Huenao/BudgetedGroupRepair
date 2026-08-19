#!/usr/bin/env python3
"""Opt-in full-row/full-LOFO single-target Router-v3 TabICLv2 smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from budgeted_group_repair_no_baran.group_gate import GroupUpliftGate
from budgeted_group_repair_no_baran.protocol import base_family, split_for_target
from budgeted_group_repair_no_baran.router_v3 import (
    MODEL_FEATURE_COLUMNS,
    ExperimentRunner,
    load_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--suite", default="tableeg")
    parser.add_argument("--dataset", default="company")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    manifest = load_json(run_dir / "run_manifest.json")
    runner = ExperimentRunner.create(
        project_root=project_root,
        data_root=project_root / "data",
        config_path=run_dir / "bound_experiment_config.json",
        llm_config_path=run_dir / "bound_llm_config.json",
        vendor_root=project_root / "vendor" / "raha_source",
        runs_root=project_root / "runs",
        run_dir=run_dir,
        resume=True,
        baran_source_run=Path(str(manifest["baran_source_run"])),
        response_reuse_run=Path(str(manifest["response_reuse_run"])),
    )
    all_pairs = runner._all_pair_features()
    labels = pd.read_csv(run_dir / "llm" / "calibration_pair_labels.csv")
    labels = labels.loc[
        :,
        [
            "cell_id",
            "query_id",
            "baran_correct",
            "llm_correct_in_query",
            "executable_propose",
        ],
    ]
    train_safe, test_safe, audit = split_for_target(
        all_pairs, args.suite, args.dataset, enforce_target_unlabeled=True
    )
    train_all = train_safe.merge(
        labels,
        how="inner",
        on=["cell_id", "query_id"],
        validate="one_to_one",
    )
    train = runner._filter_variant_pairs(
        train_all,
        (1, 4),
        context=f"smoke train 4/{args.suite}/{args.dataset}",
    )
    test = runner._filter_variant_pairs(
        test_safe,
        (1, 4),
        context=f"smoke test 4/{args.suite}/{args.dataset}",
    )
    started = time.perf_counter()
    gate = GroupUpliftGate(
        "tabiclv2",
        rho=float(runner.experiment_config["harm_penalty_rho"]),
        gamma=float(runner.experiment_config["uncertainty_penalty_gamma"]),
        random_state=int(runner.experiment_config["seed"]),
        backend_config=runner._gate_backend_config("tabiclv2"),
    ).fit(
        train.loc[:, list(MODEL_FEATURE_COLUMNS)],
        [bool(int(value)) for value in train["baran_correct"]],
        [bool(int(value)) for value in train["llm_correct_in_query"]],
        [bool(int(value)) for value in train["executable_propose"]],
        [base_family(value) for value in train["dataset"].astype(str)],
    )
    predictions = gate.predict(test.loc[:, list(MODEL_FEATURE_COLUMNS)])
    elapsed = time.perf_counter() - started
    metadata = gate.metadata()
    probabilities = np.asarray(
        [[value.q_helpful, value.q_harmful] for value in predictions], dtype=float
    )
    if len(predictions) != len(test):
        raise AssertionError("single-target smoke prediction coverage differs")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise AssertionError("single-target smoke probabilities are invalid")
    families = metadata["training"]["families"]
    if (
        metadata["training"]["lofo_replicas"] != len(families)
        or len(metadata["lofo"]) != len(families)
    ):
        raise AssertionError("single-target smoke did not retain complete LOFO")
    print(
        json.dumps(
            {
                "ok": True,
                "target": f"{args.suite}/{args.dataset}",
                "variant": "4",
                "allowed_group_sizes": [1, 4],
                "train_rows": len(train),
                "test_rows": len(test),
                "predictions": len(predictions),
                "families": families,
                "lofo_replicas": len(metadata["lofo"]),
                "elapsed_seconds": elapsed,
                "split_audit": audit.as_dict(),
                "backend": metadata["backend"],
                "backend_version": metadata["backend_version"],
                "checkpoint_sha256": metadata["full"]["helpful_head"][
                    "checkpoint_sha256"
                ],
                "full_fit_seconds": sum(
                    float(metadata["full"][head]["fit_seconds"])
                    for head in ("helpful_head", "harmful_head")
                ),
                "full_predict_seconds": sum(
                    float(metadata["full"][head]["predict_seconds"])
                    for head in ("helpful_head", "harmful_head")
                ),
                "lofo_fit_seconds": sum(
                    float(replica[head]["fit_seconds"])
                    for replica in metadata["lofo"]
                    for head in ("helpful_head", "harmful_head")
                ),
                "lofo_predict_seconds": sum(
                    float(replica[head]["predict_seconds"])
                    for replica in metadata["lofo"]
                    for head in ("helpful_head", "harmful_head")
                ),
                "feature_columns": [
                    column["name"]
                    for column in metadata["full"]["encoder"]["columns"]
                ],
                "full_effective_parameters": metadata["full"]["helpful_head"][
                    "parameters"
                ],
                "hardware": metadata["full"]["helpful_head"][
                    "runtime_environment"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
