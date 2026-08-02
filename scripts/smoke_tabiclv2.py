#!/usr/bin/env python3
"""Opt-in real-checkpoint smoke test for the strict TabICLv2 adapter."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from budgeted_group_repair_no_baran.group_gate import (
    FoundationFeatureEncoder,
    TabICLv2ClassifierAdapter,
)


def _fit_once(checkpoint: Path, checkpoint_sha256: str) -> np.ndarray:
    train = pd.DataFrame(
        {
            "numeric": [float(index % 7) if index != 5 else np.nan for index in range(32)],
            "categorical": [
                None if index == 3 else ("alpha" if index % 2 == 0 else "beta")
                for index in range(32)
            ],
        }
    )
    labels = [index % 2 for index in range(32)]
    encoder = FoundationFeatureEncoder().fit(train)
    encoded_train = encoder.transform(train)
    encoded_test = encoder.transform(
        pd.DataFrame(
            {
                "numeric": [np.nan, 4.0],
                "categorical": [None, "never-seen"],
            }
        )
    )
    if not isinstance(encoded_train["categorical"].dtype, pd.CategoricalDtype):
        raise AssertionError("categorical semantics were not preserved")
    model = TabICLv2ClassifierAdapter(
        {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "allow_auto_download": False,
            "n_estimators": 1,
            "batch_size": 8,
            "kv_cache": False,
            "random_state": 42,
            "device": "cuda",
            "use_amp": "auto",
            "use_fa3": "auto",
            "offload_mode": "auto",
        }
    ).fit(encoded_train, labels)
    probabilities = model.predict_proba(encoded_test)
    if probabilities.shape != (2, 2) or not np.isfinite(probabilities).all():
        raise AssertionError("invalid TabICLv2 smoke probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("TabICLv2 smoke probability rows do not sum to one")
    print(json.dumps(model.metadata(), sort_keys=True))
    return probabilities


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    first = _fit_once(args.checkpoint.resolve(), args.checkpoint_sha256)
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    second = _fit_once(args.checkpoint.resolve(), args.checkpoint_sha256)
    if not np.allclose(first, second, rtol=0.0, atol=1e-7):
        raise AssertionError("TabICLv2 repeated-seed predictions differ")
    print(json.dumps({"ok": True, "probabilities": first.tolist()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
