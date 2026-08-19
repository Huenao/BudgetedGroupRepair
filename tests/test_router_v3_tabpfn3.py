from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from budgeted_group_repair_no_baran.group_gate import (
    GroupUpliftGate,
    TabPFN3ClassifierAdapter,
)
from budgeted_group_repair_no_baran.router_v3 import (
    ROUTER_V3_TABPFN3_REVISION,
    ROUTER_V3_TABPFN3_VARIANTS,
    TABPFN3_GATE_BACKENDS,
    ExperimentRunner,
)
from budgeted_group_repair_no_baran import router_reporting_v3, router_v3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_NAME = "tabpfn-v3-classifier-v3_20260506_ood.ckpt"


def _runner() -> ExperimentRunner:
    runner = object.__new__(ExperimentRunner)
    runner.router_revision = ROUTER_V3_TABPFN3_REVISION
    runner.experiment_config = json.loads(
        (
            PROJECT_ROOT / "configs" / "experiment_router_v3_tabpfn3_k14.json"
        ).read_text(encoding="utf-8")
    )
    return runner


def _checkpoint(tmp_path: Path, contents: bytes = b"tabpfn-test") -> tuple[Path, str]:
    path = tmp_path / CHECKPOINT_NAME
    path.write_bytes(contents)
    return path, sha256(contents).hexdigest()


def test_tabpfn3_revision_freezes_backend_k_and_budget() -> None:
    runner = _runner()
    assert runner.is_router_v3
    assert runner.is_router_v3_tabpfn3
    assert runner.is_router_v3_foundation
    assert runner.is_router_v3_isolated_backbone
    assert runner.freezes_reused_terminal_failures
    assert runner._active_gate_backends() == TABPFN3_GATE_BACKENDS
    assert tuple(runner._router_training_variants()) == ROUTER_V3_TABPFN3_VARIANTS
    assert runner._router_budget_shares() == (0.2,)
    assert runner._comparison_backends() == ("lightgbm", "xgboost", "tabiclv2")
    specs = runner._scenario_specs()
    assert [value["scenario"] for value in specs] == ["size_conditioned"] * 2
    assert [value["group_size_variant"] for value in specs] == ["1", "4"]


def test_tabpfn3_adapter_rejects_download_missing_wrong_name_and_hash(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="forbids automatic"):
        TabPFN3ClassifierAdapter({"allow_auto_download": True})
    with pytest.raises(FileNotFoundError, match="checkpoint path is unset"):
        TabPFN3ClassifierAdapter({"allow_auto_download": False})
    wrong = tmp_path / "wrong.ckpt"
    wrong.write_bytes(b"x")
    with pytest.raises(ValueError, match="checkpoint basename"):
        TabPFN3ClassifierAdapter({"checkpoint_path": str(wrong)})
    checkpoint, _ = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        TabPFN3ClassifierAdapter(
            {"checkpoint_path": str(checkpoint), "checkpoint_sha256": "0" * 64}
        )


def test_tabpfn3_adapter_lazy_import_categories_probability_and_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, digest = _checkpoint(tmp_path)
    captured: dict[str, object] = {}

    class FakeClassifier:
        def __init__(self, **kwargs: object) -> None:
            captured["parameters"] = kwargs
            self.classes_ = np.asarray([0, 1])
            self.n_estimators_ = 8
            self.inference_precision_ = "torch.float16"
            self.forced_inference_dtype_ = "torch.float16"

        def fit(self, features: pd.DataFrame, labels: list[int]) -> "FakeClassifier":
            captured["fit_frame"] = features.copy()
            captured["labels"] = labels
            return self

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            return np.tile(np.asarray([[0.2, 0.8]]), (len(features), 1))

    monkeypatch.setitem(
        sys.modules, "tabpfn", SimpleNamespace(TabPFNClassifier=FakeClassifier)
    )
    monkeypatch.setattr(
        "budgeted_group_repair_no_baran.group_gate.package_version",
        lambda package: "8.1.0" if package == "tabpfn" else "unknown",
    )
    frame = pd.DataFrame(
        {
            "kind": pd.Series(pd.Categorical(["a", "b"])),
            "score": pd.Series([1.0, 2.0], dtype="float64"),
        }
    )
    adapter = TabPFN3ClassifierAdapter(
        {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest,
            "allow_auto_download": False,
            "n_estimators": 8,
            "auto_scale_n_estimators": False,
        }
    ).fit(frame, [0, 1])
    first = adapter.predict_proba(frame)
    second = adapter.predict_proba(frame)
    assert np.array_equal(first, second)
    assert np.allclose(first[:, 1], 0.8)
    parameters = captured["parameters"]
    assert parameters["model_path"] == str(checkpoint)
    assert parameters["categorical_features_indices"] == [0]
    assert parameters["n_estimators"] == 8
    assert parameters["auto_scale_n_estimators"] is False
    assert parameters["fit_mode"] == "fit_preprocessors"
    metadata = adapter.metadata()
    assert metadata["package"] == "tabpfn"
    assert metadata["checkpoint_sha256"] == digest
    assert metadata["parameters"]["categorical_features_indices"] == [0]
    assert metadata["parameters"]["allow_auto_download"] is False
    assert metadata["fit_seconds"] >= 0
    assert metadata["predict_seconds"] >= 0


def test_tabpfn3_adapter_rejects_invalid_probabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, digest = _checkpoint(tmp_path)

    class FakeClassifier:
        classes_ = np.asarray([0, 1])

        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def fit(self, features: pd.DataFrame, labels: list[int]) -> "FakeClassifier":
            del features, labels
            return self

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            return np.tile(np.asarray([[0.8, 0.8]]), (len(features), 1))

    monkeypatch.setitem(
        sys.modules, "tabpfn", SimpleNamespace(TabPFNClassifier=FakeClassifier)
    )
    adapter = TabPFN3ClassifierAdapter(
        {"checkpoint_path": str(checkpoint), "checkpoint_sha256": digest}
    ).fit(pd.DataFrame({"score": [1.0, 2.0]}), [0, 1])
    with pytest.raises(RuntimeError, match="do not sum to one"):
        adapter.predict_proba(pd.DataFrame({"score": [1.0]}))


def test_tabpfn3_constant_heads_do_not_construct_foundation_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "budgeted_group_repair_no_baran.group_gate.package_version",
        lambda package: "8.1.0",
    )
    features = [{"kind": "alpha", "score": float(index)} for index in range(4)]
    gate = GroupUpliftGate("tabpfn3", backend_config={}).fit(
        features,
        [False] * 4,
        [False] * 4,
        [True] * 4,
        ["family"] * 4,
    )
    metadata = gate.metadata()["full"]
    assert metadata["helpful_head"] == {"kind": "constant", "probability": 0.0}
    assert metadata["harmful_head"] == {"kind": "constant", "probability": 0.0}
    assert all(row.q_helpful == row.q_harmful == 0.0 for row in gate.predict(features))


def test_tabpfn3_report_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {"ok": True, "kind": "tabpfn3-v3"}
    monkeypatch.setattr(
        router_reporting_v3,
        "_build_tabpfn3_report",
        lambda root, validation, output_path=None: expected,
    )
    monkeypatch.setattr(
        router_v3,
        "validate_run",
        lambda run_dir, require_complete=True: {
            "router_revision": ROUTER_V3_TABPFN3_REVISION
        },
    )
    assert router_reporting_v3.build_router_v3_report(tmp_path) == expected
