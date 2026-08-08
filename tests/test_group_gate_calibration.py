from __future__ import annotations

import math

import pytest

from budgeted_group_repair_no_baran.group_gate import GroupUpliftGate


def _training_rows() -> tuple[list[dict[str, object]], list[bool], list[bool], list[bool], list[str]]:
    features: list[dict[str, object]] = []
    baran: list[bool] = []
    llm: list[bool] = []
    executable: list[bool] = []
    families: list[str] = []
    for family_index, family in enumerate(("alpha", "beta", "gamma")):
        for index in range(8):
            score = family_index * 8 + index
            features.append(
                {
                    "score": float(score),
                    "kind": f"kind-{index % 3}",
                }
            )
            baran.append(index % 4 == 0)
            llm.append(index >= 3 and index % 4 != 0)
            executable.append(index != 1)
            families.append(family)
    return features, baran, llm, executable, families


def _fit_isotonic() -> GroupUpliftGate:
    return GroupUpliftGate(
        "lightgbm",
        random_state=42,
        probability_calibration="isotonic",
    ).fit(*_training_rows())


def test_isotonic_oof_is_complete_finite_and_monotone() -> None:
    gate = _fit_isotonic()
    oof = gate.calibration_oof_predictions()
    assert len(oof) == 24
    assert [row.row_index for row in oof] == list(range(24))
    for raw_field, calibrated_field in (
        ("raw_q_helpful_oof", "q_helpful_oof"),
        ("raw_q_harmful_oof", "q_harmful_oof"),
    ):
        ordered = sorted(
            (getattr(row, raw_field), getattr(row, calibrated_field)) for row in oof
        )
        assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for pair in ordered for value in pair)
        assert all(left[1] <= right[1] + 1e-15 for left, right in zip(ordered, ordered[1:]))


def test_isotonic_clip_and_audited_prediction_schema() -> None:
    gate = _fit_isotonic()
    predictions = gate.predict_audited(
        [{"score": -1_000.0, "kind": "outside"}, {"score": 1_000.0, "kind": "outside"}]
    )
    assert len(predictions) == 2
    for row in predictions:
        values = row.as_dict()
        assert values["probability_calibration"] == "isotonic"
        for field in (
            "raw_q_helpful",
            "raw_q_harmful",
            "q_helpful",
            "q_harmful",
            "sigma",
            "conservative_uplift",
        ):
            assert math.isfinite(float(values[field]))
        assert 0.0 <= row.q_helpful <= 1.0
        assert 0.0 <= row.q_harmful <= 1.0
        assert row.conservative_uplift >= 0.0


def test_isotonic_heads_are_separate_and_metadata_is_deterministic() -> None:
    first = _fit_isotonic()
    second = _fit_isotonic()
    first_metadata = first.metadata()
    assert first_metadata == second.metadata()
    calibration = first_metadata["calibration"]
    assert calibration["target_labels_used"] is False
    assert calibration["target_responses_used"] is False
    assert calibration["oof_rows"] == 24
    assert calibration["oof_families"] == ["alpha", "beta", "gamma"]
    assert calibration["helpful"]["y_thresholds_"] != calibration["harmful"]["y_thresholds_"]
    assert first_metadata["uncertainty"] == {
        "scale": "calibrated_net_gain",
        "ddof": 1,
    }


def test_single_class_oof_uses_constant_calibrators() -> None:
    features = [{"score": float(index)} for index in range(6)]
    gate = GroupUpliftGate(
        "lightgbm", probability_calibration="isotonic"
    ).fit(
        features,
        [False] * 6,
        [False] * 6,
        [True] * 6,
        ["alpha"] * 3 + ["beta"] * 3,
    )
    calibration = gate.metadata()["calibration"]
    assert calibration["constant_fallback"] == {
        "helpful": True,
        "harmful": True,
    }
    assert all(
        row.q_helpful == row.q_harmful == 0.0 for row in gate.predict(features)
    )


def test_isotonic_requires_multiple_families_and_legacy_surface_is_unchanged() -> None:
    features, baran, llm, executable, _ = _training_rows()
    with pytest.raises(ValueError, match="at least two"):
        GroupUpliftGate(
            "lightgbm", probability_calibration="isotonic"
        ).fit(features, baran, llm, executable, ["only"] * len(features))

    legacy = GroupUpliftGate("lightgbm").fit(
        features, baran, llm, executable, ["family"] * len(features)
    )
    prediction = legacy.predict(features[:1])[0]
    assert set(prediction.as_dict()) == {
        "q_helpful",
        "q_harmful",
        "net_gain",
        "sigma",
        "conservative_uplift",
    }
    assert "calibration" not in legacy.metadata()


def test_default_none_matches_explicit_none_and_calibrated_sigma_is_ddof_one() -> None:
    training = _training_rows()
    default = GroupUpliftGate("lightgbm", random_state=42).fit(*training)
    explicit = GroupUpliftGate(
        "lightgbm", random_state=42, probability_calibration="none"
    ).fit(*training)
    target = training[0][:4]
    assert [row.as_dict() for row in default.predict(target)] == [
        row.as_dict() for row in explicit.predict(target)
    ]

    calibrated = _fit_isotonic()
    audited = calibrated.predict_audited(target)
    replica_net: list[list[float]] = []
    for replica in calibrated._replicas:
        raw_helpful, raw_harmful = calibrated._predict_heads(replica, target)
        helpful = calibrated._calibrate("helpful", raw_helpful)
        harmful = calibrated._calibrate("harmful", raw_harmful)
        replica_net.append(
            [left - calibrated.rho * right for left, right in zip(helpful, harmful)]
        )
    for index, prediction in enumerate(audited):
        values = [row[index] for row in replica_net]
        mean = sum(values) / len(values)
        expected_sigma = math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        )
        assert prediction.sigma == pytest.approx(expected_sigma, abs=1e-15)


def test_oof_replica_metadata_excludes_its_family_rows() -> None:
    gate = _fit_isotonic()
    metadata = gate.metadata()
    family_counts = {"alpha": 8, "beta": 8, "gamma": 8}
    assert {
        row["family_left_out"]: row["rows"] for row in metadata["lofo"]
    } == {
        family: 24 - count for family, count in family_counts.items()
    }
