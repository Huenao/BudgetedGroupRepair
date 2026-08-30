from __future__ import annotations

import pandas as pd
import pytest

from budgeted_group_repair_no_baran.data import SafeCell
from budgeted_group_repair_no_baran.public_fd import PublicFD
from budgeted_group_repair_no_baran.verifier import (
    GroupRepairVerifier,
    RankedRepairCandidate,
)


def _verifier_fixture() -> tuple[
    GroupRepairVerifier,
    SafeCell,
    dict[str, object],
]:
    dirty = pd.DataFrame(
        {
            "zip": ["100", "100", "100", "100", "100"],
            "city": ["Paris", "Paris", "Rome", "Rome", "Rome"],
        }
    )
    cell = SafeCell("source", "toy", 0, 1, "city", "0", "Paris")
    verifier = GroupRepairVerifier(
        dirty,
        (cell,),
        (PublicFD("zip_to_city", ("zip",), "city"),),
    )
    baran = {
        "prediction": "Lyon",
        "parse_status": "ok",
    }
    return verifier, cell, baran


def _proposal(repair: str, confidence: float = 0.9) -> dict[str, object]:
    return {
        "parse_status": "ok_item",
        "decision": "propose",
        "repair": repair,
        "confidence": confidence,
    }


def test_verifier_accepts_supported_fd_improvement() -> None:
    verifier, cell, baran = _verifier_fixture()
    decision = verifier.verify(
        cell,
        baran,
        _proposal("Rome"),
        0.5,
        query_id="q-accept",
    )

    assert decision.accept_llm is True
    assert decision.final_prediction == "Rome"
    assert decision.final_source == "llm"
    assert decision.reason == "accepted"
    assert decision.query_id == "q-accept"
    assert decision.support_advantage is True
    assert decision.fd_violations_before == 3
    assert decision.fd_violations_after == 1
    assert decision.score == pytest.approx(0.96)


@pytest.mark.parametrize(
    ("repair", "confidence", "uplift", "reason"),
    (
        ("", 0.9, 0.5, "empty_repair"),
        ("Paris", 0.9, 0.5, "unchanged_dirty_value"),
        ("Lyon", 0.9, 0.5, "equivalent_to_baran"),
        ("Rome", 0.54, 0.5, "low_llm_confidence"),
        ("Rome", 0.9, 0.0, "non_positive_predicted_gain"),
        ("Madrid", 0.9, 0.5, "public_fd_worse"),
    ),
)
def test_verifier_rejection_contract(
    repair: str,
    confidence: float,
    uplift: float,
    reason: str,
) -> None:
    verifier, cell, baran = _verifier_fixture()
    decision = verifier.verify(
        cell,
        baran,
        _proposal(repair, confidence),
        uplift,
        query_id="q-reject",
    )

    assert decision.accept_llm is False
    assert decision.final_prediction == "Lyon"
    assert decision.final_source == "baran"
    assert decision.reason == reason


def test_verifier_arbitration_order_and_base_fallback() -> None:
    verifier, cell, baran = _verifier_fixture()
    accepted = verifier.arbitrate(
        cell,
        baran,
        (
            RankedRepairCandidate("q-lower", _proposal("Rome"), 0.5, 1, 1),
            RankedRepairCandidate("q-higher", _proposal("Rome", 0.1), 0.8, 9, 4),
        ),
    )
    assert accepted.attempted_query_ids == ("q-higher", "q-lower")
    assert accepted.rejected_reasons == ("low_llm_confidence",)
    assert accepted.decision.accept_llm is True
    assert accepted.decision.query_id == "q-lower"

    low_confidence = _proposal("Rome", 0.1)
    rejected = verifier.arbitrate(
        cell,
        baran,
        (
            RankedRepairCandidate("q-cost", low_confidence, 0.4, 2, 1),
            RankedRepairCandidate("q-size", low_confidence, 0.4, 1, 2),
            RankedRepairCandidate("q-z", low_confidence, 0.4, 1, 1),
            RankedRepairCandidate("q-a", low_confidence, 0.4, 1, 1),
        ),
    )
    assert rejected.attempted_query_ids == ("q-a", "q-z", "q-size", "q-cost")
    assert rejected.rejected_reasons == ("low_llm_confidence",) * 4
    assert rejected.decision.accept_llm is False
    assert rejected.decision.final_prediction == "Lyon"
    assert rejected.decision.reason == "all_candidates_rejected"

    no_candidate = verifier.arbitrate(cell, baran, ())
    assert no_candidate.decision.final_prediction == "Lyon"
    assert no_candidate.decision.reason == "no_candidate"

    invalid_base = verifier.arbitrate(
        cell,
        {"prediction": "Lyon", "parse_status": "failed"},
        (),
    )
    assert invalid_base.decision.final_prediction == "Paris"
