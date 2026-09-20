from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from ai_job_market.training_evidence_io import EvidenceContractError
from ai_job_market.training_partitions import (
    build_expanding_monthly_folds,
    build_partition_membership,
    monthly_row_counts,
    propose_partition,
    reconcile_exposure,
    summarize_folds,
    validate_fold_summary,
)


def dated_rows(counts: list[int], *, start: str = "2025-01") -> pd.DataFrame:
    periods = pd.period_range(start, periods=len(counts), freq="M")
    rows = []
    for period, count in zip(periods, counts, strict=True):
        rows.extend(
            {"posting_year": period.year, "posting_month": period.month, "value": i}
            for i in range(count)
        )
    return pd.DataFrame(rows)


def test_partition_proposal_uses_whole_months_and_reports_actual_shares() -> None:
    frame = dated_rows([10] * 9 + [15, 20, 5])
    proposal = propose_partition(frame)
    membership = build_partition_membership(
        frame,
        dataset_id="data-1",
        train_end=proposal.train_end,
        holdout_end=proposal.holdout_end,
    )

    assert set(membership["partition"]) == {
        "TRAIN",
        "EVALUATION_HOLDOUT",
        "INFERENCE_RESERVE",
    }
    assert membership["row_id"].is_unique
    assert len(membership) == len(frame)
    by_period = membership.groupby("period")["partition"].nunique()
    assert (by_period == 1).all()
    assert proposal.actual_counts == Counter(membership["partition"])
    assert sum(proposal.actual_shares.values()) == pytest.approx(1.0)


def test_partition_rejects_missing_dates_and_stale_boundaries() -> None:
    frame = dated_rows([2] * 12)
    frame.loc[0, "posting_month"] = None
    with pytest.raises(EvidenceContractError, match="missing"):
        propose_partition(frame)

    frame = dated_rows([2] * 12)
    with pytest.raises(EvidenceContractError, match="observed whole-month"):
        build_partition_membership(
            frame, dataset_id="data-1", train_end="2024-12", holdout_end="2025-11"
        )


def test_exposure_reconciliation_never_overrides_known_history() -> None:
    status = reconcile_exposure("attested-unexposed", known_exposure_refs=["legacy-pack:row-2"])
    assert status.status == "known-exposed"
    assert status.ready is False

    status = reconcile_exposure("unknown", known_exposure_refs=[])
    assert status.status == "unknown"
    assert status.ready is False

    status = reconcile_exposure("attested-unexposed", known_exposure_refs=[])
    assert status.ready is True


def test_five_expanding_outer_folds_show_exact_unequal_counts_and_gaps() -> None:
    # Missing 2025-04 is deliberate; folds use observed months and disclose the gap.
    frame = dated_rows([2, 3, 5], start="2025-01")
    later = dated_rows([7, 11, 13, 17, 19, 23, 29, 31, 2, 3], start="2025-05")
    frame = pd.concat([frame, later], ignore_index=True)
    membership = build_partition_membership(
        frame, dataset_id="data-1", train_end="2025-12", holdout_end="2026-01"
    )
    train = membership[membership["partition"] == "TRAIN"]
    folds = build_expanding_monthly_folds(
        train, n_splits=5, scope="outer", parent_population_id="TRAIN:data-1"
    )
    summary = summarize_folds(
        folds,
        parent_membership=train,
        holdout_rows=int((membership["partition"] == "EVALUATION_HOLDOUT").sum()),
        reserve_rows=int((membership["partition"] == "INFERENCE_RESERVE").sum()),
    )

    assert list(summary["validation_period_min"]) == [
        "2025-08",
        "2025-09",
        "2025-10",
        "2025-11",
        "2025-12",
    ]
    assert list(summary["validation_rows"]) == [17, 19, 23, 29, 31]
    assert list(summary["train_rows"]) == [41, 58, 77, 100, 129]
    assert list(summary["added_train_rows"].iloc[1:]) == [17, 19, 23, 29]
    assert summary["row_overlap_count"].eq(0).all()
    assert summary["shared_month_count"].eq(0).all()
    assert summary["chronological_order_ok"].all()
    assert summary["expanding_history_ok"].iloc[1:].all()
    assert "2025-04" in summary.iloc[0]["missing_calendar_months"]
    assert summary["holdout_overlap_count"].eq(0).all()
    assert summary["reserve_overlap_count"].eq(0).all()
    validate_fold_summary(summary, folds, train)


def test_nested_folds_are_within_parent_and_monthly_counts_reconcile() -> None:
    frame = dated_rows([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    membership = build_partition_membership(
        frame, dataset_id="data-1", train_end="2025-10", holdout_end="2025-11"
    )
    train = membership[membership["partition"] == "TRAIN"]
    outer = build_expanding_monthly_folds(
        train, n_splits=5, scope="outer", parent_population_id="TRAIN:data-1"
    )
    first_outer_train_ids = set(outer[0].train_row_ids)
    parent = train[train["row_id"].isin(first_outer_train_ids)]
    inner = build_expanding_monthly_folds(
        parent,
        n_splits=3,
        scope="inner",
        parent_population_id=outer[0].fold_id,
        parent_fold_id=outer[0].fold_id,
        search_id="search-outer-1",
    )

    assert len(inner) == 3
    assert all(set(f.train_row_ids + f.validation_row_ids) <= first_outer_train_ids for f in inner)
    counts = monthly_row_counts(membership)
    assert counts["row_count"].sum() == len(frame)
    for fold in outer:
        count_lookup = counts.set_index("period")["row_count"]
        assert sum(count_lookup[p] for p in fold.train_months) == len(fold.train_row_ids)
        assert sum(count_lookup[p] for p in fold.validation_months) == len(
            fold.validation_row_ids
        )


def test_fold_summary_rejects_count_mutation() -> None:
    frame = dated_rows([2] * 12)
    membership = build_partition_membership(
        frame, dataset_id="data-1", train_end="2025-10", holdout_end="2025-11"
    )
    train = membership[membership["partition"] == "TRAIN"]
    folds = build_expanding_monthly_folds(
        train, n_splits=5, scope="outer", parent_population_id="TRAIN:data-1"
    )
    summary = summarize_folds(folds, parent_membership=train, holdout_rows=2, reserve_rows=2)
    summary.loc[0, "train_rows"] += 1
    with pytest.raises(EvidenceContractError, match="does not reconcile"):
        validate_fold_summary(summary, folds, train)
