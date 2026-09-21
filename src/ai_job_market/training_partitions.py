"""Target-independent temporal partitions and expanding monthly folds."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .training_evidence_io import EvidenceContractError

PARTITIONS = ("TRAIN", "EVALUATION_HOLDOUT", "INFERENCE_RESERVE")


@dataclass(frozen=True)
class PartitionProposal:
    train_end: str
    holdout_end: str
    actual_counts: Counter[str]
    actual_shares: dict[str, float]
    objective: float


@dataclass(frozen=True)
class ExposureStatus:
    status: str
    ready: bool
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class FoldDeclaration:
    fold_id: str
    scope: str
    parent_population_id: str
    parent_fold_id: str | None
    search_id: str | None
    ordinal: int
    requested_fold_count: int
    train_row_ids: tuple[str, ...]
    validation_row_ids: tuple[str, ...]
    train_months: tuple[str, ...]
    validation_months: tuple[str, ...]
    membership_hash: str


@dataclass(frozen=True)
class DeclaredTemporalSplitter:
    """Expose validated declared temporal folds to scikit-learn CV consumers."""

    folds: tuple[tuple[np.ndarray, np.ndarray], ...]

    def split(self, X, y=None, groups=None):
        del X, y, groups
        yield from self.folds

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        del X, y, groups
        return len(self.folds)


def temporal_fold_splitter(
    membership: pd.DataFrame, folds: list[FoldDeclaration]
) -> tuple[DeclaredTemporalSplitter, np.ndarray]:
    """Return local positions for declared folds without admitting protected rows."""
    if not folds:
        raise EvidenceContractError("GridSearchCV requires at least one declared temporal fold")
    required = {"row_id", "source_position", "period", "partition"}
    if required - set(membership.columns):
        raise EvidenceContractError("membership is missing temporal splitter fields")
    if membership["row_id"].duplicated().any() or membership["source_position"].duplicated().any():
        raise EvidenceContractError("membership row identity is not unique")
    lookup = membership.set_index("row_id")["source_position"]
    partition_lookup = membership.set_index("row_id")["partition"]
    period_lookup = membership.set_index("row_id")["period"]
    row_ids = {row_id for fold in folds for row_id in fold.train_row_ids + fold.validation_row_ids}
    try:
        if set(partition_lookup.loc[list(row_ids)]) != {"TRAIN"}:
            raise EvidenceContractError("GridSearchCV fold contains a protected partition row")
        positions = np.asarray(sorted(lookup.loc[list(row_ids)].astype(int)), dtype=int)
    except KeyError as error:
        raise EvidenceContractError("fold row identity is absent from membership") from error
    local = {int(position): index for index, position in enumerate(positions)}
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        if not fold.train_row_ids or not fold.validation_row_ids:
            raise EvidenceContractError("GridSearchCV fold requires train and validation rows")
        if max(fold.train_months) >= min(fold.validation_months):
            raise EvidenceContractError("GridSearchCV fold chronology is invalid")
        actual_train_months = set(period_lookup.loc[list(fold.train_row_ids)])
        actual_validation_months = set(period_lookup.loc[list(fold.validation_row_ids)])
        if actual_train_months != set(fold.train_months) or actual_validation_months != set(
            fold.validation_months
        ):
            raise EvidenceContractError("GridSearchCV fold periods do not match row membership")
        try:
            train = np.asarray(
                [local[int(lookup[row_id])] for row_id in fold.train_row_ids], dtype=int
            )
            validation = np.asarray(
                [local[int(lookup[row_id])] for row_id in fold.validation_row_ids], dtype=int
            )
        except KeyError as error:
            raise EvidenceContractError("fold row identity is absent from membership") from error
        if set(train) & set(validation):
            raise EvidenceContractError("GridSearchCV fold train/validation overlap")
        splits.append((train, validation))
    return DeclaredTemporalSplitter(tuple(splits)), positions


def _periods(frame: pd.DataFrame) -> pd.Series:
    required = {"posting_year", "posting_month"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise EvidenceContractError(f"missing date columns: {missing_columns}")
    if frame[list(required)].isna().any().any():
        raise EvidenceContractError("missing posting year or month")
    try:
        years = pd.to_numeric(frame["posting_year"], errors="raise")
        months = pd.to_numeric(frame["posting_month"], errors="raise")
    except (TypeError, ValueError) as error:
        raise EvidenceContractError("posting year/month must be integers") from error
    if not ((years % 1 == 0).all() and (months % 1 == 0).all()):
        raise EvidenceContractError("posting year/month must be integers")
    if not years.between(1900, 2100).all() or not months.between(1, 12).all():
        raise EvidenceContractError("posting year/month is outside the permitted range")
    return pd.Series(
        [f"{int(year):04d}-{int(month):02d}" for year, month in zip(years, months, strict=True)],
        index=frame.index,
        dtype="string",
    )


def _row_id(dataset_id: str, position: int) -> str:
    return hashlib.sha256(f"{dataset_id}:{position}".encode()).hexdigest()


def build_partition_membership(
    frame: pd.DataFrame, *, dataset_id: str, train_end: str, holdout_end: str
) -> pd.DataFrame:
    periods = _periods(frame).reset_index(drop=True)
    observed = sorted(periods.unique())
    if train_end not in observed or holdout_end not in observed:
        raise EvidenceContractError("partition boundaries must be observed whole-month values")
    if not train_end < holdout_end:
        raise EvidenceContractError("partition boundaries must be strictly chronological")
    partition = periods.map(
        lambda period: (
            "TRAIN"
            if period <= train_end
            else "EVALUATION_HOLDOUT"
            if period <= holdout_end
            else "INFERENCE_RESERVE"
        )
    )
    counts = Counter(partition)
    if any(counts[name] == 0 for name in PARTITIONS):
        raise EvidenceContractError("all three whole-month partitions must be nonempty")
    return pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "row_id": [_row_id(dataset_id, position) for position in range(len(frame))],
            "source_position": range(len(frame)),
            "period": periods,
            "partition": partition,
        }
    )


def propose_partition(
    frame: pd.DataFrame,
    targets: tuple[float, float, float] = (0.80, 0.19, 0.01),
    *,
    minimum_train_months: int = 9,
) -> PartitionProposal:
    periods = _periods(frame).reset_index(drop=True)
    counts = periods.value_counts().sort_index()
    months = list(counts.index)
    if len(months) < minimum_train_months + 2:
        raise EvidenceContractError(
            f"at least {minimum_train_months + 2} observed months are required"
        )
    total = len(frame)
    candidates: list[tuple[tuple[float, float, float, str, str], PartitionProposal]] = []
    for train_count in range(minimum_train_months, len(months) - 1):
        for holdout_end_index in range(train_count, len(months) - 1):
            train_end = months[train_count - 1]
            holdout_end = months[holdout_end_index]
            partition_counts = Counter(
                {
                    "TRAIN": int(counts.iloc[:train_count].sum()),
                    "EVALUATION_HOLDOUT": int(
                        counts.iloc[train_count : holdout_end_index + 1].sum()
                    ),
                    "INFERENCE_RESERVE": int(counts.iloc[holdout_end_index + 1 :].sum()),
                }
            )
            shares = {name: partition_counts[name] / total for name in PARTITIONS}
            objective = sum(
                (shares[name] - target) ** 2
                for name, target in zip(PARTITIONS, targets, strict=True)
            )
            proposal = PartitionProposal(
                train_end=train_end,
                holdout_end=holdout_end,
                actual_counts=partition_counts,
                actual_shares=shares,
                objective=objective,
            )
            ranking = (
                objective,
                abs(shares["TRAIN"] - targets[0]),
                abs(shares["INFERENCE_RESERVE"] - targets[2]),
                train_end,
                holdout_end,
            )
            candidates.append((ranking, proposal))
    if not candidates:
        raise EvidenceContractError("no valid whole-month partition proposal exists")
    return min(candidates, key=lambda item: item[0])[1]


def reconcile_exposure(attestation: str, *, known_exposure_refs: Iterable[str]) -> ExposureStatus:
    references = tuple(sorted(set(known_exposure_refs)))
    if references:
        return ExposureStatus("known-exposed", False, references)
    if attestation == "attested-unexposed":
        return ExposureStatus(attestation, True, ())
    if attestation == "known-exposed":
        return ExposureStatus(attestation, False, ())
    if attestation == "unknown":
        return ExposureStatus(attestation, False, ())
    raise EvidenceContractError("invalid exposure attestation")


def _membership_hash(train_ids: tuple[str, ...], validation_ids: tuple[str, ...]) -> str:
    material = "|".join(train_ids) + "::" + "|".join(validation_ids)
    return hashlib.sha256(material.encode()).hexdigest()


def build_expanding_monthly_folds(
    parent_membership: pd.DataFrame,
    *,
    n_splits: int,
    scope: str,
    parent_population_id: str,
    parent_fold_id: str | None = None,
    search_id: str | None = None,
) -> list[FoldDeclaration]:
    required = {"row_id", "period"}
    if required - set(parent_membership.columns):
        raise EvidenceContractError("parent membership is missing row_id or period")
    if parent_membership.empty:
        raise EvidenceContractError("fold parent population is empty")
    months = sorted(parent_membership["period"].unique())
    if len(months) < n_splits + 1:
        raise EvidenceContractError(
            f"{scope} folds require at least {n_splits + 1} observed parent months"
        )
    validation_months = months[-n_splits:]
    folds: list[FoldDeclaration] = []
    for ordinal, validation_month in enumerate(validation_months, start=1):
        train_months = tuple(month for month in months if month < validation_month)
        train_rows = parent_membership[parent_membership["period"].isin(train_months)]
        validation_rows = parent_membership[parent_membership["period"] == validation_month]
        train_ids = tuple(train_rows["row_id"])
        validation_ids = tuple(validation_rows["row_id"])
        fold_id = f"{scope}-{ordinal}"
        if parent_fold_id:
            fold_id = f"{parent_fold_id}:{fold_id}"
        folds.append(
            FoldDeclaration(
                fold_id=fold_id,
                scope=scope,
                parent_population_id=parent_population_id,
                parent_fold_id=parent_fold_id,
                search_id=search_id,
                ordinal=ordinal,
                requested_fold_count=n_splits,
                train_row_ids=train_ids,
                validation_row_ids=validation_ids,
                train_months=train_months,
                validation_months=(validation_month,),
                membership_hash=_membership_hash(train_ids, validation_ids),
            )
        )
    return folds


def _missing_calendar_months(months: list[str]) -> list[str]:
    expected = pd.period_range(months[0], months[-1], freq="M").astype(str).tolist()
    return [month for month in expected if month not in months]


def summarize_folds(
    folds: list[FoldDeclaration],
    *,
    parent_membership: pd.DataFrame,
    holdout_rows: int,
    reserve_rows: int,
) -> pd.DataFrame:
    parent_ids = set(parent_membership["row_id"])
    parent_rows = len(parent_membership)
    parent_periods = sorted(parent_membership["period"].unique())
    missing_months = _missing_calendar_months(parent_periods)
    rows: list[dict[str, object]] = []
    previous_train: set[str] | None = None
    for fold in folds:
        train_ids = set(fold.train_row_ids)
        validation_ids = set(fold.validation_row_ids)
        later_ids = parent_ids - train_ids - validation_ids
        added_ids = set() if previous_train is None else train_ids - previous_train
        added_months = sorted(
            parent_membership.loc[parent_membership["row_id"].isin(added_ids), "period"].unique()
        )
        train_periods = sorted(
            parent_membership.loc[parent_membership["row_id"].isin(train_ids), "period"].unique()
        )
        validation_periods = sorted(
            parent_membership.loc[
                parent_membership["row_id"].isin(validation_ids), "period"
            ].unique()
        )
        overlap = train_ids & validation_ids
        shared_months = set(train_periods) & set(validation_periods)
        expansion_ok = None if previous_train is None else previous_train < train_ids
        rows.append(
            {
                "fold_id": fold.fold_id,
                "scope": fold.scope,
                "parent_fold_id": fold.parent_fold_id,
                "search_id": fold.search_id,
                "method_id": "expanding_monthly_v1",
                "requested_fold_count": fold.requested_fold_count,
                "effective_fold_count": len(folds),
                "ordinal": fold.ordinal,
                "parent_population_id": fold.parent_population_id,
                "parent_population_rows": parent_rows,
                "train_period_min": train_periods[0],
                "train_period_max": train_periods[-1],
                "validation_period_min": validation_periods[0],
                "validation_period_max": validation_periods[-1],
                "train_months": train_periods,
                "validation_months": validation_periods,
                "train_month_count": len(train_periods),
                "validation_month_count": len(validation_periods),
                "missing_calendar_months": missing_months,
                "train_rows": len(train_ids),
                "validation_rows": len(validation_ids),
                "not_used_yet_rows": len(later_ids),
                "train_pct_of_parent": 100 * len(train_ids) / parent_rows,
                "validation_pct_of_parent": 100 * len(validation_ids) / parent_rows,
                "not_used_yet_pct_of_parent": 100 * len(later_ids) / parent_rows,
                "previous_fold_id": rows[-1]["fold_id"] if rows else None,
                "added_train_rows": None if previous_train is None else len(added_ids),
                "added_train_months": None if previous_train is None else added_months,
                "row_overlap_count": len(overlap),
                "shared_month_count": len(shared_months),
                "chronological_order_ok": max(train_periods) < min(validation_periods),
                "expanding_history_ok": expansion_ok,
                "holdout_overlap_count": 0,
                "reserve_overlap_count": 0,
                "holdout_rows_excluded": int(holdout_rows),
                "reserve_rows_excluded": int(reserve_rows),
                "membership_hash": fold.membership_hash,
                "evidence_ref": f"fold_membership.csv#fold_id={fold.fold_id}",
            }
        )
        previous_train = train_ids
    return pd.DataFrame(rows)


def validate_fold_summary(
    summary: pd.DataFrame, folds: list[FoldDeclaration], parent_membership: pd.DataFrame
) -> None:
    if len(summary) != len(folds):
        raise EvidenceContractError("fold summary does not reconcile with declarations")
    parent_rows = len(parent_membership)
    by_id = summary.set_index("fold_id")
    for fold in folds:
        if fold.fold_id not in by_id.index:
            raise EvidenceContractError("fold summary does not reconcile with declarations")
        row = by_id.loc[fold.fold_id]
        expected = (
            len(fold.train_row_ids),
            len(fold.validation_row_ids),
            parent_rows - len(fold.train_row_ids) - len(fold.validation_row_ids),
        )
        actual = (int(row.train_rows), int(row.validation_rows), int(row.not_used_yet_rows))
        if actual != expected or sum(actual) != parent_rows:
            raise EvidenceContractError("fold summary does not reconcile with row membership")
        if int(row.row_overlap_count) != 0 or int(row.shared_month_count) != 0:
            raise EvidenceContractError("fold summary does not reconcile with leakage assertions")
        if not bool(row.chronological_order_ok):
            raise EvidenceContractError("fold summary does not reconcile with chronology")


def monthly_row_counts(membership: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset_id", "period", "partition", "row_id"}
    if required - set(membership.columns):
        raise EvidenceContractError("membership is missing monthly count fields")
    grouped = (
        membership.groupby(["dataset_id", "period", "partition"], sort=True, observed=True)
        .agg(row_count=("row_id", "nunique"))
        .reset_index()
    )
    grouped["evidence_ref"] = grouped.apply(
        lambda row: f"partition_membership.csv#period={row.period}", axis=1
    )
    return grouped
