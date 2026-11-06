"""Deterministic provider-stratified outer-fold construction.

Fold construction uses participant identifiers and providers only.  Callers
should construct this table once from the pooled participant universe and join
it to every assay and model input.
"""

from __future__ import annotations

import hashlib
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

OUTER_FOLDS = 5
DEFAULT_OUTER_FOLD_SEED = "multipanel-selective-testing-outer-fold-v1"


class FoldAssignmentError(ValueError):
    """Raised when a participant registry cannot define valid outer folds."""


def _stable_scalar_key(value: Any, *, column: str, position: int) -> bytes:
    """Encode one non-missing scalar without relying on Python hash order."""

    if isinstance(value, np.generic):
        value = value.item()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError) as error:
        raise FoldAssignmentError(
            f"{column} at row position {position} is not a supported scalar"
        ) from error
    if not isinstance(missing, (bool, np.bool_)):
        raise FoldAssignmentError(
            f"{column} at row position {position} is not a supported scalar"
        )
    if bool(missing):
        raise FoldAssignmentError(f"{column} contains a missing value")
    try:
        hash(value)
    except TypeError as error:
        raise FoldAssignmentError(
            f"{column} at row position {position} is not hashable"
        ) from error

    value_type = type(value)
    type_name = f"{value_type.__module__}.{value_type.__qualname__}".encode()
    payload = value if isinstance(value, bytes) else str(value).encode()
    return (
        len(type_name).to_bytes(4, "big")
        + type_name
        + len(payload).to_bytes(8, "big")
        + payload
    )


def _sha256_digest(seed: str, purpose: str, *parts: bytes) -> bytes:
    hasher = hashlib.sha256()
    for part in (seed.encode("utf-8"), purpose.encode("utf-8"), *parts):
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.digest()


def _validate_columns(
    rows: pd.DataFrame,
    *,
    participant_column: str,
    provider_column: str,
    fold_column: str,
) -> None:
    names = (participant_column, provider_column, fold_column)
    if any(not isinstance(name, str) or not name for name in names):
        raise FoldAssignmentError("column names must be non-empty strings")
    if len(set(names)) != len(names):
        raise FoldAssignmentError("participant, provider, and fold columns must be distinct")
    missing = [
        name for name in (participant_column, provider_column) if name not in rows.columns
    ]
    if missing:
        raise FoldAssignmentError(f"missing required columns: {', '.join(missing)}")


def assign_provider_stratified_folds(
    rows: pd.DataFrame,
    *,
    participant_column: str = "participant_id",
    provider_column: str = "provider",
    fold_column: str = "outer_fold",
    n_folds: int = OUTER_FOLDS,
    seed: str = DEFAULT_OUTER_FOLD_SEED,
) -> pd.DataFrame:
    """Return one deterministic provider-balanced fold row per participant.

    Participants are ordered within provider by a seeded SHA-256 digest, then
    assigned round-robin.  A deterministic fold permutation gives priority to
    the currently smallest folds, so both the provider-specific and overall
    fold counts differ by at most one.  Extra input columns are intentionally
    ignored so assay identifiers and outcomes cannot affect assignment.

    Repeated rows for one participant are collapsed when their provider agrees.
    A participant associated with more than one provider is rejected.
    Returned participants retain first-appearance order.
    """

    if not isinstance(rows, pd.DataFrame):
        raise TypeError("rows must be a pandas DataFrame")
    _validate_columns(
        rows,
        participant_column=participant_column,
        provider_column=provider_column,
        fold_column=fold_column,
    )
    if isinstance(n_folds, bool) or not isinstance(n_folds, Integral) or n_folds < 2:
        raise FoldAssignmentError("n_folds must be an integer of at least two")
    n_folds = int(n_folds)
    if not isinstance(seed, str) or not seed:
        raise FoldAssignmentError("seed must be a non-empty string")
    if rows.empty:
        raise FoldAssignmentError("participant rows must be non-empty")

    # Keys are serialized scalars rather than Python-hash values, making both
    # duplicate detection and ordering stable across interpreter processes.
    participant_records: dict[bytes, tuple[int, bytes]] = {}
    provider_to_participants: dict[bytes, list[bytes]] = {}
    participant_values = rows[participant_column].tolist()
    provider_values = rows[provider_column].tolist()
    for position, (participant, provider) in enumerate(
        zip(participant_values, provider_values, strict=True)
    ):
        participant_key = _stable_scalar_key(
            participant,
            column=participant_column,
            position=position,
        )
        provider_key = _stable_scalar_key(
            provider,
            column=provider_column,
            position=position,
        )
        prior = participant_records.get(participant_key)
        if prior is not None:
            if prior[1] != provider_key:
                raise FoldAssignmentError(
                    "duplicate participant rows have conflicting providers"
                )
            continue
        participant_records[participant_key] = (position, provider_key)
        provider_to_participants.setdefault(provider_key, []).append(participant_key)

    if len(participant_records) < n_folds:
        raise FoldAssignmentError(
            f"at least {n_folds} distinct participants are required; "
            f"got {len(participant_records)}"
        )

    ordered_providers = sorted(
        provider_to_participants,
        key=lambda provider: (
            _sha256_digest(seed, "provider-order", provider),
            provider,
        ),
    )
    global_counts = np.zeros(n_folds, dtype=np.int64)
    participant_to_fold: dict[bytes, int] = {}
    for provider in ordered_providers:
        ordered_participants = sorted(
            provider_to_participants[provider],
            key=lambda participant: (
                _sha256_digest(seed, "participant-order", provider, participant),
                participant,
            ),
        )
        fold_order = sorted(
            range(n_folds),
            key=lambda fold: (
                int(global_counts[fold]),
                _sha256_digest(
                    seed,
                    "fold-order",
                    provider,
                    fold.to_bytes(8, "big"),
                ),
                fold,
            ),
        )
        for rank, participant in enumerate(ordered_participants):
            fold = fold_order[rank % n_folds]
            participant_to_fold[participant] = fold
            global_counts[fold] += 1

    ordered_records = list(participant_records.items())
    first_positions = [record[0] for _, record in ordered_records]
    result = rows.iloc[first_positions][[participant_column, provider_column]].copy()
    result.reset_index(drop=True, inplace=True)
    result[fold_column] = np.fromiter(
        (participant_to_fold[participant] for participant, _ in ordered_records),
        dtype=np.int64,
        count=len(ordered_records),
    )
    return result


def full_opportunity_registrations(
    registrations: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    provider_ids: set[int],
    opportunity_days: int = 365,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select people with exactly one provider having a qualifying spell.

    Exact duplicate source rows represent one canonical spell.  Distinct
    overlapping spells from the same provider are allowed; no spells are
    stitched.  More than one qualifying provider causes exclusion.
    """

    required = {"eid", "data_provider", "reg_date", "deduct_date"}
    missing = required - set(registrations.columns)
    if missing:
        raise ValueError(f"registration columns missing: {sorted(missing)}")
    if opportunity_days != 365:
        raise ValueError("Cohort requires a 365-day opportunity")
    if baseline["eid"].duplicated().any():
        raise ValueError("baseline must contain one row per participant")

    frame = registrations.loc[:, sorted(required)].copy()
    frame["eid"] = pd.to_numeric(frame["eid"], errors="coerce").astype("Int64")
    frame["provider_id"] = pd.to_numeric(
        frame["data_provider"], errors="coerce"
    ).astype("Int64")
    frame["reg_date"] = pd.to_datetime(
        frame["reg_date"].astype("string"), format="%d/%m/%Y", errors="coerce"
    )
    frame["deduct_date"] = pd.to_datetime(
        frame["deduct_date"].astype("string"), format="%d/%m/%Y", errors="coerce"
    )
    frame = frame.dropna(subset=["eid", "provider_id", "reg_date"]).copy()
    frame["eid"] = frame["eid"].astype("int64")
    frame["provider_id"] = frame["provider_id"].astype("int64")
    frame = frame.loc[frame["provider_id"].isin(provider_ids)].copy()
    frame = frame.drop_duplicates(
        ["eid", "provider_id", "reg_date", "deduct_date"], keep="first"
    )
    frame = frame.merge(baseline, on="eid", how="inner", validate="many_to_one")
    opportunity_start = frame["baseline_date"] - pd.Timedelta(days=opportunity_days)
    qualifying = frame.loc[
        frame["reg_date"].le(opportunity_start)
        & (frame["deduct_date"].isna() | frame["deduct_date"].ge(frame["baseline_date"]))
    ].copy()

    provider_counts = qualifying.groupby("eid", sort=False)["provider_id"].nunique()
    eligible_eids = provider_counts.index[provider_counts.eq(1)]
    eligible = (
        qualifying.loc[qualifying["eid"].isin(eligible_eids), ["eid", "provider_id"]]
        .drop_duplicates()
        .sort_values(["eid", "provider_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if eligible["eid"].duplicated().any():
        raise AssertionError("eligible cohort must have one row per participant")
    audit = {
        "registration_rows_loaded": int(len(registrations)),
        "canonical_registration_spells_in_current_baseline_cohort": int(len(frame)),
        "qualifying_full_opportunity_spells": int(len(qualifying)),
        "participants_with_any_qualifying_spell": int(provider_counts.size),
        "participants_with_multiple_qualifying_providers": int(
            provider_counts.gt(1).sum()
        ),
        "full_opportunity_eligible_participants": int(len(eligible)),
    }
    return eligible, audit


def build_365_day_fold_cohort(
    baseline: pd.DataFrame,
    registrations: pd.DataFrame,
    *,
    provider_ids: set[int],
    n_folds: int = 5,
    seed: str = "multipanel-selective-testing-outer-fold-v1",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the assay-blind 365-day full-opportunity cohort and audit.

    ``baseline`` must contain one row per participant and at least ``eid`` and
    ``baseline_date``. Participant fold assignment uses only ``eid`` and the
    uniquely qualifying provider. Assay values and historical test indicators
    must not be supplied to this function.
    """

    required = {"eid", "baseline_date"}
    if missing := required - set(baseline):
        raise ValueError(f"baseline is missing required columns: {sorted(missing)}")
    prohibited_prefixes = ("assay", "outcome", "target", "selected__")
    prohibited = [
        column
        for column in baseline
        if any(str(column).lower().startswith(prefix) for prefix in prohibited_prefixes)
    ]
    if prohibited:
        raise ValueError("baseline contains prohibited assay/outcome columns")
    if baseline["eid"].duplicated().any():
        raise ValueError("baseline must contain one row per participant")

    eligible, registration_audit = full_opportunity_registrations(
        registrations,
        baseline.loc[:, ["eid", "baseline_date"]],
        provider_ids=provider_ids,
        opportunity_days=365,
    )
    fold_input = eligible.rename(
        columns={"eid": "participant_id", "provider_id": "provider"}
    )
    assigned = assign_provider_stratified_folds(
        fold_input,
        participant_column="participant_id",
        provider_column="provider",
        fold_column="outer_fold",
        n_folds=n_folds,
        seed=seed,
    ).rename(columns={"participant_id": "eid", "provider": "provider_id"})
    assigned = assigned.sort_values("eid", kind="mergesort").reset_index(drop=True)

    provider_fold_counts = (
        assigned.groupby(["provider_id", "outer_fold"], sort=True)
        .size()
        .rename("participant_count")
        .reset_index()
    )
    audit: dict[str, Any] = {
        "cohort_definition": "one provider with an individual registration spell spanning "
        "baseline minus 365 days through baseline",
        "same_day_predictor_events": "excluded downstream",
        "fold_assignment_inputs": ["eid", "provider_id"],
        "fold_seed": seed,
        "n_folds": int(n_folds),
        "eligible_participants": int(len(assigned)),
        "registration_audit": registration_audit,
        "fold_counts": {
            str(int(fold)): int(count)
            for fold, count in assigned["outer_fold"].value_counts(sort=False).sort_index().items()
        },
        "provider_fold_counts": [
            {
                "provider_id": int(row.provider_id),
                "outer_fold": int(row.outer_fold),
                "participant_count": int(row.participant_count),
            }
            for row in provider_fold_counts.itertuples(index=False)
        ],
    }
    return assigned, audit


def main():
    from common import DATA, PRIVATE, setup, fresh
    setup()
    output = fresh(PRIVATE / 'participants.parquet')
    # Baseline must already exclude withdrawals and ineligible participants.
    baseline = pd.read_parquet(DATA / 'cohort_baseline.parquet')
    required = ['eid', 'baseline_date', 'age_years', 'sex_field31']
    if not set(required).issubset(baseline):
        raise ValueError('Cohort baseline columns are incomplete; see README.md.')
    baseline['baseline_date'] = pd.to_datetime(baseline['baseline_date'])
    if baseline['baseline_date'].isna().any():
        raise ValueError('Cohort baseline dates must be present.')
    registrations = pd.read_csv(DATA / 'registrations.tsv', sep='\t', dtype='string')
    folds, _ = build_365_day_fold_cohort(baseline[required], registrations, provider_ids={1, 2, 3, 4})
    participants = folds.merge(baseline[['eid', 'age_years', 'sex_field31']], on='eid', validate='one_to_one')
    participants.to_parquet(output, index=False)
    output.chmod(0o600)
    print('Cohort and fixed folds completed.')


if __name__ == '__main__':
    main()
