"""Build pre-enrollment features after excluding supplied assay and panel codes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from event_features import (
    EVENT_COLUMNS,
    SOURCES,
    EventFeatureAccumulator,
    collapse_ethnicity_top_level,
    load_feature_baseline,
    registration_tenure,
)

from preprocessing import FeatureSchema


@dataclass(frozen=True)
class SelectionCode:
    code_system: str
    code: str


@dataclass(frozen=True)
class FeatureAudit:
    participants: int
    input_event_batches: int
    input_event_rows: int
    focal_or_panel_rows_removed: int
    retained_event_rows_passed_to_builder: int
    eligible_prelandmark_rows: int
    hashed_rows: int
    unique_hashed_tokens: int
    feature_columns: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _canonical(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("").str.strip().str.casefold()


def remove_registered_test_rows(
    frame: pd.DataFrame,
    codes: tuple[SelectionCode, ...],
) -> tuple[pd.DataFrame, int]:
    """Remove focal analyte and panel aliases before utilization and hashing."""

    if not set(EVENT_COLUMNS).issubset(frame):
        raise ValueError("event batch lacks the established feature columns")
    registered = {
        (code.code_system.strip().casefold(), code.code.strip().casefold())
        for code in codes
    }
    if not registered:
        raise ValueError("at least one registered assay or panel code is required")
    gp = (
        frame["source_table"]
        .astype("string")
        .eq("gp_clinical")
        .to_numpy(dtype=bool, na_value=False)
    )
    systems = _canonical(frame["code_system"])
    raw = _canonical(frame["code_raw"])
    normalized = _canonical(frame["code_normalized"])
    focal = gp & np.fromiter(
        (
            (system, raw_code) in registered or (system, normalized_code) in registered
            for system, raw_code, normalized_code in zip(
                systems, raw, normalized, strict=True
            )
        ),
        dtype=bool,
        count=len(frame),
    )
    return frame.loc[~focal].copy(), int(np.sum(focal))


def build_process_reduced_features(
    participants: pd.DataFrame,
    *,
    baseline_path: Path,
    registrations_path: Path,
    events_path: Path,
    selection_codes: tuple[SelectionCode, ...],
    batch_size: int = 1_000_000,
) -> tuple[pd.DataFrame, FeatureSchema, FeatureAudit]:
    """Build the established representation after dynamic assay/panel removal."""

    required = {"eid", "provider_id", "age_years", "sex_field31"}
    if missing := required - set(participants):
        raise ValueError(f"participant frame is missing: {sorted(missing)}")
    if participants.empty or participants["eid"].duplicated().any() or batch_size <= 0:
        raise ValueError("participant feature input is invalid")
    ordered = participants.sort_values("eid", kind="mergesort").reset_index(drop=True).copy()
    participant_ids = ordered["eid"].astype("int64").to_numpy()
    baseline = load_feature_baseline(baseline_path, set(participant_ids.tolist()))
    baseline = baseline.set_index("eid").loc[participant_ids].reset_index()
    provider_by_eid = dict(
        zip(participant_ids, ordered["provider_id"].astype(int), strict=True)
    )
    tenure = registration_tenure(registrations_path, baseline, provider_by_eid)
    tenure = tenure.loc[participant_ids]
    accumulator = EventFeatureAccumulator(
        participant_ids,
        baseline["baseline_date"].to_numpy(dtype="datetime64[ns]"),
        set(),
    )
    scanner = ds.dataset(events_path, format="parquet").scanner(
        columns=list(EVENT_COLUMNS),
        filter=ds.field("eid").isin(sorted(int(value) for value in participant_ids))
        & ds.field("source_table").isin(list(SOURCES)),
        batch_size=batch_size,
        use_threads=True,
    )
    input_batches = 0
    input_rows = 0
    removed_rows = 0
    retained_rows = 0
    for batch in scanner.to_batches():
        input_batches += 1
        frame = batch.to_pandas()
        input_rows += len(frame)
        filtered, removed = remove_registered_test_rows(frame, selection_codes)
        removed_rows += removed
        retained_rows += len(filtered)
        accumulator.consume(filtered)
    event_features, event_columns = accumulator.finalize()
    raw = pd.DataFrame(
        {
            "age_years": pd.to_numeric(ordered["age_years"], errors="coerce"),
            "log1p_registration_days_before_landmark": tenure.to_numpy(dtype=float),
            "sex_field31": ordered["sex_field31"].astype("string"),
            "ethnicity_top_level": collapse_ethnicity_top_level(
                baseline["ethnicity_top_level"]
            ),
            "assessment_centre": baseline["assessment_centre"].astype("string"),
            "gp_provider": ordered["provider_id"].astype("string"),
            "baseline_calendar_year": baseline["baseline_date"].dt.year.astype("string"),
            "baseline_calendar_month": baseline["baseline_date"].dt.month.astype("string"),
        }
    )
    raw = pd.concat([raw.reset_index(drop=True), event_features], axis=1)
    schema = FeatureSchema(
        continuous=("age_years", "log1p_registration_days_before_landmark"),
        categorical=(
            "sex_field31",
            "ethnicity_top_level",
            "assessment_centre",
            "gp_provider",
            "baseline_calendar_year",
            "baseline_calendar_month",
        ),
        zero_fill=event_columns,
    )
    schema.validate(raw)
    audit = FeatureAudit(
        participants=len(ordered),
        input_event_batches=input_batches,
        input_event_rows=input_rows,
        focal_or_panel_rows_removed=removed_rows,
        retained_event_rows_passed_to_builder=retained_rows,
        eligible_prelandmark_rows=accumulator.eligible_prelandmark_rows,
        hashed_rows=accumulator.hashed_rows,
        unique_hashed_tokens=len(accumulator.token_cache),
        feature_columns=len(raw.columns),
    )
    return raw, schema, audit
