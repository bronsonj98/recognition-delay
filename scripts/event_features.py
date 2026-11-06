"""Pre-enrollment event features used in the case-finding experiments.

Only the baseline and event columns listed below are read.
Focal test codes are removed before both utilization and hashed-history
features, and every retained event is strictly before the participant's
baseline landmark.
"""

from __future__ import annotations

import gzip
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


WINDOWS = (30, 90, 180, 365)
SOURCES = ("gp_clinical", "gp_scripts", "hesin_diag", "hesin_oper")
CHANNEL_BY_SOURCE = {
    "gp_clinical": 0,
    "gp_scripts": 0,
    "hesin_diag": 1,
    "hesin_oper": 1,
}
HASH_DIMENSIONS_PER_CHANNEL = 256
HASH_NAMESPACE = "selective-testing-features-v1-20260827"
BASELINE_COLUMNS = (
    "f.eid",
    "f.53.0.0",
    "f.54.0.0",
    "f.21000.0.0",
)
EVENT_COLUMNS = (
    "eid",
    "event_date",
    "source_table",
    "code_system",
    "code_raw",
    "code_normalized",
)
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)




def _header(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        first = handle.readline()
    if not first:
        raise ValueError("tabular input has no header")
    return first.rstrip("\r\n").split("\t")


def load_feature_baseline(path: Path, participant_ids: set[int]) -> pd.DataFrame:
    """Read only the four baseline columns required for the feature representation."""

    header = _header(path)
    if len(header) != len(set(header)) or not set(BASELINE_COLUMNS).issubset(header):
        raise ValueError("baseline feature-column contract changed")
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=list(BASELINE_COLUMNS),
        dtype="string",
        keep_default_na=True,
        na_values=["NA"],
    ).rename(
        columns={
            "f.eid": "eid",
            "f.53.0.0": "baseline_date",
            "f.54.0.0": "assessment_centre",
            "f.21000.0.0": "ethnicity_top_level",
        }
    )
    frame["eid"] = pd.to_numeric(frame["eid"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["eid"]).copy()
    frame["eid"] = frame["eid"].astype("int64")
    frame = frame.loc[frame["eid"].isin(participant_ids)].copy()
    if frame["eid"].duplicated().any() or set(frame["eid"]) != participant_ids:
        raise ValueError("baseline feature join is not one-to-one and complete")
    frame["baseline_date"] = pd.to_datetime(
        frame["baseline_date"], format="%Y-%m-%d", errors="coerce"
    )
    if frame["baseline_date"].isna().any():
        raise ValueError("feature cohort contains a missing baseline date")
    return frame.sort_values("eid", kind="mergesort").reset_index(drop=True)


def registration_tenure(
    path: Path,
    baseline: pd.DataFrame,
    provider_by_eid: Mapping[int, int],
) -> pd.Series:
    """Return days since the earliest qualifying same-provider spell start.

    Multiple qualifying spells within the unique provider are not stitched.
    The earliest start is a fixed feature summary of the available registration
    history; eligibility itself remains the one-year coverage rule.
    """

    required = {"eid", "data_provider", "reg_date", "deduct_date"}
    header = _header(path)
    if len(header) != len(set(header)) or not required.issubset(header):
        raise ValueError("registration feature columns changed")
    frame = pd.read_csv(
        path,
        sep="\t",
        usecols=sorted(required),
        dtype="string",
        keep_default_na=True,
    )
    frame["eid"] = pd.to_numeric(frame["eid"], errors="coerce").astype("Int64")
    frame["provider_id"] = pd.to_numeric(frame["data_provider"], errors="coerce").astype(
        "Int64"
    )
    frame["reg_date"] = pd.to_datetime(
        frame["reg_date"].astype("string"), format="%d/%m/%Y", errors="coerce"
    )
    frame["deduct_date"] = pd.to_datetime(
        frame["deduct_date"].astype("string"), format="%d/%m/%Y", errors="coerce"
    )
    frame = frame.dropna(subset=["eid", "provider_id", "reg_date"]).copy()
    frame["eid"] = frame["eid"].astype("int64")
    frame["provider_id"] = frame["provider_id"].astype("int64")
    frame = frame.loc[frame["eid"].isin(provider_by_eid)].drop_duplicates(
        ["eid", "provider_id", "reg_date", "deduct_date"], keep="first"
    )
    dates = baseline.loc[:, ["eid", "baseline_date"]]
    frame = frame.merge(dates, on="eid", how="inner", validate="many_to_one")
    frame["expected_provider"] = frame["eid"].map(provider_by_eid)
    qualifying = frame.loc[
        frame["provider_id"].eq(frame["expected_provider"])
        & frame["reg_date"].le(frame["baseline_date"] - pd.Timedelta(days=365))
        & (frame["deduct_date"].isna() | frame["deduct_date"].ge(frame["baseline_date"]))
    ].copy()
    earliest = qualifying.groupby("eid", sort=False)["reg_date"].min()
    if set(earliest.index) != set(provider_by_eid):
        raise ValueError("a participant lacks a qualifying registration spell")
    baseline_lookup = baseline.set_index("eid")["baseline_date"]
    days = (baseline_lookup.loc[earliest.index] - earliest).dt.days.astype(float)
    if (days < 365).any() or not np.all(np.isfinite(days)):
        raise ValueError("registration tenure violates one-year coverage")
    return np.log1p(days).rename("log1p_registration_days_before_landmark")


ETHNICITY_TOP_LEVEL_BY_CODE = {
    1: "white",
    1001: "white",
    1002: "white",
    1003: "white",
    2: "mixed",
    2001: "mixed",
    2002: "mixed",
    2003: "mixed",
    2004: "mixed",
    3: "asian_or_asian_british",
    3001: "asian_or_asian_british",
    3002: "asian_or_asian_british",
    3003: "asian_or_asian_british",
    3004: "asian_or_asian_british",
    4: "black_or_black_british",
    4001: "black_or_black_british",
    4002: "black_or_black_british",
    4003: "black_or_black_british",
    5: "chinese",
    6: "other_ethnic_group",
}


def collapse_ethnicity_top_level(values: pd.Series) -> pd.Series:
    """Collapse Field 21000 into six frozen top-level groups plus missing.

    UK Biobank negative response codes and blank values are treated as missing.
    An unknown non-negative integer fails closed so a coding change cannot be
    silently folded into a result-sensitive feature.
    """

    text = values.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    declared_missing = text.isna() | text.fillna("").str.upper().isin(
        ["", "NA", "NAN", "NULL"]
    )
    if (numeric.isna() & ~declared_missing).any():
        raise ValueError("ethnicity contains an unparsable nonblank code")
    noninteger = numeric.notna() & numeric.ne(np.floor(numeric))
    if noninteger.any():
        raise ValueError("ethnicity contains a noninteger code")
    result = pd.Series(pd.NA, index=values.index, dtype="string")
    observed = numeric.notna() & numeric.ge(0)
    codes = numeric.loc[observed].astype("int64")
    unknown = sorted(set(codes) - set(ETHNICITY_TOP_LEVEL_BY_CODE))
    if unknown:
        raise ValueError("unmapped nonnegative ethnicity code")
    result.loc[observed] = codes.map(ETHNICITY_TOP_LEVEL_BY_CODE).astype("string")
    return result


def _hash_bucket(token: str) -> int:
    import hashlib

    namespaced = f"{HASH_NAMESPACE}|{token}"
    return (
        int.from_bytes(hashlib.sha256(namespaced.encode("utf-8")).digest()[:8], "big")
        % HASH_DIMENSIONS_PER_CHANNEL
    )


class EventFeatureAccumulator:
    """Memory-bounded event accumulator over an exact participant order."""

    def __init__(
        self,
        participant_ids: NDArray[np.int64],
        baseline_dates: NDArray[np.datetime64],
        focal_codes: set[tuple[str, str]],
    ) -> None:
        if participant_ids.ndim != 1 or baseline_dates.shape != participant_ids.shape:
            raise ValueError("participant identifiers and landmarks are not aligned")
        if len(set(participant_ids.tolist())) != participant_ids.size:
            raise ValueError("feature participants must be unique")
        self.participant_ids = participant_ids
        self.row_by_eid = {int(eid): index for index, eid in enumerate(participant_ids)}
        self.baseline_by_eid = {
            int(eid): pd.Timestamp(date)
            for eid, date in zip(participant_ids, baseline_dates, strict=True)
        }
        self.focal_codes = focal_codes
        n = participant_ids.size
        self.counts = np.zeros((n, len(SOURCES), len(WINDOWS)), dtype=np.int32)
        self.recent = np.full((n, len(SOURCES), len(WINDOWS)), 366, dtype=np.int16)
        self.day_bits = np.zeros((n, len(SOURCES), 6), dtype=np.uint64)
        self.hash_counts = np.zeros((n, 2 * HASH_DIMENSIONS_PER_CHANNEL), dtype=np.uint32)
        self.token_cache: dict[str, int] = {}
        self.event_batches = 0
        self.event_rows_scanned = 0
        self.eligible_prelandmark_rows = 0
        self.focal_rows_removed = 0
        self.hashed_rows = 0

    def consume(self, frame: pd.DataFrame) -> None:
        self.event_batches += 1
        self.event_rows_scanned += len(frame)
        if frame.empty:
            return
        if not set(EVENT_COLUMNS).issubset(frame):
            raise ValueError("event batch lacks an allowed feature column")
        work = frame.loc[:, EVENT_COLUMNS].copy()
        work["row"] = work["eid"].map(self.row_by_eid)
        work = work.dropna(subset=["row", "event_date", "source_table"]).copy()
        work = work.loc[work["source_table"].isin(SOURCES)].copy()
        if work.empty:
            return
        work["row"] = work["row"].astype("int64")
        work["baseline_date"] = work["eid"].map(self.baseline_by_eid)
        work["event_date"] = pd.to_datetime(work["event_date"], errors="coerce")
        work = work.dropna(subset=["baseline_date", "event_date"])
        work["days_before"] = (work["baseline_date"] - work["event_date"]).dt.days
        work = work.loc[work["days_before"].between(1, 365)].copy()
        if work.empty:
            return
        self.eligible_prelandmark_rows += len(work)
        system = work["code_system"].astype("string").fillna("").astype(str)
        raw = work["code_raw"].astype("string").fillna("").astype(str)
        focal = work["source_table"].eq("gp_clinical") & np.asarray(
            [(left, right) in self.focal_codes for left, right in zip(system, raw, strict=True)]
        )
        self.focal_rows_removed += int(focal.sum())
        work = work.loc[~focal].copy()
        if work.empty:
            return

        source_values = work["source_table"].astype(str).to_numpy()
        rows = work["row"].to_numpy(dtype=np.int64)
        days = work["days_before"].to_numpy(dtype=np.int16)
        for source_index, source in enumerate(SOURCES):
            source_mask = source_values == source
            if not np.any(source_mask):
                continue
            source_rows = rows[source_mask]
            source_days = days[source_mask]
            words = (source_days - 1) // 64
            bits = (source_days - 1) % 64
            for word in np.unique(words):
                word_mask = words == word
                view = self.day_bits[:, source_index, int(word)]
                np.bitwise_or.at(
                    view,
                    source_rows[word_mask],
                    np.left_shift(np.uint64(1), bits[word_mask].astype(np.uint64)),
                )
            for window_index, window in enumerate(WINDOWS):
                within = source_days <= window
                if not np.any(within):
                    continue
                np.add.at(self.counts[:, source_index, window_index], source_rows[within], 1)
                np.minimum.at(
                    self.recent[:, source_index, window_index],
                    source_rows[within],
                    source_days[within],
                )

        normalized = work["code_normalized"].astype("string").fillna("").astype(str)
        system = work["code_system"].astype("string").fillna("").astype(str)
        hashable = normalized.str.strip().ne("") & system.str.strip().ne("")
        if np.any(hashable):
            hashed = work.loc[hashable, ["row", "source_table"]].copy()
            tokens = [
                f"{source}|{code_system}|{code}"
                for source, code_system, code in zip(
                    hashed["source_table"].astype(str),
                    system.loc[hashable],
                    normalized.loc[hashable],
                    strict=True,
                )
            ]
            buckets: list[int] = []
            for token in tokens:
                bucket = self.token_cache.get(token)
                if bucket is None:
                    bucket = _hash_bucket(token)
                    self.token_cache[token] = bucket
                buckets.append(bucket)
            channels = hashed["source_table"].map(CHANNEL_BY_SOURCE).to_numpy(dtype=np.int64)
            columns = channels * HASH_DIMENSIONS_PER_CHANNEL + np.asarray(buckets, dtype=np.int64)
            hash_rows = hashed["row"].to_numpy(dtype=np.int64)
            np.add.at(self.hash_counts, (hash_rows, columns), 1)
            self.hashed_rows += len(hash_rows)

    def _unique_days(self, source_index: int, window: int) -> NDArray[np.int16]:
        full_words, remainder = divmod(window, 64)
        total = np.zeros(self.participant_ids.size, dtype=np.int16)
        if full_words:
            complete = np.ascontiguousarray(self.day_bits[:, source_index, :full_words])
            total += POPCOUNT[complete.view(np.uint8)].sum(axis=1).astype(np.int16)
        if remainder:
            mask = np.uint64((1 << remainder) - 1)
            partial = np.ascontiguousarray(
                (self.day_bits[:, source_index, full_words] & mask)[:, None]
            )
            total += POPCOUNT[partial.view(np.uint8)].sum(axis=1).astype(np.int16)
        return total

    def finalize(self) -> tuple[pd.DataFrame, tuple[str, ...]]:
        values: dict[str, NDArray[Any]] = {}
        columns: list[str] = []
        for source_index, source in enumerate(SOURCES):
            for window_index, window in enumerate(WINDOWS):
                names = (
                    f"util__{source}__{window}d__log1p_event_count",
                    f"util__{source}__{window}d__log1p_unique_event_days",
                    f"util__{source}__{window}d__days_since_most_recent_capped366",
                )
                values[names[0]] = np.log1p(self.counts[:, source_index, window_index]).astype(
                    np.float32
                )
                values[names[1]] = np.log1p(self._unique_days(source_index, window)).astype(
                    np.float32
                )
                values[names[2]] = self.recent[:, source_index, window_index].astype(np.float32)
                columns.extend(names)
        transformed_hash = np.log1p(self.hash_counts).astype(np.float32)
        for channel, offset in (("gp", 0), ("hes", HASH_DIMENSIONS_PER_CHANNEL)):
            for bucket in range(HASH_DIMENSIONS_PER_CHANNEL):
                name = f"hash__{channel}__{bucket:03d}"
                values[name] = transformed_hash[:, offset + bucket]
                columns.append(name)
        return pd.DataFrame(values), tuple(columns)
