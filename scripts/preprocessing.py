"""Training-only encoding and scaling; no fitted state is distributed."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import StandardScaler

MISSING_LEVEL = '__MISSING__'
UNSEEN_LEVEL = '__UNSEEN__'
INNER_VALIDATION_NAMESPACE = 'multipanel-stage-a-inner-validation-v1'

@dataclass(frozen=True)
class FeatureSchema:
    """Frozen roles for columns in a raw participant feature frame."""

    continuous: tuple[str, ...]
    categorical: tuple[str, ...]
    zero_fill: tuple[str, ...]

    def validate(self, frame: pd.DataFrame) -> None:
        names = (*self.continuous, *self.categorical, *self.zero_fill)
        if len(names) != len(set(names)):
            raise ValueError("feature schema contains duplicate columns")
        if not names:
            raise ValueError("feature schema must contain at least one column")
        missing = set(names) - set(frame)
        if missing:
            raise ValueError(f"raw features are missing columns: {sorted(missing)}")


@dataclass
class FrozenTabularEncoder:
    """Training-arm-fitted encoding with explicit missing and unseen states."""

    schema: FeatureSchema
    medians_: dict[str, float] | None = None
    categorical_levels_: dict[str, tuple[str, ...]] | None = None
    feature_names_: tuple[str, ...] | None = None

    def fit(self, frame: pd.DataFrame) -> FrozenTabularEncoder:
        self.schema.validate(frame)
        if frame.empty:
            raise ValueError("cannot fit an encoder on an empty frame")
        medians: dict[str, float] = {}
        names: list[str] = []
        for column in self.schema.continuous:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            medians[column] = float(np.median(finite)) if finite.size else 0.0
            names.extend((column, f"{column}__missing"))
        levels: dict[str, tuple[str, ...]] = {}
        for column in self.schema.categorical:
            values = frame[column].astype("string").fillna(MISSING_LEVEL).astype(str)
            observed = sorted(set(values) - {MISSING_LEVEL, UNSEEN_LEVEL})
            levels[column] = (MISSING_LEVEL, *observed, UNSEEN_LEVEL)
            names.extend(f"{column}=={level}" for level in levels[column])
        names.extend(self.schema.zero_fill)
        self.medians_ = medians
        self.categorical_levels_ = levels
        self.feature_names_ = tuple(names)
        return self

    def transform(self, frame: pd.DataFrame) -> NDArray[np.float32]:
        self.schema.validate(frame)
        if self.medians_ is None or self.categorical_levels_ is None or self.feature_names_ is None:
            raise ValueError("encoder must be fit before transform")
        pieces: list[NDArray[np.float32]] = []
        for column in self.schema.continuous:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            missing = ~np.isfinite(values)
            filled = values.copy()
            filled[missing] = self.medians_[column]
            pieces.extend(
                (
                    filled.astype(np.float32)[:, None],
                    missing.astype(np.float32)[:, None],
                )
            )
        for column in self.schema.categorical:
            levels = self.categorical_levels_[column]
            known = set(levels) - {UNSEEN_LEVEL}
            values = frame[column].astype("string").fillna(MISSING_LEVEL).astype(str)
            normalized = np.asarray(
                [value if value in known else UNSEEN_LEVEL for value in values], dtype=object
            )
            pieces.extend((normalized == level).astype(np.float32)[:, None] for level in levels)
        for column in self.schema.zero_fill:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
                dtype=float, copy=True
            )
            values[~np.isfinite(values)] = 0.0
            pieces.append(values.astype(np.float32)[:, None])
        matrix = np.concatenate(pieces, axis=1)
        if matrix.shape != (len(frame), len(self.feature_names_)) or not np.all(
            np.isfinite(matrix)
        ):
            raise ValueError("encoded feature matrix is invalid")
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> NDArray[np.float32]:
        return self.fit(frame).transform(frame)


@dataclass(frozen=True)
class OuterFoldEncoding:
    matrix: NDArray[np.float32]
    feature_names: tuple[str, ...]
    medians: dict[str, float]
    categorical_levels: dict[str, tuple[str, ...]]
    scaling_mean: NDArray[np.float64]
    scaling_scale: NDArray[np.float64]
    training_count: int


def encode_outer_fold(
    features: pd.DataFrame,
    schema: FeatureSchema,
    outer_folds: NDArray[np.integer],
    *,
    outer_test_fold: int,
    preprocessing_fit: NDArray[np.bool_] | None = None,
) -> OuterFoldEncoding:
    """Fit encoding/scaling on a subset of outer training and transform every row."""

    fold = np.asarray(outer_folds)
    if fold.ndim != 1 or fold.size != len(features) or fold.size == 0:
        raise ValueError("outer folds must align with the feature frame")
    if not np.all(np.isfinite(fold.astype(float))):
        raise ValueError("outer folds must be finite")
    training = fold.astype(int) != int(outer_test_fold)
    if not np.any(training) or np.all(training):
        raise ValueError("outer_test_fold must identify a non-empty held-out role")
    if preprocessing_fit is None:
        fit = training
    else:
        fit = np.asarray(preprocessing_fit)
        if fit.shape != training.shape or not np.all(np.isin(fit, [False, True, 0, 1])):
            raise ValueError("preprocessing_fit must be an aligned binary mask")
        fit = fit.astype(bool, copy=False)
        if np.any(fit & ~training):
            raise ValueError("outer-test rows cannot enter preprocessing_fit")
        if not np.any(fit):
            raise ValueError("preprocessing_fit cannot be empty")
    schema.validate(features)
    encoder = FrozenTabularEncoder(schema).fit(features.loc[fit])
    encoded = encoder.transform(features)
    scaler = StandardScaler().fit(encoded[fit])
    scaled = scaler.transform(encoded).astype(np.float32, copy=False)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("outer-fold preprocessing produced non-finite features")
    if (
        encoder.feature_names_ is None
        or encoder.medians_ is None
        or encoder.categorical_levels_ is None
    ):
        raise AssertionError("encoder did not expose fitted state")
    return OuterFoldEncoding(
        matrix=scaled,
        feature_names=encoder.feature_names_,
        medians=dict(encoder.medians_),
        categorical_levels=dict(encoder.categorical_levels_),
        scaling_mean=np.asarray(scaler.mean_, dtype=float),
        scaling_scale=np.asarray(scaler.scale_, dtype=float),
        training_count=int(np.sum(fit)),
    )


def deterministic_inner_validation(
    participant_ids: ArrayLike,
    *,
    outer_fold: int,
    fraction: float = 0.20,
    namespace: str = INNER_VALIDATION_NAMESPACE,
) -> NDArray[np.bool_]:
    """Assign a shared deterministic inner-validation role within an outer fold."""

    identifiers = np.asarray(participant_ids)
    if identifiers.ndim != 1 or identifiers.size == 0:
        raise ValueError("participant_ids must be a non-empty vector")
    if len(set(identifiers.tolist())) != identifiers.size:
        raise ValueError("participant_ids must be unique")
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must lie strictly between zero and one")
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, (int, np.integer)):
        raise ValueError("outer_fold must be an integer")
    threshold = int(fraction * (1 << 64))
    values = []
    for identifier in identifiers:
        payload = f"{namespace}|{int(outer_fold)}|{identifier}".encode()
        values.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") < threshold)
    result = np.asarray(values, dtype=bool)
    if not np.any(result) or np.all(result):
        raise ValueError("inner split produced an empty role")
    return result

