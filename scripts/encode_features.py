#!/usr/bin/env python3
"""Build reusable fold-safe encoded matrices from a raw feature cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import FeatureSchema, encode_outer_fold, deterministic_inner_validation
from common import setup, WORK


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _temp(destination: Path) -> Path:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--feature-metadata", required=True)
    parser.add_argument("--assay-matrix", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument(
        "--folds",
        default="",
        help="optional comma-separated fold subset; default builds every observed fold",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup()
    feature_path = Path(args.feature_cache).expanduser().resolve()
    metadata_path = Path(args.feature_metadata).expanduser().resolve()
    assay_path = Path(args.assay_matrix).expanduser().resolve()
    output_directory = Path(args.output_directory).expanduser().resolve()
    audit_output = Path(args.audit_output).expanduser().resolve()
    for path in (output_directory, audit_output):
        if not path.is_relative_to(WORK):
            raise ValueError('All outputs must be within ANALYSIS_WORK_ROOT.')
    if audit_output.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {audit_output}")
    output_directory.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(metadata_path.read_text())
    schema_data = metadata["schema"]
    schema = FeatureSchema(
        continuous=tuple(schema_data["continuous"]),
        categorical=tuple(schema_data["categorical"]),
        zero_fill=tuple(schema_data["zero_fill"]),
    )
    cached = pd.read_parquet(feature_path)
    assay = pd.read_parquet(assay_path, columns=["eid", "outer_fold"])
    if cached["eid"].tolist() != assay["eid"].tolist():
        raise ValueError("feature and assay caches are not identically participant aligned")
    features = cached.drop(columns="eid")
    folds = assay["outer_fold"].to_numpy(dtype=int)
    feature_cache_sha256 = _sha256(feature_path)
    observed_folds = sorted(int(value) for value in np.unique(folds))
    if args.folds:
        requested_folds = [int(value) for value in args.folds.split(",")]
        if (
            not requested_folds
            or len(set(requested_folds)) != len(requested_folds)
            or not set(requested_folds).issubset(observed_folds)
        ):
            raise ValueError("requested folds must be unique observed fold integers")
    else:
        requested_folds = observed_folds
    fold_audits = []
    for outer_test_fold in requested_folds:
        output = output_directory / f"outer_fold_{outer_test_fold}_scaled.npy"
        fitted_output = output_directory / f"outer_fold_{outer_test_fold}_preprocessing.json"
        if output.exists() or fitted_output.exists():
            raise FileExistsError(
                f"refusing to overwrite existing fold cache: {output} or {fitted_output}"
            )
        outer_training = folds != outer_test_fold
        inner_validation = np.zeros(len(folds), dtype=bool)
        inner_validation[outer_training] = deterministic_inner_validation(
            assay.loc[outer_training, "eid"].to_numpy(dtype="int64"),
            outer_fold=outer_test_fold,
        )
        preprocessing_fit = outer_training & ~inner_validation
        encoded = encode_outer_fold(
            features,
            schema,
            folds,
            outer_test_fold=outer_test_fold,
            preprocessing_fit=preprocessing_fit,
        )
        matrix_temp = _temp(output)
        fitted_temp = _temp(fitted_output)
        try:
            with matrix_temp.open("wb") as handle:
                np.save(handle, encoded.matrix, allow_pickle=False)
            os.chmod(matrix_temp, 0o600)
            matrix_sha = _sha256(matrix_temp)
            fitted = {
                "schema_version": "outer-fold-preprocessing-v1",
                "outer_test_fold": outer_test_fold,
                "training_count": encoded.training_count,
                "participant_count": int(len(features)),
                "encoded_feature_count": int(encoded.matrix.shape[1]),
                "feature_names": list(encoded.feature_names),
                "medians": encoded.medians,
                "categorical_levels": {
                    key: list(value) for key, value in encoded.categorical_levels.items()
                },
                "scaling_mean": encoded.scaling_mean.tolist(),
                "scaling_scale": encoded.scaling_scale.tolist(),

                "matrix_sha256": matrix_sha,

                "feature_cache_sha256": feature_cache_sha256,

                "fit_role": "outer_training_and_not_inner_validation",
                "inner_validation_count": int(np.sum(inner_validation)),
            }
            fitted_temp.write_text(json.dumps(fitted, indent=2, sort_keys=True) + "\n")
            os.chmod(fitted_temp, 0o600)
            os.replace(matrix_temp, output)
            os.replace(fitted_temp, fitted_output)
        finally:
            matrix_temp.unlink(missing_ok=True)
            fitted_temp.unlink(missing_ok=True)
        fold_audits.append(
            {
                "outer_test_fold": outer_test_fold,
                "training_count": encoded.training_count,
                "inner_validation_count": int(np.sum(inner_validation)),
                "participant_count": int(len(features)),
                "encoded_feature_count": int(encoded.matrix.shape[1]),

                "matrix_sha256": matrix_sha,

            }
        )
        del encoded
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_temp = _temp(audit_output)
    try:
        audit = {
            "schema_version": "outer-fold-feature-caches-v1",

            "feature_cache_sha256": feature_cache_sha256,

            "folds": fold_audits,
        }
        audit_temp.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        os.chmod(audit_temp, 0o600)
        os.replace(audit_temp, audit_output)
    finally:
        audit_temp.unlink(missing_ok=True)
    print(
        f"OUTER-FOLD FEATURE CACHES COMPLETE: folds={len(fold_audits)} "
        f"shape={len(features)}x{fold_audits[0]['encoded_feature_count']}"
    )


if __name__ == "__main__":
    main()
