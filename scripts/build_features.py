"""Build raw pre-enrollment features from external source tables and code exclusions."""
from __future__ import annotations
import json
import pandas as pd
from features import SelectionCode, build_process_reduced_features
from common import DATA, PRIVATE, setup, fresh


def main():
    setup()
    output = fresh(PRIVATE / 'raw_features.parquet')
    metadata_output = fresh(PRIVATE / 'feature_schema.json')
    participants = pd.read_parquet(PRIVATE / 'participants.parquet')
    # Codes for the laboratory tests measured at enrollment and for their panels (README).
    # It contains method code identifiers, never participant records.
    exclusions = pd.read_csv(DATA / 'feature_exclusions.tsv', sep='\t', dtype=str)
    if not {'code_system', 'code'}.issubset(exclusions):
        raise ValueError('Feature exclusions require code_system and code columns.')
    if exclusions[['code_system', 'code']].isna().any().any():
        raise ValueError('Feature exclusion codes must be complete.')
    codes = tuple(SelectionCode(row.code_system, row.code) for row in exclusions.itertuples())
    frame, schema, _ = build_process_reduced_features(
        participants, baseline_path=DATA / 'baseline.tsv.gz',
        registrations_path=DATA / 'registrations.tsv', events_path=DATA / 'events.parquet',
        selection_codes=codes)
    result = pd.concat([participants.sort_values('eid', kind='mergesort').reset_index(drop=True)[['eid']], frame], axis=1)
    result.to_parquet(output, index=False)
    output.chmod(0o600)
    metadata_output.write_text(json.dumps({'schema': {
        'continuous': list(schema.continuous), 'categorical': list(schema.categorical),
        'zero_fill': list(schema.zero_fill)}}, indent=2) + '\n')
    metadata_output.chmod(0o600)
    print('Raw features and schema completed.')


if __name__ == '__main__':
    main()
