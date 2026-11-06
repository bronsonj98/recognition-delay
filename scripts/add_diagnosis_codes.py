"""Add primary-care recognition using externally supplied code dictionaries.

Diabetes and anemia lists are the codes mapped to the ICD-10 ranges; the CKD list adds stage and renal-failure terms.
Lookup contents, expanded code lists, and participant records are not distributed.
"""
from __future__ import annotations
import os
import pandas as pd
import pyarrow.dataset as ds
from common import DATA, PRIVATE, setup, fresh


def main():
    setup()
    L = DATA / 'lookups'
    WH = DATA / 'events.parquet'
    output = fresh(PRIVATE / 'recognition_frame.parquet')

    r2 = pd.read_csv(L / 'read_v2_lkp.tsv.gz', sep='\t', dtype=str); r2 = r2[r2['term_code'] == '00']
    c3 = pd.read_csv(L / 'read_ctv3_lkp.tsv.gz', sep='\t', dtype=str); c3 = c3[c3['description_type'] == 'P']
    m2 = pd.read_csv(L / 'read_v2_icd10.tsv.gz', sep='\t', dtype=str); m3 = pd.read_csv(L / 'read_ctv3_icd10.tsv.gz', sep='\t', dtype=str)
    def icd3(s): return s.astype(str).str.strip().str[:3]
    def mapped(icd_prefixes):
        keep2 = m2[icd3(m2['icd10_code']).isin(icd_prefixes)]['read_code'].str.strip()
        keep3 = m3[icd3(m3['icd10_code']).isin(icd_prefixes)]['read_code'].str.strip()
        return set(keep2), set(keep3)
    def by_term(regex, exclude=None):
        a = r2[r2['term_description'].str.contains(regex, case=False, na=False, regex=True)]
        b = c3[c3['term_description'].str.contains(regex, case=False, na=False, regex=True)]
        if exclude:
            a = a[~a['term_description'].str.contains(exclude, case=False, na=False, regex=True)]
            b = b[~b['term_description'].str.contains(exclude, case=False, na=False, regex=True)]
        return set(a['read_code'].str.strip()), set(b['read_code'].str.strip())

    sets = {}
    # CKD: codes mapped to ICD-10 N18, codes whose term names chronic kidney disease or chronic renal failure (excluding
    # administrative, screening, family-history and acute terms), and Read 2 codes under 1Z1, which the Read 2 map omits
    r2_n18, c3_n18 = mapped({'N18'})
    r2_t, c3_t = by_term(r'chronic kidney disease|chronic renal (failure|impairment|insufficiency)|end.stage renal', exclude=r'at risk|monitor|screen|family history|fh:|review|invit|exception|declin|annual|register|education|self.management|care plan|leaflet|acute')
    sets['ckd'] = {'READ2': sorted(r2_n18 | r2_t | {c for c in r2['read_code'].str.strip() if c.startswith('1Z1')}), 'CTV3': sorted(c3_n18 | c3_t)}
    # diabetes: codes mapped to ICD-10 E10-E14
    r2_e, c3_e = mapped({'E10', 'E11', 'E12', 'E13', 'E14'})
    sets['diabetes'] = {'READ2': sorted(r2_e), 'CTV3': sorted(c3_e)}
    # anemia: codes mapped to ICD-10 D50-D64
    r2_d, c3_d = mapped({f'D{i}' for i in range(50, 65)})
    sets['anaemia'] = {'READ2': sorted(r2_d), 'CTV3': sorted(c3_d)}
    fr = pd.read_parquet(PRIVATE / 'initial_frame.parquet')
    d = ds.dataset(WH, format='parquet')
    allcodes = sorted({c for s in sets.values() for v in s.values() for c in v})
    t = d.scanner(columns=['eid', 'event_date', 'code_system', 'code_raw', 'code_normalized', 'provider_or_country'], filter=(ds.field('source_table') == 'gp_clinical') & (ds.field('code_raw').isin(allcodes) | ds.field('code_normalized').isin(allcodes))).to_table().to_pandas()
    t['event_date'] = pd.to_datetime(t['event_date'], errors='coerce'); t = t.merge(fr[['eid', 'baseline']], on='eid')
    for cond, s in sets.items():
        m = ((t['code_system'] == 'READ2') & (t['code_raw'].isin(s['READ2']) | t['code_normalized'].isin(s['READ2']))) | ((t['code_system'] == 'CTV3') & (t['code_raw'].isin(s['CTV3']) | t['code_normalized'].isin(s['CTV3'])))
        sub = t[m]
        fr[f'gpcode_first_{cond}'] = fr['eid'].map(sub.groupby('eid')['event_date'].min())
        fr[f'gpcode_post_{cond}'] = fr['eid'].map(sub[sub['event_date'] > sub['baseline']].groupby('eid')['event_date'].min())
    b = fr['baseline']
    fr['prior_ckd'] = fr['prior_ckd'] | (fr['gpcode_first_ckd'] <= b)
    fr['prior_diabetes'] = fr['prior_diabetes'] | (fr['gpcode_first_diabetes'] <= b)
    fr['prior_anaemia'] = fr['prior_anaemia'] | (fr['gpcode_first_anaemia'] <= b)
    fr['rec_ckd'] = pd.concat([fr['fo_rec_ckd'], fr['gpcode_post_ckd']], axis=1).min(axis=1)
    fr['rec_diabetes'] = pd.concat([fr['fo_rec_diabetes'], fr['gpcode_post_diabetes'], fr['rx_post_antidiabetic']], axis=1).min(axis=1)
    fr['rec_anaemia'] = pd.concat([fr['fo_rec_anaemia'], fr['gpcode_post_anaemia'], fr['rx_post_iron']], axis=1).min(axis=1)
    for c in ['rec_ckd', 'rec_diabetes', 'rec_anaemia']:
        fr[c] = fr[c].where(fr[c] > b)
    with open(output, 'xb') as handle:
        os.fchmod(handle.fileno(), 0o600)
        fr.to_parquet(handle, index=False)
    print('Analysis frame with explicit diagnosis recognition completed.')


if __name__ == '__main__':
    main()
