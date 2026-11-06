"""Private frame supplement for the sensitivity analyses.

Adds, without changing any primary definition:
  * anemia prior record and recognition rebuilt with and without hemoglobinopathy
    evidence (ICD-10 D56-D58 first-occurrence fields and primary-care Read codes
    that map only to D56-D58); the unmodified rebuild must reproduce the frame exactly;
  * hemoglobinopathy flags (any evidence, and evidence on or before enrollment);
  * the first abnormal primary-care value after enrollment and the first abnormal value at
    least 90 days later, copied from the frame (build_frame.py);
  * enrollment MCV (field 30040).
Output: <work>/sensitivity_inputs/frame_supplement.parquet (owner-only).
"""
from __future__ import annotations
import json
import time
import pandas as pd
import pyarrow.dataset as ds
from common import DATA, PRIVATE, SENS_PRIV, SENS_OUT, setup
from definitions import CFG

t0 = time.time()
setup()
OUTD = SENS_PRIV
OUTD.mkdir(mode=0o700, parents=True, exist_ok=True)
AUD = SENS_OUT
AUD.mkdir(parents=True, exist_ok=True)
REG = DATA / 'registry.tsv.gz'
ASSAY = DATA / 'assays.tsv'
WH = DATA / 'events.parquet'
LK = str(DATA / 'lookups') + '/'
HBP_FIELDS = [130632, 130634, 130636]  # D56, D57, D58 first-occurrence dates
audit = {}

fr = pd.read_parquet(PRIVATE / 'recognition_frame.parquet')
b = fr['baseline']

# ---- first-occurrence dates, all anemia fields and without D56-D58 ----
fields = CFG['conditions']['anaemia']['first_occurrence_fields']
assert all(f in fields for f in HBP_FIELDS) and len(fields) == 14
cols = ['f.eid'] + [f'f.{f}.0.0' for f in fields] + [f'f.{f + 1}.0.0' for f in fields]
reg = pd.read_csv(REG, sep='\t', usecols=cols, dtype=str, na_values=['NA', ''])
reg['eid'] = reg['f.eid'].astype(int)
reg = fr[['eid']].merge(reg, on='eid', how='left', validate='one_to_one')
rec_src = CFG['first_occurrence_sources']['recognition_default']


def fo(keep):
    D = pd.concat([pd.to_datetime(reg[f'f.{f}.0.0'], errors='coerce') for f in keep], axis=1)
    S = pd.concat([pd.to_numeric(reg[f'f.{f + 1}.0.0'], errors='coerce') for f in keep], axis=1)
    return D.min(axis=1), D.where(S.isin(rec_src).to_numpy()).min(axis=1)


fo_any_full, fo_rec_full = fo(fields)
fo_any_nohbp, fo_rec_nohbp = fo([f for f in fields if f not in HBP_FIELDS])
fo_hbp_any, _ = fo(HBP_FIELDS)
for name, a, ref in [('fo_any', fo_any_full, fr['fo_any_anaemia']), ('fo_rec', fo_rec_full, fr['fo_rec_anaemia'])]:
    same = (a.isna() & ref.isna()) | (a == ref)
    audit[f'{name}_reproduced'] = bool(same.all())
    assert same.all(), name

# ---- primary-care anemia code lists, as in gp_code_recognition.py (HEAD) ----
m2 = pd.read_csv(LK + 'read_v2_icd10.tsv.gz', sep='\t', dtype=str)
m3 = pd.read_csv(LK + 'read_ctv3_icd10.tsv.gz', sep='\t', dtype=str)


def icd3(s):
    return s.astype(str).str.strip().str[:3]


anaemia_icd = {f'D{i}' for i in range(50, 65)}
hbp_icd = {'D56', 'D57', 'D58'}
code_sets, hbp_only = {}, {}
for sysn, m in (('READ2', m2), ('CTV3', m3)):
    mm = m.assign(code=m['read_code'].str.strip(), icd=icd3(m['icd10_code']))
    mm = mm[mm['icd'].isin(anaemia_icd)]
    by_code = mm.groupby('code')['icd'].agg(set)
    code_sets[sysn] = set(by_code.index)
    hbp_only[sysn] = {c for c, s in by_code.items() if s <= hbp_icd}
    audit[f'{sysn}_anaemia_codes'] = len(code_sets[sysn])
    audit[f'{sysn}_hemoglobinopathy_only_codes'] = len(hbp_only[sysn])
    audit[f'{sysn}_codes_mapping_to_hbp_and_other_anaemia'] = int(sum(
        1 for c, s in by_code.items() if (s & hbp_icd) and not (s <= hbp_icd)))

d = ds.dataset(WH, format='parquet')
allcodes = sorted(code_sets['READ2'] | code_sets['CTV3'])
t = d.scanner(columns=['eid', 'event_date', 'code_system', 'code_raw', 'code_normalized'],
              filter=(ds.field('source_table') == 'gp_clinical') &
              (ds.field('code_raw').isin(allcodes) | ds.field('code_normalized').isin(allcodes))
              ).to_table().to_pandas()
t['event_date'] = pd.to_datetime(t['event_date'], errors='coerce')
t = t.merge(fr[['eid', 'baseline']], on='eid')


def in_set(sets):
    return (((t['code_system'] == 'READ2') & (t['code_raw'].isin(sets['READ2']) | t['code_normalized'].isin(sets['READ2']))) |
            ((t['code_system'] == 'CTV3') & (t['code_raw'].isin(sets['CTV3']) | t['code_normalized'].isin(sets['CTV3']))))


def first_post(sub):
    first = fr['eid'].map(sub.groupby('eid')['event_date'].min())
    post = fr['eid'].map(sub[sub['event_date'] > sub['baseline']].groupby('eid')['event_date'].min())
    return first, post


full_m = in_set(code_sets)
nohbp_sets = {k: code_sets[k] - hbp_only[k] for k in code_sets}
hbp_m = in_set(hbp_only)
nohbp_m = in_set(nohbp_sets)
gp_first_full, gp_post_full = first_post(t[full_m])
gp_first_nohbp, gp_post_nohbp = first_post(t[nohbp_m])
gp_first_hbp, _ = first_post(t[hbp_m])
for name, a, ref in [('gpcode_first', gp_first_full, fr['gpcode_first_anaemia']),
                     ('gpcode_post', gp_post_full, fr['gpcode_post_anaemia'])]:
    same = (a.isna() & ref.isna()) | (a == ref)
    audit[f'{name}_reproduced'] = bool(same.all())
    assert same.all(), name
audit['hbp_code_rows'] = int(hbp_m.sum())
audit['hbp_code_participants'] = int(t.loc[hbp_m, 'eid'].nunique())


def prior_rec(fo_any, fo_rec, gp_first, gp_post):
    prior = (fr['sr_anaemia'] | fr['srmed_iron'].fillna(False) | (fo_any <= b) |
             (fr['rx_first_iron'] <= b) | (gp_first <= b))
    rec = pd.concat([fo_rec, gp_post, fr['rx_post_iron']], axis=1).min(axis=1)
    return prior, rec.where(rec > b)


prior_full, rec_full = prior_rec(fo_any_full, fo_rec_full, gp_first_full, gp_post_full)
assert (prior_full == fr['prior_anaemia']).all()
same = (rec_full.isna() & fr['rec_anaemia'].isna()) | (rec_full == fr['rec_anaemia'])
assert same.all()
audit['prior_and_recognition_reproduced'] = True
prior_nohbp, rec_nohbp = prior_rec(fo_any_nohbp, fo_rec_nohbp, gp_first_nohbp, gp_post_nohbp)

SR = pd.read_csv(REG, sep='\t', usecols=lambda c: c == 'f.eid' or c.startswith('f.20002.0.'),
                 dtype=str, na_values=['NA', ''])
SR['eid'] = SR['f.eid'].astype(int)
SR = fr[['eid']].merge(SR, on='eid', how='left', validate='one_to_one')
sr_hbp = SR[[c for c in SR if c.startswith('f.20002.0.')]].apply(pd.to_numeric, errors='coerce').isin([1339, 1340]).any(axis=1)
hbp_date = pd.concat([fo_hbp_any, gp_first_hbp], axis=1).min(axis=1)
out = pd.DataFrame({'eid': fr['eid'], 'prior_anaemia_nohbp': prior_nohbp.to_numpy(bool),
                    'rec_anaemia_nohbp': rec_nohbp, 'hbp_ever': (hbp_date.notna() | sr_hbp).to_numpy(bool),
                    'hbp_prior': ((hbp_date <= b) | sr_hbp).to_numpy(bool)})

# ---- primary-care laboratory values after enrollment, from the frame (build_frame.py) ----
for an in ('hba1c', 'egfr', 'hb'):
    out[f'gp_post_first_abn_{an}'] = fr[f'gp_post_first_abn_{an}'].to_numpy()
    out[f'gp_post_confirm90_{an}'] = fr[f'gp_post_confirm90_{an}'].to_numpy()

# ---- enrollment MCV ----
mcv = pd.read_csv(ASSAY, sep='\t', usecols=['f.eid', 'f.30040.0.0'], na_values=['NA', ''])
mcv = mcv.rename(columns={'f.eid': 'eid', 'f.30040.0.0': 'mcv'})
out = out.merge(mcv, on='eid', how='left', validate='one_to_one')

dest = OUTD / 'frame_supplement.parquet'
out.to_parquet(dest, index=False)
dest.chmod(0o600)
linked = fr['model_cohort'].to_numpy() & fr['hb'].notna().to_numpy() & fr['pos_anaemia'].fillna(False).to_numpy()
audit.update(
    linked_anaemia_positive=int(linked.sum()),
    linked_unrecorded_full=int((linked & ~prior_full.to_numpy()).sum()),
    linked_unrecorded_nohbp=int((linked & ~prior_nohbp.to_numpy()).sum()),
    linked_positive_with_hbp_ever=int((linked & out['hbp_ever'].to_numpy()).sum()),
    linked_positive_with_hbp_prior=int((linked & out['hbp_prior'].to_numpy()).sum()),
    seconds=round(time.time() - t0))
(AUD / 'frame_supplement_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit, indent=1))
