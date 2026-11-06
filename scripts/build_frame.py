"""Build the three-condition frame from external harmonized source tables.

Run only in an approved data environment. Input extraction and dictionary
acquisition are external; the README documents every expected input.
"""
from __future__ import annotations
import gzip, os, re, time, urllib.parse
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from common import DATA, PRIVATE, setup, fresh
from definitions import CFG, DRUG_CLASSES


def drug_match(names, cls):
    """True where a lower-cased drug name is in the class: matches 'include' and not 'exclude'."""
    names = pd.Series(names).fillna('').astype(str)
    spec = DRUG_CLASSES[cls]
    return (names.str.contains(spec['include'], regex=True) & ~names.str.contains(spec['exclude'], regex=True)).to_numpy()


def main():
    setup()
    t0 = time.time()
    REG = DATA / 'registry.tsv.gz'
    ASSAY = DATA / 'assays.tsv'
    BASE = DATA / 'baseline.tsv.gz'
    EXTRA = DATA / 'extra_baseline.tsv'
    WH = DATA / 'events.parquet'
    COV = DATA / 'coverage.parquet'
    AM = PRIVATE / 'participants.parquet'
    SEL = DATA / 'prior_testing.parquet'
    LK = DATA / 'lookups'
    ENC = DATA / 'medication_encoding.txt'
    OUT_FRAME = fresh(PRIVATE / 'initial_frame.parquet')

    # ---------------- registry slice ----------------
    hdr = gzip.open(REG, 'rt').readline().rstrip('\n').split('\t')
    sr_cols = [c for c in hdr if c.startswith('f.20002.0.')]
    fo_fields = sorted({f for c in CFG['conditions'].values() for f in c['first_occurrence_fields']})
    fo_cols = [f'f.{f}.0.0' for f in fo_fields] + [f'f.{f+1}.0.0' for f in fo_fields]
    want = ['f.eid', 'f.31.0.0', 'f.34.0.0', 'f.52.0.0', 'f.53.0.0', 'f.53.1.0', 'f.40000.0.0'] + sr_cols + fo_cols
    missing = [c for c in want if c not in hdr]; assert not missing, missing
    reg = pd.read_csv(REG, sep='\t', usecols=want, dtype=str, na_values=['NA', ''])
    fr = pd.DataFrame({'eid': reg['f.eid'].astype(int)})
    fr['sex'] = pd.to_numeric(reg['f.31.0.0'])
    fr['baseline'] = pd.to_datetime(reg['f.53.0.0']); fr['visit1'] = pd.to_datetime(reg['f.53.1.0'])
    fr['death'] = pd.to_datetime(reg['f.40000.0.0'])
    dob = pd.to_datetime(dict(year=pd.to_numeric(reg['f.34.0.0']), month=pd.to_numeric(reg['f.52.0.0']), day=15))
    fr['age'] = (fr['baseline'] - dob).dt.days / 365.25
    fr['age_v1'] = (fr['visit1'] - dob).dt.days / 365.25
    SR = reg[sr_cols].apply(pd.to_numeric, errors='coerce')
    for cond, codes in CFG['self_report_illness_codes'].items():
        fr[f'sr_{cond}'] = SR.isin(codes).any(axis=1)
    src_map = CFG['first_occurrence_sources']
    for cond, c in CFG['conditions'].items():
        D = pd.concat([pd.to_datetime(reg[f'f.{f}.0.0'], errors='coerce') for f in c['first_occurrence_fields']], axis=1)
        S = pd.concat([pd.to_numeric(reg[f'f.{f+1}.0.0'], errors='coerce') for f in c['first_occurrence_fields']], axis=1)
        fr[f'fo_any_{cond}'] = D.min(axis=1)
        fr[f'fo_rec_{cond}'] = D.where(S.isin(src_map['recognition_default']).to_numpy()).min(axis=1)
    del reg, SR
    print('registry loaded', len(fr), round(time.time() - t0), flush=True)

    # ---------------- assays (instances 0/1) ----------------
    amap = {30750: 'hba1c', 30700: 'creat', 30720: 'cysc', 30020: 'hb'}
    ah = open(ASSAY).readline().rstrip('\n').split('\t')
    acols = ['f.eid'] + [f'f.{f}.{i}.0' for f in amap for i in (0, 1) if f'f.{f}.{i}.0' in ah]
    ass = pd.read_csv(ASSAY, sep='\t', usecols=acols, na_values=['NA', ''])
    ren = {'f.eid': 'eid'}
    for f, n in amap.items():
        for i in (0, 1):
            if f'f.{f}.{i}.0' in ass: ren[f'f.{f}.{i}.0'] = n if i == 0 else f'{n}_v1'
    ass = ass.rename(columns=ren)
    fr = fr.merge(ass, on='eid', how='left')

    def egfr_cr(creat, age, sex):
        scr = creat / 88.4; k = np.where(sex == 0, 0.7, 0.9); a = np.where(sex == 0, -0.241, -0.302)
        return 142 * np.minimum(scr / k, 1) ** a * np.maximum(scr / k, 1) ** -1.2 * 0.9938 ** age * np.where(sex == 0, 1.012, 1.0)
    def egfr_cys(cys, age, sex):  # CKD-EPI 2012 cystatin C
        return 133 * np.minimum(cys / 0.8, 1) ** -0.499 * np.maximum(cys / 0.8, 1) ** -1.328 * 0.996 ** age * np.where(sex == 0, 0.932, 1.0)
    fr['egfr'] = egfr_cr(fr['creat'], fr['age'], fr['sex']); fr['egfr_cys'] = egfr_cys(fr['cysc'], fr['age'], fr['sex'])
    if 'creat_v1' in fr: fr['egfr_v1'] = egfr_cr(fr['creat_v1'], fr['age_v1'], fr['sex'])

    # ---------------- baseline demographics + extra fields ----------------
    base = pd.read_csv(BASE, sep='\t', usecols=['f.eid', 'f.189.0.0', 'f.21000.0.0'], na_values=['NA', '']).rename(columns={'f.eid': 'eid', 'f.189.0.0': 'townsend', 'f.21000.0.0': 'ethnicity'})
    fr = fr.merge(base, on='eid', how='left')
    eth = fr['ethnicity'].astype(float)
    fr['eth_group'] = np.select([eth.between(1000, 1003) | (eth == 1), eth.between(3000, 3004) | (eth == 3), eth.between(4000, 4003) | (eth == 4), eth.between(2000, 2004) | (eth == 2), eth == 5, eth == 6],
                                ['white', 'asian', 'black', 'mixed', 'chinese', 'other'], default='missing')
    fr['tq'] = pd.qcut(fr['townsend'], 5, labels=False)
    ex = pd.read_csv(EXTRA, sep='\t', dtype=str, na_values=['NA', ''])
    ex['eid'] = ex['f.eid'].astype(int)
    enc = open(ENC, encoding='utf-8', errors='replace').read()
    i = enc.find('encoding_id=4&'); k = enc.find('encoding_id=', i + 15)
    med_pairs = re.findall(r's#va=([^&]+)&i#se=\d&s#mn=([^\]&]+)', enc[i:k])
    med_terms = {va: urllib.parse.unquote(mn.replace('+', ' ')).lower() for va, mn in med_pairs}   # coding text is URL-encoded
    mcols = [c for c in ex.columns if c.startswith('f.20003.0.')]
    M = ex[mcols].apply(lambda s: s.map(med_terms).fillna(''))
    for cls in ('antidiabetic', 'iron'):
        ex[f'srmed_{cls}'] = M.apply(lambda s: drug_match(s, cls)).any(axis=1)
    ex['dr_diabetes'] = pd.to_numeric(ex['f.2443.0.0'], errors='coerce') == 1
    ins = ex[[c for c in ex.columns if c.startswith('f.6153.0.') or c.startswith('f.6177.0.')]].apply(pd.to_numeric, errors='coerce')
    ex['insulin_med'] = (ins == 3).any(axis=1)
    fr = fr.merge(ex[['eid', 'srmed_antidiabetic', 'srmed_iron', 'dr_diabetes', 'insulin_med']], on='eid', how='left')
    print('extra fields merged', round(time.time() - t0), flush=True)

    # ---------------- cohort / coverage ----------------
    am = pq.read_table(AM, columns=['eid', 'provider_id', 'outer_fold']).to_pandas()
    sel = pq.read_table(SEL, columns=['eid', 'selected__hba1c__365d', 'selected__creatinine__365d', 'selected__haemoglobin__365d']).to_pandas().rename(columns={'selected__hba1c__365d': 'gp_test1y_hba1c', 'selected__creatinine__365d': 'gp_test1y_creat', 'selected__haemoglobin__365d': 'gp_test1y_hb'})
    fr = fr.merge(am, on='eid', how='left').merge(sel, on='eid', how='left'); fr['model_cohort'] = fr['provider_id'].notna()
    # Suppliers whose numeric results are stored in value1 (England TPP and Vision, Wales). Scottish results often carry an
    # operator in value1 and the number in value2, and part of them cannot be decoded; value-based follow-up excludes them.
    fr['gp_values_complete'] = fr['provider_id'].isin([1, 3, 4])
    cov = pq.read_table(COV).to_pandas()
    hes = cov[cov['source'] == 'hes'].groupby('eid').agg(hes_censor=('censor_primary', 'max')).reset_index()
    gp = cov[cov['source'] == 'gp'].groupby('eid').agg(gp_end=('coverage_end', 'max')).reset_index()
    fr = fr.merge(hes, on='eid', how='left').merge(gp, on='eid', how='left')
    fr['hes_censor'] = pd.to_datetime(fr['hes_censor'], errors='coerce'); fr['gp_end'] = pd.to_datetime(fr['gp_end'], errors='coerce')

    # ---------------- prescriptions ----------------
    d = ds.dataset(WH, format='parquet')
    rx = d.scanner(columns=['eid', 'event_date', 'drug_name', 'code_system', 'code_raw'], filter=(ds.field('source_table') == 'gp_scripts')).to_table().to_pandas()
    lk = pd.read_csv(LK / 'read_v2_drugs_lkp.tsv.gz', sep='\t', dtype=str)
    lkmap = dict(zip(lk['read_code'].str.strip(), lk['term_description'].fillna('').str.lower()))
    r2 = rx['code_system'] == 'READ2'
    rx.loc[r2, 'drug_name'] = rx.loc[r2, 'code_raw'].astype(str).str.strip().map(lkmap)
    rx['drug_name'] = rx['drug_name'].fillna('').str.lower()
    rx['event_date'] = pd.to_datetime(rx['event_date'], errors='coerce')
    rx = rx.merge(fr[['eid', 'baseline']], on='eid', how='left')
    for cls in ('antidiabetic', 'iron'):
        names = pd.Series(rx['drug_name'].unique())
        m = rx['drug_name'].isin(set(names[drug_match(names, cls)]))
        sub = rx[m]
        # Do not export drug-code frequencies or participant examples.
        first = sub.groupby('eid')['event_date'].min(); post = sub[sub['event_date'] > sub['baseline']].groupby('eid')['event_date'].min()
        fr[f'rx_first_{cls}'] = fr['eid'].map(first); fr[f'rx_post_{cls}'] = fr['eid'].map(post)
    del rx
    print('prescriptions done', round(time.time() - t0), flush=True)

    # ---------------- GP test values after baseline ----------------
    HB = ['42W5.', 'XaPbt', '42W4.', 'XaERp', '42W..']; CR = ['44J3.', 'XE2q5']; HG = ['423..', 'XE2m6', 'Xa96v']
    tb = d.scanner(columns=['eid', 'event_date', 'code_raw', 'value1_num'], filter=(ds.field('source_table') == 'gp_clinical') & (ds.field('code_raw').isin(HB + CR + HG))).to_table().to_pandas()
    tb['event_date'] = pd.to_datetime(tb['event_date'], errors='coerce'); tb = tb.merge(fr[['eid', 'baseline', 'age', 'sex']], on='eid'); tb['dt'] = (tb['event_date'] - tb['baseline']).dt.days
    v = tb['value1_num']
    tb['analyte'] = np.select([tb['code_raw'].isin(HB), tb['code_raw'].isin(CR), tb['code_raw'].isin(HG)], ['hba1c', 'egfr', 'hb'], '')
    pct = tb['code_raw'].isin(['42W4.', 'XaERp']) | ((tb['code_raw'] == '42W..') & (v < 20))
    # DCCT percentages are converted to IFCC mmol/mol and rounded to an integer (6.5% -> 48).
    val = np.where(tb['analyte'] == 'hba1c', np.where(pct, np.round((v - 2.15) * 10.929), v), np.where(tb['analyte'] == 'egfr', egfr_cr(v, tb['age'] + tb['dt'] / 365.25, tb['sex']), v))
    tb['val'] = val
    ok = ((tb['analyte'] == 'hba1c') & tb['val'].between(15, 200)) | ((tb['analyte'] == 'egfr') & v.between(20, 2000)) | ((tb['analyte'] == 'hb') & ((tb['val'].between(4, 25)) | (tb['val'].between(40, 250))))
    tb.loc[(tb['analyte'] == 'hb') & (tb['val'] >= 40), 'val'] /= 10.0   # g/L -> g/dL, including exactly 40 g/L
    tb['abn'] = np.select([tb['analyte'] == 'hba1c', tb['analyte'] == 'egfr', tb['analyte'] == 'hb'], [tb['val'] >= 48, tb['val'] < 60, np.where(tb['sex'] == 0, tb['val'] < 12, tb['val'] < 13)], False)
    tbv = tb[ok & tb['val'].notna()]
    for an in ('hba1c', 'egfr', 'hb'):
        s = tb[tb['analyte'] == an]; sv = tbv[tbv['analyte'] == an]
        post = s[s['dt'] > 0]; fr[f'gp_post_first_test_{an}'] = fr['eid'].map(post.groupby('eid')['event_date'].min())
        ab = sv[(sv['dt'] > 0) & sv['abn']]; first_abn = ab.groupby('eid')['event_date'].min(); fr[f'gp_post_first_abn_{an}'] = fr['eid'].map(first_abn)
        ab = ab.merge(first_abn.rename('first_abn'), left_on='eid', right_index=True)
        fr[f'gp_post_confirm90_{an}'] = fr['eid'].map(ab[ab['event_date'] >= ab['first_abn'] + pd.Timedelta(days=90)].groupby('eid')['event_date'].min())
    print('GP values done', round(time.time() - t0), flush=True)

    # ---------------- condition flags ----------------
    b = fr['baseline']
    fr['pos_diabetes'] = fr['hba1c'] >= 48
    fr['prior_diabetes'] = fr['sr_diabetes'] | fr['dr_diabetes'].fillna(False) | fr['insulin_med'].fillna(False) | fr['srmed_antidiabetic'].fillna(False) | (fr['fo_any_diabetes'] <= b) | (fr['rx_first_antidiabetic'] <= b)
    fr['rec_diabetes'] = pd.concat([fr['fo_rec_diabetes'], fr['rx_post_antidiabetic']], axis=1).min(axis=1)
    fr['pos_ckd'] = fr['egfr'] < 60; fr['pos_ckd_confirmed'] = (fr['egfr'] < 60) & (fr['egfr_cys'] < 60)
    fr['prior_ckd'] = fr['sr_ckd'] | (fr['fo_any_ckd'] <= b)
    fr['rec_ckd'] = fr['fo_rec_ckd']
    fr['pos_anaemia'] = ((fr['sex'] == 0) & (fr['hb'] < 12)) | ((fr['sex'] == 1) & (fr['hb'] < 13))
    fr['prior_anaemia'] = fr['sr_anaemia'] | fr['srmed_iron'].fillna(False) | (fr['fo_any_anaemia'] <= b) | (fr['rx_first_iron'] <= b)
    fr['rec_anaemia'] = pd.concat([fr['fo_rec_anaemia'], fr['rx_post_iron']], axis=1).min(axis=1)

    for column in [c for c in fr if c.startswith('rec_')]:
        fr[column] = fr[column].where(fr[column] > b)
    with open(OUT_FRAME, 'xb') as handle:
        os.fchmod(handle.fileno(), 0o600)
        fr.to_parquet(handle, index=False)
    print('Initial analysis frame completed; run add_diagnosis_codes.py next.')


if __name__ == '__main__':
    main()
