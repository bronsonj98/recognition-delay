"""Descriptive analyses for the sensitivity appendix (participant-free outputs).

Uses the main analysis frame, cohort, follow-up endpoint, and estimators
(describe_recognition.estimate; compare_groups.build/contrasts/holm), plus the
private frame supplement from sensitivity_inputs.py.
Outputs: <work>/summaries/sensitivity/*.tsv
"""
from __future__ import annotations
import os
import time
import numpy as np
import pandas as pd
from survival import aj_curve, ci_at, aj_with_bootstrap
from common import DATA, SENS_PRIV, SENS_OUT, PRIMARY, load_frame, arrays, time_event
from describe_recognition import estimate
import compare_groups as rsd

NB = int(os.environ.get('NBOOT', '1000'))
ROUT = SENS_OUT
AN = {'diabetes': 'hba1c', 'ckd_confirmed': 'egfr', 'anaemia': 'hb'}


def suppress(n):
    return int(n) if n == 0 or n >= 5 else np.nan


def load():
    fr = load_frame()
    sup = pd.read_parquet(SENS_PRIV / 'frame_supplement.parquet')
    # First abnormal and confirmation dates are copied from the frame, so keep the frame's columns.
    sup = sup.drop(columns=[c for c in sup if c.startswith(('gp_post_first_abn_', 'gp_post_confirm90_'))])
    fr = fr.merge(sup, on='eid', how='left', validate='one_to_one')
    f = pd.read_csv(DATA / 'menopause_pregnancy.tsv', sep='\t', na_values=['NA', ''])
    f = f.rename(columns={'f.eid': 'eid', 'f.2724.0.0': 'menopause', 'f.3140.0.0': 'pregnant'})
    fr = fr.merge(f, on='eid', how='left', validate='one_to_one')
    return fr


def five_year(fr, mask, rec, label, extra):
    """Five-year recognition among a linked, enrollment-positive, unrecorded subset."""
    g = fr.loc[mask]
    t, e, usable = time_event(g, rec.loc[mask])
    t, e = t[usable], e[usable]
    if len(t) < 30:
        return dict(label=label, n_followup=suppress(len(t)), **extra)
    est, lo, hi = aj_with_bootstrap(t, e, [1, 5], n_boot=NB, seed=31)
    return dict(label=label, n_followup=suppress(len(t)), recognized=suppress((e == 1).sum()),
                ci1=est[0], ci5=est[1], lo5=lo[1], hi5=hi[1], **extra)


def strata_rows(fr, condition, valid, L, A, rec, groups, tag):
    rows = []
    linked = fr.model_cohort.to_numpy() & valid
    for name, m in groups:
        m = np.asarray(m, bool)
        pos = linked & A & m
        unrec = pos & ~L
        rows.append(five_year(fr, unrec, rec, name, dict(
            condition=condition, analysis=tag, n_positive=suppress(pos.sum()),
            n_unrecorded=suppress(unrec.sum()),
            unrecorded_share=unrec.sum() / pos.sum() if pos.sum() else np.nan)))
    return rows


def landmark(fr, mask, rec, start_col):
    """Recognition within one year of a primary-care landmark, as in describe_recognition."""
    g = fr.loc[mask].copy()
    rr = rec.loc[mask]
    t, e, usable = time_event(g, rr)
    g, rr = g.loc[usable].copy(), rr.loc[usable]
    end = g[['coverage_end', 'death']].min(axis=1)
    start = g[start_col].where(g[start_col] <= end)
    risk = start.notna() & (rr.isna() | (rr >= start))
    before = start.notna() & rr.notna() & (rr < start)
    ar = g.loc[risk].copy()
    ar['baseline'] = start[risk] - pd.Timedelta(days=1)
    ta, ea, ua = time_event(ar, rr[risk])
    ta = np.maximum(0, ta[ua] - 1 / 365.25)
    ea = ea[ua]
    est, lo, hi = aj_with_bootstrap(ta, ea, [1], n_boot=NB, seed=7)
    never = rr.isna() | (rr > end)
    return dict(n=suppress(len(g)), with_landmark=suppress(start.notna().sum()),
                recognized_before_landmark=suppress(before.sum()), at_risk=suppress(risk.sum()),
                one_year=est[0], lo=lo[0], hi=hi[0],
                never_recognized_share_of_landmark=float(never[start.notna()].mean()))


def main():
    start_time = time.time()
    ROUT.mkdir(parents=True, exist_ok=True)
    fr = load()
    linked = fr.model_cohort.to_numpy()
    summary, bands, confirm, anaemia_rows, provider_rows = [], [], [], [], []

    # 1. Stricter thresholds, primary endpoint
    for condition, strict, label in [('diabetes', fr.hba1c >= 53, 'HbA1c >= 53'),
                                     ('ckd_confirmed', (fr.egfr < 45) & (fr.egfr_cys < 45), 'both eGFR < 45'),
                                     ('anaemia', fr.hb < 11, 'hemoglobin < 11')]:
        valid, L, A, rec = arrays(fr, condition)
        S = strict.fillna(False).to_numpy(bool)
        for name, D in [(f'{condition} primary', A), (f'{condition} {label}', S)]:
            base = linked & valid
            mask = base & D & ~L
            r, _ = estimate(fr, mask, rec, 'coverage_end', name, 'linked')
            r.update(n_positive=suppress((base & D).sum()), n_unrecorded=suppress(mask.sum()),
                     unrecorded_share=mask.sum() / (base & D).sum())
            summary.append(r)

    # 2. Severity bands among enrollment-positive participants
    for condition in PRIMARY:
        valid, L, A, rec = arrays(fr, condition)
        # Bands are left-closed; A restricts every band to enrollment positives.
        if condition == 'diabetes':
            x, cuts, labels = fr.hba1c, [48, 53, 64, np.inf], ['48-52.9', '53-63.9', '>=64']
        elif condition == 'ckd_confirmed':
            x, cuts, labels = fr.egfr, [-np.inf, 30, 45, 60], ['<30', '30-44.9', '45-59.9']
        else:
            x = pd.Series(np.where(fr.sex == 0, 12, 13) - fr.hb, index=fr.index)
            cuts, labels = [-np.inf, .5, 1, 2, np.inf], ['<0.5', '0.5-0.99', '1-1.99', '>=2']
        band = pd.cut(x, cuts, right=False, labels=labels)
        groups = [(lab, (band == lab).to_numpy()) for lab in labels]
        bands += strata_rows(fr, condition, valid, L, A, rec, groups, 'severity')

    # 3. Primary-care confirmation landmarks
    for condition in PRIMARY:
        valid, L, A, rec = arrays(fr, condition)
        # Post-enrollment values: suppliers whose numeric results are fully parsed (England and Wales).
        mask = linked & valid & A & ~L & fr.gp_values_complete.to_numpy(bool)
        an = AN[condition]
        for tag, col in [('first abnormal value', f'gp_post_first_abn_{an}'),
                         ('second abnormal value >= 90 days later', f'gp_post_confirm90_{an}')]:
            r = landmark(fr, pd.Series(mask, index=fr.index), rec, col)
            r.update(condition=condition, landmark=tag)
            confirm.append(r)

    # 4. Anemia composition: pregnancy, MCV class, menopause
    valid, L, A, rec = arrays(fr, 'anaemia')
    preg = fr.pregnant.isin([1, 2]).to_numpy()
    mcv = fr.mcv
    women, men = (fr.sex == 0).to_numpy(), (fr.sex == 1).to_numpy()
    groups = [('All', np.ones(len(fr), bool)),
              ('Pregnant or unsure at enrollment', preg),
              ('Excluding pregnant or unsure', ~preg),
              ('MCV < 80 fL', (mcv < 80).to_numpy()),
              ('MCV 80-100 fL', mcv.between(80, 100).to_numpy()),
              ('MCV > 100 fL', (mcv > 100).to_numpy()),
              ('Women, premenopausal', women & (fr.menopause == 0).to_numpy()),
              ('Women, postmenopausal', women & (fr.menopause == 1).to_numpy()),
              ('Women, hysterectomy or unsure', women & fr.menopause.isin([2, 3]).to_numpy()),
              ('Women, age < 50', women & (fr.age < 50).to_numpy()),
              ('Women, age >= 50', women & (fr.age >= 50).to_numpy()),
              ('Men', men),
              ('Any hemoglobinopathy evidence', fr.hbp_ever.to_numpy(bool)),
              ('Hemoglobinopathy evidence before enrollment', fr.hbp_prior.to_numpy(bool))]
    anaemia_rows += strata_rows(fr, 'anaemia', valid, L, A, rec, groups, 'composition')

    # 5. Provider (1 England Vision, 2 Scotland, 3 England TPP, 4 Wales)
    for condition in PRIMARY:
        valid, L, A, rec = arrays(fr, condition)
        groups = [(f'provider {p}', (fr.provider_id == p).to_numpy()) for p in [1, 2, 3, 4]]
        provider_rows += strata_rows(fr, condition, valid, L, A, rec, groups, 'provider')

    for name, data in [('strict_thresholds', summary), ('severity_bands', bands),
                       ('confirmation_landmarks', confirm), ('anaemia_composition', anaemia_rows),
                       ('provider_recognition', provider_rows)]:
        pd.DataFrame(data).to_csv(ROUT / f'{name}.tsv', sep='\t', index=False)
    print('core descriptives done', round(time.time() - start_time), flush=True)

    # 6. Anemia without hemoglobinopathy evidence: recognition and group contrasts
    variants = {}
    v = fr.copy()
    v['prior_anaemia'] = v['prior_anaemia_nohbp']
    v['rec_anaemia'] = v['rec_anaemia_nohbp']
    variants['nohbp'] = v
    variants['exhbp'] = fr.loc[~fr.hbp_ever.to_numpy(bool)].copy()
    variants['primary'] = fr
    hrows, crows = [], []
    for tag, f in variants.items():
        valid, L, A, rec = arrays(f, 'anaemia')
        base = f.model_cohort.to_numpy() & valid
        mask = base & A & ~L
        r, _ = estimate(f, mask, rec, 'coverage_end', f'anaemia {tag}', 'linked')
        r.update(n_positive=suppress((base & A).sum()), n_unrecorded=suppress(mask.sum()),
                 unrecorded_share=mask.sum() / (base & A).sum())
        hrows.append(r)
        g, t, e = rsd.build(f, 'anaemia')
        rng = np.random.default_rng(7)
        n = len(g)
        W = rng.multinomial(n, np.full(n, 1 / n), size=NB)
        cres = []
        for name, a, b in rsd.contrasts(g):
            a = a.fillna(False).to_numpy(bool)
            b = b.fillna(False).to_numpy(bool)

            def ci5(mask_, w=None):
                u, c = aj_curve(t[mask_], e[mask_], None if w is None else w[mask_])
                return ci_at(u, c, [5])[0]
            boots = np.array([ci5(a, W[i]) - ci5(b, W[i]) for i in range(NB)])
            p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
            p = min(1.0, max(p, 1 / NB))
            cres.append(dict(variant=tag, contrast=name, n_a=suppress(a.sum()), n_b=suppress(b.sum()),
                             ci5_a=ci5(a), ci5_b=ci5(b), diff=ci5(a) - ci5(b),
                             lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5), p_boot=p))
        for r_, pa in zip(cres, rsd.holm([x['p_boot'] for x in cres])):
            r_['p_holm'] = pa
            crows.append(r_)
    pd.DataFrame(hrows).to_csv(ROUT / 'anaemia_hbp_recognition.tsv', sep='\t', index=False)
    pd.DataFrame(crows).to_csv(ROUT / 'anaemia_hbp_contrasts.tsv', sep='\t', index=False)
    print('done', round(time.time() - start_time), flush=True)


if __name__ == '__main__':
    main()
