"""Subgroup recognition: unadjusted contrasts and a severity decomposition.

Follow-up to describe_recognition.py (same frame, same linked-cohort censoring at
the earlier of the practice extract end and the hospital endpoint, same
participants). Written so that the paper can show, side by side, the raw
five-year recognition gaps between groups and how much of each gap remains once
enrollment assay severity is held fixed.

Runtime summaries in the external work directory:
  recognition_strata_full.tsv  five-year recognition by stratum (Aalen-Johansen, death competing)
  stratum_contrasts.tsv        paired participant-bootstrap differences for prespecified
                               contrasts, Holm-adjusted within condition
  severity_decomposition.tsv   Fine-Gray hazard ratios for the same contrasts
                               without and with enrollment-severity splines
  severity_by_group.tsv        enrollment severity distribution by stratum
  risk_6y.tsv                  participants still at risk at six years
"""
from __future__ import annotations
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from survival import aj_curve, ci_at, aj_with_bootstrap
from common import OUT, PRIMARY, setup, load_frame, arrays, time_event

NB = int(os.environ.get('NBOOT', '1000'))
NONWHITE = ['asian', 'black', 'mixed', 'chinese', 'other']
TCOL = {'diabetes': 'gp_test1y_hba1c', 'ckd': 'gp_test1y_creat', 'anaemia': 'gp_test1y_hb'}


def suppress(n):
    return int(n) if n == 0 or n >= 5 else np.nan


def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def build(fr, condition):
    valid, L, A, rec = arrays(fr, condition)
    linked = fr.model_cohort.to_numpy() & valid & A & ~L
    g = fr.loc[linked].copy()
    rr = rec.loc[linked]
    t, e, usable = time_event(g, rr)
    g, t, e = g.loc[usable].copy(), t[usable], e[usable]
    base = 'ckd' if condition.startswith('ckd') else condition
    g['tested'] = g[TCOL[base]].fillna(False).astype(bool)
    g['nonwhite'] = g.eth_group.isin(NONWHITE)
    g['white'] = g.eth_group == 'white'
    # Severity on the scale a clinician sees: HbA1c itself, creatinine eGFR
    # (lower is worse), and hemoglobin deficit below the sex-specific threshold.
    g['severity'] = (g.hba1c if condition == 'diabetes' else
                     g.egfr if condition.startswith('ckd') else
                     np.where(g.sex == 0, 12, 13) - g.hb)
    g['mild'] = (g.hba1c < 53 if condition == 'diabetes' else
                 g.egfr >= 45 if condition.startswith('ckd') else
                 g.severity < 1.0)
    return g, t, e


def levels(g):
    tq = g.tq.astype(float)
    return [('Deprivation', 'Q1', tq == 0), ('Deprivation', 'Q2', tq == 1), ('Deprivation', 'Q3', tq == 2),
            ('Deprivation', 'Q4', tq == 3), ('Deprivation', 'Q5', tq == 4),
            ('Sex', 'Women', g.sex == 0), ('Sex', 'Men', g.sex == 1),
            ('Ethnicity', 'White', g.white), ('Ethnicity', 'Non-white', g.nonwhite),
            ('Age', '<55', g.age < 55), ('Age', '55-64', (g.age >= 55) & (g.age < 65)), ('Age', '65+', g.age >= 65),
            ('Tested', 'No', ~g.tested), ('Tested', 'Yes', g.tested)]


def contrasts(g):
    tq = g.tq.astype(float)
    return [('Women vs. men', g.sex == 0, g.sex == 1),
            ('Most vs. least deprived quintile', tq == 4, tq == 0),
            ('Non-white vs. white', g.nonwhite, g.white),
            ('Age 65+ vs. <55', g.age >= 65, g.age < 55),
            ('Tested vs. not tested in prior year', g.tested, ~g.tested)]


def main():
    setup()
    start = time.time()
    fr = load_frame()
    strata, crows, sev, risk, inputs = [], [], [], [], []
    for condition in PRIMARY:
        g, t, e = build(fr, condition)
        risk.append(dict(condition=condition, n_followup=suppress(len(t)), at_risk_6y=suppress((t >= 6).sum()),
                         at_risk_8y=suppress((t >= 8).sum())))
        for group, label, m in levels(g):
            m = m.fillna(False).to_numpy(bool)
            if m.sum() < 30:
                continue
            est, lo, hi = aj_with_bootstrap(t[m], e[m], [5], n_boot=NB, seed=31)
            strata.append(dict(condition=condition, group=group, level=label, n=suppress(m.sum()),
                               ci5=est[0], lo=lo[0], hi=hi[0]))
            s = g.severity[m]
            sev.append(dict(condition=condition, group=group, level=label, n=suppress(m.sum()),
                            median=float(s.median()), q25=float(s.quantile(.25)), q75=float(s.quantile(.75)),
                            mild_share=float(g.mild[m].mean())))
        rng = np.random.default_rng(7)
        n = len(g)
        W = rng.multinomial(n, np.full(n, 1 / n), size=NB)
        cres = []
        for name, a, b in contrasts(g):
            a = a.fillna(False).to_numpy(bool)
            b = b.fillna(False).to_numpy(bool)
            if a.sum() < 30 or b.sum() < 30:
                continue

            def ci5(mask, w=None):
                u, ci = aj_curve(t[mask], e[mask], None if w is None else w[mask])
                return ci_at(u, ci, [5])[0]
            est = ci5(a) - ci5(b)
            boots = np.array([ci5(a, W[i]) - ci5(b, W[i]) for i in range(NB)])
            p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
            p = min(1.0, max(p, 1 / NB))
            cres.append(dict(condition=condition, contrast=name, n_a=suppress(a.sum()), n_b=suppress(b.sum()),
                             ci5_a=ci5(a), ci5_b=ci5(b), diff=est,
                             lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5), p_boot=p))
        for r, pa in zip(cres, holm([r['p_boot'] for r in cres])):
            r['p_holm'] = pa
            crows.append(r)
        ex = g[['age', 'sex', 'tq', 'provider_id']].copy()
        ex['t'], ex['e'], ex['condition'] = t, e, condition
        ex['nonwhite'] = g.nonwhite.astype(int)
        ex.loc[g.eth_group == 'missing', 'nonwhite'] = np.nan
        ex['tested'] = g.tested.astype(int)
        ex['year'] = g.baseline.dt.year
        # Severity is oriented so that higher values are worse (higher is worse).
        ex['severity'] = (g.hba1c if condition == 'diabetes' else
                          -g.egfr if condition.startswith('ckd') else
                          np.where(g.sex == 0, 12, 13) - g.hb)
        ex['cystatin_egfr'] = g.egfr_cys
        inputs.append(ex)
        print(condition, 'strata and contrasts done', round(time.time() - start), flush=True)
    pd.DataFrame(strata).to_csv(OUT / 'recognition_strata_full.tsv', sep='\t', index=False)
    pd.DataFrame(crows).to_csv(OUT / 'stratum_contrasts.tsv', sep='\t', index=False)
    pd.DataFrame(sev).to_csv(OUT / 'severity_by_group.tsv', sep='\t', index=False)
    pd.DataFrame(risk).to_csv(OUT / 'risk_6y.tsv', sep='\t', index=False)
    inp = pd.concat(inputs, ignore_index=True)
    inp['id'] = np.arange(len(inp))
    run = subprocess.run(['Rscript', str(Path(__file__).with_name('severity_models.R')), str(OUT)],
                         input=inp.to_csv(index=False), text=True, capture_output=True)
    # Do not echo subprocess diagnostics, which may contain input values.
    print('Severity model subprocess completed.', flush=True)
    if run.returncode:
        raise RuntimeError('Severity decomposition models failed.')
    print('done', round(time.time() - start))


if __name__ == '__main__':
    main()
