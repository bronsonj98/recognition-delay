"""Recompute recognition estimates, chronology, and clinical-definition checks."""
from __future__ import annotations
import json
import os
import time
import numpy as np
import pandas as pd
from survival import aj_curve, ci_at, aj_with_bootstrap
from common import OUT, PRIMARY, SENSITIVITY, setup, load_frame, arrays, time_event

H = [1, 2, 5, 8]
NB = int(os.environ.get('NBOOT', '1000'))
GRID = np.arange(0, 8.001, .125)


def suppress(n):
    return int(n) if n == 0 or n >= 5 else np.nan


def estimate(fr, mask, rec, endpoint, name, scope, nb=NB):
    g = fr.loc[mask]
    t, e, usable = time_event(g, rec.loc[mask], endpoint)
    t, e = t[usable], e[usable]
    est, lo, hi = aj_with_bootstrap(t, e, H, n_boot=nb, seed=20260905)
    row = dict(condition=name, scope=scope, endpoint=endpoint,
               n_followup=suppress(len(t)), excluded_no_followup=suppress((~usable).sum()),
               recognized=suppress((e == 1).sum()), deaths=suppress((e == 2).sum()),
               median_event_years=float(np.median(t[e == 1])) if (e == 1).any() else np.nan)
    for h, v, l, u in zip(H, est, lo, hi):
        row.update({f'ci_{h}y': v, f'lo_{h}y': l, f'hi_{h}y': u,
                    f'at_risk_{h}y': suppress((t >= h).sum())})
    u, ci = aj_curve(t, e)
    # The bootstrap band uses the same participants and competing-risk definition.
    rng = np.random.default_rng(20260905)
    bands = []
    for _ in range(min(nb, 300)):
        w = rng.multinomial(len(t), np.full(len(t), 1 / len(t)))
        ub, cb = aj_curve(t, e, w)
        bands.append(ci_at(ub, cb, GRID))
    bands = np.asarray(bands)
    curve = pd.DataFrame(dict(condition=name, scope=scope, endpoint=endpoint,
                              years=GRID, ci=ci_at(u, ci, GRID),
                              lo=np.percentile(bands, 2.5, axis=0),
                              hi=np.percentile(bands, 97.5, axis=0)))
    return row, curve


def main():
    setup()
    start = time.time()
    fr = load_frame()
    rows, curves, decomposition, repeats, composition = [], [], [], [], []
    for condition in PRIMARY + SENSITIVITY + ['ckd']:
        valid, L, A, rec = arrays(fr, condition)
        for scope, sm in [('linked', fr.model_cohort.to_numpy()), ('whole', np.ones(len(fr), bool))]:
            if scope == 'whole' and condition not in PRIMARY:
                continue
            base = sm & valid
            mask = base & A & ~L
            end = 'coverage_end' if scope == 'linked' else 'hes_censor'
            r, c = estimate(fr, mask, rec, end, condition, scope)
            r.update(n_measured=suppress(base.sum()), n_positive=suppress((base & A).sum()),
                     n_unrecorded=suppress(mask.sum()),
                     unrecorded_share=mask.sum() / (base & A).sum())
            rows.append(r)
            curves.append(c)
        if condition in PRIMARY:
            mask = fr.model_cohort.to_numpy() & valid & A & ~L
            r, _ = estimate(fr, mask, rec, 'hes_censor', condition, 'linked_hospital_end', nb=300)
            rows.append(r)
        linked = fr.model_cohort.to_numpy() & valid & A & ~L
        g = fr.loc[linked].copy()
        rr = rec.loc[linked]
        t, e, usable = time_event(g, rr)
        g, rr, t, e = g.loc[usable].copy(), rr.loc[usable], t[usable], e[usable]
        if condition in PRIMARY + ['ckd']:
            # Post-enrollment values: suppliers whose numeric results are fully parsed (England and Wales).
            gv = g.gp_values_complete.to_numpy(bool)
            g_all, rr_all = g, rr
            g, rr = g.loc[gv].copy(), rr.loc[gv]
            an = {'diabetes': 'hba1c', 'ckd': 'egfr', 'ckd_confirmed': 'egfr', 'anaemia': 'hb'}[condition]
            end = g[['coverage_end', 'death']].min(axis=1)
            recognized = rr.notna() & (rr <= end)
            first = g['gp_post_first_test_' + an].where(g['gp_post_first_test_' + an] <= end)
            abnormal = g['gp_post_first_abn_' + an].where(g['gp_post_first_abn_' + an] <= end)
            nr = ~recognized
            # Same-day coding counts as recognition at day zero. People coded
            # before their first abnormal clinical value are excluded from this clock.
            risk = abnormal.notna() & (rr.isna() | (rr >= abnormal))
            ar = g.loc[risk].copy()
            ar['baseline'] = abnormal[risk] - pd.Timedelta(days=1)
            rec_ar = rr[risk]
            ta, ea, ua = time_event(ar, rec_ar)
            ta = np.maximum(0, ta[ua] - 1 / 365.25)
            ea = ea[ua]
            est, lo, hi = aj_with_bootstrap(ta, ea, [1], n_boot=NB, seed=7)
            decomposition.append(dict(condition=condition, n=suppress(len(g)),
                recognized_fraction=recognized.mean(), tested_fraction=first.notna().mean(),
                abnormal_fraction=abnormal.notna().mean(), n_never_recognized=suppress(nr.sum()),
                never_retested_fraction_of_never=first[nr].isna().mean(),
                normal_retests_fraction_of_never=(first[nr].notna() & abnormal[nr].isna()).mean(),
                abnormal_uncoded_fraction_of_never=abnormal[nr].notna().mean(),
                recognized_before_first_abnormal=suppress((abnormal.notna() & rr.notna() & (rr < abnormal)).sum()),
                n_at_first_abnormal=suppress(risk.sum()), one_year_recognition=est[0],
                one_year_recognition_lo=lo[0], one_year_recognition_hi=hi[0],
                n_all_suppliers=suppress(len(g_all))))
            g, rr = g_all, rr_all
            # Repeat-visit evidence is presented with its actual unrecorded denominator.
            col = {'diabetes': 'hba1c_v1', 'ckd': 'egfr_v1',
                   'ckd_confirmed': 'egfr_v1', 'anaemia': 'hb_v1'}[condition]
            base_repeat = valid & A & ~L & fr[col].notna().to_numpy() & fr.visit1.notna().to_numpy()
            h = fr.loc[base_repeat]
            hr = rec.loc[base_repeat]
            unrec = hr.isna() | (hr > h.visit1)
            pos = (h[col] >= 48 if condition == 'diabetes' else
                   h[col] < 60 if condition.startswith('ckd') else
                   h[col] < np.where(h.sex == 0, 12, 13))
            repeats.append(dict(condition=condition, n_remeasured=suppress(len(h)),
                                n_still_unrecorded=suppress(unrec.sum()),
                                n_persistent_unrecorded=suppress((unrec & pos).sum()),
                                persistent_unrecorded_fraction=pos[unrec].mean()))
        if condition in PRIMARY:
            # Prospective composition uses linked records within the same observable interval.
            all_hidden = fr.model_cohort.to_numpy() & valid & ~L
            h = fr.loc[all_hidden]
            rh = rec.loc[all_hidden].where(rec.loc[all_hidden] <= h[['coverage_end', 'death']].min(axis=1))
            years = (rh - h.baseline).dt.days / 365.25
            if condition == 'diabetes':
                state = pd.cut(h.hba1c, [-np.inf, 42, 48, np.inf], right=False,
                               labels=['<42', '42-47.9', '>=48'])
            elif condition == 'ckd_confirmed':
                state = pd.cut(h.egfr, [-np.inf, 60, 90, np.inf], right=False,
                               labels=['<60', '60-89', '>=90'])
            else:
                state = pd.Series(np.where(A[all_hidden], 'anemic', 'not_anemic'), index=h.index)
            for lo, hi in [(0, 1), (1, 2), (2, 5), (5, 8)]:
                m = (years > lo) & (years <= hi)
                for label, n in state[m].value_counts().items():
                    composition.append(dict(condition=condition, window=f'{lo}-{hi}',
                                            state=str(label), n=suppress(n), share=n / m.sum()))
        print(condition, 'descriptives finished', round(time.time()-start), flush=True)
    for name, data in [('recognition_summary', rows), ('tested_but_uncoded', decomposition),
                       ('repeat_validation', repeats), ('incident_composition', composition)]:
        pd.DataFrame(data).to_csv(OUT / f'{name}.tsv', sep='\t', index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / 'recognition_curves.tsv', sep='\t', index=False)
    audit = {'participants': len(fr), 'linked': int(fr.model_cohort.sum()), 'n_boot': NB,
             'linked_without_valid_gp_endpoint': int((fr.model_cohort & (
                 fr.coverage_end.isna() | (fr.coverage_end <= fr.baseline))).sum()),
             'seconds': round(time.time()-start),
             'gp_endpoint': 'minimum of participant GP coverage end and hospital endpoint',
             'post_abnormal_clock': 'exclude recognition before abnormal clinical value; same-day events at zero',
             'severity_models': 'run compare_groups.py separately'}
    (OUT / 'descriptive_manifest.json').write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    main()
