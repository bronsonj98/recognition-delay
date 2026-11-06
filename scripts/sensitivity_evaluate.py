"""Held-out evaluations for the sensitivity appendix (participant-free outputs).

Reuses the evaluation machinery (evaluate_models.capacity_bootstrap,
bootstrap_weights with the same seed, fold-local top fractions, ties by participant
ID). Models are the main held-out predictions plus the sensitivity_models.py fits.
Outputs (<work>/summaries/sensitivity/): eval_operating, eval_calibration, and
eval_net_benefit (operating points and net benefit), eval_strict_contrasts (stricter
thresholds as the target), eval_representation_contrasts (feature sets), and
eval_provider_contrasts (selection within GP data supplier).
"""
from __future__ import annotations
import json
import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from common import SENS_PRIV, SENS_OUT, PRED, PRIMARY, load_frame, top_mask
from evaluate_models import capacity_bootstrap, bootstrap_weights, ci

NB = int(os.environ.get('NBOOT', '1000'))
RPRED = SENS_PRIV / 'preds'
ROUT = SENS_OUT
BASE = {'diabetes': 'diabetes', 'ckd_confirmed': 'ckd', 'anaemia': 'anaemia'}
ARMS = ['recorded_hgb', 'direct_hgb']
# Risk threshold for net benefit (Appendix table), per condition.
THRESHOLD = {'diabetes': .01, 'ckd_confirmed': .01, 'anaemia': .05}


def primary(condition):
    f = pd.concat([pd.read_parquet(PRED / f'{condition}_fold{k}.parquet') for k in range(5)], ignore_index=True)
    assert f.eid.is_unique
    return f


def contrast(rows, boots, condition, analysis, a, b, extra=None):
    ra = next(r for r in rows if r['arm'] == a)
    rb = next(r for r in rows if r['arm'] == b)
    d = boots[a]['q10'] - boots[b]['q10']
    lo, hi = ci(d)
    out = dict(condition=condition, analysis=analysis, arm=a, comparator=b,
               yield_arm=ra['yield_q10'], yield_comparator=rb['yield_q10'],
               difference=ra['yield_q10'] - rb['yield_q10'], lo=lo, hi=hi,
               relative_loss=1 - ra['yield_q10'] / rb['yield_q10'])
    out.update(extra or {})
    return out


def yields(f, y, arms, W, condition, analysis):
    """Found per 1,000 tested at the top 10%, with capacity-preserving bootstrap draws."""
    fold, ids = f.fold.to_numpy(int), f.eid.to_numpy()
    rows, boots = [], {}
    qs = np.array([.1])
    for arm in arms:
        score = f[arm].to_numpy()
        bs, _ = capacity_bootstrap(y, score, fold, ids, W, qs)
        boots[arm] = {'q10': bs[:, 0]}
        lo, hi = ci(bs[:, 0])
        rows.append(dict(condition=condition, analysis=analysis, arm=arm, n=len(y), positives=int(y.sum()),
                         yield_q10=1000 * y[top_mask(score, fold, ids, .1).astype(bool)].mean(), lo=lo, hi=hi,
                         auroc=roc_auc_score(y, score), ap=average_precision_score(y, score)))
    return rows, boots


def recall_curve(y, score, fold, ids, grid):
    found = np.zeros(len(grid))
    for k in np.unique(fold):
        ix = np.flatnonzero(fold == k)
        order = ix[np.lexsort((ids[ix], -score[ix]))]
        cum = np.cumsum(y[order])
        caps = np.maximum(1, np.rint(grid * len(ix)).astype(int))
        found += cum[np.minimum(caps, len(ix)) - 1]
    return found / y.sum()


def operating_points(condition, f):
    y, fold, ids = f.y.to_numpy(int), f.fold.to_numpy(int), f.eid.to_numpy()
    grid = np.round(np.arange(1, 1001) / 1000, 3)
    op, cal, nb_rows = [], [], []
    pt = THRESHOLD[condition]
    odds = pt / (1 - pt)
    for arm in ARMS:
        score = f[arm].to_numpy()
        rc = recall_curve(y, score, fold, ids, grid)
        top1 = top_mask(score, fold, ids, .01).astype(bool)
        top10 = top_mask(score, fold, ids, .1).astype(bool)
        op.append(dict(condition=condition, arm=arm, prevalence_per_100=100 * y.mean(),
                       cases_per_100_tests_q1=100 * y[top1].mean(), tests_per_case_q10=1 / y[top10].mean(),
                       fraction_tested_for_recall_50=float(grid[np.argmax(rc >= .5)])))
        # Cross-fitted isotonic recalibration to the enrollment assay: each fold's
        # scores are mapped with a calibrator fitted on the other folds' held-out scores.
        r_ = np.zeros(len(y))
        for k in range(5):
            te, tr = fold == k, fold != k
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(score[tr], y[tr])
            r_[te] = iso.predict(score[te])
        cal.append(dict(condition=condition, arm=arm, expected_per_1000_top10=1000 * r_[top10].mean(),
                        observed_per_1000_top10=1000 * y[top10].mean()))
        s = (r_ >= pt).astype(float)
        nb_rows.append(dict(condition=condition, threshold=pt, arm=arm,
                            net_benefit_per_1000=1000 * (np.sum(y * s) - odds * np.sum((1 - y) * s)) / len(y)))
    return op, cal, nb_rows


def evaluate_condition(condition):
    start = time.time()
    f = primary(condition)
    y, fold = f.y.to_numpy(int), f.fold.to_numpy(int)
    W = bootstrap_weights(fold, NB)
    out = {}
    # 1. Operating points, calibration, net benefit
    out['operating'], out['calibration'], out['net_benefit'] = operating_points(condition, f)
    # 2. Stricter threshold as the evaluation target (same held-out people and models). The stricter
    #    kidney threshold leaves too few unrecorded participants for case finding.
    if condition != 'ckd_confirmed':
        base = BASE[condition]
        s = pd.concat([pd.read_parquet(RPRED / f'strict_{base}_fold{k}.parquet') for k in range(5)], ignore_index=True)
        assert np.array_equal(s.eid.to_numpy(), f.eid.to_numpy())
        g = f.merge(s.drop(columns=['fold']), on='eid', validate='one_to_one')
        ys = g.y_strict.to_numpy(int)
        assert (ys <= y).all()
        rows_s, boots_s = yields(g, ys, ARMS, W, condition, 'strict_target')
        out['strict_contrasts'] = [contrast(rows_s, boots_s, condition, 'strict_target', 'recorded_hgb', 'direct_hgb')]
    # 3. Representation: full (627), no code history (115), demographics only (67)
    d = pd.concat([pd.read_parquet(RPRED / f'demo_{BASE[condition]}_fold{k}.parquet') for k in range(5)], ignore_index=True)
    assert np.array_equal(d.eid.to_numpy(), f.eid.to_numpy()) and np.array_equal(d.y.to_numpy(), y)
    h = f.merge(d.drop(columns=['fold', 'y']), on='eid', validate='one_to_one')
    arms = [f'{a}{v}_{fam}' for fam in ('hgb', 'logit') for v in ('', '_lowdim', '_demo') for a in ('recorded', 'direct')]
    rows_r, boots_r = yields(h, y, arms, W, condition, 'representation')
    out['representation_contrasts'] = [
        contrast(rows_r, boots_r, condition, 'representation', f'recorded{v}_{fam}', f'direct{v}_{fam}',
                 dict(representation={'': '627', '_lowdim': '115', '_demo': '67'}[v]))
        for fam in ('hgb', 'logit') for v in ('', '_lowdim', '_demo')]
    # 4. GP data supplier: selection within supplier and fold
    demo = load_frame(model_only=True)[['eid', 'provider_id']]
    pf = f.merge(demo, on='eid', validate='one_to_one')
    prov_con = []
    for p in (1, 2, 3, 4):
        sub = pf.loc[(pf.provider_id == p).to_numpy()].reset_index(drop=True)
        Wp = bootstrap_weights(sub.fold.to_numpy(int), NB, seed=11 + p)
        r_p, b_p = yields(sub, sub.y.to_numpy(int), ARMS, Wp, condition, f'provider_{p}')
        prov_con.append(contrast(r_p, b_p, condition, f'provider_{p}', 'recorded_hgb', 'direct_hgb'))
    out['provider_contrasts'] = prov_con
    print(condition, 'done', round(time.time() - start), flush=True)
    return out


if __name__ == '__main__':
    ROUT.mkdir(parents=True, exist_ok=True)
    res = Parallel(n_jobs=len(PRIMARY), backend='loky')(delayed(evaluate_condition)(c) for c in PRIMARY)
    merged = {}
    for r in res:
        for key, rows in r.items():
            merged.setdefault(key, []).extend(rows)
    for key, rows in merged.items():
        pd.DataFrame(rows).to_csv(ROUT / f'eval_{key}.tsv', sep='\t', index=False)
    (ROUT / 'eval_manifest.json').write_text(json.dumps({'n_boot': NB, 'bootstrap_seed': 11,
        'provider_bootstrap_seed': '11 + provider', 'recalibration': 'cross-fitted isotonic across outer folds',
        'net_benefit_threshold': THRESHOLD}, indent=2) + '\n')
