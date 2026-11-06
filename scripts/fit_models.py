"""Fit recorded, combined, measured, incident, sensitivity, and calibrated PU models.

All fits start from externally supplied inputs; outputs remain in the restricted
work directory. Logistic regression is capped at 1,000 iterations for the recorded, combined,
and measured labels of the three primary conditions, and at 2,000 iterations otherwise.
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from common import (CACHE, PRED, PRIMARY, SENSITIVITY, OUT,
    HORIZONS, setup, load_frame, arrays, time_event, censor_strata,
    censor_models, horizon_weights)


def estimator(family, propensity=False):
    if family == 'logit':
        return LogisticRegression(C=1.0 if propensity else 0.1,
                                  max_iter=2000, solver='lbfgs')
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=.05,
        max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1., random_state=0)


def fit_predict(X, y, Xtest, family, weights=None):
    weights = None if weights is None else weights / np.mean(weights)
    return estimator(family).fit(X, y, sample_weight=weights).predict_proba(Xtest)[:, 1]


def calibrated(X, y, Xtest, family, propensity=False):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260905)
    fit = CalibratedClassifierCV(estimator=estimator(family, propensity),
                                method='sigmoid', cv=cv, n_jobs=1)
    return fit.fit(X, y).predict_proba(Xtest)[:, 1]


def run_fold(condition, k):
    setup()
    start = time.time()
    dest = PRED / f'{condition}_fold{k}.parquet'
    meta_path = PRED / f'{condition}_fold{k}.json'
    required = [dest, meta_path]
    if condition in PRIMARY:
        required.append(PRED / f'{condition}_propensity_fold{k}.parquet')
    if all(p.exists() for p in required):
        return json.loads(meta_path.read_text())
    if any(p.exists() for p in required):
        raise FileExistsError('Incomplete fold output; select a fresh work directory.')
    fr = load_frame(model_only=True)
    valid, L, A, rec = arrays(fr, condition)
    fold = fr.outer_fold.to_numpy(int)
    tr = valid & (fold != k)
    te_all = valid & (fold == k)
    te = te_all & ~L
    trh = tr & ~L
    X = np.load(CACHE / f'features/outer_fold_{k}_scaled.npy', mmap_mode='r')
    info = json.loads((CACHE / f'features/outer_fold_{k}_preprocessing.json').read_text())
    low = [i for i, name in enumerate(info['feature_names']) if 'hash' not in name]
    assert len(X) == len(fr) and len(info['feature_names']) == X.shape[1]
    assert X.shape[1] - len(low) == 512
    out = pd.DataFrame({'eid': fr.eid.to_numpy()[te], 'fold': k, 'y': A[te].astype(int)})
    meta = dict(condition=condition, fold=k, n_test=len(out), n_positive=int(A[te].sum()),
                n_train=int(tr.sum()), n_train_unrecorded=int(trh.sum()))
    Xte = np.asarray(X[te])
    for family in ['logit', 'hgb']:
        # Primary baseline logistic fits use 1,000 iterations; sensitivity,
        # incident, and calibrated fits use 2,000.
        def baseline_fit(train, y, columns=None):
            model = estimator(family)
            if condition in PRIMARY and family == 'logit':
                model.set_params(max_iter=1000)
            xtrain, xtest = np.asarray(X[train]), Xte
            if columns is not None:
                xtrain, xtest = xtrain[:, columns], xtest[:, columns]
            return model.fit(xtrain, y).predict_proba(xtest)[:, 1]
        out[f'recorded_{family}'] = baseline_fit(tr, L[tr])
        out[f'reference_{family}'] = baseline_fit(tr, (L | A)[tr])
        out[f'direct_{family}'] = baseline_fit(trh, A[trh])
        if condition in PRIMARY:
            out[f'recorded_lowdim_{family}'] = baseline_fit(tr, L[tr], low)
            out[f'direct_lowdim_{family}'] = baseline_fit(trh, A[trh], low)
    meta['baseline_predictions'] = 'fitted from supplied features and labels'
    t, e, usable = time_event(fr, rec)
    z = censor_strata(fr)
    obs_train = trh & usable
    km = censor_models(t, e, z, obs_train)
    meta['followup'] = {}
    for horizon in HORIZONS:
        if condition not in PRIMARY and horizon != 5:
            continue
        y, w, known = horizon_weights(t, e, usable, z, km, horizon)
        train = trh & known
        out[f'inc{horizon}y'] = y[te]
        out[f'weight_inc{horizon}y'] = w[te]
        out[f'known_inc{horizon}y'] = known[te]
        for family in ['logit', 'hgb']:
            out[f'incident{horizon}y_{family}'] = fit_predict(
                np.asarray(X[train]), y[train], Xte, family, w[train])
        meta['followup'][str(horizon)] = dict(
            train_known=int(train.sum()), train_unknown=int((trh & ~known).sum()),
            test_known=int(known[te].sum()), test_unknown=int((~known[te]).sum()),
            max_training_weight=float(w[train].max()),
            max_test_weight=float(w[te].max()),
            observed_deaths_by_horizon=int((te & usable & (e == 2) & (t <= horizon)).sum()))
    if condition in PRIMARY:
        # Propensities are predicted for all held-out participants with the assay, including
        # those already recorded, so that they can be validated (evaluate_models.py).
        Y = L | A
        Xa = np.asarray(X[te_all])[:, low]
        Ztr = np.asarray(X[tr])[:, low]
        ZY = np.asarray(X[tr & Y])[:, low]
        ZA = np.asarray(X[tr & A])[:, low]
        cY = calibrated(ZY, L[tr & Y], Xa, 'logit', propensity=True)
        cA = calibrated(ZA, L[tr & A], Xa, 'logit', propensity=True)
        prop = pd.DataFrame({'eid': fr.eid.to_numpy()[te_all], 'fold': k,
                            'L': L[te_all].astype(int), 'A': A[te_all].astype(int),
                            'Y': Y[te_all].astype(int), 'c_union': cY, 'c_assay': cA,
                            'constant_union': L[tr & Y].mean(),
                            'constant_assay': L[tr & A].mean()})
        hidden_in_test = ~L[te_all]
        c = np.clip(cY[hidden_in_test], .02, 1.)
        out['propensity_union'] = cY[hidden_in_test]
        out['propensity_assay'] = cA[hidden_in_test]
        for family in ['logit', 'hgb']:
            g = calibrated(Ztr, L[tr], Xa, family)
            gh = np.clip(g[hidden_in_test], 1e-8, 1 - 1e-8)
            out[f'pu_base_{family}'] = gh
            p = np.clip(gh / c, gh, 1.)
            out[f'pu_population_{family}'] = p
            out[f'pu_hidden_{family}'] = (p - gh) / (1 - gh)
            # A weaker floor diagnoses dependence on the stabilization choice.
            cf = np.clip(cY[hidden_in_test], .005, 1.)
            pf = np.clip(gh / cf, gh, 1.)
            out[f'pu_hidden_floor005_{family}'] = (pf - gh) / (1 - gh)
            meta[f'pu_g_gt_c_{family}'] = float((gh > c).mean())
        # Assays for a random 10% of training participants: the same participants serve the
        # PU correction and measured-label training. Recorded status remains available for all.
        rng = np.random.default_rng(20260905 + k)
        indices = np.flatnonzero(tr)
        selected = rng.choice(indices, size=int(round(.1 * len(indices))), replace=False)
        anchor = np.zeros(len(fr), bool)
        anchor[selected] = True
        ca = calibrated(np.asarray(X[anchor & Y])[:, low],
                        L[anchor & Y], Xa, 'logit', propensity=True)
        ca = np.clip(ca[hidden_in_test], .02, 1.)
        gh = out.pu_base_logit.to_numpy()
        pp = np.clip(gh / ca, gh, 1.)
        out['pu_budget10_logit'] = (pp - gh) / (1 - gh)
        out['direct_budget10_logit'] = fit_predict(
            np.asarray(X[anchor & ~L])[:, low], A[anchor & ~L],
            Xte[:, low], 'logit')
        meta['budget10_assays'] = int(anchor.sum())
        meta['budget10_unrecorded_positive'] = int((anchor & ~L & A).sum())
        prop.to_parquet(PRED / f'{condition}_propensity_fold{k}.parquet', index=False)
        (PRED / f'{condition}_propensity_fold{k}.parquet').chmod(0o600)
    out.to_parquet(dest, index=False)
    dest.chmod(0o600)
    meta['seconds'] = round(time.time() - start)
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    meta_path.chmod(0o600)
    print(condition, k, 'finished', meta['seconds'], 'seconds', flush=True)
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', default=','.join(PRIMARY + SENSITIVITY))
    ap.add_argument('--folds', default='0,1,2,3,4')
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()
    setup()
    tasks = [(c, int(k)) for c in args.conditions.split(',') for k in args.folds.split(',')]
    metas = Parallel(n_jobs=args.jobs, backend='loky')(delayed(run_fold)(c, k) for c, k in tasks)
    # All manifests, including complete runs, remain in the restricted work directory.
    if len(tasks) == 30:
        (OUT / 'model_manifest.json').write_text(json.dumps({
            'seed': 20260905, 'outer_folds': 5, 'inner_calibration_folds': 3,
            'paper_features': 627, 'paper_propensity_features': 115,
            'incident_target': 'recognition before death by horizon; IPCW by provider and enrollment year',
            'folds': metas}, indent=2) + '\n')
