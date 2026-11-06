"""Additional held-out model inputs for the sensitivity appendix.

Use the package versions listed in requirements.txt, with which these fits reproduce the paper.

Tasks
  strict_*        targets at the stricter thresholds (HbA1c >= 53; hemoglobin < 11),
                  for the same held-out unrecorded participants as the primary condition.
  demo_*          recorded and measured labels on the 67 demographic, GP data supplier,
                  calendar, and registration columns (no healthcare-use summaries and no
                  code history).
"""
from __future__ import annotations
import argparse
import json
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from common import SENS_PRIV, SENS_OUT, CACHE, PRED, load_frame, arrays

RPRED = SENS_PRIV / 'preds'
ROUT = SENS_OUT
BASE = {'diabetes': 'diabetes', 'ckd': 'ckd_confirmed', 'anaemia': 'anaemia'}


def fit_model(X, y, Xtest, family):
    """Estimators exactly as in fit_models.py for the primary conditions."""
    if family == 'logit':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            m = LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs').fit(X, y)
    else:
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=50, l2_regularization=1.0,
                                           random_state=0).fit(X, y)
    return m.predict_proba(Xtest)[:, 1]


def load():
    fr = load_frame(model_only=True)
    sup = pd.read_parquet(SENS_PRIV / 'frame_supplement.parquet')
    fr = fr.merge(sup, on='eid', how='left', validate='one_to_one')
    assert len(fr) == 186449
    return fr


def features(k):
    X = np.load(CACHE / f'features/outer_fold_{k}_scaled.npy', mmap_mode='r')
    info = json.loads((CACHE / f'features/outer_fold_{k}_preprocessing.json').read_text())
    names = info['feature_names']
    demo = [i for i, n in enumerate(names) if 'hash' not in n and not n.startswith('util__')]
    assert len(names) == 627 and len(demo) == 67
    return X, demo


def strict_fold(fr, base, k, dest):
    condition = BASE[base]
    valid, L, A, _ = arrays(fr, condition)
    S = {'diabetes': fr.hba1c >= 53, 'anaemia': fr.hb < 11}[base].fillna(False).to_numpy(bool)
    assert not (S & ~A & valid).any()
    fold = fr.outer_fold.to_numpy(int)
    te = valid & (fold == k) & ~L
    ref = pd.read_parquet(PRED / f'{condition}_fold{k}.parquet', columns=['eid', 'y'])
    assert np.array_equal(ref.eid.to_numpy(), fr.eid.to_numpy()[te])
    assert np.array_equal(ref.y.to_numpy(), A[te].astype(int))
    out = pd.DataFrame({'eid': fr.eid.to_numpy()[te], 'fold': k, 'y_strict': S[te].astype(int)})
    out.to_parquet(dest, index=False)
    dest.chmod(0o600)
    return dict(n_test=len(out), strict_positive=int(out.y_strict.sum()))


def demo_fold(fr, base, k, dest):
    condition = BASE[base]
    valid, L, A, _ = arrays(fr, condition)
    fold = fr.outer_fold.to_numpy(int)
    tr = valid & (fold != k)
    te = valid & (fold == k) & ~L
    trh = tr & ~L
    ref = pd.read_parquet(PRED / f'{condition}_fold{k}.parquet', columns=['eid', 'y'])
    assert np.array_equal(ref.eid.to_numpy(), fr.eid.to_numpy()[te])
    X, demo = features(k)
    Xtr = np.asarray(X[tr])[:, demo]
    Xtrh = np.asarray(X[trh])[:, demo]
    Xte = np.asarray(X[te])[:, demo]
    out = pd.DataFrame({'eid': fr.eid.to_numpy()[te], 'fold': k, 'y': A[te].astype(int)})
    for family in ['logit', 'hgb']:
        out[f'recorded_demo_{family}'] = fit_model(Xtr, L[tr], Xte, family)
        out[f'direct_demo_{family}'] = fit_model(Xtrh, A[trh], Xte, family)
    out.to_parquet(dest, index=False)
    dest.chmod(0o600)
    return dict(n_test=len(out), positives=int(out.y.sum()))


def run(task, k):
    start = time.time()
    fr = load()
    dest = RPRED / f'{task}_fold{k}.parquet'
    if task.startswith('strict_'):
        meta = strict_fold(fr, task.split('_', 1)[1], k, dest)
    elif task.startswith('demo_'):
        meta = demo_fold(fr, task.split('_', 1)[1], k, dest)
    else:
        raise ValueError(task)
    meta.update(task=task, fold=k, seconds=round(time.time() - start))
    print(json.dumps(meta), flush=True)
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='strict_diabetes,strict_anaemia,demo_diabetes,demo_ckd,demo_anaemia')
    ap.add_argument('--jobs', type=int, default=41)
    args = ap.parse_args()
    RPRED.mkdir(parents=True, exist_ok=True)
    RPRED.chmod(0o700)
    ROUT.mkdir(parents=True, exist_ok=True)
    jobs = [(t, k) for t in args.tasks.split(',') for k in range(5)]
    metas = Parallel(n_jobs=args.jobs, backend='loky')(delayed(run)(t, k) for t, k in jobs)
    (ROUT / 'model_manifest.json').write_text(json.dumps(metas, indent=2) + '\n')
