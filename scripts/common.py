"""Shared definitions, censoring weights, exact-capacity selection, and external paths."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import os

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('ANALYSIS_DATA_ROOT', 'RESTRICTED_INPUTS')).expanduser().resolve()
WORK = Path(os.environ.get('ANALYSIS_WORK_ROOT', 'RESTRICTED_WORK')).expanduser().resolve()
PRIVATE = WORK / 'inputs'
CACHE = WORK
OUT = WORK / 'summaries'
PRED = WORK / 'predictions'
SENS_PRIV = WORK / 'sensitivity_inputs'
SENS_OUT = OUT / 'sensitivity'
PRIMARY = ['diabetes', 'ckd_confirmed', 'anaemia']
SENSITIVITY = ['diabetes_calibrated', 'diabetes_no_rx', 'anaemia_no_rx']
HORIZONS = (1, 2, 5)


def setup():
    """Keep every input and runtime artifact outside the source release."""
    if not all(os.environ.get(k) for k in ('ANALYSIS_DATA_ROOT', 'ANALYSIS_WORK_ROOT')):
        raise ValueError('Set ANALYSIS_DATA_ROOT and ANALYSIS_WORK_ROOT; see README.md.')
    for path in (DATA, WORK, PRIVATE, OUT, PRED, WORK / 'features', WORK / 'rendered'):
        if path.resolve().is_relative_to(ROOT):
            raise ValueError('Data and runtime directories must be outside the source release.')
    if DATA == WORK or DATA.is_relative_to(WORK) or WORK.is_relative_to(DATA):
        raise ValueError('Input and work directories must be separate, non-nested locations.')
    os.umask(0o077)
    for path in (WORK, PRIVATE, OUT, PRED):
        path.mkdir(parents=True, mode=0o700, exist_ok=True)


def fresh(path):
    """Refuse to overwrite an existing artifact."""
    if path.exists():
        raise FileExistsError('Output already exists; select a fresh work directory.')
    return path


def load_frame(model_only=False):
    setup()
    fr = pd.read_parquet(PRIVATE / 'recognition_frame.parquet')
    assert fr.eid.is_unique and fr.eid.notna().all()
    fr['coverage_end'] = fr[['gp_end', 'hes_censor']].min(axis=1)
    # A linked analysis needs a known GP endpoint; do not replace missing GP
    # coverage by the hospital endpoint or clip negative observation durations.
    fr.loc[fr.gp_end.isna(), 'coverage_end'] = pd.NaT
    if model_only:
        order = pq.read_table(PRIVATE / 'participants.parquet',
                              columns=['eid', 'outer_fold']).to_pandas()
        assert order.eid.is_monotonic_increasing and order.eid.is_unique
        assert order.outer_fold.notna().all() and set(order.outer_fold) == set(range(5))
        fr = order.merge(fr.drop(columns=['outer_fold']), on='eid', validate='one_to_one', how='left')
        assert fr.model_cohort.notna().all() and fr.model_cohort.all()
    return fr


def arrays(fr, condition):
    base = condition.split('_calibrated')[0].split('_no_rx')[0]
    base = 'ckd' if base == 'ckd_confirmed' else base
    assay = {'diabetes': 'hba1c', 'ckd': 'creat', 'anaemia': 'hb'}[base]
    valid = fr[assay].notna().to_numpy()
    L = fr['prior_' + base].fillna(False).to_numpy(bool)
    pos = 'pos_ckd_confirmed' if condition == 'ckd_confirmed' else 'pos_' + base
    A = fr[pos].fillna(False).to_numpy(bool)
    rec = fr['rec_' + base].copy()
    if condition == 'ckd_confirmed':
        valid &= fr.cysc.notna().to_numpy()
    if condition == 'diabetes_calibrated':
        A = (0.9696 * fr.hba1c + 3.3595 >= 48).to_numpy()
    if condition.endswith('_no_rx'):
        L = (fr['sr_' + base].fillna(False) |
             (fr['fo_any_' + base] <= fr.baseline) |
             (fr['gpcode_first_' + base] <= fr.baseline)).to_numpy(bool)
        if base == 'diabetes':
            L |= fr.dr_diabetes.fillna(False).to_numpy(bool)
        rec = fr[['fo_rec_' + base, 'gpcode_post_' + base]].min(axis=1)
        rec = rec.where(rec > fr.baseline)
    return valid, L, A, rec


def time_event(fr, rec, endpoint='coverage_end'):
    """Return nonnegative times only for valid observation intervals."""
    admin = fr[endpoint]
    end = pd.concat([admin, fr.death], axis=1).min(axis=1)
    usable = admin.notna() & (admin > fr.baseline) & (end > fr.baseline)
    event = rec.where((rec > fr.baseline) & (rec <= end) & usable)
    stop = pd.concat([event, end], axis=1).min(axis=1)
    t = ((stop - fr.baseline).dt.days / 365.25).to_numpy(float)
    e = np.where(event.notna(), 1,
                 np.where(fr.death.notna() & (fr.death <= admin), 2, 0))
    return t, e, usable.to_numpy(bool)


def censor_strata(fr):
    # Enrollment date and GP provider determine much of administrative follow-up.
    return (fr.provider_id.astype('Int64').astype(str) + '_' +
            fr.baseline.dt.year.astype('Int64').astype(str)).to_numpy()


def km_censor(t, e):
    """KM survival of censoring time; clinical events/deaths censor this process."""
    order = np.argsort(t, kind='stable')
    tt, ee = np.asarray(t)[order], np.asarray(e)[order]
    u, first, count = np.unique(tt, return_index=True, return_counts=True)
    dc = np.add.reduceat((ee == 0).astype(float), first)
    risk = len(tt) - first
    survival = np.cumprod(1 - dc / risk)
    return u, survival


def censor_models(t, e, strata, train):
    models = {'pooled': km_censor(t[train], e[train])}
    for s in np.unique(strata[train]):
        m = train & (strata == s)
        if m.sum() >= 200:
            models[s] = km_censor(t[m], e[m])
    return models


def horizon_weights(t, e, usable, strata, models, horizon):
    """IPCW binary cumulative-incidence target, including competing deaths.

    Events/deaths observed by H have known status; otherwise follow-up must reach
    H. Weights use G(t-) for terminal events and G(H-) for follow-up through H.
    G is estimated only in the outer-training population.
    """
    t, e = np.asarray(t), np.asarray(e)
    terminal = (e > 0) & (t <= horizon)
    known = usable & (terminal | (t >= horizon))
    y = ((e == 1) & (t <= horizon)).astype(int)
    when = np.where(terminal, t, horizon)
    w = np.zeros(len(t))
    for s in np.unique(strata[known]):
        m = known & (strata == s)
        u, surv = models.get(s, models['pooled'])
        ix = np.searchsorted(u, when[m], side='left') - 1
        g = np.where(ix < 0, 1.0, surv[np.maximum(ix, 0)])
        if (g <= 0).any():
            raise ValueError('Censoring positivity failed at the requested horizon.')
        w[m] = 1 / g
    return y, w, known


def top_mask(scores, fold, ids, q=0.1, weights=None):
    """Exact fold-local capacity; multiplicity weights reselect bootstrap top-q."""
    weights = np.ones(len(scores), dtype=np.int64) if weights is None else np.asarray(weights)
    chosen = np.zeros(len(scores), dtype=np.int64)
    for k in np.unique(fold):
        ix = np.flatnonzero(fold == k)
        order = ix[np.lexsort((ids[ix], -np.asarray(scores)[ix]))]
        capacity = max(1, int(round(q * weights[ix].sum())))
        cum = np.cumsum(weights[order])
        before = cum - weights[order]
        chosen[order] = np.clip(capacity - before, 0, weights[order])
    return chosen
