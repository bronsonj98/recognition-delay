"""Vectorized Aalen-Johansen cumulative incidence with a competing event and weighted bootstrap."""
from __future__ import annotations
import numpy as np


def aj_curve(t, e, w=None):
    """Return (unique event times, cumulative incidence of e==1) treating e==2 as competing, e==0 censored.

    ``w`` are non-negative participant weights (bootstrap multiplicities); default 1.
    """
    t = np.asarray(t, float); e = np.asarray(e, int)
    w = np.ones_like(t) if w is None else np.asarray(w, float)
    order = np.argsort(t, kind='stable'); t, e, w = t[order], e[order], w[order]
    total = w.sum()
    cum_w = np.concatenate(([0.0], np.cumsum(w)))
    ev = e > 0
    if not ev.any():
        return np.zeros(0), np.zeros(0)
    u, inv = np.unique(t[ev], return_inverse=True)
    d1 = np.bincount(inv, weights=w[ev] * (e[ev] == 1), minlength=len(u))
    d2 = np.bincount(inv, weights=w[ev] * (e[ev] == 2), minlength=len(u))
    at_risk = total - cum_w[np.searchsorted(t, u, side='left')]
    at_risk = np.where(at_risk <= 0, np.nan, at_risk)
    haz_all = (d1 + d2) / at_risk
    surv_prev = np.concatenate(([1.0], np.cumprod(1 - haz_all)[:-1]))
    ci = np.cumsum(np.nan_to_num(surv_prev * d1 / at_risk))
    return u, ci


def ci_at(u, ci, horizons):
    out = []
    for h in horizons:
        idx = np.searchsorted(u, h, side='right') - 1
        out.append(float(ci[idx]) if idx >= 0 else 0.0)
    return np.array(out)


def aj_with_bootstrap(t, e, horizons, n_boot=200, seed=0):
    """Point estimate and percentile 95% interval of cumulative incidence at horizons."""
    t = np.asarray(t, float); e = np.asarray(e, int); n = len(t)
    u, ci = aj_curve(t, e)
    est = ci_at(u, ci, horizons)
    if n_boot <= 0 or n < 20:
        return est, np.full_like(est, np.nan), np.full_like(est, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, len(horizons)))
    for b in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n))
        ub, cb = aj_curve(t, e, w)
        boots[b] = ci_at(ub, cb, horizons)
    return est, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)

