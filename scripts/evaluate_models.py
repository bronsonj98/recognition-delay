"""Evaluate held-out models with capacity-preserving paired bootstrap."""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import xlogy
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from statsmodels.stats.proportion import proportion_confint
from common import (OUT, PRED, PRIMARY, SENSITIVITY, setup, load_frame, top_mask)

Q = np.array([.01, .02, .03, .05, .075, .1, .15, .2])
NB = int(os.environ.get('NBOOT', '1000'))


def bootstrap_weights(fold, nb, seed=11):
    rng = np.random.default_rng(seed)
    W = np.zeros((nb, len(fold)), dtype=np.int16)
    for k in np.unique(fold):
        ix = np.flatnonzero(fold == k)
        W[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1/len(ix)), size=nb)
    return W


def capacity_bootstrap(y, score, fold, ids, W, qs, groups=None):
    """Reselect exact capacity from each empirical bootstrap population.

    Cumulative multiplicities implement selection of resampled duplicates without
    expanding them. Ties are broken by participant identifier.
    """
    total = np.zeros((len(W), len(qs)))
    denominator = np.zeros(len(qs))
    groups = {} if groups is None else groups
    group_found = {name: np.zeros(len(W)) for name in groups}
    j10 = list(qs).index(.1) if .1 in qs else None
    row = np.arange(len(W))
    for k in np.unique(fold):
        ix = np.flatnonzero(fold == k)
        order = ix[np.lexsort((ids[ix], -score[ix]))]
        w = W[:, order]
        cum = np.cumsum(w, axis=1, dtype=np.int32)
        caps = np.maximum(1, np.rint(qs * len(ix)).astype(int))
        locs = np.stack([np.searchsorted(cum[b], caps, side='left') for b in row])
        before = np.maximum(locs - 1, 0)
        prev_n = np.take_along_axis(cum, before, axis=1)
        prev_n = np.where(locs == 0, 0, prev_n)
        extra = caps[None, :] - prev_n
        positives = np.cumsum(w * y[order], axis=1, dtype=np.int32)
        prev_y = np.take_along_axis(positives, before, axis=1)
        prev_y = np.where(locs == 0, 0, prev_y)
        total += prev_y + extra * y[order][locs]
        denominator += caps
        del positives
        if j10 is not None:
            for name, mask in groups.items():
                yg = (y.astype(bool) & mask)[order]
                prefix = np.cumsum(w * yg, axis=1, dtype=np.int32)
                n0 = prefix[row, before[:, j10]]
                n0 = np.where(locs[:, j10] == 0, 0, n0)
                group_found[name] += n0 + extra[:, j10] * yg[locs[:, j10]]
    return 1000 * total / denominator, group_found


def bootstrap_auc(y, score, W, outcome_weights=None):
    """Tie-correct weighted AUC, with fixed scores and bootstrap multiplicities."""
    ow = np.ones(len(y)) if outcome_weights is None else outcome_weights
    keep = ow > 0
    ix = np.flatnonzero(keep)
    order = ix[np.argsort(score[ix], kind='stable')]
    _, first = np.unique(score[order], return_index=True)
    ans = []
    for start in range(0, len(W), 50):
        w = W[start:start+50, order] * ow[order]
        pos = np.add.reduceat(w * y[order], first, axis=1)
        neg = np.add.reduceat(w * (1-y[order]), first, axis=1)
        nprev = np.cumsum(neg, axis=1) - neg
        num = np.sum(pos * (nprev + .5 * neg), axis=1)
        den = pos.sum(axis=1) * neg.sum(axis=1)
        ans.extend(num / den)
    return np.asarray(ans)


def ci(x):
    return np.nanpercentile(x, [2.5, 97.5])


def evaluate(condition):
    start = time.time()
    files = [PRED / f'{condition}_fold{k}.parquet' for k in range(5)]
    assert all(p.exists() for p in files), condition
    f = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    demo = load_frame(model_only=True)[['eid', 'sex', 'age', 'tq', 'eth_group', 'provider_id']]
    f = f.merge(demo, on='eid', validate='one_to_one')
    assert f.eid.is_unique
    y, fold, ids = f.y.to_numpy(int), f.fold.to_numpy(int), f.eid.to_numpy()
    arms = [c for c in f if c.endswith(('_hgb', '_logit'))]
    W = bootstrap_weights(fold, NB)
    groups = {'Women': f.sex.to_numpy() == 0, 'Men': f.sex.to_numpy() == 1,
              'White': f.eth_group.to_numpy() == 'white',
              'Non-white': f.eth_group.isin(['asian','black','mixed','chinese','other']).to_numpy(),
              'Least deprived': f.tq.to_numpy() == 0, 'Most deprived': f.tq.to_numpy() == 4,
              'Age <55': f.age.to_numpy() < 55, 'Age 65+': f.age.to_numpy() >= 65}
    groups = {g: m for g, m in groups.items() if (m & (y == 1)).sum() >= 30}
    rows, curves, contrasts, group_rows = [], [], [], []
    boots, group_boots = {}, {}
    for arm in arms:
        score = f[arm].to_numpy()
        vals, recalls = [], []
        masks = {}
        for q in Q:
            m = top_mask(score, fold, ids, q).astype(bool)
            masks[q] = m
            vals.append(1000 * y[m].mean())
            recalls.append(y[m].sum()/y.sum())
            curves.append(dict(condition=condition, arm=arm, q=q,
                               yield_per_1000=vals[-1], recall=recalls[-1]))
        is_core = arm in [a+'_'+fam for a in ['recorded','reference','direct','incident5y']
                         for fam in ['logit','hgb']]
        qs = Q if is_core else np.array([.1])
        gs = groups if arm in ['recorded_hgb','reference_hgb','direct_hgb','incident5y_hgb'] else {}
        bs, gb = capacity_bootstrap(y, score, fold, ids, W, qs, gs)
        j = list(qs).index(.1)
        boots[arm] = {'q10': bs[:, j]}
        if is_core:
            boots[arm]['integrated'] = np.trapz(bs, Q, axis=1)/(Q[-1]-Q[0])
        m = masks[.1]
        lo, hi = ci(bs[:, j])
        r = dict(condition=condition, arm=arm, n=len(f), positives=int(y.sum()),
                 prevalence_per_1000=1000*y.mean(), yield_q10=1000*y[m].mean(),
                 yield_q10_lo=lo, yield_q10_hi=hi, recall_q10=y[m].sum()/y.sum(),
                 auroc_assay=roc_auc_score(y, score), ap_assay=average_precision_score(y,score),
                 integrated_yield=np.trapz(vals,Q)/(Q[-1]-Q[0]))
        for h in [1,2,5]:
            wc = f'weight_inc{h}y'
            if wc not in f:
                continue
            ow = f[wc].to_numpy()
            yh = f[f'inc{h}y'].to_numpy(int)
            known = ow > 0
            r[f'auroc_incident{h}y'] = roc_auc_score(yh[known],score[known],sample_weight=ow[known])
        rows.append(r)
        for group, nums in gb.items():
            mask = groups[group]
            den = W @ (mask & (y == 1)).astype(np.int64)
            recall_boot = nums / den
            gl, gh = ci(recall_boot)
            group_rows.append(dict(condition=condition, arm=arm, group=group,
                                   positives=int((mask & (y == 1)).sum()),
                                   recall=float(y[m & mask].sum()/y[mask].sum()),
                                   lo=gl, hi=gh, n_group=int(mask.sum()),
                                   selected=int((m & mask).sum()), share_selected=float((m & mask).sum()/mask.sum()),
                                   precision=float(y[m & mask].sum()/max(1, (m & mask).sum()))))
            group_boots[(arm,group)] = recall_boot
    for family in ['logit','hgb']:
        pairs = [('recorded','direct'),('recorded','reference'),('reference','direct'),
                 ('incident5y','direct'),
                 ('recorded_lowdim','direct_lowdim'),('pu_hidden','pu_base'),
                 ('pu_population','pu_base'),('pu_hidden','pu_population'),
                 ('pu_hidden','direct_lowdim'),('pu_budget10','direct_budget10')]
        for a,b in pairs:
            aa,bb = a+'_'+family,b+'_'+family
            if aa not in boots or bb not in boots:
                continue
            ra = next(r for r in rows if r['arm']==aa)
            rb = next(r for r in rows if r['arm']==bb)
            diff = boots[aa]['q10']-boots[bb]['q10']
            lo,hi = ci(diff)
            row = dict(condition=condition,arm=aa,comparator=bb,
                       difference=ra['yield_q10']-rb['yield_q10'],lo=lo,hi=hi,
                       relative_loss=1-ra['yield_q10']/rb['yield_q10'])
            if 'integrated' in boots[aa] and 'integrated' in boots[bb]:
                bi=boots[aa]['integrated']-boots[bb]['integrated']
                il,ih=ci(bi)
                row.update(integrated_difference=ra['integrated_yield']-rb['integrated_yield'],
                           integrated_lo=il,integrated_hi=ih)
            contrasts.append(row)
    group_contrasts=[]
    for group in groups:
        if ('direct_hgb',group) in group_boots and ('recorded_hgb',group) in group_boots:
            a=next(r for r in group_rows if r['arm']=='direct_hgb' and r['group']==group)
            b=next(r for r in group_rows if r['arm']=='recorded_hgb' and r['group']==group)
            lo,hi=ci(group_boots[('direct_hgb',group)]-group_boots[('recorded_hgb',group)])
            group_contrasts.append(dict(condition=condition,group=group,difference=a['recall']-b['recall'],lo=lo,hi=hi))
    auc_rows=[]
    for target, yt, ow in [('assay',y,None),('recognition_5y',f.inc5y.to_numpy(int),f.weight_inc5y.to_numpy())]:
        bb={}
        for arm in ['direct_hgb','incident5y_hgb']:
            score=f[arm].to_numpy()
            bb[arm]=bootstrap_auc(yt,score,W,ow)
            m=np.ones(len(y),bool) if ow is None else ow>0
            val=roc_auc_score(yt[m],score[m],sample_weight=None if ow is None else ow[m])
            lo,hi=ci(bb[arm])
            auc_rows.append(dict(condition=condition,target=target,arm=arm,value=val,lo=lo,hi=hi))
        lo,hi=ci(bb['direct_hgb']-bb['incident5y_hgb'])
        point=auc_rows[-2]['value']-auc_rows[-1]['value']
        auc_rows.append(dict(condition=condition,target=target,arm='direct_minus_incident',value=point,lo=lo,hi=hi))
    dest=OUT/'parts'
    dest.mkdir(exist_ok=True)
    for name,data in [('model_summary',rows),('capacity_curves',curves),('paired_contrasts',contrasts),
                      ('group_recall',group_rows),('group_contrasts',group_contrasts),
                      ('dual_target_auc',auc_rows)]:
        pd.DataFrame(data).to_csv(dest/f'{condition}_{name}.tsv',sep='\t',index=False)
    print(condition,'evaluation finished',round(time.time()-start),flush=True)


def propensity_audit():
    demo=load_frame(model_only=True)[['eid','sex','age','tq','eth_group','provider_id',
                                    'gp_test1y_hba1c','gp_test1y_creat','gp_test1y_hb']]
    metrics,group_rows,cal_rows=[],[],[]
    for condition in PRIMARY:
        f=pd.concat([pd.read_parquet(PRED/f'{condition}_propensity_fold{k}.parquet') for k in range(5)])
        f=f.merge(demo,on='eid',validate='one_to_one')
        for target,flag,col,const in [('union','Y','c_union','constant_union'),('assay','A','c_assay','constant_assay')]:
            h=f[f[flag]==1].copy()
            y=h.L.to_numpy(int);p=h[col].to_numpy();p0=h[const].to_numpy()
            W=bootstrap_weights(h.fold.to_numpy(),NB,seed=91)
            ll=-(xlogy(y,p)+xlogy(1-y,1-p))
            ll0=-(xlogy(y,p0)+xlogy(1-y,1-p0))
            delta=(W@(ll0-ll))/len(h);lo,hi=ci(delta)
            ab=bootstrap_auc(y,p,W);al,ah=ci(ab)
            metrics.append(dict(condition=condition,reference=target,n=len(h),
                                recorded=int(y.sum()),recorded_share=y.mean(),
                                auroc=roc_auc_score(y,p),auroc_lo=al,auroc_hi=ah,
                                logloss=log_loss(y,p),constant_logloss=log_loss(y,p0),
                                logloss_improvement=(ll0-ll).mean(),improvement_lo=lo,improvement_hi=hi,
                                brier=brier_score_loss(y,p),constant_brier=brier_score_loss(y,p0)))
            bins=pd.qcut(h[col],10,duplicates='drop')
            for label,ix in h.groupby(bins,observed=True).groups.items():
                s=h.loc[ix];cal_rows.append(dict(condition=condition,reference=target,n=len(s),
                    predicted=s[col].mean(),observed=s.L.mean()))
            testcol={'diabetes':'gp_test1y_hba1c','ckd_confirmed':'gp_test1y_creat','anaemia':'gp_test1y_hb'}[condition]
            levels={'sex':{'Women':h.sex==0,'Men':h.sex==1},
                    'deprivation':{f'Q{i+1}':h.tq==i for i in range(5)},
                    'ethnicity':{'White':h.eth_group=='white','Non-white':h.eth_group.isin(['asian','black','mixed','chinese','other'])},
                    'provider':{f'P{i}':h.provider_id==i for i in [1,2,3,4]},
                    'prior_test':{'Tested':h[testcol].fillna(False),'Not tested':~h[testcol].fillna(False)}}
            for group,labels in levels.items():
                for label,m in labels.items():
                    if m.sum()<30:continue
                    n=int(m.sum());k=int(h.loc[m,'L'].sum());l,u=proportion_confint(k,n,method='wilson')
                    group_rows.append(dict(condition=condition,reference=target,group=group,
                                           level=label,n=n,recorded=k,observed=k/n,lo=l,hi=u))
    pd.DataFrame(metrics).to_csv(OUT/'propensity_validation.tsv',sep='\t',index=False)
    pd.DataFrame(group_rows).to_csv(OUT/'propensity_observed_strata.tsv',sep='\t',index=False)
    pd.DataFrame(cal_rows).to_csv(OUT/'propensity_calibration.tsv',sep='\t',index=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--conditions',default=','.join(PRIMARY+SENSITIVITY))
    ap.add_argument('--jobs',type=int,default=3)
    ap.add_argument('--reuse-completed',action='store_true',
                    help='Reuse complete per-condition evaluation tables already in the work directory')
    args=ap.parse_args()
    setup()
    conditions=args.conditions.split(',')
    names=['model_summary','capacity_curves','paired_contrasts','group_recall',
           'group_contrasts','dual_target_auc']
    todo=[c for c in conditions if not (args.reuse_completed and
          all((OUT/'parts'/f'{c}_{name}.tsv').exists() for name in names))]
    Parallel(n_jobs=args.jobs,backend='loky')(delayed(evaluate)(c) for c in todo)
    for name in names:
        pd.concat([pd.read_csv(OUT/'parts'/f'{c}_{name}.tsv',sep='\t') for c in conditions],
                  ignore_index=True).to_csv(OUT/f'{name}.tsv',sep='\t',index=False)
    if all((PRED/f'{c}_propensity_fold4.parquet').exists() for c in PRIMARY) and not (
            args.reuse_completed and (OUT/'propensity_validation.tsv').exists()):
        propensity_audit()
    (OUT/'evaluation_manifest.json').write_text(json.dumps({
        'n_boot':NB,'capacities':Q.tolist(),'selection':'reselected exact top fraction within each bootstrap fold',
        'models':'fixed outer-fold fits','censoring_weights':'fixed outer-training estimates',
        'conditions':conditions},indent=2)+'\n')
