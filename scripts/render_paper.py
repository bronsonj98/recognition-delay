"""Render the manuscript figures and tables from the summaries in the work directory.

Reads <work>/summaries/*.tsv and <work>/summaries/sensitivity/*.tsv and writes <work>/rendered/{figures,tables}.
The name of the R-or-D training label is set once, in COMBINED in figstyle.py.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import WORK, OUT, SENS_OUT, PRIMARY, setup
from figstyle import (TEXTWIDTH, COLWIDTH, COMBINED, COND, LABEL, EDGE, figure, axes, letter,  # noqa: E402
                      check_layout)
setup()

STAGE = WORK / 'rendered'
FIG = STAGE / 'figures'; TAB = STAGE / 'tables'
FIG.mkdir(parents=True, mode=0o700, exist_ok=True); TAB.mkdir(parents=True, mode=0o700, exist_ok=True)
NAMES = {'diabetes': 'Diabetes range', 'ckd_confirmed': 'CKD range', 'anaemia': 'Anemia range',
         'ckd': 'Creatinine-only CKD range', 'diabetes_calibrated': 'Calibrated HbA1c',
         'diabetes_no_rx': 'Diabetes, no medication', 'anaemia_no_rx': 'Anemia, no medication'}
ARM = {'recorded': 'Recorded', 'reference': COMBINED, 'direct': 'Measured', 'incident5y': 'Incident 5 y',
       'incident1y': 'Incident 1 y', 'incident2y': 'Incident 2 y', 'recorded_lowdim': 'Recorded, 115 columns',
       'direct_lowdim': 'Measured, 115 columns', 'pu_base': 'Calibrated recorded',
       'pu_population': 'PU population', 'pu_hidden': 'PU unrecorded', 'pu_hidden_floor005': 'PU floor 0.005',
       'pu_budget10': 'PU, 10% assays', 'direct_budget10': 'Measured, 10% assays'}
CORE = ['recorded', 'reference', 'incident5y', 'direct']
NONWHITE_NOTE = ''


def read(name, d=OUT):
    return pd.read_csv(d / f'{name}.tsv', sep='\t')


def pct(x):
    return f'{100 * x:.0f}\\%'


def intv(v, l, h, scale=1, digits=1):
    return f'{v * scale:.{digits}f} [{l * scale:.{digits}f}, {h * scale:.{digits}f}]'


def arm_name(s):
    root, fam = s.rsplit('_', 1)
    return ARM.get(root, root.replace('_', ' ')) + ' (' + ('boosting' if fam == 'hgb' else 'logistic') + ')'


def table(name, df, caption=None, label=None, size='small', sep=3.5, colfmt=None):
    text = df.to_latex(index=False, escape=False, longtable=caption is not None,
                       caption=caption, label=label, column_format=colfmt or 'l' + 'r' * (len(df.columns) - 1))
    if caption is not None:
        text = '{\\' + size + f'\\setlength{{\\tabcolsep}}{{{sep}pt}}\n' + text + '}\n'
    (TAB / f'{name}.tex').write_text(text)


s = read('recognition_summary'); c = read('recognition_curves'); decomp = read('tested_but_uncoded')
m = read('model_summary'); curves = read('capacity_curves'); du = read('dual_target_auc')
pr = read('propensity_validation'); cal = read('propensity_calibration')
primary = s[s.scope.eq('linked') & s.condition.isin(PRIMARY)].set_index('condition').loc[PRIMARY]

# ---------------------------------------------------------------- Table 2 (main): burden
rows = []
for cond, r in primary.iterrows():
    rows.append([NAMES[cond], f'{int(r.n_positive):,}', f'{int(r.n_unrecorded):,} ({100 * r.unrecorded_share:.0f}\\%)',
                 pct(r.ci_1y), intv(r.ci_5y, r.lo_5y, r.hi_5y, 100, 0) + r'\%'])
table('burden', pd.DataFrame(rows, columns=['Condition', 'Assay-positive', 'Unrecorded', '1 y', '5 y [95\\% CI]']), colfmt='@{}lrrrr@{}')

# ---------------------------------------------------------------- Figure 2: recognition, follow-up, recording propensity
sf = read('recognition_strata_full')
W, H = TEXTWIDTH, 3.47
fig = figure(W, H)
XL1, XL2, XR = .42, 2.98, 6.4                  # left edges of the axes in the two columns; right edge
Y1, H1, Y2, H2 = 1.98, 1.15, .34, 1.0          # bottom and height of the axes in rows 1 and 2
axA = axes(fig, XL1, Y1, 1.88, H1)
axB = axes(fig, XL2, Y1, XR - XL2, H1)
axC = axes(fig, XL1, Y2, 1.3, H2)
GD = .18; WD = (XR - XL2 - 2 * GD) / 3
axD = [axes(fig, XL2 + j * (WD + GD), Y2, WD, H2) for j in range(3)]
for cond in PRIMARY:
    d = c[c.condition.eq(cond) & c.scope.eq('linked') & c.years.le(6.0001)]
    axA.fill_between(d.years, 100 * d.lo, 100 * d.hi, color=COND[cond], alpha=.2, lw=0)
    axA.plot(d.years, 100 * d.ci, color=COND[cond], lw=1.3)
axA.axvline(5, color='.5', lw=.7, ls='--')
axA.set(xlim=(0, 6), ylim=(0, 100), xticks=[0, 2, 4, 6], yticks=[0, 25, 50, 75, 100],
        xlabel='Years since enrollment', ylabel='Recognized (%)')
axA.grid(axis='y')
order = [('Deprivation', ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']), ('Sex', ['Women', 'Men']), ('Ethnicity', ['White', 'Non-white']),
         ('Age', ['<55', '55-64', '65+']), ('Tested', ['No', 'Yes'])]
x = 0; ticks = []; labels = []; seps = []; heads = []
for group, levels in order:
    start = x
    for lv in levels:
        for k, cond in enumerate(PRIMARY):
            r = sf[sf.condition.eq(cond) & sf.group.eq(group) & sf.level.eq(lv)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            axB.errorbar(x + (k - 1) * .26, 100 * r.ci5, yerr=[[100 * (r.ci5 - r.lo)], [100 * (r.hi - r.ci5)]], fmt='o', ms=3.2,
                         color=COND[cond], elinewidth=.8, capsize=1.5, capthick=.8, **EDGE)
        ticks.append(x); labels.append(lv); x += 1
    heads.append(((start + x - 1) / 2, group)); seps.append(x - .25); x += .5
for s_ in seps[:-1]:
    axB.axvline(s_, color='.85', lw=.7)
for xc, gname in heads:
    axB.text(xc, 1.03, gname, ha='center', va='bottom', fontsize=7, transform=axB.get_xaxis_transform())
axB.set(xticks=ticks, ylim=(0, 100), xlim=(-.6, x - 1.5 + .6), yticks=[0, 25, 50, 75, 100], ylabel='Recognized by 5 y (%)')
axB.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor')
axB.grid(axis='y')
fig.legend(handles=[Line2D([0], [0], color=COND[cd], marker='o', ms=3.2, lw=1.3, label=NAMES[cd], **EDGE) for cd in PRIMARY],
           loc='upper center', bbox_to_anchor=((XL2 + XR) / 2 / W, 1), ncol=3, handlelength=1.8, columnspacing=1.6,
           borderaxespad=0, borderpad=.1)
dd = decomp.set_index('condition')
segs = [('abnormal_uncoded_fraction_of_never', 'Abnormal value', '.25'),
        ('normal_retests_fraction_of_never', 'Normal retests', '.6'),
        ('never_retested_fraction_of_never', 'No retest', '.88')]
for i, cond in enumerate(PRIMARY):
    bottom = 0
    for col, lab, colr in segs:
        v = 100 * dd.loc[cond, col]
        axC.bar(i, v, bottom=bottom, width=.6, color=colr, edgecolor='k', lw=.5)
        if cond == 'anaemia':
            axC.text(2.4, bottom + v / 2, lab, ha='left', va='center', fontsize=7, clip_on=False)
        bottom += v
axC.set(xlim=(-.5, 2.5), ylim=(0, 100), yticks=[0, 25, 50, 75, 100], xticks=range(3), ylabel='Never recognized (%)')
axC.set_xticklabels(['Diabetes', 'CKD', 'Anemia'])
for j, cond in enumerate(PRIMARY):
    ax = axD[j]
    d = cal[cal.condition.eq(cond) & cal.reference.eq('assay')]
    r = pr[pr.condition.eq(cond) & pr.reference.eq('assay')].iloc[0]
    ax.plot([0, 1], [0, 100], color='.7', lw=.7)
    ax.axhline(100 * r.recorded_share, color='.35', ls='--', lw=.8)
    ax.plot(d.predicted, 100 * d.observed, '-', color=COND[cond], lw=1)
    ax.plot(d.predicted, 100 * d.observed, 'o', color=COND[cond], ms=3, ls='', **EDGE)
    ax.set(xlim=(0, 1), ylim=(0, 100), xticks=[0, .5, 1], yticks=[0, 50, 100])
    ax.set_xticklabels(['0', '0.5', '1'])
    if j:
        ax.set_yticklabels([])
    ax.text(.97, 4, f'AUROC {r.auroc:.2f}', ha='right', va='bottom', fontsize=7)
    ax.set_title(NAMES[cond], pad=3)
axD[0].set_ylabel('With prior record (%)')
axD[1].set_xlabel('Predicted recording propensity')
TOP1, TOP2 = H - .02, Y2 + H2 + .25
for s_, xx, yy in [('A', 0, TOP1), ('B', XL2 - .45, TOP1), ('C', 0, TOP2), ('D', XL2 - .45, TOP2)]:
    letter(fig, xx, yy, s_)
check_layout(fig, 'Figure 2')
fig.savefig(FIG / 'recognition.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Figure 3: case finding (both model families)
W, H = TEXTWIDTH, 2.76
fig = figure(W, H)
XL, GX = .6, .28; WX = (6.4 - XL - 2 * GX) / 3
YB, HB, YA, HA = .5, .85, 1.88, .72
axTop = [axes(fig, XL + j * (WX + GX), YA, WX, HA) for j in range(3)]
axBot = [axes(fig, XL + j * (WX + GX), YB, WX, HB) for j in range(3)]
ylab = {a: (ARM[a] if a != 'incident5y' else 'Incident') for a in CORE}
for j, cond in enumerate(PRIMARY):
    ax = axTop[j]
    xmax = 0
    for i, a in enumerate(CORE):
        y = len(CORE) - 1 - i
        for fam, off, face in [('hgb', .16, LABEL[a]), ('logit', -.16, 'white')]:
            r = m[m.condition.eq(cond) & m.arm.eq(f'{a}_{fam}')].iloc[0]
            ax.errorbar(r.yield_q10, y + off, xerr=[[r.yield_q10 - r.yield_q10_lo], [r.yield_q10_hi - r.yield_q10]], fmt='o', ms=3.4,
                        color=LABEL[a], mfc=face, elinewidth=.8, capsize=1.5, capthick=.8, **EDGE)
            xmax = max(xmax, r.yield_q10_hi)
    ax.set(ylim=(-.6, len(CORE) - .4), yticks=range(len(CORE)), xlim=(0, xmax * 1.08))
    ax.set_yticklabels([ylab[a] for a in CORE[::-1]] if j == 0 else [])
    ax.grid(axis='x')
    ax.set_title(NAMES[cond], pad=4)
    ax = axBot[j]
    d = {a: curves[curves.condition.eq(cond) & curves.arm.eq(a + '_hgb')].sort_values('q') for a in CORE}
    ax.fill_between(100 * d['recorded'].q.to_numpy(), d['recorded'].yield_per_1000.to_numpy(), d['direct'].yield_per_1000.to_numpy(),
                    color=LABEL['direct'], alpha=.12, lw=0)
    for a in CORE:
        ax.plot(100 * d[a].q, d[a].yield_per_1000, color=LABEL[a], lw=1.3)
    prev = m[m.condition.eq(cond) & m.arm.eq('recorded_hgb')].prevalence_per_1000.iloc[0]
    ax.axhline(prev, color='.4', ls='--', lw=.8)
    ax.set(xscale='log', xlim=(1, 20), ylim=(0, None))
    ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(['1', '2', '5', '10', '20']); ax.minorticks_off()
    ax.grid(axis='y')
axTop[1].set_xlabel('Found per 1,000 tested, top 10%')
axBot[1].set_xlabel('People tested (%)')
axBot[0].set_ylabel('Found per 1,000 tested')
handles = [Line2D([0], [0], color=LABEL[a], marker='o', ms=3.4, lw=1.3, label=ylab[a], **EDGE) for a in CORE]
handles += [Line2D([0], [0], color='k', marker='o', ms=3.4, ls='', label='Boosting'),
            Line2D([0], [0], color='k', marker='o', mfc='white', ms=3.4, ls='', label='Logistic')]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 0), ncol=6, handlelength=1.8, columnspacing=1.6, borderaxespad=0,
           borderpad=.1)
letter(fig, 0, H - .02, 'A'); letter(fig, 0, YB + HB + .17, 'B')
check_layout(fig, 'Figure 3')
fig.savefig(FIG / 'casefinding.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: who is found (boosting)
gr = read('group_recall'); gc = read('group_contrasts')
groups = ['Least deprived', 'Most deprived', 'Women', 'Men', 'White', 'Non-white', 'Age <55', 'Age 65+']
glab = ['Q1', 'Q5', 'Women', 'Men', 'White', 'Non-white', 'Age <55', 'Age 65+']
W, H = TEXTWIDTH, 1.98
fig = figure(W, H)
XL, GX = .58, .2; WX = (6.4 - XL - 2 * GX) / 3
YB, HB = .48, 1.3
axs = [axes(fig, XL + j * (WX + GX), YB, WX, HB) for j in range(3)]
ny = len(groups)
for j, (ax, cond) in enumerate(zip(axs, PRIMARY)):
    d = gr[gr.condition.eq(cond)]
    for arm, off in [('recorded', .17), ('direct', -.17)]:
        v = d[d.arm.eq(arm + '_hgb')].set_index('group').reindex(groups)
        for gi, g in enumerate(groups):
            y = ny - 1 - gi + off
            if pd.isna(v.loc[g, 'recall']):
                if arm == 'recorded':
                    ax.text(2, ny - 1 - gi, 'n < 30', ha='left', va='center', fontsize=6.5, color='.4')
                continue
            ax.errorbar(100 * v.loc[g, 'recall'], y, xerr=[[100 * (v.loc[g, 'recall'] - v.loc[g, 'lo'])], [100 * (v.loc[g, 'hi'] - v.loc[g, 'recall'])]],
                        fmt='o', ms=3.2, color=LABEL[arm], elinewidth=.8, capsize=1.5, capthick=.8, **EDGE)
    cc = gc[gc.condition.eq(cond)].set_index('group')
    hi = d[d.arm.isin(['recorded_hgb', 'direct_hgb'])].groupby('group').hi.max()
    for gi, g in enumerate(groups):
        if g in cc.index and (cc.loc[g, 'lo'] > 0 or cc.loc[g, 'hi'] < 0):
            ax.text(100 * hi[g] + 3, ny - 1 - gi - .15, '*', ha='left', va='center', fontsize=8)
    for yy in (1.5, 3.5, 5.5):
        ax.axhline(yy, color='.85', lw=.7)
    ax.set(xlim=(0, 105), ylim=(-.6, ny - .4), xticks=[0, 25, 50, 75, 100], yticks=range(ny))
    ax.set_yticklabels(glab[::-1] if j == 0 else [])
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x')
    ax.set_title(NAMES[cond], pad=4)
axs[1].set_xlabel('Recall (%)')
fig.legend(handles=[Line2D([0], [0], color=LABEL[a], marker='o', ms=3.2, ls='', label=f'{ARM[a]} label', **EDGE) for a in ('recorded', 'direct')],
           loc='lower center', bbox_to_anchor=(.5, 0), ncol=2, columnspacing=1.6, borderaxespad=0, borderpad=.1, handletextpad=.3)
check_layout(fig, 'Figure 4')
fig.savefig(FIG / 'groups.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Figure 5: two scoring targets (paired AUROC differences)
W, H = COLWIDTH, 1.42
fig = figure(W, H)
ax = axes(fig, .9, .32, W - .9 - .12, .88)
tl = {'assay': 'Enrollment value', 'recognition_5y': 'Recognition within 5 y'}
tc = {'assay': LABEL['direct'], 'recognition_5y': LABEL['incident5y']}
for i, cond in enumerate(PRIMARY):
    for target, off in [('assay', .14), ('recognition_5y', -.14)]:
        r = du[du.condition.eq(cond) & du.target.eq(target) & du.arm.eq('direct_minus_incident')].iloc[0]
        ax.errorbar(r.value, 2 - i + off, xerr=[[r.value - r.lo], [r.hi - r.value]], fmt='o', ms=3.4, color=tc[target], elinewidth=.8,
                    capsize=1.5, capthick=.8, label=tl[target] if i == 0 else None, **EDGE)
ax.axvline(0, color='.4', lw=.8)
ax.set(yticks=range(3), ylim=(-.6, 2.6), xlim=(-.09, .11), xticks=[-.05, 0, .05, .1], xlabel='AUROC, measured minus incident label')
ax.set_xticklabels(['\u22120.05', '0', '0.05', '0.10'])
ax.set_yticklabels([NAMES[cd] for cd in PRIMARY[::-1]])
ax.tick_params(axis='y', length=0)
ax.grid(axis='x')
fig.legend(loc='upper center', bbox_to_anchor=(.5, 1), ncol=2, handlelength=1.6, columnspacing=1.2, borderaxespad=0, borderpad=.1,
           handletextpad=.3)
check_layout(fig, 'Figure 5')
fig.savefig(FIG / 'targets.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Appendix tables (recognition)
rows = []
scope_name = {'linked': 'study cohort', 'linked_hospital_end': 'to hospital end', 'whole': 'all participants'}
for r in s[s.scope.isin(['linked', 'linked_hospital_end', 'whole'])].sort_values(['condition', 'scope'], key=lambda c: c.map({k: i for i, k in enumerate(list(NAMES) + list(scope_name))})).itertuples():
    hosp = r.scope == 'linked_hospital_end'
    rows.append([NAMES[r.condition], scope_name[r.scope], '--' if hosp else f'{int(r.n_measured):,}', '--' if hosp else f'{int(r.n_positive):,}',
                 '--' if hosp else f'{int(r.n_unrecorded):,}', f'{int(r.n_followup):,}', intv(r.ci_5y, r.lo_5y, r.hi_5y, 100, 1)])
table('flow', pd.DataFrame(rows, columns=['Condition', 'Cohort', 'Assayed', 'Assay-positive', 'Unrecorded', 'Followed', '5-y recognition (\\%)']),
      'Assays, disease-range values, and recognition in the study cohort (participants with primary-care records covering the year before enrollment) and among all UK Biobank participants. The row "to hospital end" extends follow-up of the same unrecorded study-cohort participants to the end of hospital records. Recognition among all participants also runs to the end of hospital records and is lower because many lack primary-care records. Creatinine-only CKD range requires only creatinine eGFR below 60; the other added rows are the definition variants in Appendix~\\ref{app:defs}.', 'tab:flow', size='footnotesize', sep=2.5)
rows = []; r6 = read('risk_6y').set_index('condition')
for cond, r in primary.iterrows():
    rows.append([NAMES[cond]] + [f'{int(r[f"at_risk_{h}y"]):,}' for h in [1, 2, 5]] + [f'{int(r6.loc[cond, "at_risk_6y"]):,}', f'{int(r["at_risk_8y"]):,}'])
table('risk', pd.DataFrame(rows, columns=['Condition', 'At risk 1 y', '2 y', '5 y', '6 y', '8 y']),
      'Participants still at risk just before each horizon in the recognition curves of Figure~\\ref{fig:recognition}A, which are drawn to six years.', 'tab:risk')
comp = read('incident_composition'); rows = []
states = {'diabetes': ['>=48', '42-47.9', '<42'], 'ckd_confirmed': ['<60', '60-89', '>=90'], 'anaemia': ['anemic', None, 'not_anemic']}
for cond in PRIMARY:
    for w in ['0-1', '1-2', '2-5', '5-8']:
        d = comp[comp.condition.eq(cond) & comp.window.eq(w)].set_index('state')
        cells = [pct(d.share.get(k, 0.0)) if k is not None else '--' for k in states[cond]]
        name = {'diabetes': 'Diabetes', 'ckd_confirmed': 'CKD', 'anaemia': 'Anemia'}[cond]
        rows.append([name, w.replace('-', '--'), f'{int(d.n.fillna(0).sum()):,}'] + cells)
table('composition', pd.DataFrame(rows, columns=['Recognized condition', 'Years to recognition', 'Recognitions', 'Disease range at enrollment', 'Intermediate', 'Normal']),
      'Enrollment value at first recognition in each window after enrollment, among study-cohort participants without a prior record. The kidney rows classify creatinine eGFR alone ($<60$, 60--89, or $\\geq90$). Intermediate HbA1c is 42--47.9 mmol/mol; anemia has no intermediate state.', 'tab:composition')
sc = read('stratum_contrasts'); rows = []
short = {'Women vs. men': 'Women vs. men', 'Most vs. least deprived quintile': 'Most vs. least deprived', 'Non-white vs. white': 'Non-white vs. white',
         'Age 65+ vs. <55': 'Age 65+ vs. $<$55', 'Tested vs. not tested in prior year': 'Prior-year testing, yes vs. no'}
for r in sc.itertuples():
    rows.append([NAMES[r.condition], short.get(r.contrast, r.contrast), f'{int(r.n_a):,} vs. {int(r.n_b):,}', f'{100 * r.ci5_a:.0f} vs. {100 * r.ci5_b:.0f}',
                 intv(r.diff, r.lo, r.hi, 100, 1), f'{r.p_holm:.3g}'])
table('strata', pd.DataFrame(rows, columns=['Condition', 'Comparison', 'N', '5-y rec. (\\%)', 'Diff. (pp) [95\\% CI]', 'Holm p']),
      'Five-year recognition by group among unrecorded assay-positive participants (Aalen--Johansen, death competing), with paired bootstrap differences. Comparisons with fewer than 30 participants in a group are omitted. Holm adjustment is within each condition. P-values are bootstrap two-sided tail probabilities with a floor of 0.001.', 'tab:strata', size='footnotesize')
sd = read('severity_decomposition'); rows = []
terms = {'sexwomen': 'Women vs. men', 'tqQ5': 'Most vs. least deprived', 'nonwhite': 'Non-white vs. white', 'tested': 'Prior-year testing'}
for cond in PRIMARY:
    for term, name in terms.items():
        d = sd[sd.condition.eq(cond) & sd.model.eq('fine_gray') & sd.term.eq(term)].set_index('adjustment')
        if len(d) < 2:
            continue
        u, a = d.loc['unadjusted'], d.loc['severity']
        rows.append([NAMES[cond], name, intv(u.hr, u.lo, u.hi, 1, 2), f'{u.p_holm:.3g}', intv(a.hr, a.lo, a.hi, 1, 2), f'{a.p_holm:.3g}'])
table('severity', pd.DataFrame(rows, columns=['Condition', 'Comparison', 'HR [95\\% CI]', 'Holm p', 'With severity, HR [95\\% CI]', 'Holm p']),
      'Fine--Gray subdistribution hazard ratios for recognition, fitted on the same participants without and with a three-degree-of-freedom spline of severity (HbA1c; creatinine and cystatin C eGFR; hemoglobin deficit below the sex-specific threshold). Both models adjust for age, sex, deprivation quintile, ethnicity, GP data supplier, enrollment year, and prior-year testing, with standard errors clustered by participant. Holm adjustment is over the four comparisons within each model. Hazard ratios are not collapsible, so differences between columns are read qualitatively.', 'tab:severity')
sg = read('severity_by_group'); rows = []
unit = {'diabetes': 'HbA1c, mmol/mol', 'ckd_confirmed': 'Creatinine eGFR', 'anaemia': 'Hb deficit, g/dL'}
for r in sg[sg.level.isin(['Q1', 'Q5', 'Women', 'Men', 'White', 'Non-white', 'No', 'Yes'])].itertuples():
    rows.append([NAMES[r.condition], unit[r.condition], 'Prior-year testing' if r.group == 'Tested' else r.group, r.level, f'{int(r.n):,}', f'{r.median:.1f} [{r.q25:.1f}, {r.q75:.1f}]', pct(r.mild_share)])
table('severity_groups', pd.DataFrame(rows, columns=['Condition', 'Severity scale', 'Grouping', 'Level', 'N', 'Median [IQR]', 'Mild (\\%)']),
      'Severity among unrecorded assay-positive participants by group. Groups with fewer than 30 participants are omitted. Mild denotes HbA1c below 53 mmol/mol, creatinine eGFR of 45 or more, or a hemoglobin deficit below 1 g/dL.', 'tab:severitygroups', size='footnotesize')
rows = []
for r in decomp.itertuples():
    rows.append([NAMES[r.condition], f'{r.n:,}', pct(r.tested_fraction), pct(r.abnormal_fraction),
                 f'{r.n_at_first_abnormal:,}', intv(r.one_year_recognition, r.one_year_recognition_lo, r.one_year_recognition_hi, 100, 1)])
table('uncoded', pd.DataFrame(rows, columns=['Condition', 'Followed', 'Retested', 'Abnormal value', 'At abnormal value', '1-y recognition (\\%)']),
      'Primary-care testing after enrollment among unrecorded assay-positive participants in practices in England and Wales. Retested and abnormal-value shares use the Followed count as denominator. The last two columns start a new clock at the first abnormal primary-care value, exclude people recognized before it, and count same-day recognition. For kidney function, an abnormal value is creatinine eGFR below 60.', 'tab:uncoded')
t = read('repeat_validation'); t['condition'] = t.condition.map(NAMES)
t['persistent_unrecorded_fraction'] = t.persistent_unrecorded_fraction.map(pct)
t.columns = ['Condition', 'Remeasured', 'Still unrecorded', 'Still in range', 'Persistence']
table('repeat', t, 'Persistence at the 2012--2013 repeat assessment among assay-positive participants without a prior record, using all UK Biobank participants. Persistence is the share still in the disease range among those still unrecorded; for kidney function it refers to creatinine eGFR below 60.', 'tab:repeat')

# ---------------------------------------------------------------- Appendix tables (models)
rows = []
for r in m[m.arm.str.startswith(('recorded_', 'reference_', 'direct_', 'incident'))].itertuples():
    if r.arm.startswith('direct_budget10'):
        continue
    rows.append([NAMES[r.condition], arm_name(r.arm).replace('%', r'\%'), intv(r.yield_q10, r.yield_q10_lo, r.yield_q10_hi), f'{r.auroc_assay:.3f}', f'{r.ap_assay:.3f}', f'{r.auroc_incident5y:.3f}'])
table('models', pd.DataFrame(rows, columns=['Condition', 'Training label', 'Found per 1,000 [95\\% CI]', 'AUROC', 'AP', 'AUROC, 5-y rec.']),
      'Held-out models among unrecorded participants. Found per 1,000 tested when the highest-scoring 10\\% are tested. AUROC and average precision (AP) are scored against the enrollment value. The last column scores five-year recognition with censoring weights, a different target.', 'tab:allmodels', size='footnotesize')
pairs = read('paired_contrasts'); rows = []
for r in pairs[pairs.arm.isin(['recorded_hgb', 'reference_hgb', 'incident5y_hgb', 'recorded_logit', 'reference_logit', 'incident5y_logit', 'recorded_lowdim_hgb', 'recorded_lowdim_logit'])].itertuples():
    rows.append([NAMES[r.condition], arm_name(r.arm), ARM.get(r.comparator.rsplit('_', 1)[0]), intv(r.difference, r.lo, r.hi),
                 intv(r.integrated_difference, r.integrated_lo, r.integrated_hi) if pd.notna(r.integrated_difference) else '--'])
table('contrasts', pd.DataFrame(rows, columns=['Condition', 'Model', 'Comparator', 'Diff. at 10\\% tested', 'Averaged, 1--20\\%']),
      'Paired differences in the number found per 1,000 tested, with the top 10\\% reselected within each bootstrap sample. Averaged differences integrate over tested fractions of 1--20\\%. The comparator uses the same model family.', 'tab:contrasts', size='footnotesize', sep=2.5)
g = read('group_recall'); rows = []
for r in gc.itertuples():
    a = g[g.condition.eq(r.condition) & g.group.eq(r.group)].set_index('arm')
    rows.append([NAMES[r.condition], r.group, f'{int(a.loc["direct_hgb", "positives"]):,}',
                 f'{pct(a.loc["recorded_hgb", "share_selected"])} / {pct(a.loc["direct_hgb", "share_selected"])}',
                 f'{100 * a.loc["recorded_hgb", "precision"]:.1f} / {100 * a.loc["direct_hgb", "precision"]:.1f}',
                 f'{pct(a.loc["recorded_hgb", "recall"])} / {pct(a.loc["direct_hgb", "recall"])}', intv(r.difference, r.lo, r.hi, 100, 1)])
table('groups', pd.DataFrame(rows, columns=['Condition', 'Group', 'Assay-positive', 'Selected', 'Precision (\\%)', 'Recall', 'Recall diff. (pp) [95\\% CI]']),
      'Groups when the same overall 10\\% are tested, recorded-label / measured-label boosting. Selected is the share of the group that is tested; precision is the share of those tested who are assay-positive; recall is the share of the group\'s assay-positive participants who are tested. Groups with fewer than 30 assay-positive participants are omitted. Intervals are not adjusted for multiplicity.', 'tab:groups', size='scriptsize')
rows = []
for cond in PRIMARY:
    for target, tname in [('assay', 'Enrollment value'), ('recognition_5y', 'Recognition within 5 y')]:
        d = du[du.condition.eq(cond) & du.target.eq(target)].set_index('arm')
        rows.append([NAMES[cond], tname, intv(d.loc['direct_hgb', 'value'], d.loc['direct_hgb', 'lo'], d.loc['direct_hgb', 'hi'], 1, 3),
                     intv(d.loc['incident5y_hgb', 'value'], d.loc['incident5y_hgb', 'lo'], d.loc['incident5y_hgb', 'hi'], 1, 3),
                     intv(d.loc['direct_minus_incident', 'value'], d.loc['direct_minus_incident', 'lo'], d.loc['direct_minus_incident', 'hi'], 1, 3)])
table('targets', pd.DataFrame(rows, columns=['Condition', 'Scored against', 'Measured', 'Incident 5 y', 'Difference']),
      'AUROC of the measured-label and five-year incident-label boosting models against two targets in the same held-out unrecorded participants, with paired bootstrap intervals. Recognition is scored with inverse-probability-of-censoring weights, and deaths before recognition count as negatives.', 'tab:targets')
rows = []
for r in m[m.arm.str.startswith(('pu_', 'direct_budget10'))].itertuples():
    rows.append([NAMES[r.condition], arm_name(r.arm).replace('%', r'\%'), intv(r.yield_q10, r.yield_q10_lo, r.yield_q10_hi), f'{r.auroc_assay:.3f}'])
table('pu', pd.DataFrame(rows, columns=['Condition', 'Score', 'Found per 1,000 [95\\% CI]', 'AUROC']),
      'PU correction on the 115-column representation. Calibrated recorded ranks by $g$, PU population by $p=g/c$, and PU unrecorded by $(p-g)/(1-g)$, the probability of a case among participants with $R=0$; all three use the same $g$ model (Appendix~\\ref{app:pu}). The 10\\%-assay comparisons use the same randomly chosen assay subset.', 'tab:pu')
rows = []
for r in pairs[pairs.arm.str.startswith('pu_')].itertuples():
    rows.append([NAMES[r.condition], arm_name(r.arm).replace('%', r'\%'), arm_name(r.comparator).replace('%', r'\%'), intv(r.difference, r.lo, r.hi)])
table('pu_contrasts', pd.DataFrame(rows, columns=['Condition', 'Score', 'Comparator', 'Difference [95\\% CI]']),
      'Paired comparisons of PU scores when the highest-scoring 10\\% are tested (found per 1,000 tested).', 'tab:pucontrasts')
rows = []
for r in pr.itertuples():
    rows.append([NAMES[r.condition], {'union': 'Recorded or assay-positive', 'assay': 'Assay-positive'}[r.reference], f'{r.n:,}',
                 intv(r.auroc, r.auroc_lo, r.auroc_hi, 1, 3), intv(r.logloss_improvement, r.improvement_lo, r.improvement_hi, 1, 3)])
table('propensity', pd.DataFrame(rows, columns=['Condition', 'Participants', 'N', 'AUROC', 'Log-loss improvement']),
      'Held-out prediction of a prior record, among assay-positive participants (the recording propensity) and among participants who are recorded or assay-positive (the PU propensity $c(X)$). The comparator for log loss is a constant estimated in each training fold.', 'tab:propensity')
gq = read('propensity_observed_strata'); rows = []
for r in gq[gq.reference.eq('assay') & gq.group.isin(['sex', 'prior_test', 'deprivation', 'ethnicity'])].itertuples():
    if r.group == 'deprivation' and r.level not in ['Q1', 'Q5']:
        continue
    rows.append([NAMES[r.condition], r.group.replace('prior_test', 'prior-year testing'), r.level, f'{r.n:,}', intv(r.observed, r.lo, r.hi, 100, 1)])
table('recording_groups', pd.DataFrame(rows, columns=['Condition', 'Grouping', 'Level', 'Assay-positive', 'With a prior record (\\%) [95\\% CI]']),
      'Prior recording among assay-positive participants, with Wilson intervals. These associations do not show whether contact with care precedes or follows a diagnosis.', 'tab:recordinggroups')

# ---------------------------------------------------------------- Appendix tables (sensitivity analyses)
st = read('strict_thresholds', SENS_OUT); sc2 = read('eval_strict_contrasts', SENS_OUT); rows = []
for cond, lab in [('diabetes', 'HbA1c >= 53'), ('anaemia', 'hemoglobin < 11'), ('ckd_confirmed', 'both eGFR < 45')]:
    for name, tag in [(f'{cond} primary', 'Primary threshold'), (f'{cond} {lab}', {'diabetes': 'HbA1c $\\geq$53', 'anaemia': 'Hb $<$11', 'ckd_confirmed': 'Both eGFR $<$45'}[cond])]:
        r = st[st.condition.eq(name)].iloc[0]
        if cond != 'ckd_confirmed' or tag != 'Primary threshold':
            pass
        row = [NAMES[cond], tag, f'{int(r.n_positive):,}', f'{int(r.n_unrecorded):,} ({100 * r.unrecorded_share:.0f}\\%)', intv(r.ci_5y, r.lo_5y, r.hi_5y, 100, 0)]
        k = sc2[sc2.condition.eq(cond) & sc2.arm.eq('recorded_hgb') & sc2.comparator.eq('direct_hgb')]
        if tag != 'Primary threshold' and len(k) and cond != 'ckd_confirmed':
            k = k.iloc[0]; row += [f'{k.yield_arm:.1f} vs. {k.yield_comparator:.1f}', f'{100 * k.relative_loss:.0f}\\%']
        elif tag == 'Primary threshold':
            mm = m[m.condition.eq(cond)].set_index('arm')
            row += [f'{mm.loc["recorded_hgb", "yield_q10"]:.1f} vs. {mm.loc["direct_hgb", "yield_q10"]:.1f}',
                    f'{100 * (1 - mm.loc["recorded_hgb", "yield_q10"] / mm.loc["direct_hgb", "yield_q10"]):.0f}\\%']
        else:
            row += ['--', '--']
        rows.append(row)
table('strict', pd.DataFrame(rows, columns=['Condition', 'Threshold', 'Assay-positive', 'Unrecorded', '5-y rec. (\\%)', 'Rec. vs. meas.', 'Fewer']),
      'Stricter thresholds. Case finding uses the same held-out unrecorded participants and the same models as the primary analysis (boosting, top 10\\% tested), scored against the stricter value; the measured label is the primary one. Kidney function at the stricter threshold has too few unrecorded participants for case finding. Rec.\\ vs.\\ meas.: found per 1,000 tested with the recorded and measured labels.', 'tab:strict', size='footnotesize')
cl = read('confirmation_landmarks', SENS_OUT); rows = []
for r in cl.itertuples():
    rows.append([NAMES[r.condition], {'first abnormal value': 'First abnormal value', 'second abnormal value >= 90 days later': 'Second abnormal value, $\\geq$90 days later'}[r.landmark],
                 f'{int(r.with_landmark):,}', f'{int(r.recognized_before_landmark):,}', f'{int(r.at_risk):,}', intv(r.one_year, r.lo, r.hi, 100, 0)])
table('confirm', pd.DataFrame(rows, columns=['Condition', 'Clock starts at', 'With value', 'Rec. before', 'At risk', '1-y rec. (\\%)']),
      'Recognition within one year of an abnormal primary-care value among unrecorded assay-positive participants in practices in England and Wales. Same-day recognition counts.', 'tab:confirm')
sb = read('severity_bands', SENS_OUT); rows = []
unitb = {'diabetes': 'HbA1c (mmol/mol)', 'ckd_confirmed': 'Creatinine eGFR', 'anaemia': 'Hb deficit (g/dL)'}
for r in sb.itertuples():
    if pd.isna(r.ci5):
        continue
    rows.append([NAMES[r.condition], unitb[r.condition], r.label.replace('>=', '$\\geq$').replace('<', '$<$'), f'{int(r.n_positive):,}',
                 f'{100 * r.unrecorded_share:.0f}\\%', intv(r.ci5, r.lo5, r.hi5, 100, 0)])
table('bands', pd.DataFrame(rows, columns=['Condition', 'Scale', 'Band', 'Assay-positive', 'Unrecorded', '5-y rec. (\\%) [95\\% CI]']),
      'Prior recording and five-year recognition by severity. Bands with fewer than 30 unrecorded participants are omitted.', 'tab:bands')
ac = read('anaemia_composition', SENS_OUT); rows = []
for r in ac.itertuples():
    rows.append([r.label, f'{int(r.n_positive):,}', f'{100 * r.unrecorded_share:.0f}\\%', intv(r.ci5, r.lo5, r.hi5, 100, 0) if pd.notna(r.ci5) else 'n $<$ 30'])
table('anemia_groups', pd.DataFrame(rows, columns=['Anemia-range participants', 'Assay-positive', 'Unrecorded', '5-y rec. (\\%) [95\\% CI]']),
      'Anemia-range participants by pregnancy status, mean corpuscular volume (MCV), menopause, and hemoglobinopathy evidence.', 'tab:anemiagroups')
hb = read('anaemia_hbp_contrasts', SENS_OUT); hr = read('anaemia_hbp_recognition', SENS_OUT); rows = []
vn = {'primary': 'Primary', 'nohbp': 'Hemoglobinopathy codes removed', 'exhbp': 'Hemoglobinopathy evidence excluded'}
for v in ['primary', 'nohbp', 'exhbp']:
    r = hr[hr.condition.eq(f'anaemia {v}')].iloc[0]
    e = hb[hb.variant.eq(v) & hb.contrast.eq('Non-white vs. white')].iloc[0]
    dpv = hb[hb.variant.eq(v) & hb.contrast.str.startswith('Most vs. least')].iloc[0]
    rows.append([vn[v], f'{int(r.n_unrecorded):,} ({100 * r.unrecorded_share:.0f}\\%)', intv(r.ci_5y, r.lo_5y, r.hi_5y, 100, 0),
                 intv(e['diff'], e.lo, e.hi, 100, 1), intv(dpv['diff'], dpv.lo, dpv.hi, 100, 1)])
table('hbp', pd.DataFrame(rows, columns=['Anemia definition', 'Unrecorded', '5-y rec. (\\%)', 'Ethnicity (pp)', 'Deprivation (pp)']),
      'Anemia without hemoglobinopathy evidence (ICD-10 D56--D58 and primary-care codes that map only to them). Ethnicity and Deprivation are the differences in five-year recognition between non-white and white participants and between the most and least deprived quintiles, with 95\\% bootstrap intervals.', 'tab:hbp', size='footnotesize')
ry = read('eval_representation_contrasts', SENS_OUT); rows = []
for r in ry.itertuples():
    rows.append([NAMES[r.condition], str(r.representation), 'boosting' if r.arm.endswith('_hgb') else 'logistic',
                 f'{r.yield_arm:.1f} vs. {r.yield_comparator:.1f}', intv(r.difference, r.lo, r.hi), f'{100 * r.relative_loss:.0f}\\%'])
table('representation', pd.DataFrame(rows, columns=['Condition', 'Columns', 'Model', 'Recorded vs. measured', 'Difference [95\\% CI]', 'Fewer']),
      'Feature sets. The 627 columns are the full representation, the 115 columns drop the code history, and the 67 columns keep only demographic, GP data supplier, assessment-center, calendar, and registration information. Found per 1,000 tested, top 10\\%.', 'tab:representation', size='footnotesize')
op = read('eval_operating', SENS_OUT); ca = read('eval_calibration', SENS_OUT); rows = []
for cond in PRIMARY:
    o = op[op.condition.eq(cond)].set_index('arm'); cc = ca[ca.condition.eq(cond)].set_index('arm')
    for arm in ['recorded_hgb', 'direct_hgb']:
        rows.append([NAMES[cond], ARM[arm[:-4]], f'{100 / o.loc[arm, "prevalence_per_100"]:.0f}', f'{o.loc[arm, "tests_per_case_q10"]:.0f}',
                     f'{10 * o.loc[arm, "cases_per_100_tests_q1"]:.0f}', f'{100 * o.loc[arm, "fraction_tested_for_recall_50"]:.0f}\\%',
                     f'{cc.loc[arm, "expected_per_1000_top10"]:.1f} / {cc.loc[arm, "observed_per_1000_top10"]:.1f}'])
table('operating', pd.DataFrame(rows, columns=['Condition', 'Label', 'Random', 'Top 10\\%', 'Top 1\\%', 'Half found', 'Exp./obs.']),
      'Operating points for boosting among unrecorded participants. Random and Top 10\\%: tests per case found with random testing and with the highest-scoring 10\\% tested. Top 1\\%: cases per 1,000 tests among the highest-scoring 1\\%. Half found: share tested to find half of the assay-positive participants. Exp./obs.: expected and observed cases per 1,000 tested in the top 10\\%, with expected cases from isotonic recalibration fitted on the other folds.', 'tab:operating', size='footnotesize')
nb = read('eval_net_benefit', SENS_OUT); rows = []
for cond, thr in [('diabetes', .01), ('ckd_confirmed', .01), ('anaemia', .05)]:
    d = nb[nb.condition.eq(cond) & np.isclose(nb.threshold, thr)].set_index('arm')
    rows.append([NAMES[cond], f'{100 * thr:.0f}\\%', f'{d.loc["recorded_hgb", "net_benefit_per_1000"]:.1f}', f'{d.loc["direct_hgb", "net_benefit_per_1000"]:.1f}'])
table('netbenefit', pd.DataFrame(rows, columns=['Condition', 'Risk threshold', 'Recorded', 'Measured']),
      'Net benefit per 1,000 unrecorded participants when testing everyone whose recalibrated risk exceeds the threshold (boosting).', 'tab:netbenefit')
pv = read('provider_recognition', SENS_OUT); pc = read('eval_provider_contrasts', SENS_OUT); rows = []
pname = {1: 'England (Vision)', 2: 'Scotland', 3: 'England (TPP)', 4: 'Wales'}
for cond in PRIMARY:
    for p in [3, 1, 2, 4]:
        r = pv[pv.condition.eq(cond) & pv.label.eq(f'provider {p}')].iloc[0]
        k = pc[pc.condition.eq(cond) & pc.analysis.eq(f'provider_{p}') & pc.arm.eq('recorded_hgb') & pc.comparator.eq('direct_hgb')].iloc[0]
        rows.append([NAMES[cond], pname[p], f'{int(r.n_positive):,}', f'{100 * r.unrecorded_share:.0f}\\%', intv(r.ci5, r.lo5, r.hi5, 100, 0) if pd.notna(r.ci5) else '--',
                     f'{k.yield_arm:.1f} vs. {k.yield_comparator:.1f}', intv(k.difference, k.lo, k.hi)])
table('provider', pd.DataFrame(rows, columns=['Condition', 'GP data supplier', 'Assay-positive', 'Unrecorded', '5-y rec. (\\%)', 'Recorded vs. measured', 'Difference [95\\% CI]']),
      'Results by GP data supplier. Models are trained on all suppliers, and the top 10\\% are selected within each supplier and fold (boosting, found per 1,000 tested).', 'tab:provider', size='footnotesize')
print('rendered figures and tables from', OUT, 'and', SENS_OUT)
