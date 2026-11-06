"""Figure 1: study design (schematic, no data).

A, timeline for one participant (records more than a year before enrollment are not drawn to scale). B, value of each training label by prior record (R)
and enrollment assay (D), and the role of each group in the held-out evaluation.
"""
import sys
from common import WORK, setup
from figstyle import TEXTWIDTH, COMBINED, figure, axes, letter, check_layout

setup()
OUTF = WORK / 'rendered' / 'figures' / 'design_timeline.pdf'
OUTF.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
ASSAY_Y, REC_T = .53, 3.1                  # height of the assay dot (inches, level with recognition); time of the example recognition (years)
W, H = TEXTWIDTH, 1.4
YOFF = .1                                  # drawing coordinates are inches above y = YOFF
fig = figure(W, H)

# ---------------------------------------------------------------- A: timeline (axes in inches)
WA = 2.75
ax = axes(fig, 0, 0, WA, H)
ax.set_xlim(0, WA); ax.set_ylim(YOFF, H + YOFF); ax.axis('off')
K = dict(color='k', lw=.8, solid_capstyle='butt')
y0 = .36                                   # time axis
t0 = 1.2                                   # enrollment
YR = .235                                  # inches per year; the axis is broken only before the year preceding enrollment
x = lambda t: t0 + t * YR
ax.plot([.08, .52], [y0, y0], **K)
ax.text(.59, y0, '//', ha='center', va='center', fontsize=7)
ax.annotate('', xy=(WA - .02, y0), xytext=(.66, y0), arrowprops=dict(arrowstyle='-|>', lw=.8, color='k', shrinkA=0, shrinkB=0,
                                                                     mutation_scale=7))
for t, lab in [(-1, '−1'), (0, '0'), (1, '1'), (2, ''), (3, ''), (4, ''), (5, '5')]:
    ax.plot([x(t), x(t)], [y0 - .035, y0], **K)
    if lab:
        ax.text(x(t), y0 - .06, lab, ha='center', va='top', fontsize=7)
ax.text(.3, y0 - .06, 'earlier', ha='center', va='top', fontsize=7, color='.3')
ax.text(WA - .02, y0 - .06, 'years', ha='right', va='top', fontsize=7)


def bracket(xa, xb, y, h=.05):
    ax.plot([xa, xa, xb, xb], [y - h, y, y, y - h], **K)


# enrollment
ax.plot([t0, t0], [y0, 1.27], color='k', lw=1.2)
ax.text(t0, 1.38, 'Enrollment (2006–2010)', ha='center', va='center', fontsize=7)
# before enrollment
bracket(.08, t0 - .05, 1.02)
ax.text((.08 + t0 - .05) / 2, 1.07, 'Prior record ($R$)', ha='center', va='bottom', fontsize=7)
bracket(x(-1), t0 - .05, .72)
ax.text(x(-1) - .06, .72, 'Features', ha='right', va='center', fontsize=7)
# at enrollment
ax.plot([t0], [ASSAY_Y], 'o', ms=4.5, color='k')
ax.text(t0 + .08, ASSAY_Y, 'Assay ($D$)', ha='left', va='center', fontsize=7)
# after enrollment
bracket(t0 + .05, WA - .08, 1.02)
ax.text((t0 + .05 + WA - .08) / 2, 1.07, 'Follow-up (to 2016–2017)', ha='center', va='bottom', fontsize=7)
xr = x(REC_T)
ax.plot([xr], [.53], marker='x', ms=4.5, mew=1.1, color='k', ls='')
ax.text(xr + .08, .53, 'Recognition', ha='left', va='center', fontsize=7)

# ---------------------------------------------------------------- B: training labels by group (axes in inches)
XB = WA + .2
WB = W - XB
bx = axes(fig, XB, 0, WB, H)
bx.set_xlim(0, WB); bx.set_ylim(YOFF, H + YOFF); bx.axis('off')
names = ['R', 'D', 'Recorded', COMBINED, 'Measured', 'Incident', 'Evaluation']
rows = [('1', '1', ['1', '1', '–', '–'], '–'),
        ('1', '0', ['1', '1', '–', '–'], '–'),
        ('0', '1', ['0', '1', '1', '0 or 1'], 'case'),
        ('0', '0', ['0', '0', '0', '0 or 1'], 'non-case')]
# column centers from measured text widths, with equal gaps between columns
fig.canvas.draw(); rend = fig.canvas.get_renderer()
def width(s):
    t = bx.text(0, 0, s, fontsize=7); w_ = t.get_window_extent(rend).width / fig.dpi; t.remove(); return w_
colw = [max(width(f'${n}$' if n in 'RD' else n), *(width(r[0] if i == 0 else r[1] if i == 1 else (r[2][i - 2] if i < 6 else r[3]))
                                                   for r in rows)) for i, n in enumerate(names)]
colw[0] = colw[1] = max(colw[0], .12)
gaps = [.1, .22, .13, .13, .13, .17]          # wider gaps separate R, D from the labels and the labels from evaluation
total = sum(colw) + sum(gaps)
xl = (WB - total) / 2
cols = {}
for i, n in enumerate(names):
    cols[n] = xl + colw[i] / 2
    xl += colw[i] + (gaps[i] if i < len(gaps) else 0)
x0r, x1r = cols['R'] - colw[0] / 2 - .03, cols['Evaluation'] + colw[-1] / 2 + .03
assert x0r >= 0 and x1r <= WB, (x0r, x1r, WB)
ytop, yh1, yh2, ymid = 1.48, 1.39, 1.2, 1.1
ys = [.98, .77, .56, .35]
ybot = .245
L = dict(color='k', lw=.8)
bx.plot([x0r, x1r], [ytop, ytop], **L)
xa_, xb_ = cols['Recorded'] - colw[2] / 2, cols['Incident'] + colw[5] / 2
bx.plot([xa_, xb_], [yh1 - .08, yh1 - .08], color='k', lw=.5)
bx.text((xa_ + xb_) / 2, yh1, 'Training label', ha='center', va='center', fontsize=7)
for name, cx in cols.items():
    bx.text(cx, yh2, f'${name}$' if name in 'RD' else name, ha='center', va='center', fontsize=7)
bx.plot([x0r, x1r], [ymid, ymid], color='k', lw=.5)
bx.fill_between([x0r, x1r], ys[2] - .105, ys[2] + .105, color='.9', lw=0, zorder=0)
for y, (r, d, labs, ev) in zip(ys, rows):
    bx.text(cols['R'], y, r, ha='center', va='center', fontsize=7)
    bx.text(cols['D'], y, d, ha='center', va='center', fontsize=7)
    for name, v in zip(['Recorded', COMBINED, 'Measured', 'Incident'], labs):
        bx.text(cols[name], y, v, ha='center', va='center', fontsize=7, color='.45' if v == '–' else 'k')
    bx.text(cols['Evaluation'], y, ev, ha='center', va='center', fontsize=7, color='.45' if ev == '–' else 'k')
bx.plot([x0r, x1r], [ybot, ybot], **L)

letter(fig, 0, H, 'A')
letter(fig, XB - .15, H, 'B')
problems = check_layout(fig, 'Figure 1')
fig.savefig(OUTF)
if problems and '--strict' in sys.argv:
    sys.exit(1)
