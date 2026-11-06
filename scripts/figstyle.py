"""Shared figure style and layout checks for the manuscript figures.

Figures are drawn at their printed size (text width 6.5 in, column width 3.15 in) with axes placed in inches, so
panel edges and panel letters line up exactly. check_layout() reports text that overlaps or nearly touches other
text, axes it does not belong to, or the figure edge.
"""
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TEXTWIDTH, COLWIDTH = 6.5, 3.15
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': .8,
    'xtick.major.width': .8, 'ytick.major.width': .8, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.pad': 2, 'ytick.major.pad': 2, 'axes.labelpad': 3,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'grid.linestyle': ':', 'grid.linewidth': .6, 'grid.color': '.75',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300})

COMBINED = 'Combined'   # name of the R-or-D label in figures and tables; keep in step with \comblabel in main.tex
COND = {'diabetes': 'tab:blue', 'ckd_confirmed': 'tab:orange', 'anaemia': 'tab:red'}
LABEL = {'recorded': 'tab:gray', 'reference': 'tab:purple', 'incident5y': 'tab:brown', 'direct': 'tab:green'}
EDGE = dict(mec='k', mew=.5)


def figure(w, h):
    fig = plt.figure(figsize=(w, h))
    fig._size_in = (w, h)
    return fig


def axes(fig, left, bottom, width, height, **kw):
    """Axes placed in inches from the lower-left corner of the figure."""
    w, h = fig._size_in
    return fig.add_axes([left / w, bottom / h, width / w, height / h], **kw)


def letter(fig, x, y, s):
    """Bold panel letter with its top-left corner at (x, y) inches."""
    w, h = fig._size_in
    return fig.text(x / w, y / h, s, fontsize=9, fontweight='bold', ha='left', va='top')


def _texts(fig):
    """Visible, non-empty text, leaving out tick labels outside the axis limits (matplotlib does not draw them)."""
    hidden = set()
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tick in axis.get_major_ticks():
                if not (lo - 1e-9 <= tick.get_loc() <= hi + 1e-9):
                    hidden.update({id(tick.label1), id(tick.label2)})
    out = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip() and t.get_figure() is fig and id(t) not in hidden:
            out.append(t)
    return out


def _poly(t, r):
    """Corners (display units) of the rectangle a text item occupies, rotated with the text."""
    bb = t.get_window_extent(r)
    theta = np.deg2rad(t.get_rotation())
    if abs(np.sin(2 * theta)) < 1e-6:
        return np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]])
    tmp = t.figure.text(0, 0, t.get_text(), fontproperties=t.get_fontproperties())
    e = tmp.get_window_extent(r); tmp.remove()
    w, h = e.width, e.height
    c, s = np.cos(theta), np.sin(theta)
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
    return corners @ np.array([[c, s], [-s, c]]) + [(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2]


def _separated(p, q, gap):
    """Separating-axis test for two convex polygons; True if some axis separates them by at least gap."""
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]) / np.hypot(*e)
            a, b = p @ n, q @ n
            if a.max() + gap <= b.min() or b.max() + gap <= a.min():
                return True
    return False


def check_layout(fig, name, pad_pt=1.5):
    """Print overlaps between text items, text crossing another axes, and text outside the figure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pad = pad_pt * fig.dpi / 72
    W, H = fig.bbox.width, fig.bbox.height
    texts = _texts(fig)
    polys = [(t, _poly(t, r)) for t in texts]
    problems = []
    for (t1, p1), (t2, p2) in itertools.combinations(polys, 2):
        if not _separated(p1, p2, pad):
            problems.append(f'text overlap: {t1.get_text()!r} / {t2.get_text()!r}')
    for t, p in polys:
        if p[:, 0].min() < 0 or p[:, 1].min() < 0 or p[:, 0].max() > W or p[:, 1].max() > H:
            problems.append(f'outside figure: {t.get_text()!r}')
        for ax in fig.axes:
            if ax is t.axes or not ax.get_visible() or not ax.axison:
                continue
            b = ax.bbox
            if not _separated(p, np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]]), 0):
                problems.append(f'text {t.get_text()!r} crosses another axes')
    print(f'{name}: ' + ('layout check passed' if not problems else f'{len(problems)} problem(s)'))
    for p in problems:
        print('   ', p)
    return problems
