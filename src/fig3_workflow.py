"""fig3_workflow.py — Figure 3 (Section 3.4): two-phase deployment-matched validation workflow."""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

FIG = os.path.join(os.path.dirname(__file__), '..', 'output', 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'Liberation Sans', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
                     'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold'})
BLUE, GREEN, RED, GOLD, TEAL, GREY, BROWN = '#2B5C9E', '#2E7D4F', '#B0302A', '#B8860B', '#1F6F78', '#5F6670', '#A0631A'


def rbox(ax, x, y, w, h, ec, fc='white', lw=2.2, ls='-', r=0.18, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw,
                                ls=ls, zorder=z))


def arr(ax, xy0, xy1, c, lw=2.0, ls='-'):
    ax.annotate('', xy=xy1, xytext=xy0, arrowprops=dict(arrowstyle='-|>', color=c, lw=lw, ls=ls, mutation_scale=17,
                                                         shrinkA=0, shrinkB=0), zorder=3)


def param(ax, x, y, w, h, txt, fs=10.5):
    rbox(ax, x, y, w, h, BROWN, '#FDF2E0', lw=1.6, ls=(0, (3, 2)), r=0.15, z=4)
    ax.text(x + w / 2, y + h / 2, txt, ha='center', va='center', fontsize=fs, color='#7A4A10', zorder=5,
            linespacing=1.15)


def step(ax, x, y, w, h, c, title, body, fs=10.5, tfs=13):
    rbox(ax, x, y, w, h, c)
    ax.text(x + w / 2, y + h - 0.45, title, ha='center', va='center', fontsize=tfs, weight='bold', color=c, zorder=5)
    ax.text(x + w / 2, y + (h - 0.8) / 2, body, ha='center', va='center', fontsize=fs, zorder=5, linespacing=1.3)


def main():
    fig, ax = plt.subplots(figsize=(12.6, 19.8))
    ax.set_xlim(0, 20); ax.set_ylim(0, 31.5); ax.axis('off')
    # top definition box
    rbox(ax, 2.6, 29.6, 14.8, 1.6, '#222222', lw=2.4)
    ax.text(10, 30.75, 'Define the deployment unit first (station area, county, submarket)', ha='center',
            fontsize=13, weight='bold')
    ax.text(10, 30.0, '→ the held-out validation unit must be the same unit', ha='center', fontsize=11.5)
    arr(ax, (10, 29.6), (10, 28.75), '#222222', lw=2.2)
    # PHASE A panel
    rbox(ax, 0.3, 20.6, 19.4, 8.1, 'none', '#EBF1F8', lw=0, r=0.3, z=1)
    ax.text(0.7, 28.15, 'PHASE A  ·  Development audit', fontsize=15, weight='bold', color=BLUE)
    ax.text(19.4, 28.15, 'once per model version, on labelled source groups', fontsize=11, style='italic',
            color=BLUE, ha='right')
    A = [('A1  LOGO (+ future)', 'Hold out each group $g$;\ntrain on the other $G$−1\n(before the split date);\n'
          'score $g$ after it.\n\nFailure if  $R^2_g \\leq 0$\nor MAPE ≥ global-mean\nbenchmark.'),
         ('A2  Error decomposition', '$e_{ig} = b_g + u_{ig}$\nMSE$_g = b_g^2 + $Var$_g(u)$\n\nLevel bias $b_g$ vs.\n'
          'within-group error $u_{ig}$;\nrecord AOA share, adv. AUC\nand target-range excess.'),
         ('A3  Method selection', 'Compare under LOGO + future:\nlevel shift · EB shrinkage ·\nweighted retraining ·\n'
          'random intercept (GPBoost) ·\nmultiplicative · local mean.\n\nNested LOGO to choose\nτ and the method.'),
         ('A4  Probe design', 'Estimate $\\sigma_w$, $\\bar{\\rho}$ and drift $d$\nfrom held-out groups;\n\n'
          '$k_{eff} \\geq [z_{1-\\beta}\\,\\sigma_w/(\\delta - |d|)]^2$\n(Eqs 4–5)  →  $k_{min}$, τ.')]
    xs = [0.75, 5.5, 10.25, 15.0]
    for i, (t, b) in enumerate(A):
        step(ax, xs[i], 23.2, 4.3, 4.5, BLUE, t, b, fs=10.3, tfs=12.2)
        if i < 3:
            arr(ax, (xs[i] + 4.3, 25.45), (xs[i + 1], 25.45), BLUE)
        arr(ax, (xs[i] + 2.15, 23.2), (xs[i] + 2.15, 22.55), BLUE)
    rbox(ax, 0.75, 20.95, 18.55, 1.6, BLUE, lw=2.6)
    ax.text(1.2, 21.75, 'Frozen deployment rule', fontsize=12.5, weight='bold', color=BLUE, va='center')
    for x, w, t in [(5.9, 1.0, 'τ'), (7.1, 1.2, 'c·τ'), (8.5, 1.3, '$k_{min}$'), (10.0, 2.2, '$a_{min}$, $r_{max}$'),
                    (12.4, 2.1, 'method, $w$'), (14.7, 1.0, '$m$'), (15.9, 1.3, '$n_{mon}$')]:
        param(ax, x, 21.3, w, 0.9, t)
    arr(ax, (10, 20.95), (10, 20.15), BLUE, lw=2.2)
    ax.text(10.25, 20.5, 'rule applied to $h$', fontsize=10.5, style='italic', color=BLUE, va='center')
    # PHASE B panel
    rbox(ax, 0.3, 2.6, 19.4, 17.5, 'none', '#EEF6F0', lw=0, r=0.3, z=1)
    ax.text(0.7, 19.5, 'PHASE B  ·  Deployment gate', fontsize=15, weight='bold', color=GREEN)
    ax.text(19.4, 19.5, 'for each unseen group h, before it goes live', fontsize=11, style='italic', color=GREEN,
            ha='right')
    L, W = 1.2, 8.0
    step(ax, L, 16.6, W, 2.3, GREEN, 'B1  Minimum local sample', 'Local labels available for h:  $n_h \\geq k_{min}$ ?')
    param(ax, 9.8, 17.0, 2.8, 1.1, '$k_{min}$ = 30\nprovisional ≥ 10')
    rbox(ax, 13.4, 16.1, 5.8, 2.8, GREY)
    ax.text(16.3, 18.35, 'No → hold', ha='center', fontsize=13, weight='bold', color=GREY)
    ax.text(16.3, 17.1, '10 ≤ $n_h$ < $k_{min}$: provisional use only,\nlabelled \'level not verified\';\n'
            'keep collecting labels.', ha='center', va='center', fontsize=10.3, linespacing=1.3)
    arr(ax, (L + W, 18.3), (13.4, 18.3), GREY); ax.text(11.4, 18.42, 'no', fontsize=10, style='italic', color=GREY)
    arr(ax, (5.2, 16.6), (5.2, 15.85), GREEN); ax.text(5.45, 16.2, 'yes', fontsize=10, style='italic', color=GREY)
    step(ax, L, 13.75, W, 2.1, GREEN, 'B2  Covariate and range flags',
         'Flag h if  AOA share < $a_{min}$  or  > $r_{max}$\nof probe outside the training target range.\n'
         'A flag forces calibration; it never grants release.', fs=10.2)
    param(ax, 9.8, 14.25, 2.8, 1.1, '$a_{min}$ = 90%\n$r_{max}$ = 20%')
    arr(ax, (5.2, 13.75), (5.2, 13.05), GREEN)
    step(ax, L, 10.95, W, 2.1, GREEN, 'B3  Local probe bias',
         'Probe $P_h$: the $k$ most recent local labels, spread\nover projects, buildings and dates.\n'
         '$\\hat{b}_h$ = mean relative error $(\\hat{y}-y)/y$ on $P_h$.', fs=10.2)
    param(ax, 9.8, 11.45, 2.8, 1.1, '$k$ = $k_{min}$\ncheck $k_{eff}$ (Eq. 4)')
    arr(ax, (5.2, 10.95), (5.2, 10.25), GREEN)
    # B4 diamond
    ax.add_patch(Polygon([(5.2, 10.25), (8.8, 8.65), (5.2, 7.05), (1.6, 8.65)], closed=True, fc='white', ec=GREEN,
                         lw=2.4, zorder=2))
    ax.text(5.2, 9.0, 'B4  Gate', ha='center', fontsize=13, weight='bold', color=GREEN, zorder=5)
    ax.text(5.2, 8.25, '$|\\hat{b}_h|$  vs.  c·τ, τ', ha='center', fontsize=11, zorder=5)
    param(ax, 9.8, 8.9, 2.8, 1.1, 'τ = 20%\nc = 0.75 (band)')
    outs = [(10.55, GREEN, 'Release', '$|\\hat{b}_h|$ ≤ c·τ  and no flag'),
            (8.95, GOLD, 'Release + watch', 'c·τ < $|\\hat{b}_h|$ ≤ τ  and no flag'),
            (7.35, RED, 'Calibrate', '$|\\hat{b}_h|$ > τ  or any flag')]
    ax.plot([8.8, 12.9], [8.65, 8.65], color=GREY, lw=2)
    for yb, c, t, b in outs:
        rbox(ax, 13.4, yb, 5.8, 1.45, c)
        ax.text(16.3, yb + 1.0, t, ha='center', fontsize=12.5, weight='bold', color=c, zorder=5)
        ax.text(16.3, yb + 0.38, b, ha='center', fontsize=10.2, zorder=5)
        ax.plot([12.9, 12.9], [8.65, yb + 0.72], color=c, lw=2)
        arr(ax, (12.9, yb + 0.72), (13.4, yb + 0.72), c)
    # B5 / B6
    step(ax, L, 5.15, W, 1.7, RED, 'B5  Calibrate', 'Apply the Phase-A method using $C_h = P_h$\n'
         '(no extra labels; $P_h$ cannot validate it).', fs=10.2)
    ax.plot([16.3, 16.3, 9.2], [7.35, 7.05, 7.05], color=RED, lw=2)
    arr(ax, (9.2, 7.05), (9.2, 6.85), RED)
    param(ax, 9.8, 5.45, 2.8, 1.0, 'method from A3\n$w$ = 10')
    arr(ax, (5.2, 5.15), (5.2, 4.8), RED)
    step(ax, L, 3.1, W, 1.7, RED, 'B6  Independent validation',
         'Next $m$ new labels $V_h$ ($V_h \\cap P_h = \\emptyset$):  $|\\hat{b}_V|$ ≤ τ ?', fs=10.2)
    param(ax, 9.8, 3.45, 2.8, 0.9, '$m$ = 10')
    ax.plot([L, 0.75, 0.75], [3.95, 3.95, 6.0], ls='--', color=RED, lw=1.6)
    arr(ax, (0.75, 6.0), (L, 6.0), RED, lw=1.6, ls='--')
    ax.text(0.5, 4.9, 'no', fontsize=10, style='italic', color=RED, rotation=90, va='center')
    arr(ax, (5.2, 3.1), (5.2, 2.35), GREEN)
    ax.text(5.45, 2.65, 'yes → release', fontsize=10, style='italic', color=GREEN)
    # B7
    rbox(ax, L, 0.25, 18.0, 2.1, TEAL, lw=2.6)
    ax.text(1.6, 1.8, 'B7  Continuous monitoring (all released groups)', fontsize=13, weight='bold', color=TEAL,
            va='center')
    ax.text(1.6, 0.85, 'Rolling bias every $n_{mon}$ new sales or quarterly.  $|\\hat{b}_{roll}|$ > τ  →  recalibrate (B5).\n'
            'Persistent or structural error (low within-group rank correlation)  →  new model version (Phase A).',
            fontsize=10.3, va='center', linespacing=1.35)
    param(ax, 14.2, 1.45, 4.6, 0.75, '$n_{mon}$ = 30 sales / quarter')
    ax.plot([19.2, 19.55, 19.55], [11.25, 11.25, 1.0], color=GREEN, lw=1.6)
    arr(ax, (19.55, 1.0), (19.2, 1.0), GREEN, lw=1.6)
    ax.plot([16.3, 16.3], [2.35, 6.6], ls='--', color=TEAL, lw=1.6)
    arr(ax, (16.3, 6.6), (16.3, 7.35), TEAL, lw=1.6, ls='--')
    ax.text(16.45, 4.6, 'drift > τ:\nrecalibrate', fontsize=10, style='italic', color=TEAL, va='center')
    ax.plot([L, 0.12, 0.12], [0.9, 0.9, 25.45], ls=':', color=BLUE, lw=1.8)
    arr(ax, (0.12, 25.45), (0.75, 25.45), BLUE, lw=1.8)
    ax.text(0.32, 12.0, 'new model version', rotation=90, fontsize=10, style='italic', color=BLUE, va='center')
    # legend
    param(ax, 0.3, -0.75, 1.2, 0.5, '', fs=8)
    ax.text(1.75, -0.5, 'Adjustable parameter (value used in this study): set from the use-case tolerance and '
            'Eqs (4)–(5); re-derive for each new market or domain.', fontsize=10, va='center')
    ax.set_ylim(-1.0, 31.5)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'Figure3_workflow.{ext}'), dpi=400, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
