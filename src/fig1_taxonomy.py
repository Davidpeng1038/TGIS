"""fig1_taxonomy.py — Figure 1 (Section 3.2): taxonomy of geographic transfer failures."""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = os.path.join(os.path.dirname(__file__), '..', 'output', 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'Liberation Sans', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
                     'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold'})

BLUE, RED, ORANGE, GREY, DARK = '#2F67B1', '#C93A33', '#D98322', '#555555', '#222222'
FILL = {BLUE: '#EDF2F9', RED: '#FAEDEC', ORANGE: '#FBF3E8'}


def box(ax, xc, yc, w, h, edge, face='white', lw=2.4):
    ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.12',
                                fc=face, ec=edge, lw=lw))


def arrow(ax, x0, y0, x1, y1, c, lw=2):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw, mutation_scale=18, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(12, 8.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10.5); ax.axis('off')
    # top
    box(ax, 7.75, 9.5, 6.2, 1.0, DARK)
    ax.text(7.75, 9.5, 'Geographic transfer failure', ha='center', va='center', fontsize=16, weight='bold', color=DARK)
    cols = [(3.65, BLUE, 'Covariate shift', '$P(x)$ changes\n$P_g(x) \\neq P_{train}(x)$',
             'observable before deployment', 'AOA / adversarial validation', 'label-free flag (B2)\nforces calibration'),
            (7.75, RED, 'Level shift', '$\\alpha_g$ changes\n$\\alpha_g \\neq 0$',
             'needs local labels', 'Local probe → calibration',
             '$k_{eff} \\geq k_{min}$ labels (B3–B4)\nlevel shift / EB (B5–B6)'),
            (11.85, ORANGE, 'Structural shift', 'slopes change\n$f_g \\neq f$',
             'needs labels and time', 'Monitoring → retraining',
             'within-group rank correlation (B7)\nweighted retraining / new model')]
    ax.plot([7.75, 7.75], [9.0, 8.4], color=GREY, lw=2)
    ax.plot([3.65, 11.85], [8.4, 8.4], color=GREY, lw=2)
    for x, c, t, sub, lab, t2, sub2 in cols:
        arrow(ax, x, 8.4, x, 7.55, c)
        box(ax, x, 6.8, 3.45, 1.5, c, FILL[c])
        ax.text(x, 7.2, t, ha='center', va='center', fontsize=14, weight='bold', color=c)
        ax.text(x, 6.6, sub, ha='center', va='center', fontsize=12, linespacing=1.3)
        arrow(ax, x, 6.05, x, 4.7, c)
        ax.text(x + 0.13, 5.38, lab, ha='left', va='center', fontsize=10.5, style='italic', color=c)
        box(ax, x, 3.95, 3.75, 1.5, c)
        ax.text(x, 4.35, t2, ha='center', va='center', fontsize=12, weight='bold', color=c)
        ax.text(x, 3.72, sub2, ha='center', va='center', fontsize=11, linespacing=1.3)
        ax.plot([x, x], [3.2, 2.35], color=c, lw=2)
    ax.plot([3.65, 11.85], [2.35, 2.35], color=GREY, lw=2)
    arrow(ax, 7.75, 2.35, 7.75, 1.75, GREY)
    box(ax, 7.75, 1.15, 6.7, 1.2, DARK, '#F3F3F3')
    ax.text(7.75, 1.4, 'Deployment decision', ha='center', va='center', fontsize=14, weight='bold', color=DARK)
    ax.text(7.75, 0.88, 'Release  /  Observe  /  Calibrate', ha='center', va='center', fontsize=13)
    for yy, t in [(6.8, 'Source of\nfailure'), (3.95, 'Detection and\nresponse'), (1.15, 'Gate\noutcome')]:
        ax.text(0.05, yy, t, ha='left', va='center', fontsize=11.5, style='italic', color='#666666', linespacing=1.0)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'Figure1_taxonomy.{ext}'), dpi=400, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
