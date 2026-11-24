"""fig2_framework.py — Figure 2 (Section 3.4): conceptual framework of deployment-matched validation."""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

FIG = os.path.join(os.path.dirname(__file__), '..', 'output', 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'Liberation Sans', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
                     'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold'})
BLUE, PURPLE, GREEN, RED = '#2F67B1', '#6A4C9C', '#2E8B57', '#C93A33'
FILL = {BLUE: '#EDF2F9', PURPLE: '#F2EEF8', GREEN: '#EDF6F0'}

STEPS = [
    (BLUE, 'Deployment unit', 'Where will the model be used?',
     'Define the unit before validating:\nstation area (Taiwan) or county (California)'),
    (PURPLE, 'Validation unit', 'Hold out what will be deployed',
     'Leave-one-group-out (+ future period);\nNNDM / kNNDM only for interpolation within a unit'),
    (PURPLE, 'Failure diagnosis', 'Does transfer fail, and why?',
     '$R^2_g \\leq 0$ or MAPE ≥ benchmark;  $e_{ig} = b_g + u_{ig}$ (Eqs. 1–2);\nAOA / adversarial AUC used as flags only'),
    (GREEN, 'Local probe', 'Is the target level shifted here?',
     '$k$ local labels → probe bias $\\hat{b}_g$;\nSE $= \\sigma_w/\\sqrt{k_{eff}}$ (Eqs. 4–5): size the probe by $k_{eff}$, not $k$'),
    (GREEN, 'Calibration', 'Correct the level with few labels',
     'Level shift / EB shrinkage / weighted retraining,\nmethod chosen in Phase A; verified on new labels $V_g$'),
    (GREEN, 'Governance decision', 'Release, watch, or recalibrate?',
     '$|\\hat{b}_g| \\leq 0.75\\tau$ release;  $\\leq\\tau$ watch;  $>\\tau$ or flag calibrate;\nrolling monitoring and an auditable log'),
]


def main():
    fig, ax = plt.subplots(figsize=(12.6, 10.3))
    ax.set_xlim(0, 13); ax.set_ylim(0, 10.8); ax.axis('off')
    ys = [9.85 - i * 1.72 for i in range(6)]
    for i, (c, t, q, desc) in enumerate(STEPS):
        y = ys[i]
        ax.add_patch(FancyBboxPatch((0.9, y - 0.6), 4.3, 1.2, boxstyle='round,pad=0,rounding_size=0.1',
                                    fc=FILL[c], ec=c, lw=2.6))
        ax.add_patch(Circle((1.45, y), 0.28, fc=c, ec='none'))
        ax.text(1.45, y, str(i + 1), ha='center', va='center', color='white', fontsize=15, weight='bold')
        ax.text(1.95, y + 0.25, t, ha='left', va='center', color=c, fontsize=14.5, weight='bold')
        ax.text(1.95, y - 0.25, q, ha='left', va='center', fontsize=11.5, style='italic', color='#333333')
        ax.add_patch(FancyBboxPatch((5.5, y - 0.52), 5.6, 1.04, boxstyle='round,pad=0,rounding_size=0.08',
                                    fc='white', ec='#BBBBBB', lw=1.3))
        ax.text(5.7, y, desc, ha='left', va='center', fontsize=10.8, linespacing=1.45)
        ax.plot([5.2, 5.5], [y, y], ls='--', color='#BBBBBB', lw=1.2)
        if i < 5:
            ax.annotate('', xy=(3.05, ys[i + 1] + 0.6), xytext=(3.05, y - 0.6),
                        arrowprops=dict(arrowstyle='-|>', lw=2.2, color='#444444', mutation_scale=18))
    # feedback loops
    ax.plot([0.9, 0.62, 0.62], [ys[5] - 0.42, ys[5] - 0.42, ys[3] + 0.18], ls='--', color=RED, lw=1.8)
    ax.annotate('', xy=(0.9, ys[3] + 0.18), xytext=(0.62, ys[3] + 0.18),
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.8, ls='--'))
    ax.text(0.47, (ys[5] + ys[3]) / 2, 'rolling bias > τ: re-probe, recalibrate', rotation=90, ha='center',
            va='center', color=RED, fontsize=10, style='italic')
    ax.plot([0.9, 0.3, 0.3], [ys[5] - 0.5, ys[5] - 0.5, ys[2]], ls='--', color='#333333', lw=1.8)
    ax.annotate('', xy=(0.9, ys[2]), xytext=(0.3, ys[2]),
                arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1.8, ls='--'))
    ax.text(0.15, (ys[5] + ys[2]) / 2 - 0.3, 'new model version: re-audit', rotation=90, ha='center', va='center',
            color='#333333', fontsize=10, style='italic')
    # brackets
    for (y0, y1, c, lab) in [(ys[0] - 0.6, ys[0] + 0.6, BLUE, 'Design'),
                             (ys[2] - 0.6, ys[1] + 0.6, PURPLE, 'Phase A\ndevelopment audit'),
                             (ys[5] - 0.6, ys[3] + 0.6, GREEN, 'Phase B\ndeployment gate')]:
        ax.plot([11.35, 11.5, 11.5, 11.35], [y1, y1, y0, y0], color=c, lw=2.6)
        ax.text(11.95, (y0 + y1) / 2, lab, rotation=90, ha='center', va='center', color=c, fontsize=13,
                weight='bold', linespacing=1.1)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'Figure2_framework.{ext}'), dpi=400, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
