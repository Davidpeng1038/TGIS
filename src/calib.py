"""calib.py — Table 3 and Figure 4 (Section 4.5): few-shot local calibration, Taiwan.

Design: leave-one-station-area-out + future. For each held-out area h the model is trained on the other 21 areas'
transactions before the split date (6 June 2023); the k most recent pre-split local transactions (probe P_h; Keelung
has fewer than 300) are used for calibration; evaluation uses all post-split local transactions.
Methods: level shift; empirical-Bayes shrinkage (Morris, 1983); weighted retraining (w = 10); GPBoost random
intercept (untuned: LightGBM hyperparameters, standardised target); F3 multiplicative correction (log target,
object-level features); local mean. Oracle level correction uses the post-split mean residual.
Output: output/tables/Table3.csv, output/tables/Table3_by_station.csv, output/figures/Figure4_calibration.png
"""
import numpy as np, pandas as pd, os, time, warnings
import gpboost as gpb
from common import *  # noqa

warnings.filterwarnings('ignore')
KS = [0, 10, 30, 100, 300]
W = 10
GP_FEATS = ['City', 'District'] + DENS + OBJ      # station enters as a random intercept


def gpboost_pred(tr, te, seed=0):
    mu, sd = tr.y.mean(), tr.y.std()
    gp = gpb.GPModel(group_data=tr.station.cat.codes.to_numpy(), likelihood='gaussian')
    gp.set_optim_params(params={'trace': False})
    ds = gpb.Dataset(tr[GP_FEATS], (tr.y - mu) / sd, categorical_feature=['City', 'District'])
    params = dict(learning_rate=LGB['learning_rate'], num_leaves=LGB['num_leaves'],
                  min_data_in_leaf=LGB['min_child_samples'], verbose=-1, num_threads=2, seed=seed)
    b = gpb.train(params=params, train_set=ds, gp_model=gp, num_boost_round=LGB['n_estimators'])
    pr = b.predict(data=te[GP_FEATS], group_data_pred=te.station.cat.codes.to_numpy(), predict_var=False,
                   pred_latent=False)
    return pr['response_mean'] * sd + mu


def main():
    t0 = time.time()
    d, split = load_tw()
    P = pd.read_parquet(os.path.join(OUT, 'tw_preds.parquet'))
    d = pd.concat([d, P], axis=1)
    # EB variance components from pre-split LOSAO+future residuals (raw scale)
    pre_all = d[~d.test].copy(); pre_all['res'] = pre_all.y - pre_all.losaof
    gm = pre_all.groupby('station', observed=True).res.agg(['mean', 'var', 'count'])
    rows, by_station = [], []
    for s in ORDER:
        tr = d[(d.station != s) & ~d.test]
        loc_pre = d[(d.station == s) & ~d.test].sort_values('date')
        ev = d[(d.station == s) & d.test]
        oth = gm.drop(s)
        s_a2 = max(oth['mean'].var() - np.mean(oth['var'] / oth['count']), 1e-9); sig2 = np.average(oth['var'], weights=oth['count'])
        base, base3 = ev.losaof.values, ev.losaof_f3.values
        oracle = base + (ev.y - base).mean()
        rows.append(dict(station=s, method='Oracle level correction', k=np.nan, MAPE=mape(ev.y, oracle), R2=r2(ev.y, oracle)))
        for k in KS:
            pr = loc_pre.tail(k) if k else loc_pre.iloc[0:0]
            preds = {}
            if k == 0:
                preds['Level shift'] = preds['EB shrinkage'] = preds['Weighted retraining (w = 10)'] = base
                preds['F3 multiplicative correction'] = base3
                preds['Local mean'] = np.full(len(ev), tr.y.mean())
            else:
                c = (pr.y - pr.losaof).mean()
                preds['Level shift'] = base + c
                Bk = s_a2 / (s_a2 + sig2 / len(pr))
                preds['EB shrinkage'] = base + Bk * c
                Xtr = pd.concat([tr, pr]); wts = np.r_[np.ones(len(tr)), np.full(len(pr), W)]
                preds['Weighted retraining (w = 10)'] = fit_predict(Xtr[F0], Xtr.y, ev[F0], weight=wts)
                preds['F3 multiplicative correction'] = base3 * np.exp(np.mean(np.log(pr.y) - np.log(pr.losaof_f3)))
                preds['Local mean'] = np.full(len(ev), pr.y.mean())
            preds['GPBoost (untuned)'] = gpboost_pred(pd.concat([tr, pr]), ev)
            for mth, p in preds.items():
                rows.append(dict(station=s, method=mth, k=k, MAPE=mape(ev.y, p), R2=r2(ev.y, p)))
        print(f'{s:10s} done ({time.time() - t0:.0f} s)', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(TAB, 'Table3_by_station.csv'), index=False)
    order = ['Level shift', 'EB shrinkage', 'Weighted retraining (w = 10)', 'GPBoost (untuned)',
             'F3 multiplicative correction', 'Local mean']
    agg = R[R.method != 'Oracle level correction'].groupby(['method', 'k'])[['MAPE', 'R2']].mean().reset_index()
    T = pd.DataFrame({'Method': order})
    for k in KS:
        a = agg[agg.k == k].set_index('method')
        T[f'k = {k}'] = [f"{a.loc[m, 'MAPE']:.1f} / {a.loc[m, 'R2']:.2f}".replace('-', '−') for m in order]
    T.to_csv(os.path.join(TAB, 'Table3.csv'), index=False)
    o = R[R.method == 'Oracle level correction'][['MAPE', 'R2']].mean()
    # paired bootstrap CI for the k = 30 weighted-retraining improvement over k = 0
    a = R[(R.method == 'Weighted retraining (w = 10)')].pivot(index='station', columns='k', values='MAPE')
    diff = (a[0] - a[30]).values; rng = np.random.default_rng(0)
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(10000)]
    note = (f'Oracle level correction: {o.MAPE:.1f} / {o.R2:.2f}.\n'
            f'Weighted retraining k = 30 vs k = 0: MAPE reduction {diff.mean():.1f} percentage points '
            f'[{np.percentile(bs, 2.5):.1f}, {np.percentile(bs, 97.5):.1f}] (station bootstrap, 10,000 resamples).\n')
    open(os.path.join(TAB, 'Table3_notes.txt'), 'w').write(note)
    print(T.to_string(index=False)); print(note)
    figure4(R, o.MAPE)


def figure4(R, oracle):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Liberation Sans'})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
    xs = np.arange(len(KS))
    agg = R.groupby(['method', 'k']).MAPE.mean()
    for m, c, mk, ls, lab in [('Level shift', '#2F67B1', 'o', '-', 'F0 + level shift'),
                              ('Weighted retraining (w = 10)', '#169B74', 'D', '-', 'F0 + weighted retraining'),
                              ('F3 multiplicative correction', '#D7301F', 's', '-', 'F3 + multiplicative correction'),
                              ('Local mean', '#333333', '^', '--', 'Local mean')]:
        a1.plot(xs, [agg[(m, k)] for k in KS], marker=mk, ls=ls, color=c, lw=2.2, ms=7, label=lab)
    a1.axhline(oracle, ls=':', color='#2F67B1', lw=1.4, label='Oracle level correction (F0)')
    a1.set_title('(a) Average MAPE across 22 station areas', loc='left')
    wr = R[R.method == 'Weighted retraining (w = 10)'].pivot(index='station', columns='k', values='MAPE').reindex(ORDER)
    fails = set(pd.read_csv(os.path.join(TAB, 'Table2_raw.csv')).query('fail').station) \
        if os.path.exists(os.path.join(TAB, 'Table2_raw.csv')) else set()
    cmap = plt.get_cmap('tab20')
    for i, s in enumerate(ORDER):
        a2.plot(xs, wr.loc[s, KS].values, marker='o', ms=4, lw=1.3, color=cmap(i % 20) if i < 20 else ['#333333', '#E6550D'][i - 20],
                label=s + (' (F)' if s in fails else ''))
    a2.set_title('(b) Weighted retraining (w = 10) by station area', loc='left')
    a2.legend(title='Held-out station (F = failure)', fontsize=7.5, title_fontsize=8, ncol=3, loc='upper right')
    for a in (a1, a2):
        a.set_xticks(xs, [str(k) for k in KS]); a.grid(alpha=.3)
        a.set_xlabel('k = most recent local transactions used for calibration'); a.set_ylabel('MAPE (%)')
    a1.legend(fontsize=9)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'Figure4_calibration.{ext}'), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
