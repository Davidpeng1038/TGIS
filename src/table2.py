"""table2.py — Table 2 (Section 4.2): failure diagnostics by held-out station area (Taiwan).

Columns
  MAPE random → LOSAO : random five-fold vs leave-one-station-area-out MAPE (full period)
  Bench. MAPE⁰        : predicting every transaction by the mean unit price of the other 21 areas
  LOSAO R² (F)        : failure if R² ≤ 0 or MAPE ≥ MAPE⁰
  AOA % / adv. AUC    : object-level (F2) features. AOA follows Meyer & Pebesma (2021): importance-weighted,
                        standardised features; DI = distance to nearest training point / mean training distance;
                        threshold = upper whisker (Q3 + 1.5 IQR) of training DI computed across training areas.
                        Adversarial AUC: LightGBM classifier (held-out area vs training areas), five-fold CV.
  Bias / probe / post : mean (ŷ − y)/y, full-period LOSAO; LOSAO+future model on the 30 most recent
                        pre-split local transactions; same model on all post-split local transactions
  MAE reduction / ρ   : oracle removal of the area's mean raw residual; within-area Spearman correlation
Output: output/tables/Table2.csv
"""
import numpy as np, pandas as pd, os, lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.neighbors import KDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import *  # noqa

N_REF = 40000      # reference subsample of training transactions for nearest-neighbour distances
rng = np.random.default_rng(0)


def aoa_share(d, s):
    tr, ho = d[d.station != s], d[d.station == s]
    m = lgb.LGBMRegressor(**LGB, random_state=0).fit(tr[F2], tr.y)
    w = m.booster_.feature_importance('gain'); w = w / w.sum()
    mu, sd = tr[F2].mean(), tr[F2].std()
    Z = lambda X: ((X[F2] - mu) / sd).fillna(0).values * w
    ref = tr.iloc[rng.choice(len(tr), min(N_REF, len(tr)), replace=False)]
    Zr = Z(ref)
    sub = Zr[rng.choice(len(Zr), 3000, replace=False)]
    dbar = np.mean(np.sqrt(((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1))[np.triu_indices(len(sub), 1)])
    # training DI: distance to the nearest reference point belonging to another training area
    di_tr = np.empty(len(ref))
    st = ref.station.astype(str).values
    for g in np.unique(st):
        a, b = st == g, st != g
        di_tr[a] = KDTree(Zr[b]).query(Zr[a], k=1)[0][:, 0]
    di_tr /= dbar
    q1, q3 = np.percentile(di_tr, [25, 75]); thr = min(q3 + 1.5 * (q3 - q1), di_tr.max())
    di_new = KDTree(Zr).query(Z(ho), k=1)[0][:, 0] / dbar
    return 100 * np.mean(di_new <= thr)


def adv_auc(d, s, cap=20000):
    ho, tr = d[d.station == s], d[d.station != s]
    ho = ho.iloc[rng.choice(len(ho), min(cap, len(ho)), replace=False)]
    tr = tr.iloc[rng.choice(len(tr), min(cap, len(tr)), replace=False)]
    X = pd.concat([ho[F2], tr[F2]]).values; lab = np.r_[np.ones(len(ho)), np.zeros(len(tr))]
    p = np.zeros(len(lab))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(X, lab):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, verbose=-1, n_jobs=2,
                               is_unbalance=True, random_state=0).fit(X[a], lab[a])
        p[b] = m.predict_proba(X[b])[:, 1]
    return roc_auc_score(lab, p)


def main():
    d, split = load_tw()
    P = pd.read_parquet(os.path.join(OUT, 'tw_preds.parquet'))
    d = pd.concat([d, P], axis=1)
    rows = []
    for s in ORDER:
        g = d[d.station == s]
        y, pl = g.y.values, g.losao.values
        bench = mape(y, np.full(len(y), d.y[d.station != s].mean()))
        R, M = r2(y, pl), mape(y, pl)
        pre = g[~g.test].sort_values('date')
        probe = pre.tail(30); post = g[g.test]
        res = y - pl
        mae0 = np.mean(np.abs(res)); mae1 = np.mean(np.abs(res - res.mean()))
        rows.append(dict(station=s, mape_random=mape(y, g.random), mape_losao=M, bench_mape0=bench, losao_r2=R,
                         fail=(R <= 0) or (M >= bench), aoa_pct=aoa_share(d, s), adv_auc=adv_auc(d, s),
                         bias_full=rel_bias(y, pl), bias_probe=rel_bias(probe.y, probe.losaof),
                         bias_post=rel_bias(post.y, post.losaof),
                         mae_reduction=100 * (mae0 - mae1) / mae0, spearman=spearmanr(y, pl).statistic,
                         share_above_p99=100 * np.mean(y > np.percentile(d.y[d.station != s], 99))))
        print(s, f"{R:.2f}", f"{rows[-1]['aoa_pct']:.1f}", f"{rows[-1]['adv_auc']:.3f}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(TAB, 'Table2_raw.csv'), index=False)
    # formatted version in the manuscript's layout
    f = lambda v, n=1, sign=False: (f'{v:+.{n}f}' if sign else f'{v:.{n}f}').replace('-', '−')
    out = pd.DataFrame({
        'Station area': T.station,
        'MAPE (%) random → LOSAO': [f'{a:.1f} → {b:.1f}' for a, b in zip(T.mape_random, T.mape_losao)],
        'Bench. MAPE⁰ (%)': T.bench_mape0.map(lambda v: f'{v:.1f}'),
        'LOSAO R²': [f(r, 2) + (' (F)' if fl else '') for r, fl in zip(T.losao_r2, T.fail)],
        'AOA (%) / adv. AUC': [f'{a:.1f} / {b:.3f}' for a, b in zip(T.aoa_pct, T.adv_auc)],
        'Bias / probe / post-split (%)': [f'{f(a, 1, True)} / {f(b, 1, True)} / {f(c, 1, True)}'
                                          for a, b, c in zip(T.bias_full, T.bias_probe, T.bias_post)],
        'MAE reduction (%) / ρ': [f'{a:.1f} / {b:.2f}' for a, b in zip(T.mae_reduction, T.spearman)]})
    out.to_csv(os.path.join(TAB, 'Table2.csv'), index=False)
    # '22-area MAE reduction': 1 - mean_g MAE_g(after removing the area mean residual) / mean_g MAE_g(before)
    res = d.y - d.losao
    mae_b = res.abs().groupby(d.station, observed=True).mean()
    mae_a = (res - res.groupby(d.station, observed=True).transform('mean')).abs().groupby(d.station, observed=True).mean()
    tot = 100 * (1 - mae_a.mean() / mae_b.mean())
    print(out.to_string(index=False))
    print(f'failures {int(T.fail.sum())}/22; 22-area MAE reduction {tot:.1f}%; mean Spearman {T.spearman.mean():.3f}')
    open(os.path.join(TAB, 'Table2_notes.txt'), 'w').write(
        f'Failures: {int(T.fail.sum())}/22 ({", ".join(T.station[T.fail])}).\n'
        f'22-area MAE reduction after removing station-area mean residual: {tot:.1f}% '
        f'(mean of per-area reductions {T.mae_reduction.mean():.1f}%); mean within-area Spearman {T.spearman.mean():.3f}.\n'
        f'Share of unit prices above the training 99th percentile: Taipei {T.set_index("station").share_above_p99["Taipei"]:.1f}%, '
        f'Songshan {T.set_index("station").share_above_p99["Songshan"]:.1f}%.\n')


if __name__ == '__main__':
    main()
