"""table1.py — Table 1 (Section 4.1): validation designs that do and do not match the deployment unit.

Taiwan rows are scored on the test set (final 20% of date-sorted transactions); cross-validation designs use
their out-of-fold predictions for those transactions. California rows are scored on the 29 held-out counties.
Also reports unit means/medians of R², failure counts and residual Moran's I (k = 10 nearest neighbours).
Output: output/tables/Table1.csv, output/tables/Table1_notes.txt
"""
import numpy as np, pandas as pd, os
from sklearn.neighbors import BallTree
from common import *  # noqa


def morans_i(lat, lon, resid, k=10, n_max=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(lat), min(n_max, len(lat)), replace=False)
    X = np.radians(np.c_[lat[idx], lon[idx]]); z = resid[idx] - resid[idx].mean()
    nb = BallTree(X, metric='haversine').query(X, k=k + 1, return_distance=False)[:, 1:]
    num = np.sum(z[:, None] * z[nb]) / k
    return num / np.sum(z ** 2)   # row-standardised weights: I = sum_ij w_ij z_i z_j / sum_i z_i^2


def unit_stats(df, pred, group, bench):
    rows = []
    for g, s in df.groupby(group, observed=True):
        r, m = r2(s.y, s[pred]), mape(s.y, s[pred])
        rows.append((g, r, m, (r <= 0) or (m >= bench[g])))
    u = pd.DataFrame(rows, columns=['unit', 'R2', 'MAPE', 'fail'])
    return u


def main():
    d, split = load_tw()
    P = pd.read_parquet(os.path.join(OUT, 'tw_preds.parquet'))
    d = pd.concat([d, P], axis=1)
    te = d[d.test]
    designs = [('Random five-fold', 'random'), ('Location-grouped five-fold', 'locgroup'),
               ('Spatial-block five-fold', 'block'), ('Temporal holdout (final 20%)', 'temporal'),
               ('Rolling origin (2018–2026)', 'rolling'), ('Leave-one-city-out', 'loco'),
               ('Leave-one-group-out (station area / county)', 'losao'),
               ('Leave-one-group-out + future period', 'losaof')]
    T = pd.DataFrame({'Validation design': [a for a, _ in designs]})
    T['Taiwan R²'] = [r2(te.y, te[c]) for _, c in designs]
    T['Taiwan MAPE (%)'] = [mape(te.y, te[c]) for _, c in designs]
    c = load_ca(); C = pd.read_parquet(os.path.join(OUT, 'ca_preds.parquet')); c = pd.concat([c, C], axis=1)
    h = c[c.held_out]
    ca = {'random': 'random', 'block': 'block', 'losao': 'loco'}
    T['California R²'] = [r2(h.y, h[ca[k]]) if k in ca else np.nan for _, k in designs]
    T['California MAPE (%)'] = [mape(h.y, h[ca[k]]) if k in ca else np.nan for _, k in designs]
    T.round(3).to_csv(os.path.join(TAB, 'Table1.csv'), index=False)
    # notes: unit-level summaries (full period for Taiwan LOSAO)
    bench_tw = {s: mape(d.y[d.station == s], np.full((d.station == s).sum(), d.y[d.station != s].mean())) for s in ORDER}
    u_tw = unit_stats(d, 'losao', 'station', bench_tw)
    hc = h.copy(); hc['county'] = hc.county.astype(str)
    bench_ca = {k: mape(hc.y[hc.county == k], np.full((hc.county == k).sum(), c.y[c.county.astype(str) != k].mean()))
                for k in hc.county.unique()}
    u_ca = unit_stats(hc, 'loco', 'county', bench_ca)
    mi = {k: morans_i(te.lat.values, te.lon.values, (te.y - te[k]).values) for k in ['random', 'block']}
    notes = (f"Test set n = {len(te):,} (from {split.date()}).\n"
             f"Taiwan LOSAO (full period): unit mean R² = {u_tw.R2.mean():.2f}, median = {u_tw.R2.median():.2f}, "
             f"mean MAPE = {u_tw.MAPE.mean():.1f}%, failures = {int(u_tw.fail.sum())}/22.\n"
             f"California LOCO: unit mean R² = {u_ca.R2.mean():.2f}, median = {u_ca.R2.median():.2f}, "
             f"mean MAPE = {u_ca.MAPE.mean():.1f}%, failures = {int(u_ca.fail.sum())}/{len(u_ca)}.\n"
             f"Residual Moran's I (test set, k = 10): random five-fold = {mi['random']:.3f}, "
             f"spatial-block = {mi['block']:.3f}.\n")
    open(os.path.join(TAB, 'Table1_notes.txt'), 'w').write(notes)
    u_ca.to_csv(os.path.join(TAB, 'TableS6_county_LOCO.csv'), index=False)
    print(T.round(3).to_string(index=False)); print(notes)


if __name__ == '__main__':
    main()
