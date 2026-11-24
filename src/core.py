"""core.py — out-of-sample predictions for every validation design (Tables 1–3).

Taiwan (F0, LightGBM, two seeds):
  random / location-grouped / spatial-block (0.01 deg) five-fold, temporal holdout (final 20%),
  rolling origin (test years 2018–2026), leave-one-city-out, leave-one-station-area-out (LOSAO,
  full period), LOSAO + future (train other areas before the split date; predict the held-out area).
  Also LOSAO with F2 (object-level only) and the F3 log-target model used in Table 3.
California (LightGBM, two seeds): random five-fold, spatial-block (0.1 deg) five-fold, leave-one-county-out.
Output: out/tw_preds.parquet, out/ca_preds.parquet
"""
import numpy as np, pandas as pd, os, time
from sklearn.model_selection import KFold, GroupKFold
from common import *  # noqa


def kfold_pred(X, y, folds, feats, **kw):
    p = np.full(len(X), np.nan)
    for tr, te in folds:
        p[te] = fit_predict(X.iloc[tr][feats], y.iloc[tr], X.iloc[te][feats], **kw)
    return p


def group_folds(groups, n=5, seed=0):
    """Random assignment of whole groups to n folds."""
    u = pd.unique(groups)
    rng = np.random.default_rng(seed)
    f = dict(zip(u, rng.integers(0, n, len(u))))
    fid = np.array([f[g] for g in groups])
    return [(np.where(fid != k)[0], np.where(fid == k)[0]) for k in range(n)]


def taiwan():
    d, split = load_tw()
    y = d.y
    P = pd.DataFrame(index=d.index)
    t0 = time.time()
    P['random'] = kfold_pred(d, y, list(KFold(5, shuffle=True, random_state=0).split(d)), F0)
    loc = d.station.astype(str) + '_' + d.lat.round(6).astype(str) + '_' + d.lon.round(6).astype(str)
    P['locgroup'] = kfold_pred(d, y, group_folds(loc.values), F0)
    cell = (np.floor(d.lat / 0.01)).astype(int).astype(str) + '_' + (np.floor(d.lon / 0.01)).astype(int).astype(str)
    P['block'] = kfold_pred(d, y, group_folds(cell.values), F0)
    tr, te = np.where(~d.test)[0], np.where(d.test)[0]
    P['temporal'] = np.nan
    P.loc[te, 'temporal'] = fit_predict(d.iloc[tr][F0], y.iloc[tr], d.iloc[te][F0])
    P['rolling'] = np.nan
    yr = d.date.dt.year
    for Y in range(2018, 2027):
        tr, te = np.where(yr < Y)[0], np.where(yr == Y)[0]
        if len(te):
            P.loc[te, 'rolling'] = fit_predict(d.iloc[tr][F0], y.iloc[tr], d.iloc[te][F0])
    P['loco'] = kfold_pred(d, y, [(np.where(d.City != c)[0], np.where(d.City == c)[0])
                                  for c in d.City.unique()], F0)
    st = [(np.where(d.station != s)[0], np.where(d.station == s)[0]) for s in ORDER]
    P['losao'] = kfold_pred(d, y, st, F0)
    P['losao_f2'] = kfold_pred(d, y, st, F2)
    # LOSAO + future: train on other areas before the split; predict the whole held-out area
    P['losaof'] = np.nan
    P['losaof_f3'] = np.nan   # F3: F2 with log target
    for s in ORDER:
        tr = np.where((d.station != s) & ~d.test)[0]
        te = np.where(d.station == s)[0]
        P.loc[te, 'losaof'] = fit_predict(d.iloc[tr][F0], y.iloc[tr], d.iloc[te][F0])
        P.loc[te, 'losaof_f3'] = np.exp(fit_predict(d.iloc[tr][F2], np.log(y.iloc[tr]), d.iloc[te][F2]))
    print(f'Taiwan predictions done in {time.time() - t0:.0f} s')
    P.to_parquet(os.path.join(OUT, 'tw_preds.parquet'))


def california():
    c = load_ca()
    y = c.y
    P = pd.DataFrame(index=c.index)
    P['random'] = kfold_pred(c, y, list(KFold(5, shuffle=True, random_state=0).split(c)), CA_F)
    cell = (np.floor(c.latitude / 0.1)).astype(int).astype(str) + '_' + (np.floor(c.longitude / 0.1)).astype(int).astype(str)
    P['block'] = kfold_pred(c, y, group_folds(cell.values), CA_F)
    P['loco'] = np.nan
    for k in c[c.held_out].county.astype(str).unique():
        tr, te = np.where(c.county != k)[0], np.where(c.county == k)[0]
        P.loc[te, 'loco'] = fit_predict(c.iloc[tr][CA_F], y.iloc[tr], c.iloc[te][CA_F])
    P.to_parquet(os.path.join(OUT, 'ca_preds.parquet'))
    print('California predictions done')


if __name__ == '__main__':
    taiwan()
    california()
