"""common.py — shared settings, model fitting and metrics."""
import os, numpy as np, pandas as pd, lightgbm as lgb

ROOT = os.path.join(os.path.dirname(__file__), '..')
OUT = os.path.join(ROOT, 'out')                     # intermediate files (cleaned data, predictions)
OUTPUT = os.path.join(ROOT, 'output')               # deliverables
FIG = os.path.join(OUTPUT, 'figures'); TAB = os.path.join(OUTPUT, 'tables')
for p in (OUT, OUTPUT, FIG, TAB):
    os.makedirs(p, exist_ok=True)

from prep import ORDER, HUBS  # noqa: E402

# Feature sets (Table S1a)
ID = ['City', 'District', 'station']
DENS = ['dens500', 'dens1000', 'dens1500']
OBJ = ['dist_km', 't', 'land_area', 'bldg_area', 'floor', 'bedrooms', 'living', 'parking', 'age']
F0 = ID + DENS + OBJ
F1 = DENS + OBJ
F2 = OBJ
CA_F = ['housing_median_age', 'rooms_per_hh', 'bedrooms_per_room', 'pop_per_hh', 'population',
        'households', 'median_income', 'ocean_proximity', 'county']

LGB = dict(n_estimators=300, learning_rate=0.1, num_leaves=31, min_child_samples=20, verbose=-1, n_jobs=2)
SEEDS = (0, 1)
TAU = 0.20


def fit_predict(Xtr, ytr, Xte, seeds=SEEDS, weight=None, params=None):
    """LightGBM averaged over seeds; categorical columns handled natively (unseen levels -> 'other' side)."""
    p = dict(LGB, **(params or {}))
    pred = np.zeros(len(Xte))
    for s in seeds:
        m = lgb.LGBMRegressor(**p, random_state=s)
        m.fit(Xtr, ytr, sample_weight=weight)
        pred += m.predict(Xte)
    return pred / len(seeds)


def load_tw():
    d = pd.read_parquet(os.path.join(OUT, 'tw_clean.parquet'))
    n = len(d)
    cut = int(round(n * 0.8))            # final 20% of date-sorted transactions = test set
    d['test'] = False
    d.loc[cut:, 'test'] = True
    split_date = d.date.iloc[cut]
    return d, split_date


def load_ca():
    return pd.read_parquet(os.path.join(OUT, 'ca_clean.parquet'))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 100 * np.mean(np.abs(p - y) / y)


def rel_bias(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return 100 * np.mean((p - y) / y)
