"""compare.py — manuscript (V2.1) values vs reproduced values for Tables 1–3."""
import os, re, pandas as pd, numpy as np
from common import TAB, ORDER

MS_T1 = {  # design: (TW R2, TW MAPE, CA R2, CA MAPE)
 'Random five-fold': (0.88, 13.2, 0.78, 18.2), 'Location-grouped five-fold': (0.85, 14.2, None, None),
 'Spatial-block five-fold': (0.80, 16.6, 0.75, 19.7), 'Temporal holdout (final 20%)': (0.79, 16.7, None, None),
 'Rolling origin (2018–2026)': (0.84, 14.9, None, None), 'Leave-one-city-out': (0.01, 28.2, None, None),
 'Leave-one-group-out (station area / county)': (-0.08, 27.4, -0.38, 39.3),
 'Leave-one-group-out + future period': (-0.26, 31.0, None, None)}
MS_T2 = """Taipei 16.2 68.7 62.9 -4.65 91.0 0.969 -68.1 -67.0 -68.3 67.1 0.41
Nangang 14.8 67.2 52.8 -5.25 93.9 0.987 -66.5 -66.5 -67.9 70.8 0.64
Banqiao 13.8 43.4 41.2 -0.81 91.8 0.962 -23.2 -38.1 -39.0 39.3 0.32
Hsinchu 10.8 27.6 35.8 -0.18 96.8 0.961 -23.6 -40.2 -44.1 24.4 0.74
Miaoli 11.2 38.5 70.2 -1.13 98.9 0.996 38.2 46.0 17.5 57.9 0.75
Taichung 8.6 19.3 51.4 0.54 99.2 0.992 16.3 16.5 9.5 25.5 0.84
Tainan 7.9 17.8 44.4 0.67 99.6 0.999 9.3 -3.6 -12.2 3.7 0.89
Zuoying 13.2 18.3 61.1 0.54 97.2 0.972 -2.4 8.9 -17.1 4.6 0.80
Keelung 14.8 45.7 37.3 -0.84 98.0 0.994 44.4 66.3 44.7 52.3 0.66
Yilan 12.9 19.8 57.7 0.53 98.1 0.993 9.4 22.7 4.1 2.9 0.73
Hualien 11.4 16.9 52.2 0.37 97.2 0.989 8.9 11.3 12.8 3.3 0.66
Taitung 15.3 33.8 99.4 -0.23 75.7 0.998 30.6 40.5 18.1 34.9 0.69
Pingtung 14.8 46.0 126.6 -0.10 98.9 0.992 45.3 39.3 31.3 53.7 0.84
Songshan 16.3 67.3 62.0 -2.62 82.2 0.985 -66.5 -68.0 -63.8 60.5 0.67
Songzhu 10.6 19.4 44.8 0.27 98.5 0.945 -13.3 -26.1 -28.1 15.1 0.78
Kaohsiung 11.3 19.2 57.7 0.64 94.8 0.979 8.9 10.1 -4.5 2.1 0.81
Gangshan 13.9 37.8 102.0 0.33 95.5 0.995 36.3 34.8 16.4 50.1 0.85
Yingge 9.1 19.1 40.8 0.36 98.6 0.992 14.2 15.6 4.6 15.9 0.71
Taoyuan 11.2 20.5 45.9 0.37 76.5 0.983 -12.1 -16.0 -19.9 14.9 0.84
Changhua 22.7 74.4 156.1 -1.46 26.3 0.999 73.4 67.0 55.7 60.1 0.67
Yunlin 13.9 51.1 105.8 -1.71 97.1 0.998 51.0 48.5 29.1 59.1 0.72
Chiayi 10.2 40.7 76.6 -0.41 89.8 0.999 39.7 27.5 9.8 65.5 0.86"""
T2_COLS = ['mape_random', 'mape_losao', 'bench_mape0', 'losao_r2', 'aoa_pct', 'adv_auc', 'bias_full', 'bias_probe',
           'bias_post', 'mae_reduction', 'spearman']
MS_T3 = {  # method: MAPE at k = 0,10,30,100,300
 'Level shift': [33.5, 20.3, 20.4, 20.5, 20.4], 'EB shrinkage': [33.5, 20.2, 20.3, 20.5, 20.4],
 'Weighted retraining (w = 10)': [33.5, 19.7, 18.9, 18.2, 17.5], 'GPBoost (untuned)': [49.7, 32.8, 31.8, 29.6, 30.7],
 'F3 multiplicative correction': [39.8, 24.9, 25.0, 25.1, 24.9], 'Local mean': [40.9, 26.4, 26.4, 26.6, 27.0]}


def main():
    rows = []
    t1 = pd.read_csv(os.path.join(TAB, 'Table1.csv')).set_index('Validation design')
    for k, v in MS_T1.items():
        for j, col in enumerate(t1.columns):
            if v[j] is not None:
                rows.append(('Table 1', k, col, v[j], round(t1.loc[k, col], 3)))
    t2 = pd.read_csv(os.path.join(TAB, 'Table2_raw.csv')).set_index('station')
    for line in MS_T2.splitlines():
        p = line.split(); s = p[0]
        for col, v in zip(T2_COLS, map(float, p[1:])):
            rows.append(('Table 2', s, col, v, round(float(t2.loc[s, col]), 3)))
    r = pd.read_csv(os.path.join(TAB, 'Table3_by_station.csv'))
    agg = r.groupby(['method', 'k']).MAPE.mean()
    for m, vals in MS_T3.items():
        for k, v in zip([0, 10, 30, 100, 300], vals):
            rows.append(('Table 3', m, f'MAPE k = {k}', v, round(agg[(m, k)], 2)))
    C = pd.DataFrame(rows, columns=['table', 'row', 'column', 'manuscript', 'reproduced'])
    C['difference'] = (C.reproduced - C.manuscript).round(3)
    C.to_csv(os.path.join(TAB, 'Comparison.csv'), index=False)
    print(C.groupby(['table', 'column']).difference.agg(lambda x: np.abs(x).max()).to_string())


if __name__ == '__main__':
    main()
