"""prep.py — Taiwan sample construction (Section 3.7; Supplementary S2, S11).

Input : data/housing_tw.csv  (houseing_price_cleaned_eng_1120.csv)
        data/shops.csv       (TR_shops_density_1030.csv; commercial density per km2, 500/1000/1500 m)
Output: out/tw_clean.parquet
"""
import numpy as np, pandas as pd, os, re

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
os.makedirs(OUT, exist_ok=True)

# Corrected hub coordinates (Table S2a)
HUBS = {
 'Taipei': (25.0478, 121.5172), 'Nangang': (25.0523, 121.6065), 'Banqiao': (25.0143, 121.4637),
 'Taichung': (24.1091, 120.6150), 'Zuoying': (22.6870, 120.3076), 'Hsinchu': (24.8082, 121.0403),
 'Miaoli': (24.6058, 120.8253), 'Tainan': (22.9246, 120.2857), 'Keelung': (25.1316, 121.7384),
 'Yilan': (24.7546, 121.7581), 'Hualien': (23.9929, 121.6011), 'Taitung': (22.7933, 121.1231),
 'Pingtung': (22.6689, 120.4860), 'Songshan': (25.0493, 121.5778), 'Songzhu': (24.1801, 120.7019),
 'Kaohsiung': (22.6394, 120.3028), 'Gangshan': (22.7927, 120.3008), 'Yingge': (24.9545, 121.3542),
 'Taoyuan': (25.0132, 121.2149), 'Changhua': (23.8743, 120.5746), 'Yunlin': (23.7361, 120.4165),
 'Chiayi': (23.4595, 120.3233)}
# Table 2 row order
ORDER = ['Taipei','Nangang','Banqiao','Hsinchu','Miaoli','Taichung','Tainan','Zuoying','Keelung','Yilan',
         'Hualien','Taitung','Pingtung','Songshan','Songzhu','Kaohsiung','Gangshan','Yingge','Taoyuan',
         'Changhua','Yunlin','Chiayi']

def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arcsin(np.sqrt(a))

def first_floor(s):
    m = re.findall(r'\d+', str(s))
    return float(m[0]) if m else np.nan

def main():
    d = pd.read_csv(os.path.join(DATA, 'housing_tw.csv'), encoding='utf-8-sig', low_memory=False)
    n0 = len(d)
    d['station'] = d['Station Name'].str.strip().str.replace(' Station', '', regex=False)
    d['date'] = pd.to_datetime(d['Transaction Date'], format='%Y/%m/%d', errors='coerce')
    # district mislabel: Taipei station-area transactions coded as Nangang District -> Zhongzheng District
    m = (d.station == 'Taipei') & (d.District.str.strip() == 'Nangang District')
    d.loc[m, 'District'] = 'Zhongzheng District'
    d['District'] = d['District'].str.strip(); d['City'] = d['City'].str.strip()
    # filters
    d = d[(d.date >= '2012-08-01') & (d.date <= '2026-05-31')]
    d = d[d['Unit Price'] >= 10000]
    d = d[(d['Total Land Area'] > 0) & (d['Main Building Area'] > 0)]
    d = d.dropna(subset=['Latitude2', 'Longitude2'])
    hub = np.array([HUBS[s] for s in d.station])
    d['dist_km'] = haversine(d.Latitude2.values, d.Longitude2.values, hub[:, 0], hub[:, 1])
    d = d[d.dist_km <= 10].copy()
    # commercial density (per km2) for the three buffers
    sh = pd.read_csv(os.path.join(DATA, 'shops.csv'))
    sh['station'] = sh['Station Name'].str.strip().str.replace(' Station', '', regex=False)
    sh = sh.rename(columns={'Density 500m (per km2)': 'dens500', 'Density 1000m (per km2)': 'dens1000',
                            'Density 1500m (per km2)': 'dens1500'})[['station', 'dens500', 'dens1000', 'dens1500']]
    d = d.merge(sh, on='station', how='left')
    d['t'] = d.date.dt.year + (d.date.dt.dayofyear - 1) / 365.25
    d['floor'] = d['Floor'].map(first_floor)
    d = d.rename(columns={'Unit Price': 'y', 'Total Land Area': 'land_area', 'Main Building Area': 'bldg_area',
                          'Bedrooms': 'bedrooms', 'Living Rooms': 'living', 'Parking Lot': 'parking',
                          'Building Age': 'age', 'Latitude2': 'lat', 'Longitude2': 'lon'})
    keep = ['City', 'District', 'station', 'date', 't', 'y', 'land_area', 'bldg_area', 'floor', 'bedrooms',
            'living', 'parking', 'age', 'dist_km', 'dens500', 'dens1000', 'dens1500', 'lat', 'lon']
    d = d[keep].sort_values(['date']).reset_index(drop=True)
    for c in ['City', 'District', 'station']:
        d[c] = d[c].astype('category')
    d.to_parquet(os.path.join(OUT, 'tw_clean.parquet'))
    print(f'raw {n0:,} -> analysis sample {len(d):,}')
    print(d.station.value_counts().reindex(ORDER).to_string())
    return d

if __name__ == '__main__':
    main()
