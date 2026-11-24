"""prep_ca.py — California block groups -> counties (Section 3.7; Supplementary S6, S11).

Point-in-polygon assignment with the 2010 Census cartographic boundary file for counties
(gz_2010_us_050_00_500k; California subset bundled as data/ca_counties_2010.geojson). Block groups outside
every county polygon (coastal points) go to the nearest county boundary. Censored values (>= 500,000) and
missing total_bedrooms are dropped. Only numpy, pandas and matplotlib are required.
Output: out/ca_clean.parquet
"""
import os, json, numpy as np, pandas as pd
from matplotlib.path import Path

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT = os.path.join(os.path.dirname(__file__), '..', 'out')
os.makedirs(OUT, exist_ok=True)

HELD_OUT = ['Alameda', 'Butte', 'Contra Costa', 'Fresno', 'Humboldt', 'Kern', 'Los Angeles', 'Marin', 'Merced',
            'Monterey', 'Orange', 'Placer', 'Riverside', 'Sacramento', 'San Bernardino', 'San Diego', 'San Francisco',
            'San Joaquin', 'San Luis Obispo', 'San Mateo', 'Santa Barbara', 'Santa Clara', 'Santa Cruz', 'Shasta',
            'Solano', 'Sonoma', 'Stanislaus', 'Tulare', 'Ventura']


def load_counties():
    """Return list of (name, [outer rings], [hole rings]) as numpy arrays of lon/lat."""
    gj = json.load(open(os.path.join(DATA, 'ca_counties_2010.geojson'), encoding='utf-8'))
    out = []
    for f in gj['features']:
        g = f['geometry']
        polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        outer = [np.asarray(p[0]) for p in polys]
        holes = [np.asarray(h) for p in polys for h in p[1:]]
        out.append((f['properties']['NAME'], outer, holes))
    return out


def inside(pts, rings, holes):
    m = np.zeros(len(pts), bool)
    for r in rings:
        m |= Path(r).contains_points(pts)
    for h in holes:
        m &= ~Path(h).contains_points(pts)
    return m


def dist_to_rings_km(pts, rings):
    """Approximate distance (km) from points to polygon boundaries (point-to-segment, local equirectangular)."""
    best = np.full(len(pts), np.inf)
    for r in rings:
        a, b = r[:-1], r[1:]
        kx = 111.32 * np.cos(np.radians(pts[:, 1].mean())); ky = 110.57
        P = pts[:, None, :]; A = a[None]; B = b[None]
        AB = (B - A) * [kx, ky]; AP = (P - A) * [kx, ky]
        t = np.clip((AP * AB).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-12), 0, 1)
        dd = np.sqrt(((AP - t[..., None] * AB) ** 2).sum(-1)).min(1)
        best = np.minimum(best, dd)
    return best


def main():
    d = pd.read_csv(os.path.join(DATA, 'california.csv'))
    n0 = len(d)
    d = d[(d.median_house_value < 500000) & d.total_bedrooms.notna()].reset_index(drop=True)
    pts = d[['longitude', 'latitude']].to_numpy(float)
    counties = load_counties()
    county = np.array([None] * len(d), dtype=object)
    for name, rings, holes in counties:
        m = inside(pts, rings, holes) & (county == None)  # noqa: E711
        county[m] = name
    out_idx = np.where(county == None)[0]  # noqa: E711
    dist = np.full((len(out_idx), len(counties)), np.inf)
    for j, (name, rings, _) in enumerate(counties):
        dist[:, j] = dist_to_rings_km(pts[out_idx], rings)
    county[out_idx] = [counties[j][0] for j in dist.argmin(1)]
    dmin = dist.min(1)
    d['county'] = county
    d['rooms_per_hh'] = d.total_rooms / d.households
    d['bedrooms_per_room'] = d.total_bedrooms / d.total_rooms
    d['pop_per_hh'] = d.population / d.households
    d['y'] = d.median_house_value
    d['held_out'] = d.county.isin(HELD_OUT)
    d['county'] = d.county.astype('category'); d['ocean_proximity'] = d.ocean_proximity.astype('category')
    d.to_parquet(os.path.join(OUT, 'ca_clean.parquet'))
    print(f'raw {n0:,} -> {len(d):,} block groups; outside polygons {len(out_idx)} '
          f'(median {np.median(dmin):.1f} km, max {np.max(dmin):.1f} km); '
          f'held-out counties {d[d.held_out].county.nunique()} (n = {int(d.held_out.sum()):,})')


if __name__ == '__main__':
    main()
