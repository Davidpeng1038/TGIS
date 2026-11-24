# Code for Figures 1–4 and Tables 1–3 (GeoAI deployment-matched validation, V2.1)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24489743.svg)](https://doi.org/10.5281/zenodo.24489743)

Run everything with `python run_all.py` (Python 3.11+, about 35 minutes on a two-core machine); `python run_all.py table2` resumes from a given step.

```
pip install pandas pyarrow numpy scipy scikit-learn lightgbm gpboost matplotlib python-docx openpyxl
```

## Input files (`data/`)

| File | Source file name | Used for |
|---|---|---|
| housing_tw.csv | houseing_price_cleaned_eng_1120.csv | Taiwan transactions (all tables) |
| shops.csv | TR_shops_density_1030.csv | commercial density, 500/1000/1500 m (F0) |
| california.csv | California_Housing_Dataset.csv | California case (Table 1) |
| ca_counties_2010.geojson | 2010 Census county boundaries (California) | county assignment (prep_ca.py) |
| thsr.csv, mrt.csv, tra.csv | THSR_eng_1107.csv, MRT_eng_1029.csv, TR_eng_1028.csv | not used by these figures/tables (passenger volumes are not model features in F0–F3) |

County boundaries for California: `data/ca_counties_2010.geojson`, the California subset of the 2010 Census
cartographic boundary file for counties (`gz_2010_us_050_00_500k`). Point-in-polygon uses matplotlib only.

## Output folder

All deliverables are written to `output/`:

```
output/
├── figures/   Figure1_taxonomy, Figure2_framework, Figure3_workflow, Figure4_calibration (.png 400/300 dpi, .pdf)
├── tables/    Table1–3 (.csv), notes, per-station results, Comparison.csv
├── Figures_Tables_by_section.docx
└── Tables_1-3.xlsx
```

Intermediate files (cleaned data, out-of-sample predictions) go to `out/`.

## Scripts, in section order

| Section | Output | Script |
|---|---|---|
| 3.7 data | out/tw_clean.parquet (231,825 transactions), out/ca_clean.parquet (19,448 block groups) | `prep.py`, `prep_ca.py` |
| 3.2 | Figure 1 (taxonomy) | `fig1_taxonomy.py` |
| 3.4 | Figure 2 (framework), Figure 3 (two-phase workflow) | `fig2_framework.py`, `fig3_workflow.py` |
| 4.1–4.5 | out-of-sample predictions for every validation design | `core.py` |
| 4.1 | Table 1 | `table1.py` |
| 4.2 | Table 2 | `table2.py` |
| 4.5 | Table 3, Figure 4 | `calib.py` |
| — | manuscript vs reproduced values; output/*.docx, output/*.xlsx | `compare.py`, `make_outputs.py` |

Shared settings (feature sets F0/F2, LightGBM hyperparameters, seeds 0 and 1, metrics) are in `common.py`.

## Implementation notes

* Hub coordinates are the corrected ones of Table S2a; distance to the hub is recomputed (great-circle) and
  transactions beyond 10 km are dropped. Taipei-area transactions mislabelled as Nangang District are recoded
  to Zhongzheng District (Supplementary S11).
* Test set and split date: the final 20% of date-sorted transactions (from 6 June 2023, n = 46,365).
* California held-out counties: the 29 counties of Table S6. The county assignment here differs by a few dozen
  block groups for some counties (for example Alameda and Contra Costa), so California values can differ
  slightly from the manuscript.
* AOA (Table 2) uses a 40,000-transaction reference subsample of the training areas for nearest-neighbour
  distances; the training-DI threshold is computed across training areas (area-grouped CV).

## Citation

Peng, S.-C. (2026). *TGIS: Code and data for GeoAI deployment-matched validation (Figures 1–4, Tables 1–3)* (v1.0.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.24489744

All versions: https://doi.org/10.5281/zenodo.24489743
