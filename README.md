# treescreen

Find the columns worth keeping before you train a gradient-boosted tree model (LightGBM,
XGBoost, CatBoost, sklearn HGB). Fast and model-free: no model is trained, every decision
is a calibrated statistical test.

## Quick start

From a shell, on a CSV / Parquet / Feather / NDJSON file:

```
python -m treescreen data.csv --target price
```

```
treescreen: data.csv: 5,000 rows x 62 feature columns, target 'price' (numeric); screened 5,000 rows in 0.14 s

KEEP 5 of 62: 5 alone
  strongest (ic): f0 0.64, f1 0.54, f2 0.18, f3 0.14
  ...
DROP 57 of 62:
    f43, f7, f44, ...

CHECK (may score well through a leak, e.g. rows sorted by the target):
    row_id: id-like, dropped
```

Useful flags: `--exclude id other_target` (leave columns out), `--out screen.csv` (full table
with p-values), `--print-keep` (kept names only, one per line), `--classes` (numeric target
holding class codes), `--max-rows 0` (screen every row; default is a 200k-row sample).

From Python, with a polars or pandas DataFrame:

```python
from treescreen import screen_features

table = screen_features(df, target_col="price")       # every other column is a candidate
keep = table.filter("keep")["feature"].to_list()
```

Inside an sklearn pipeline (the screen is refit on each training fold, so CV stays honest):

```python
from treescreen import TreeScreen
pipe = make_pipeline(TreeScreen(), LGBMRegressor())
```

The target can be numeric, boolean or class labels (multiclass works). Nulls, NaNs,
booleans, dates, strings and categoricals are handled.

## What it keeps

A column is dropped only when nothing hints that it is useful. `reason` says why a column
was kept:

| reason | meaning |
|---|---|
| `alone` | signal on its own (any shape: mutual information, Chatterjee's xi, Spearman, a coarse table, or whether it is missing) |
| `pair` | signal only together with another column (`pair_with`), e.g. a product or an XOR |
| `masked` | a weak effect of its own that shows once the stronger columns are accounted for |
| `inconclusive` | too few rows to rule out a useful effect (small data keeps nearly everything) |
| `untested` | a categorical whose categories are all too small to test |

Flags (`id_like`, `row_order`) mark columns that can score well through a leak; they never
change `keep`. Correlated copies are all kept on purpose: a tree model handles them.

## What to expect (measured)

- **Wide data with many junk columns** (1,000 columns, 20 real, 5k-20k rows): keeps 14-20 of
  the 20 real columns and 1-5% of the junk. LightGBM then fits 11-16x faster and is 2-8%
  more accurate than on all columns.
- **Typical real datasets** (73 public benchmarks, ChimeraBoost): drops 4-11% of columns,
  median accuracy change within +-0.05%, no real speed gain. Use it to clear out junk, not
  to squeeze accuracy out of clean data.
- **Speed**: 0.25 s for 5,000 rows x 1,000 columns, 0.45 s for 20,000 x 1,000, 0.6 s for
  1M x 100 (a 200k-row sample; 3 s on every row). Mixed real-world-shaped data (300
  columns: numbers with nulls, counts, strings): 0.6 s at 50,000 rows, 2.6 s at 200,000.
  Same data in, same columns out, every run.
- **Misses**: value that lives in a handful of rows (a rare category that matters only in
  combination with two other columns) can't be seen by any column test at small sizes.
  Integer-coded IDs are treated as numbers: cast them to strings.

## Knobs

All optional. `keep_p` (0.2) and `keep_q` (0.5) set how much evidence keeps a column;
`min_ic` (0.1) keeps columns the data can't yet rule out; `pair_alpha` (0.05) and
`pair_partners` (64) control the pair and masked tests (`pair_partners=0` turns them off);
`max_rows` (200,000) is the sample size. See `help(screen_features)`.

## Other functions

| function | use it for |
|---|---|
| `recursive_ic_score` / `RecursiveICSelector` | a short, non-redundant ranking (explanation, linear models); drops redundant copies too, which can cost a tree model accuracy |
| `nonlinear_corr` | one score per feature (`ic` in [0, 1], xi, Spearman, p-values), nothing dropped |

## Development

```
pip install -e ".[dev]"
pytest
python benchmarks/wide_junk_lgbm.py
```

Design notes, measurements and open questions: `CLAUDE.md`, `docs/`.
