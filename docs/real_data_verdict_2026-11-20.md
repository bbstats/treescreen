# screen_features on real data: ChimeraBoost, Grinsztajn + high-card

Run 2026-11-20. `benchmarks/real_data_chimera.py`, results in `benchmarks/results/`
(`real_data_20261120-222916.*` = round 1 code; `-213809.*` = before the "untested" fix;
round 2 below). `benchmarks/compare_runs.py OLD NEW` lists datasets that moved.
ChimeraBoost 0.34.0, defaults, 3 seeds, same splits / row caps / 25%-rows shrink as the
external report. Dropped columns are removed before the fit. Error = RMSE (regression),
Brier (classification). ChimeraBoost is deterministic: unchanged rows reproduce exactly.

## Verdict
The screen no longer costs accuracy on real data, and it clearly removes junk rather than
signal. But it buys no speed: real datasets carry little junk, so it drops a median 6-12%
of columns and the fit takes 0.96-1.04x as long.

| stratum | datasets (any drop) | screen vs all columns | report's strict rule | random vs all | screen vs random | time |
|---|---|---|---|---|---|---|
| Grinsztajn, full | 59 (16) | 12 vs 4 (p=0.08) | 7 vs 18 (p=0.04) | 6 vs 10 | 12 vs 4 | 1.00x |
| Grinsztajn, 25% | 59 (27) | 10 vs 16 (p=0.33) | 10 vs 26 (p=0.01) | 3 vs 24 | 22 vs 4 | 1.04x |
| High-card, full | 14 (6) | 3 vs 3 | 4 vs 6 | 2 vs 4 | 4 vs 2 | 1.00x |
| High-card, 25% | 14 (11) | 8 vs 2 (p=0.11) | 7 vs 6 | 2 vs 9 | 8 vs 3 | 0.96x |

Cells: datasets better vs worse (error averaged over seeds; ties = nothing dropped, left out;
two-sided sign test). Median error change where something was dropped: -0.05%, +0.03%,
-0.02%, -0.14%.

## Fixed in this round
1. Untestable categoricals were dropped. Categories under 10 rows (or under ~5 n^(1/3) rows
   with a categorical target) are pooled; when that left one bin the column scored ic 0,
   p 1.0 and was dropped without a test. Now kept as reason "untested" unless `id_like`.
   colleges `zip` (full rows, was +0.96%) and eucalyptus `Sp`/`Abbrev` at 138 rows (was
   +3.4%) are kept; high-card went from 3 vs 4 to 3 vs 3 and from 8 vs 3 to 8 vs 2.
2. `row_order` flag fired on mostly-constant columns (ordinal ranks broke ties by row
   position). Now average ranks. Flags never change `keep`.

## Round 2 (2026-11-21): `min_ic` equivalence rule
Drop needs no hint of signal AND the data ruling out ic >= `min_ic` (0.1; one-sided 5%,
noncentral chi-square of G). Mostly-constant columns with < 5 rows per target bin outside
their modal value are kept; constant columns dropped. Applied after pair rescue, so it only
adds keeps. Results: `real_data_20261121-125807.*`, `wide_junk_20261121-125807.log`.

| stratum | screen vs all, before | after | median change after | time after |
|---|---|---|---|---|
| Grinsztajn, full | 12 vs 4 | 8 vs 5 | -0.02% | 1.02x |
| Grinsztajn, 25% | 10 vs 16 | 9 vs 16 | +0.04% | 1.04x |
| High-card, full | 3 vs 3 | 3 vs 2 | -0.00% | 1.02x |
| High-card, 25% | 8 vs 2 | 6 vs 3 | -0.02% | 1.01x |

- The biggest losses shrank: Bioresponse 25% +2.06% -> +0.61%, Mercedes 25% +0.76% -> +0.16%,
  topo_2_1 25% +0.33% -> +0.14%. Small wins shrank too (house_prices 25% -2.49% -> -0.81%,
  Moneyball, colleges). The full-row count moved on changes under 0.1% (covertype, Allstate,
  Mercedes, nyc-taxi went from tiny wins to ties).
- Net on real data: a no-op either way (median change within +-0.04%).
- Decision (maintainer, 2026-11-21): keep `min_ic=0.1` as the default.
- Synthetic wide junk (LightGBM): regression 5k rows 0.67x time, -0.4% error (was ~0.5x,
  -0.8%); regression 20k 0.52x, +0.0% (unchanged); binary 5k 0.50x, -1.2% (unchanged).
- Simulation (`min_ic` off vs on, regression target): a real column with ic 0.1 is dropped
  44% / 36% / 26% of the time at 650 / 1,300 / 2,500 rows without the rule, 1% / 1% / 6% with
  it. Pure noise is dropped at the old rate from ~5k rows (regression) / ~1.3k (binary).

## Round 3 (2026-11-21): less junk on junk-heavy data
Three changes, found with `benchmarks/junk_by_reason.py` (which rule keeps the junk):
1. Two more hint tests: Spearman p-value (monotone, 1 dof) and ic on the coarse 5 x 5 table
   (smooth shapes, 16 dof). The fine table and xi missed weak linear columns (ic ~0.04)
   that Spearman sees at z ~5. All four p-values: 2-7% of noise below 0.05.
2. `keep_q=0.5`: a column kept alone also needs a Storey q-value < 0.5 (smallest p x 4;
   pi0 = lowest of the four tests' estimates). Pair rescues need a Benjamini-Hochberg
   q < 0.5 across leftovers. Offline on the saved real-data tables: 0 of 4,125 "alone"
   columns would fail at 0.5 (first failures at 0.3).
3. Equivalence test on the coarse 5 x 5 table instead of the fine one (power to rule out
   smooth effects). Plus a fix: a pair hit for a candidate with its own signal no longer
   rescues its junk partner.
Results: `wide_junk_20261121-134106.log`, `real_data_20261121-134106.*`.

Synthetic wide junk (LightGBM, time incl. screen, error vs all columns):

| case | round 2 | round 3 | old strict rule | perfect |
|---|---|---|---|---|
| regression 5k | 0.67x, -0.4% | 0.19x, -4.8% | 0.12x, -4.5% | 0.03x, -7.6% |
| regression 20k | 0.52x, +0.0% | 0.25x, -1.4% | 0.15x, -0.9% | 0.03x, -3.3% |
| binary 5k | 0.50x, -1.2% | 0.19x, -7.5% | 0.12x, -5.3% | 0.02x, -10.1% |

Real columns kept 13-18 of 20 (the c5 x c6 pure product pair is always missed), junk 26-57
of 980. The screen now beats the strict rule's error everywhere; the strict rule is still
faster. The screen itself is now most of the time: 1.7 s of ~2.1 s at regression 5k.

Real data: Grinsztajn full 5 vs 4, 25% 10 vs 12 (was 9 vs 16), high-card 3 vs 2 and 7 vs 2;
median change within +-0.05%. Still a no-op there, now with fewer losses.

Speed follow-up (same day): the pair step got a work cap (leftovers x partners x rows
<= 1e8). It never binds on these datasets (max 8.6e7): a screen-only replay reproduced all
4,992 keep/reason values of `real_data_20261121-134106`. Synthetic end-to-end time fell to
0.12-0.15x (errors -4.6%, -1.5%, -7.3%), level with the old strict rule's speed.

## Round 4 (2026-11-21): block scoring, missingness, row sampling
Block scoring (faster; xi tie-breaking from a new random stream), a missingness hint, a
200k-row default sample, multiclass recursive scoring, sklearn checks. Results:
`real_data_20261121-144755.*`, `wide_junk_20261121-144755.log`.

- Synthetic: 0.10x / 0.10x / 0.08x time (regression 5k / 20k, binary 5k), errors -4.6% /
  -1.5% / -7.3%. Old strict rule: 0.09x / 0.14x / 0.08x, -4.5% / -0.9% / -5.3%.
- Real data: Grinsztajn full 5 vs 4, 25% 7 vs 13 (p = 0.26; round 3 was 10 vs 12), high-card
  3 vs 2 and 7 vs 2; median change -0.05% to +0.08%. Per-dataset moves vs round 3 are all
  within +-0.5% with near-equal drop counts: xi's random tie-breaking alone swings the 25%
  count by ~3 datasets. That is this benchmark's noise floor for the screen.
- Big losses kept shrinking: Bioresponse full +1.10% -> +0.33%, topo_2_1 full +0.70% -> +0.47%.

## Round 5 (2026-11-21): deterministic xi
xi on tied columns is now the exact average over every tie-break (was one random draw),
so the screen no longer depends on a random stream. `real_data_20261121-150759.*`:
Grinsztajn full 5 vs 4, 25% 10 vs 15 (p = 0.42), high-card 3 vs 2 and 7 vs 2; median change
within +-0.05%. Fewer false xi hints on tied junk, so slightly more is dropped
(Bioresponse 25%: 67 -> 88 columns, error +0.17% -> -0.22%).

## Round 6 (2026-11-21): calibrated xi p-value on tied columns
The tie-averaged xi was tested against the no-ties null variance, which made it nearly
silent on tied columns (0-6% of noise below 0.2). Now: exact permutation variance plus a
Pearson III tail (the null tends to the k-sample Cramer-von Mises law); noise below 0.2 at
18-25%, and it sees spread-only effects on 0/1 columns that Spearman misses.
`real_data_20261121-213127.*`: Grinsztajn full 4 vs 5, 25% 8 vs 15 (p = 0.21), high-card
3 vs 2 and 7 vs 2; median change within +-0.05%; time 0.99-1.06x. Every dataset that moved
drops fewer columns. Losses shrank: Bioresponse full +1.10% -> +0.34%, topo_2_1 full
+0.75% -> +0.35%, pol 25% +0.90% -> 0 (nothing dropped). The full-row count flipped on
Mercedes (-0.01% -> +0.03%). Bioresponse 25% moved -0.22% -> +0.34% while dropping fewer
columns (88 -> 71), inside this benchmark's noise (round 4: 67 columns, +0.17%).

## Round 7 (2026-11-22): product pair test
A second pair test (y vs rank(a) x rank(b), every leftover against every numeric null-free
column, 1 dof) catches pure products / XORs among many columns. `real_data_20261122-123907.*`:
Grinsztajn full 5 vs 4 (was 4 vs 5), 25% 8 vs 14 (was 8 vs 15), high-card 3 vs 2 and 7 vs 2
(unchanged); median change within +-0.05%; time 0.97-1.07x. Seven datasets moved, all
dropping fewer columns: yprop_4_1 full +0.07% -> -0.52%, topo_2_1 25% +0.11% -> -0.04%,
Bioresponse 25% +0.34% -> +0.81% (67 vs 71 columns dropped; all of it seed 0, -1.75% ->
-0.38%, still better than all columns; seeds 1-2 unchanged; per-seed swings of +-2-3% at 644
rows are this dataset's noise floor, random drop +0.5% to +4%).

## Round 8 (2026-11-22): masked test
Diagnosis of the remaining losses (ChimeraBoost, 10-20 train/test splits, the column dropped
vs shuffled; shuffling costs the same as dropping in every case, so the value is information):
- house_sales 25%, `date_month`: a weak seasonal main effect (month means 12.99-13.10 in log
  price) hidden inside grade / size / location. Against y it reads p 0.0006-0.17 (dropped on
  1 split of 10); against y minus a LightGBM fit of the other columns, p 1e-11 to 1e-19 on
  every split. Fixable.
- analcatdata_supreme 25%, `Unanimous` (drop +3.1%, worse on 15 of 20 splits): a three-way
  effect, Year x Actions_taken (non-zero on 4% of rows) x Unanimous; its cost appears only
  when both partners are in the model. Nothing sees it at 760 rows: alone, pair tests,
  recursive scorer (p 0.02-1.0), or even the LightGBM-residual test (p 0.12-0.82).
- covertype 25%: five of the six dropped soil types are all zero in train and test; the
  +0.5% (worse on 8 of 12 splits) is `Soil_Type27`, 20 ones in 9,375 rows, class split 8/12.
  LightGBM-residual test p 0.003-0.9. Not testable.
- topo_2_1 full, `oz225`: drop +0.06% (4 of 6 splits). Noise.

New leftover test ("masked"): each leftover's linear and quadratic trend against y (normal
scores of its ranks) minus an additive piecewise-linear fit of the strongest kept columns.
`real_data_20261122-195453.*`: Grinsztajn full 5 vs 4, 25% 7 vs 14 (was 8 vs 14),
high-card 3 vs 2 and 7 vs 2 (unchanged); median change within +-0.05%; time 0.93-1.07x.
Moved: Bike_Sharing 25% +0.30% -> 0.00% (`holiday` kept as masked), house_sales 25%
+0.56% -> +0.27% (reg_cat) and +0.51% -> +0.06% (reg_num), Bioresponse 25% +0.81% -> +0.72%,
kdd_ipums 25% +0.45% -> +0.34%; topo_2_1 25% -0.04% -> +0.28% with fewer columns dropped
(12.3 -> 12.0 of 255: noise; this moved the 25% count by one).

## Where it still loses
Value that lives in a few rows: analcatdata_supreme (`Unanimous`, a three-way effect on 4%
of rows, +2.8%), covertype 25% (`Soil_Type27`, 20 ones, +0.8%). No test of a column can see
these at 760-9,375 rows (see round 8). The only fix would be a policy, e.g. never drop rare
or sparse columns (cheap for histogram GBDTs), not a better test.

## Noise floor (`benchmarks/perturbation_floor.py`)
On numeric data ChimeraBoost is exactly invariant to dropping constant columns and to
column order (0.00% on Mercedes, covertype, house_sales, analcatdata). On categorical data
it is not: kdd_ipums 25% moves +6 / +1 / +4% from reordering columns alone, and +12 / -3 / +2%
from dropping one constant column. High-card single-dataset results are noise at that size.
