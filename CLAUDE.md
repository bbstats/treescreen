# treescreen: handoff notes

## Goal
Sit on the Pareto front of **speed x selection quality** (keep the good columns, remove the
bad ones), with speed as the priority. Model-free: never train a model to select features.
Primary use: a pre-fit screen for gradient-boosted trees, used inside sklearn CV pipelines.

## Layout
- `src/treescreen/core.py`     all algorithms (numpy + polars only)
- `src/treescreen/sklearn.py`  `TreeScreen`, `RecursiveICSelector` (sklearn optional, lazy import)
- `src/treescreen/__main__.py` command line: `python -m treescreen DATA -t TARGET [-x COL ..]
  [-o table.csv] [--print-keep] [--classes]` (csv / tsv / parquet / feather / ndjson); also
  the `treescreen` console script in pyproject. README quick start shows its output.
- `tests/`                   41 pytest tests, ~6 s (incl. sklearn check_estimator on both)
- `.github/workflows/ci.yml`  pytest on push / PR (Ubuntu py3.10-3.14, Windows + macOS py3.13),
  plus build + `twine check`. `publish.yml`: a published GitHub release `vX.Y.Z` (must match
  `version` in pyproject) builds and uploads to PyPI by trusted publishing (no token). To
  release: bump `version`, commit, push, create the release.
- `benchmarks/wide_junk_lgbm.py`  1,000 uniform columns, 20 real; old rule vs screen vs perfect.
  Takes screen kwargs: `wide_junk_lgbm.py keep_q=0.8`. ~4 min.
- `benchmarks/junk_by_reason.py`  same data, no fits: real/junk kept by reason. ~10 s.
- `benchmarks/pure_pairs.py [seeds]`  one pure pair (product, XOR, sine x sine, U x U) among
  100 / 1,000 junk columns: share of runs keeping both. ~3 min at 10 seeds.
- `benchmarks/xi_ties_calibration.py`  xi p-values on tied columns: null shares and power,
  old (no-ties variance) vs current. ~3 min.
- `benchmarks/keep_q_offline.py`  how many real-data "alone" columns a keep_q would cut,
  from a saved real_data JSONL. Instant.
- `benchmarks/replay_screen.py SAVED.jsonl`  screen only (no fits) on all 73 datasets, seed 0,
  compared column by column with a saved run. ~2 min. Use for changes meant to be no-ops.
- `benchmarks/profile_screen.py [n] [--profile]`  wall time of the screen's parts + cProfile.
- `benchmarks/messy_speed.py [n] [binary|numeric]`  screen time on 300 mixed columns (floats
  with nulls, counts, low / high-card strings), and by column kind. ~10 s.
- `benchmarks/real_data_chimera.py`  screen vs all columns vs random drop, ChimeraBoost on
  Grinsztajn + high-card, via ChimeraBoost's own harness. ~12-17 min. `dig_losses.py` lists
  what was dropped where it hurt; `compare_runs.py OLD NEW` lists datasets that moved;
  `perturbation_floor.py` measures ChimeraBoost's error change from information-free column
  changes (constant columns dropped, column order shuffled).
- `docs/gbdt_findings_2026-11-20.md`  external test on 73 real datasets with ChimeraBoost. Read it.
- `docs/real_data_verdict_2026-11-20.md`  screen_features on the same 73 datasets. Read it.

## Environment
Machine-specific setup (interpreter, paths, git rules) lives in `CLAUDE.local.md` (gitignored):
read it first. Generic: Python >= 3.10 with polars and numpy; tests add pytest, scipy,
scikit-learn, pandas; benchmarks add lightgbm, and the real-data runs ChimeraBoost's benchmark
harness (`CHIMERA_BENCH`, default `~/Documents/chimeraboost/benchmarks`). Run from source with
`PYTHONPATH=src`. One benchmark at a time. Keep personal names and machine paths out of tracked
files: the repo is public.

## The three functions
All take a polars DataFrame / LazyFrame or a pandas DataFrame (`_inputs`: `pl.from_pandas`,
column names made strings); `feature_cols` defaults to every column but `target_col`.
Unsupported dtypes (list, struct, binary, object): skipped with a warning when the feature
list is the default, TypeError when named. All-null (`pl.Null`) columns are fine (dropped).
Fewer than 10 rows with a target value, or a single-valued target: ValueError (all three
functions; the screen used to keep or drop everything silently). Messy-input sweep
(dates, durations, times, decimals, +-inf, enums, uint64, non-ASCII names; targets with
nulls / inf / 50 classes / bool / dates): no crashes.
- `nonlinear_corr`: per feature `ic` (Miller-Madow MI on equal-frequency rank bins, mapped by
  sqrt(1-exp(-2 MI)); Williams-corrected G-test p-value), Chatterjee `xi` (ties in y: its
  tie-aware null variance, matches scipy; ties in x: averaged over every tie-break, p-value
  from the exact permutation variance + a Pearson III tail, `_xi_tied_pvalue`), `spearman`.
- `recursive_ic_score`: greedy. Score = (ic vs residual after removing picked features'
  out-of-fold binned-mean effects, rescued inside bins of likely partners) x (share of
  information surviving inside bins of its closest picked features). Bonferroni p-values.
  Class-label targets: one class-vs-rest pass per class (top 10 + pooled rest) at alpha/K,
  merged by step (`_recursive_by_class`).
- `screen_features`: keep "alone" if any of 5 p-values (fine ic, xi, Spearman, coarse 5x5
  ic, and for columns with nulls: null indicator vs y) < `keep_p` (0.2) AND Storey q
  (smallest p x tests) < `keep_q` (0.5; only bites when most columns look like junk); or
  "pair" (`pair_alpha` 0.05 Bonferroni over its pairs, AND BH q < keep_q across leftovers;
  two tests: a coarse table inside up to 64 partners' bins, and a product test, y vs
  rank(a) x rank(b), against every numeric null-free column in one matrix product,
  `_product_pairs`); or "masked" (a leftover's linear / quadratic trend vs y minus an
  additive piecewise-linear fit of the strongest kept columns, y as normal scores of its
  ranks: `_residual`, `_masked_tests`; same cut as a pair). Pair and masked tests share
  `pair_alpha` by weighted Bonferroni over the tests that ran (table 4, product 1, masked 1);
  or "untested" (categorical whose categories are all too small to test, unless `id_like`);
  or "inconclusive" (data can't rule out ic >= `min_ic` 0.1 on the coarse table, noncentral
  chi-square, exact for 1 dof; or too sparse for a chi-square). Inconclusive runs last, so
  min_ic only adds keeps. Constant columns are dropped (unless their nulls carry a test).
  Screens a 200k-row sample when larger (`max_rows`, order kept). Flags `id_like` / `row_order`.

Shared kernel: `_score_many` batches candidates into one `np.bincount`, `_table_ic` scores a
stack of [table, stratum, y, x] count tables. Numeric null-free columns are scored 64 at a
time by `_score_block` (polars ranks in parallel; one bincount per table type; xi without
sorting x; its fine bins are reused by the pair step). Numeric columns with nulls are
ranked by polars in batches (one per thread) and scored one at a time by `_score` from
those ranks, against target stats for their rows built from the full target's sort order
(no sort; cached per null pattern). Categoricals go through `_score` one at a time; string
codes are their string order (polars `rank("dense")`). High-cardinality categoricals use
ordered (CatBoost-style) target encoding after pooling categories under 10 rows. The pair
step reuses the scoring's bins for null-free numeric columns ("bins"), numeric columns
with nulls and one-bin-per-category categoricals ("pair_bins").

## Measured so far (2-core sandbox)
- `nonlinear_corr`: 1M rows x 100 features 5.7 s.
- `recursive_ic_score`: 1M x 100 11.4 s; 10k x 2,000 2.2 s. Cost grows with number of picks
  (Covertype 581k x 52 took 25 s on a laptop).
- `screen_features`: 5k x 1,000 2.2 s; 20k x 1,000 3.9 s; 6,663 x 255 0.75 s.
- Wide-junk benchmark with LightGBM (2 seeds), current defaults (2026-11-22, speed round:
  pair cap 3e7, coverage-weighted table share): screen keeps 14-20 of 20 real and 11-46 of 980
  junk; 0.06-0.09x time, error 2.0-8.0% better (regression 5k: 20/20 real, -5.3%; 20k keeps
  more junk than before, 29 / 46 vs 15 / 39: the product test's share grew there). Matches
  the old strict rule's speed (0.07-0.10x) and beats its error everywhere (strict: 11-15
  real, 0.9-5.3% better). Perfect screen: 0.02-0.03x, 3.3-10.1% better. Masked round (same
  day): 14-20 real, 13-39 junk, 0.07-0.09x, 2.1-8.2%. Product-pair round: 13-20 real, 11-40 junk,
  0.07-0.08x, 2.3-7.4%. Round 4 (2026-11-21): 13-18 real, 25-57 junk, 0.08-0.10x, 1.5-7.3%.
  Before keep_q / Spearman / coarse tests: 0.50-0.67x, 0-1.2%.
  Log: `wide_junk_20261122-speed.log`.
- Screen time on 5k x 1,000 / 20k x 1,000 (12 cores): 0.25 s / 0.44 s after the speed round
  (2026-11-22; 0.46 / 0.64 s measured just before it in the same session: row-order flag
  from block ranks -0.08 s, a missing-column check that rebuilt the name list per column
  -0.04 s, table pair cap 1e8 -> 3e7 -0.1 s). Now at 5k: scoring ~0.13 s, table pairs
  ~0.03 s, product ~0.03 s, masked ~0.02 s; at 20k scoring ~0.29 s (polars ranking 0.14 s,
  `_score_block` 0.11 s). 0.42 s / 0.59 s with the masked
  test (+0.02 / +0.04 s), 0.40 s / 0.55 s with the product pair test (+0.03 s),
  0.36 s / 0.52 s before it (0.58 / 0.96 before
  the two kernel changes below, 1.69 / 3.8 before the pair cap, 1.14 / 1.27 before block
  scoring). At 5k: scoring 0.13 s, pairs ~0.15 s, rest 0.08 s. At 20k: scoring 0.29 s,
  pairs ~0.15 s. Single-threaded at 20k: row counting (`_count_tables`, block tables +
  pairs) 0.25 s, polars ranking 0.2 s, `_score_block`'s own lines 0.14 s.
  Profile: `benchmarks/profile_screen.py [n] --profile`.
- Pair / recursive count-table kernel (`_score_many`, 2026-11-22): G by lookup of i ln i
  (`_xlogx`; no per-cell log or division, 2-3x faster `_table_ic`); the [stratum, y] margins
  computed once per call; temporaries kept under ~0.5 MB (larger numpy allocations page-fault
  on every call here, ~1 ns per element); one reused index buffer per thread; at most 4
  parallel tasks (12 threads were slower than 4: GIL contention between short numpy calls).
  Pair prep reuses the scoring's fine bins through lookup tables and no longer re-reads
  the columns that have them. 1.8-2x faster per partner at 5k-200k rows; values within
  4e-12. `replay_screen.py`: 0 of 4,992 keep/reason values changed, all 146 real screens
  8.5 s -> 6.7 s. `recursive_ic_score` 10k x 2,000 1.92 -> 1.34 s, 200k x 100 2.25 -> 1.81 s,
  same picks and order.
- Block scoring without sorting x (2026-11-22): average ranks of different tie groups differ
  by >= 1, so their integer parts (1..n) order the rows. A column without ties places y's
  ranks by that key (one scatter) for xi; one with ties gets its tie groups from a bincount
  of the key (no x order needed: `_xi_num` sorts by group label in y order). Row-by-row
  reused buffers instead of [k, n] temporaries; Spearman from A @ yc and sum A^2 (no
  centred copy). The argsort was 40% of a block. Bit-for-bit identical ic / xi / Spearman /
  p-values vs the previous commit (3k-120k rows); `replay_screen.py` 0 of 4,992 changed,
  146 real screens 6.7 -> 5.8 s.
- 1M rows x 100: full screen 3.0 s, the default 200k sample 0.60 s (0.70 s before the speed
  round; at 200k: scoring 0.44 s, of it ranking 0.2 s and `_score_block` 0.19 s, memory-bound:
  blocks of 20 -> 2 columns, i.e. more parallel tasks, gave 438 -> 430-457 ms), 2.9 / 0.64 s before the
  masked test, 2.8 / 0.61 s before the product pair test, 4.5 s / 0.9 s before the kernel
  changes). When first measured the
  sample kept the same real columns (19/20).
- Product pair test (2026-11-22, `_product_pairs`, open question 1): S = sum y * R_a * R_b
  (R = centred fine bins, y = centred target ranks; class targets: one-vs-rest indicators
  of up to 8 classes) for every leftover against every numeric null-free column, one
  float32 matrix product. p-value: exact permutation variance and skewness of S (checked
  on all 5,040 permutations, test) under three nulls (permute a; permute b; permute y,
  needed when a and b are dependent, e.g. share a spread), Pearson III tail, the largest
  p. Columns with rare values (excess kurtosis of bins > 1, e.g. 0/1 under ~17% ones) stay
  out: 3% 0/1 x 3% 0/1 pairs read 1.7-3.5x nominal below 1e-3 / 1e-4. Null calibration of
  the Bonferroni p (selection included), 300-2,000 rows, uniform / mixed / correlated /
  shared-spread columns, numeric / 10% binary / 4-class / tied targets: share below
  0.01 / 0.05 / 0.2 at 0.001-0.010 / 0.006-0.045 / 0.03-0.165. Rows subsampled when
  leftovers x columns x rows > 5e9 or columns x rows > 5e6 (1,000 columns: ~5k rows at any
  n), ~30 ms. `pure_pairs.py` (10 seeds, share of runs keeping both): 1,000 columns, 5k rows,
  pair at 1% / 3% of var(y): product 1.0 / 1.0, XOR 0.7 / 1.0, sine 0.3 / 1.0, U x U 0 / 0
  (was 0 everywhere at 1,000 columns); 100 columns, 5k: product 1.0 / 1.0 (was 0 / 0.9),
  U x U 0 / 0.7 (the table test's, unchanged). On pure-noise data the screen keeps ~2 junk
  columns per run as "pair" (500 columns, 3k rows; was 0.5-1.3): calibrated p-values, but
  BH at keep_q 0.5 lets a second family through about as often as the first.
  `replay_screen.py`: 8 of 4,992 keep/drop changes, all new keeps (Bioresponse 25% 6,
  yprop 25% 2); an equal split (x2) instead of 80/20 lost 2 table rescues (house_sales
  date_year, Higgs lepton_phi). Wide junk (LightGBM, `wide_junk_20261122-product.log`):
  c5 x c6 caught in 5 of 6 runs; regression 5k 0.08x, -4.9% (round 4: 0.10x, -4.6%);
  20k 0.08x, -2.3%, 20/20 real (0.10x, -1.5%, 18/20); binary 5k 0.07x, -7.4% (0.08x,
  -7.3%). Time ratios also include the earlier kernel speed-ups.
- Masked test (2026-11-22, `_residual` + `_masked_tests`, open question 2): each leftover's
  linear and quadratic (orthogonal) contrast of its fine bins vs r = y minus an additive
  piecewise-linear fit (per fine bin: mean + slope in rank; 2 backfitting sweeps) of the
  strongest kept columns (by ic, up to 32 and n / (8 B)). y as normal scores of its ranks
  (centred ranks made additive y non-additive: synthetic median p 0.13 vs 5e-4); class
  targets: one-vs-rest indicators of up to 8 classes. Piecewise-linear vs bin means only:
  median p 5e-9 vs 5e-4 (3 smooth strong columns + one weak). r never uses the leftover, so
  permuting it is an exact null: exact variance and skewness, Pearson III tail (`_tail_p`
  now vectorized: 67 -> 6 ms per 4,000 values, same values to 3e-14), Bonferroni over its
  2 contrasts x scores (1 contrast for 2-bin columns). Null: 6 junk kinds (uniform, 0/1 30% /
  3%, 5 levels, lognormal, zero-inflated) x 5 targets (numeric, 10% binary, 4-class, tied,
  Cauchy noise) x 300-3,000 rows: share below 0.01 / 0.05 / 0.2 at 0.002-0.017 /
  0.025-0.078 / 0.13-0.24 (means 0.009 / 0.046 / 0.186). 200k x 100 strong-signal data: 427
  junk leftovers 0.014 / 0.052 / 0.19, but BH at keep_q 0.5 lets 0-2 junk per run through as
  masked once junk pairs (which come in twos) are in the set. Rows capped at 50k and
  leftovers x rows x scores at 1e7 (1,000 leftovers: ~10k rows). Cost 5k x 1,000 +0.02 s,
  20k +0.04 s, 1M x 100 sample +0.06 s. junk_by_reason: regression 5k 17 -> 20 of 20 real
  (c13, c18, c19 masked), junk +1; binary 5k unchanged (16/20). `replay_screen.py`: 11 of
  4,992 changed, 4 keep/drop: Bike_Sharing 25% `holiday` newly kept; 3 marginal Bioresponse
  25% pair keeps lost (pair p x 1.2 from the 4:1:1 split, just over BH 0.5).
- Messy-data speed round (2026-11-22, `benchmarks/messy_speed.py`: 120 floats, 80 floats with
  nulls (own pattern each), 40 counts, 40 + 10 low / high-card strings, 10 counts with
  nulls): 50k x 300 0.76 -> 0.61 s, 200k x 300 3.26 -> 2.64 s (12 cores). Columns with nulls:
  polars ranks instead of an argsort per column, subset target stats from the full sort
  order (and r sorted without a sort), xi's tie-variance terms only when a tied column
  needs them, pair-step bins reused; `_codes` by bincount lookup instead of np.unique;
  `_ordered_encode` sorts 8/16-bit keys (radix). All bit-identical (old vs new outputs on 12
  messy cases, nulls in y too). Batching null columns 64 at a time put them on 2 threads
  and was slower (0.25 -> 0.35 s); one batch per thread fixed it. Left at 200k: ~10 ms target
  stats + ~6 ms scoring per null column (memory-bound numpy passes), `_xi_num` on tied
  counts (~9 ms per column), categorical encoding.
- Category codes (same round): string / categorical codes were polars' physical codes,
  which differ between processes (and could differ between queries in one process): ties
  between equal-count categories (`_lump`, the native-bin cut) and summation order changed
  with them, so the same data could give different pair p-values (e.g. 0.0019 vs 0.0015)
  and 1e-12 jitter run to run. Now codes are the string order (`rank("dense")`, test):
  identical output in 3 fresh processes (HEAD: 3 different). Cost: +2.7 ms per string
  column at 200k rows (polars, parallel). `replay_screen.py`: 0 of 4,992 changed; all 146
  real screens 5.7 s.
- Block scoring matches per-column scoring exactly, xi included (test). So does the null
  path (test).
- xi with ties in x (2026-11-21): exact average over every tie-break (`_xi_num`, checked
  against brute-force enumeration), so the whole screen is deterministic (categorical
  codes only since the messy-data round, see above).
- xi p-value with ties in x (2026-11-21, round 6): the averaged numerator is sum over row
  pairs of w |r_a - r_b| (w = 2/g in a group, 1/(g h) between neighbouring groups), a
  quadratic assignment statistic: exact permutation variance by Mantel's formula (checked on
  all 5,040 permutations, test). It is not normal: with G large groups it tends to the
  k-sample Cramer-von Mises law sum_j chi2_{G-1} / (j pi)^2, skewness 2.56 / sqrt(G-1)
  (measured 2.3-2.5 at G = 2, 0.84 at 10, 0.37 at 50); exact variance + normal gave 2-5% of
  noise below 0.01. Pearson III with that skewness: noise below 0.01 / 0.05 / 0.2 at
  0.000-0.027 / 0.033-0.080 / 0.18-0.25 over 0/1, levels, rounded, zero-inflated columns at
  300-20,000 rows (`xi_ties_calibration.py`); a minority group of 1-2 rows reads ~0.25 below
  0.2 (the screen keeps those as inconclusive anyway). Before, with the no-ties variance:
  0-6% below 0.2. Power below 0.2 on 0/1 columns: monotone 0.70-0.74 (was ~0, Spearman
  0.70-0.75); spread-only (y's scale depends on x) 0.69-0.86 (Spearman 0.21-0.40). Many
  small groups are less skewed than the formula says, so the tail errs conservative there.
- `_xi_num` speed (2026-11-21): the target's sort order is computed once (`t["ro"]`,
  `t["rs"]`); per tied column one radix sort of uint8/uint16 group labels per pass (3
  passes), 0/1 columns none. 3-14x faster per column than the lexsort version, identical
  values in all 18 cases checked. `replay_screen.py`, all 146 real screens: 12.4 s -> 8.5 s,
  0 of 4,992 keep/reason values changed.
- Table pair cap (`_PAIR_BUDGET` leftovers x partners x rows): 3e7 since 2026-11-22 (was 1e8;
  the largest real case is 8.6e7, porto-seguro). `replay_screen.py`: 0 of 4,992 keep/reason
  values changed at 3e7 (1e7 lost one jannis pair keep). Cost of 1e8 over 3e7: ~0.1 s per
  screen at any shape (100 x 20k: 247 vs 125 ms). What 1e8 bought: pure U x U pairs among 100
  columns at 20k rows (`pure_pairs.py` squares: 0.3 / 0.9 -> 0.0 / 0.2; 5k: 0.7 -> 0.6). With
  fewer partners the table test kept its 4/6 share of pair_alpha and two weak binary masked
  columns lost BH support; so its weight now scales with partners tried / partners wanted
  (6 of 64 at 1,000 x 5k): both back (14 / 17 real). Pure noise: rescue keeps per run 0.1-1.0
  -> 0.6-2.6 (500-1,000 columns; total junk kept 2.6-4.9 per run).
- `min_ic` simulation (regression target): a real ic-0.1 column is dropped 44/36/26% of the
  time at 650/1,300/2,500 rows with min_ic=0, 1/1/6% with 0.1. Noise is dropped at the old
  rate (~55-70%) from ~5k rows (regression) or ~1.3k (binary).
- p-values: noise flagged at 4-7% for alpha 0.05 across the cases tested.
- Real data (ChimeraBoost, 73 datasets, resolved 2026-11-20): screen vs all columns wins
  12 vs 4 on Grinsztajn full rows (strict rule was 7 vs 18), 10 vs 16 at 25% rows (was 10 vs
  26); high-card 3 vs 3 and 8 vs 2. Beats a random drop of the same size 12 vs 4 and 22 vs 4.
  No speed gain on real data: drops 6-12% of columns, 0.96-1.04x fit time.
  With `min_ic=0.1` (2026-11-21): 8 vs 5, 9 vs 16, 3 vs 2, 6 vs 3; median change within
  +-0.04%; biggest losses shrank (Bioresponse 25% +2.06% -> +0.61%, Mercedes +0.76% -> +0.16%).
  On real data the screen is a no-op either way. See the verdict doc. The maintainer kept
  `min_ic=0.1` as the default (2026-11-21): resolve doubt toward keeping.
  With keep_q / Spearman / coarse tests (round 3): 5 vs 4, 10 vs 12, 3 vs 2, 7 vs 2; still a
  no-op on real data (median change within +-0.05%).
  Round 4 (block scoring etc.): 5 vs 4, 7 vs 13, 3 vs 2, 7 vs 2; median -0.05% to +0.08%.
  The 25% count moving 10 vs 12 -> 7 vs 13 is xi tie-breaking noise (all moves <= 0.5%).
  Round 5 (exact tie-averaged xi, deterministic): 5 vs 4, 10 vs 15, 3 vs 2, 7 vs 2; median
  within +-0.05%. Slightly more junk dropped (Bioresponse 25%: 67 -> 88 columns, error +0.17%
  -> -0.22%). Real screens took 10.6 s for all 146 (was 5.2 s): `_xi_num` on tied columns
  (since sped up, see above); time ratio 1.06-1.07x where something is dropped (was
  1.01-1.03x).
  Results: `real_data_20261121-150759.*`.
  Round 6 (calibrated xi p-value on tied columns): 4 vs 5, 8 vs 15, 3 vs 2, 7 vs 2; median
  within +-0.05%; time 0.99-1.06x. Every moved dataset drops fewer columns; losses shrank
  (Bioresponse full +1.10% -> +0.34%, topo_2_1 full +0.75% -> +0.35%, pol 25% +0.90% -> 0),
  the full-row count moved on one change under 0.05% (Mercedes -0.01% -> +0.03%).
  Results: `real_data_20261121-213127.*`.
  Round 7 (product pair test, 2026-11-22): 5 vs 4, 8 vs 14, 3 vs 2, 7 vs 2; median within
  +-0.05%; time 0.97-1.07x. Seven datasets moved, all dropping fewer columns (yprop full
  +0.07% -> -0.52%; Bioresponse 25% +0.34% -> +0.81%, one seed, inside its noise).
  Results: `real_data_20261122-123907.*`.
  Round 8 (masked test, 2026-11-22): 5 vs 4, 7 vs 14, 3 vs 2, 7 vs 2; median within +-0.05%;
  time 0.93-1.07x. Bike_Sharing 25% +0.30% -> 0.00%, house_sales 25% +0.56% -> +0.27%
  (reg_cat) and +0.51% -> +0.06% (reg_num); topo_2_1 25% -0.04% -> +0.28% with fewer
  columns dropped (noise; moved the 25% count by one). Results: `real_data_20261122-195453.*`
  (resumed once after a memory kill; 438 tasks, no errors).

## Open questions, in priority order
1. **Pure pairs among many columns**: bilinear ones (product, XOR of signs, partly sine x
   sine) are now caught by the product test (c5 x c6 in 5 of 6 wide-junk runs). Still
   missed among many columns: U x U shapes (no bilinear part; only the 64-partner table
   test sees them), pairs involving columns with nulls, categoricals or rare values, and
   weak pairs at 20k+ rows on 1,000 columns (the test samples ~5k rows there to stay at
   ~30 ms; its cost grows with the rows used). Real data: a no-op (round 7 in the verdict doc).
2. **Value that lives in a few rows** (round 8 diagnosis, verdict doc): analcatdata_supreme
   `Unanimous` (three-way effect on 4% of rows, +2.8% at 25%) and covertype `Soil_Type27`
   (20 ones in 9,375 rows, +0.8%). Shuffling them costs as much as dropping them, but no
   column test sees them at those sizes, even against a LightGBM residual. house_sales dates
   (a masked main effect) are fixed by the masked test. Left: a policy question (never drop
   rare / sparse columns? cheap for histogram GBDTs). Wide-junk binary 5k still misses 3-4
   weak real columns (c12/c13/c18/c19), but at the information limit: their masked p
   (0.001-0.004) is within 2x of a test against an out-of-fold LightGBM residual of the other
   real columns; with ~960 junk leftovers, BH at 0.5 can't keep p 0.001.
3. **Speed**, now small: 0.25-0.44 s on 1,000 numeric columns, 0.6 s on a 200k-row sample,
   0.6 s / 2.6 s on 300 mixed columns at 50k / 200k rows. Scoring is now half or more
   (ranking + `_score_block`, memory-bound at large n; more parallel blocks don't help).
   Left: polars ranking, row counting (~1.7 ns per row x table), `_xi_num` on tied
   columns, per-column target stats for columns with nulls. A compiled kernel is off the
   table (numpy + polars only). A product-test U x U basis: maintainer, 2026-11-22: pure U x U
   pairs among many columns are not a real-data case; not worth work.

## Tried and rejected (2026-11-21)
- Spread test for product interactions (x vs |rank(y) - median|, coarse 5x5): no power on the
  synthetic c5 x c6 (p 0.1-0.9 even at 100k rows, same as junk); the product moves y's spread
  ~2% there. Even product + noise alone: p 0.78 at 5k rows.
- Pre-binned integer-matrix input: after block scoring, ranking costs <= 0.05 s at 5k and
  ~0.15 s at 20k on 1,000 columns; a second input path would also change xi / Spearman.
- xi order from polars rank("random") + inverse permutation: slower (the numpy scatter holds
  the GIL; numpy's sort releases it).
- Pair kernel, 2026-11-22: two candidates packed per bincount element (codes a * X + b,
  uint8): half the row work, but splitting the joint [S, Y, X, X] tables back into margins
  cost about as much (same single-thread time). Counting by one-hot matmul (BLAS, no GIL):
  not run; 2 x S x Y x X flops per row-pair (~5e10 at 5k x 1,000 vs 1e8 bincount
  increments), so it can only win on tiny tables. Fresh arrays per chunk
  without a reused buffer: the mixed-type add costs 1.5 ns per element instead of 0.84.
- `_xi_num` batched over a block's tied columns ([k, n] arrays, row-wise sorts): 2-4x faster
  at 644 rows but 1.2-2x slower from ~3k rows (2-D fancy gathers / cumsums are slower and
  the temporaries leave cache, even in 3-column batches). Saves <= ~0.2 s over all 146 real
  screens.

## Known limits (measured)
- Recursive scorer misses perfectly balanced sign-flip interactions (sign(x1*x2), 50/50 XOR).
  The screen's pair rescue does catch these.
- Pair rescue: the table test tries at most `pair_partners` (64) partners per leftover;
  the product test tries all but sees only bilinear pairs of numeric null-free columns
  without rare values. A pure U x U pair among many columns is still missed. The masked
  test covers numeric null-free leftovers only, against numeric null-free kept columns.
- Categories under 10 rows are pooled; integer-coded IDs are treated as numbers. A
  categorical pooled down to one bin is kept untested by the screen (unless `id_like`);
  `nonlinear_corr` still reports it as ic 0, p 1.
- Recursive fitted effects use raw y, so they are not outlier-proof (the scores are).
- Nulls are dropped before the value tests; the null indicator has its own test (`na_p`),
  valid when both groups have ~5 rows per coarse y bin.
- ChimeraBoost on categorical data is sensitive to column order and to dropping constant
  columns (kdd_ipums 25%: up to +12%); on numeric data it is exactly invariant. Judge
  high-card results by counts over many datasets, never one dataset.
- pandas input goes through `pl.from_pandas` (pyarrow; tested); without pyarrow, a
  per-column numpy conversion (untested). numpy input: use `TreeScreen`.

## Conventions
- Verify claims with a test or a simulation before stating them; several early guesses were wrong.
- Any new statistic needs a null-calibration check (pure-noise features, share with p < 0.05).
