r"""Screen time on real-world-shaped data: numeric columns with nulls, integer counts
(ties), low- and high-cardinality strings, counts with nulls. Prints total time and the time
by column kind (scored alone through nonlinear_corr on each subset).

    python benchmarks/messy_speed.py [n_rows] [binary|numeric]
"""
import sys
import time

import numpy as np
import polars as pl

from treescreen import nonlinear_corr, screen_features

n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
kind = sys.argv[2] if len(sys.argv) > 2 else "binary"
rng = np.random.default_rng(0)
cols = {}
for i in range(120):                                   # clean floats
    cols[f"num{i}"] = rng.normal(size=n)
for i in range(80):                                    # floats with nulls, own pattern each
    v = rng.normal(size=n)
    cols[f"nul{i}"] = pl.Series(v).scatter(np.flatnonzero(rng.random(n) < rng.uniform(0.02, 0.4)), None)
for i in range(40):                                    # integer counts (ties)
    cols[f"cnt{i}"] = rng.poisson(rng.uniform(0.3, 20), size=n)
for i in range(40):                                    # low-card strings
    k = int(rng.integers(2, 40))
    cols[f"cat{i}"] = pl.Series(rng.integers(0, k, size=n).astype(str))
for i in range(10):                                    # high-card strings
    k = int(rng.integers(1_000, 20_000))
    cols[f"hic{i}"] = pl.Series(rng.integers(0, k, size=n).astype(str))
for i in range(10):                                    # counts with nulls
    v = pl.Series(rng.poisson(3, size=n))
    cols[f"cnl{i}"] = v.scatter(np.flatnonzero(rng.random(n) < 0.2), None)
df = pl.DataFrame(cols)
sig = df["num0"].to_numpy() + 0.5 * df["cnt0"].to_numpy() / 10 + (df["cat0"] == "1").to_numpy()
y = sig + rng.normal(size=n)
df = df.with_columns(y=(y > np.median(y)) if kind == "binary" else pl.Series(y))
feats = [c for c in df.columns if c != "y"]

screen_features(df.head(2000), feats, "y")              # warm up
t0 = time.perf_counter()
out = screen_features(df, feats, "y")
print(f"{n:,} rows x {len(feats)} cols, {kind}: screen {time.perf_counter() - t0:.2f} s, "
      f"kept {out['keep'].sum()}")
for pre in ["num", "nul", "cnt", "cat", "hic", "cnl"]:
    sub = [c for c in feats if c.startswith(pre)]
    t0 = time.perf_counter()
    nonlinear_corr(df, sub, "y")
    print(f"  {pre}: {len(sub):3d} cols  nonlinear_corr {time.perf_counter() - t0:.3f} s")
