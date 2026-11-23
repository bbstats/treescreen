"""Why does the screen keep junk? Synthetic wide-junk data (same generator as
wide_junk_lgbm.py), no model fits: real / junk columns kept, by reason.

    python benchmarks/junk_by_reason.py [key=value screen kwargs ...]
"""
import sys
import time
from collections import Counter

import numpy as np
import polars as pl

from treescreen import screen_features


def make(n, seed, p=1000):
    rng = np.random.default_rng(seed); X = rng.uniform(size=(n, p))
    f = (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-.5)**2 + 10*X[:,3] + 5*X[:,4]
         + 8*(X[:,5]-.5)*(X[:,6]-.5)
         + sum(4*0.75**k*X[:,7+k] for k in range(7))
         + 2*(X[:,14] > .5) + 1.5*(X[:,15] > .7) + 2*np.sin(6*X[:,16]) + np.sin(9*X[:,17])
         + 0.8*X[:,18] + 0.6*X[:,19])
    return X, f + rng.normal(size=n)*2.0


def main():
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("=")
        kw[k] = float(v) if "." in v else int(v)
    print("screen kwargs:", kw)
    for n, task in [(5000, "reg"), (20000, "reg"), (5000, "binary"), (1500, "reg")]:
        X, y = make(n, 0)
        if task == "binary":
            y = (y > np.median(y)).astype(int)
        names = [f"c{i}" for i in range(X.shape[1])]
        df = pl.DataFrame(X, schema=names).with_columns(y=pl.Series(y))
        t = time.perf_counter()
        out = screen_features(df, names, "y", **kw)
        dt = time.perf_counter() - t
        real = out.filter(pl.col("feature").is_in(names[:20]))
        junk = out.filter(~pl.col("feature").is_in(names[:20]))
        rk = Counter(real.filter("keep")["reason"].to_list())
        jk = Counter(junk.filter("keep")["reason"].to_list())
        missed = sorted(real.filter(~pl.col("keep"))["feature"].to_list(), key=lambda c: int(c[1:]))
        print(f"{task} {n:>6,} rows ({dt:.1f}s): real kept {real['keep'].sum()}/20 {dict(rk)} "
              f"missed {missed} | junk kept {junk['keep'].sum()}/980 {dict(jk)}")


if __name__ == "__main__":
    main()
