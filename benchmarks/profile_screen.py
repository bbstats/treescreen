"""Where does screen_features spend its time? Synthetic wide-junk data.

    python benchmarks/profile_screen.py [n_rows] [--profile]
"""
import cProfile
import pstats
import sys
import time

import numpy as np
import polars as pl

from treescreen import screen_features, nonlinear_corr

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from junk_by_reason import make  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5000
X, y = make(n, 0)
names = [f"c{i}" for i in range(X.shape[1])]
df = pl.DataFrame(X, schema=names).with_columns(y=pl.Series(y))
screen_features(df, names, "y")                       # warm-up

for label, fn in [("screen_features", lambda: screen_features(df, names, "y")),
                  ("  nonlinear_corr part", lambda: nonlinear_corr(df, names, "y")),
                  ("  screen, no pairs", lambda: screen_features(df, names, "y", pair_partners=0))]:
    ts = []
    for _ in range(3):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    print(f"{label:24s} wall {min(ts):.2f}s (all cores)")

if "--profile" in sys.argv:
    pr = cProfile.Profile()
    pr.enable()
    screen_features(df, names, "y", n_threads=1)
    pr.disable()
    st = pstats.Stats(pr).sort_stats("tottime")
    st.print_stats(18)
