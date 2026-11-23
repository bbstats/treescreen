"""Re-run only the screen (no model fits) on the real datasets, seed 0, and compare each
column's keep / reason with a saved real_data_chimera.py run. Use after a change that
should not move real data (e.g. a speed change).

    python benchmarks/replay_screen.py SAVED.jsonl
"""
import json
import sys
import time

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

import real_data_chimera as rdc
from treescreen import screen_features

saved, flags_old = {}, {}
flag_diff = []
for s in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(s)
    if r.get("dataset") and "error" not in r and r["seed"] == 0:
        saved[(r["dataset"], r["frac"])] = {t["feature"]: (t["keep"], t["reason"]) for t in r["screen_table"]}
        flags_old[(r["dataset"], r["frac"])] = {t["feature"]: t["flag"] for t in r["screen_table"]}

rb = rdc._rb()
diff, flips, total, t_all = 0, 0, 0, 0.0
for key in rdc.dataset_keys():
    rdc._COLS.clear()
    X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(1000))
    names = rdc._COLS.get("names") or [f"c{i}" for i in range(X.shape[1])]
    strat = y if task != "regression" else None
    Xtr0, _, ytr0, _ = train_test_split(X, y, test_size=0.25, random_state=0, stratify=strat)
    for frac in (1.0, 0.25):
        Xtr, ytr = (Xtr0, ytr0) if frac == 1 else rb._subsample_train(Xtr0, ytr0, frac, task)
        cat_set = set(cat or [])
        cols = {f"c{i}": pl.Series(Xtr[:, i].astype(str) if i in cat_set else Xtr[:, i].astype(np.float64))
                for i in range(X.shape[1])}
        yser = pl.Series(ytr.astype(str)) if task == "multiclass" else pl.Series(ytr)
        df = pl.DataFrame(cols).with_columns(__y__=yser)
        t = time.perf_counter()
        sc = screen_features(df, list(cols), "__y__", n_threads=2)
        t_all += time.perf_counter() - t
        now = {names[int(f[1:])]: (k, rs) for f, k, rs in sc.select("feature", "keep", "reason").iter_rows()}
        fl = {names[int(f[1:])]: g for f, g in sc.select("feature", "flag").iter_rows()}
        flag_diff += [(key, frac, c, flags_old[(key, frac)].get(c), g) for c, g in fl.items()
                      if g != flags_old[(key, frac)].get(c)]
        old = saved[(key, frac)]
        bad = [c for c in now if now[c] != old.get(c)]
        total += len(now)
        diff += len(bad)
        flips += sum(now[c][0] != (old.get(c) or (None,))[0] for c in bad)
        if bad:
            print(f"{key} frac {frac}: {len(bad)} differ, e.g. {[(c, old.get(c), now[c]) for c in bad[:8]]}")
print(f"{diff} of {total} columns differ from the saved run ({flips} change keep/drop); "
      f"screen time {t_all:.1f}s total")
print(f"flags changed: {len(flag_diff)}", flag_diff[:10])
