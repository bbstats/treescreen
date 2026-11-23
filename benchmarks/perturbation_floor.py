"""How much does ChimeraBoost's test error move when the column set changes without
losing any information? Arms per (dataset, seed): all columns; constant columns removed
(constant in the training rows: zero information); same columns in a shuffled order.

    python benchmarks/perturbation_floor.py [dataset-substring ...] [--frac 0.25]
"""
import argparse
import sys
import zlib

import numpy as np

import real_data_chimera as rdc

ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="+")
ap.add_argument("--frac", type=float, default=0.25)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()

from sklearn.model_selection import train_test_split

rb = rdc._rb()
keys = [k for k in rdc.dataset_keys() if any(s in k for s in a.datasets)]
print(f"{'dataset':50s} seed {'consts':>6s} {'drop consts':>12s} {'shuffle order':>14s}")
for key in keys:
    for seed in range(a.seeds):
        X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(1000 + seed))
        strat = y if task != "regression" else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=strat)
        if a.frac < 1:
            Xtr, ytr = rb._subsample_train(Xtr, ytr, a.frac, task)
        cat_set = set(cat or [])
        p = X.shape[1]

        def fit(idx):
            Xa, Xb = Xtr[:, idx], Xte[:, idx]
            sub_cat = [j for j, i in enumerate(idx) if i in cat_set] or None
            if sub_cat is None and Xa.dtype == object:
                Xa, Xb = Xa.astype(np.float64), Xb.astype(np.float64)
            m = rb._run_chimera(task, Xa, ytr, Xb, yte, sub_cat, a.threads)[0]
            return m["rmse"] if task == "regression" else m["brier"]

        const = [i for i in range(p) if len(set(Xtr[:, i].tolist())) < 2]
        base = fit(list(range(p)))
        dc = fit([i for i in range(p) if i not in set(const)]) if const else base
        perm = np.random.default_rng([zlib.crc32(key.encode()), seed]).permutation(p).tolist()
        sh = fit(perm)
        print(f"{key:50s} {seed:4d} {len(const):6d} {100 * (dc / base - 1):+11.2f}% "
              f"{100 * (sh / base - 1):+13.2f}%", flush=True)
