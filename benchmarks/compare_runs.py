"""Per-dataset screen error change (vs all columns) in two real_data runs, for rows that moved.

    python benchmarks/compare_runs.py OLD.jsonl NEW.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np


def load(path):
    g = defaultdict(list)
    for s in open(path, encoding="utf-8"):
        r = json.loads(s)
        if r.get("dataset") and "error" not in r:
            g[(r["dataset"], r["frac"])].append(r)
    out = {}
    for k, rs in g.items():
        b = np.mean([r["base"]["err"] for r in rs])
        s = np.mean([r["screen"]["err"] for r in rs])
        out[k] = (100 * (s / b - 1), np.mean([r["p"] - r["n_keep"] for r in rs]), rs[0]["p"])
    return out


old, new = load(sys.argv[1]), load(sys.argv[2])
print(f"{'dataset':55s} {'rows':>5s} {'old change':>10s} {'new change':>10s} {'dropped old -> new':>20s}")
for k in sorted(set(old) & set(new), key=lambda k: (k[1], k[0])):
    (co, do, p), (cn, dn, _) = old[k], new[k]
    if abs(co - cn) > 1e-9 or abs(do - dn) > 1e-9:
        print(f"{k[0]:55s} {'full' if k[1] == 1 else '25%':>5s} {co:+9.2f}% {cn:+9.2f}% "
              f"{do:8.1f} -> {dn:5.1f} of {p}")
