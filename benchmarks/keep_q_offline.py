"""Offline: how many 'alone' columns would fail keep_q on the real datasets?
Uses the screen tables saved by real_data_chimera.py (seed 0). No model fits.
Columns that fail would still get the pair and inconclusive checks, so this is
an upper bound on what keep_q drops.

    python benchmarks/keep_q_offline.py RESULTS.jsonl [keep_q]
"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 2)[0] + "\\src")
from treescreen.core import _pi0, _qvalues  # noqa: E402

path = sys.argv[1]
keep_q = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
rows = []
for s in open(path, encoding="utf-8"):
    r = json.loads(s)
    if not r.get("dataset") or "error" in r or r["seed"] != 0:
        continue
    tab = [t for t in r["screen_table"] if t["ic_p"] is not None
           and not (t["ic_p"] == 1.0 and t["ic"] == 0.0)]          # ~ non-constant
    if not tab:
        continue
    hints = [[p for p in (t["ic_p"], t["xi_p"]) if p is not None] for t in tab]
    p2 = np.array([min(1.0, len(h) * min(h)) for h in hints])
    pi0 = _pi0(np.array([t["ic_p"] for t in tab]))
    q = _qvalues(p2, pi0)
    alone = [t["reason"] == "alone" for t in tab]
    fail = sum(a and qi >= keep_q for a, qi in zip(alone, q))
    rows.append((r["dataset"], r["frac"], len(tab), pi0, sum(alone), fail))

print(f"keep_q={keep_q}: datasets where some 'alone' column fails (seed 0)")
print(f"{'dataset':52s} {'rows':>5s} {'cols':>5s} {'pi0':>5s} {'alone':>6s} {'fail':>5s}")
for d, f, m, pi0, a, fl in sorted(rows, key=lambda x: -x[5]):
    if fl:
        print(f"{d:52s} {'full' if f == 1 else '25%':>5s} {m:5d} {pi0:5.2f} {a:6d} {fl:5d}")
print(f"\n{sum(1 for x in rows if x[5])} of {len(rows)} dataset-sizes affected; "
      f"{sum(x[5] for x in rows)} of {sum(x[4] for x in rows)} 'alone' columns fail")
