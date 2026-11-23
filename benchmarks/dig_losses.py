"""Show what the screen dropped on the datasets where it hurt most.

    python benchmarks/dig_losses.py benchmarks/results/<file>.jsonl [min_loss_pct]
"""
import json
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1]
min_loss = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
recs = [json.loads(s) for s in open(path, encoding="utf-8") if s.strip()]
print("errors:", sum("error" in r for r in recs))
recs = [r for r in recs if "error" not in r and r.get("dataset")]
g = defaultdict(list)
for r in recs:
    g[(r["dataset"], r["frac"])].append(r)

for (ds, frac), rs in sorted(g.items()):
    b = np.mean([r["base"]["err"] for r in rs])
    s = np.mean([r["screen"]["err"] for r in rs])
    ch = 100 * (s / b - 1)
    if ch < min_loss:
        continue
    print(f"\n{ds} frac {frac}: {ch:+.2f}%  n_train {rs[0]['n_train']}  p {rs[0]['p']}  "
          f"per-seed {[round(100 * (r['screen']['err'] / r['base']['err'] - 1), 2) for r in rs]}")
    drops = defaultdict(int)
    info = {}
    for r in rs:
        for t in r["screen_table"]:
            if not t["keep"]:
                drops[t["feature"]] += 1
                info[t["feature"]] = t
    for f, k in sorted(drops.items(), key=lambda kv: -kv[1])[:12]:
        t = info[f]
        fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"
        print(f"   dropped in {k}/3 seeds  {f:28s} ic {fmt(t['ic'])}  ic_p {fmt(t['ic_p'])}  "
              f"xi_p {fmt(t['xi_p'])}  pair_p {fmt(t['pair_p'])}  flag {t['flag']}")
