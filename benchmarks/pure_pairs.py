"""Pure pairs (no signal alone) among junk columns: how often the screen keeps both.

y = a * g(x1) * g(x2) + noise, x uniform, a set so the pair explains `share` of
var(y); every other column is junk. Shapes: product of centred values, XOR of
signs, sine x sine (non-monotone, partly bilinear in ranks), centred squares
(U x U: no bilinear part). Cells: share of runs keeping both pair columns, and
junk kept per run (any reason).

    python benchmarks/pure_pairs.py [seeds]     (~3 min at 10)
"""
import sys
import time

import numpy as np
import polars as pl

from treescreen import screen_features

SHAPES = {
    "product": lambda u: u - 0.5,
    "xor": lambda u: np.sign(u - 0.5),
    "sine": lambda u: np.sin(2 * np.pi * u),
    "squares": lambda u: (u - 0.5) ** 2 - 1 / 12,
}


def run(n, p, shape, share, seed):
    r = np.random.default_rng(seed)
    X = r.uniform(size=(n, p))
    g = SHAPES[shape]
    f = g(X[:, 0]) * g(X[:, 1])
    f = f / f.std() * np.sqrt(share / (1 - share))
    y = f + r.normal(size=n)
    names = [f"c{i}" for i in range(p)]
    out = screen_features(pl.DataFrame(X, schema=names).with_columns(y=pl.Series(y)), names, "y")
    kept = set(out.filter("keep")["feature"])
    return {"c0", "c1"} <= kept, len(kept - {"c0", "c1"})


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t0 = time.perf_counter()
    for p in (100, 1000):
        for n in (2000, 5000, 20000):
            line = []
            for shape in SHAPES:
                for share in (0.01, 0.03):
                    res = [run(n, p, shape, share, s) for s in range(seeds)]
                    line.append(f"{shape[:4]} {share:.0%}: {np.mean([a for a, _ in res]):.1f}")
                    junk = np.mean([b for _, b in res])
            print(f"p={p:5d} n={n:6d} | " + " | ".join(line) + f" | junk kept ~{junk:.0f}", flush=True)
    print(f"cells: share of {seeds} runs keeping both pair columns ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
