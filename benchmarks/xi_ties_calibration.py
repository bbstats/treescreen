"""xi on columns with ties in x: null calibration and power of the p-value.

"usual" = Chatterjee's no-ties null variance (what the screen used before 2026-11-21:
far too conservative on the tie-averaged xi); "exact" = the current p-value (exact
permutation variance, Pearson III tail, `_xi_tied_pvalue`). Null by permuting x.
Signal: a weak monotone effect, and a spread-only effect (y's scale depends on x).

    python benchmarks/xi_ties_calibration.py      (~3 min)
"""
import math

import numpy as np

from treescreen import core

KINDS = {
    "0/1 30%": lambda r, n: (r.random(n) < 0.3) * 1.0,
    "0/1 3%": lambda r, n: (r.random(n) < 0.03) * 1.0,
    "5 levels": lambda r, n: r.integers(0, 5, n) * 1.0,
    "50 levels": lambda r, n: r.integers(0, 50, n) * 1.0,
    "rounded normal": lambda r, n: np.round(r.normal(size=n) * 3),
    "zero-inflated 70%": lambda r, n: np.where(r.random(n) < 0.7, 0, np.round(r.normal(size=n), 1)),
}


def pvals(x, y):
    """(usual, exact) xi p-values and the Spearman p-value of x vs y."""
    n = x.size
    t = core._target_stats(y, True, None)
    avg, order, gid, starts, counts = core._ranks(x)
    lab = np.empty(n, dtype=np.intp)
    lab[order] = gid
    xi = 1 - n * core._xi_num(lab, starts.size, t) / (2 * t["den"])
    sd = math.sqrt(t["tau2"] / n)
    sp = np.corrcoef(avg, t["avg"])[0, 1]
    return (0.5 * math.erfc(xi / sd / math.sqrt(2)), core._xi_tied_pvalue(xi, sd, counts, t),
            math.erfc(abs(sp) * math.sqrt((n - 1) / 2)))


def main():
    r = np.random.default_rng(0)
    for n, reps in [(300, 1000), (2000, 500), (20000, 150)]:
        print(f"n = {n}")
        for name, gen in KINDS.items():
            null, mono, spread = [], [], []
            for _ in range(reps):
                x = gen(r, n)
                if np.unique(x).size < 2:
                    continue
                xs = (x - x.mean()) / x.std()
                e = r.normal(size=n)
                null.append(pvals(x, e))
                mono.append(pvals(x, 2.0 / math.sqrt(n) * xs + e))
                spread.append(pvals(x, e * np.exp(3.0 / math.sqrt(n) * xs)))
            a, m, s = np.array(null), np.array(mono), np.array(spread)
            share = lambda v, c: f"{np.mean(v < c):.3f}"
            print(f"  {name:18s} null <.01 {share(a[:, 0], .01)}/{share(a[:, 1], .01)}"
                  f"  <.05 {share(a[:, 0], .05)}/{share(a[:, 1], .05)}"
                  f"  <.2 {share(a[:, 0], .2)}/{share(a[:, 1], .2)}"
                  f" | power <.2 monotone {share(m[:, 0], .2)}/{share(m[:, 1], .2)} (spearman {share(m[:, 2], .2)})"
                  f"  spread {share(s[:, 0], .2)}/{share(s[:, 1], .2)} (spearman {share(s[:, 2], .2)})")
    print("cells: usual / exact")


if __name__ == "__main__":
    main()
