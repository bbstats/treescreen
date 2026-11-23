"""
Fast, robust nonlinear correlation: Polars feature columns vs. one target.

    from treescreen import nonlinear_corr
    nonlinear_corr(df, feature_cols, target_col)

Returns one row per feature, sorted by `ic` (best first):

    feature     name of the feature column
    n           rows used (nulls / NaNs in that feature or the target dropped)
    ic          MAIN SCORE, 0..1. Rank-binned mutual information mapped to a
                correlation scale: ic = sqrt(1 - exp(-2*MI)).
                For a linear-Gaussian relationship ic ~= |pearson r|.
                Catches ANY dependence: curves, U-shapes, circles, steps,
                binary/categorical features, heteroskedastic noise.
    ic_pvalue   p-value that ic is just noise (Williams-corrected G-test).
    xi          Chatterjee's xi, ~0..1. Measures how well the target is a
                FUNCTION of the feature (1 = perfectly determined).
                Not on the same scale as ic: noisy linear r=0.5 -> xi ~= 0.15.
    xi_pvalue   p-value that xi is just noise (tie-aware asymptotic null).
                With ties in x, xi is averaged exactly over every way of
                breaking them (deterministic); its p-value is then
                conservative (never below the nominal rate on noise).
    spearman    signed rank correlation, for direction of monotone effects.

Robust:
  * Rank-based throughout -> immune to outliers, skew, infinities, and any
    monotone transform of a feature or the target (log, exp, winsorizing...).
  * Ties handled correctly (tie-aware xi + null variance; ties share a bin).
  * Binned MI is bias-corrected (Miller-Madow), so pure noise scores ~0.
  * Nulls, NaNs, constant columns, booleans, dates, strings/categoricals OK.
    High-cardinality categoricals are target-encoded without leakage, so an
    ID-like column scores ~0 instead of looking informative.
  * p-values hold ~5% false-positive rate at alpha=.05 on pure noise.

Fast:
  * Target-only work (ranks, bins, xi null variance) is done once and shared.
  * Per feature: one argsort + a few O(n) numpy passes, run on all cores.
  * No O(n^2) step anywhere.

Also here (see each docstring):
  screen_features(df, feature_cols, target_col)
      Pre-fit screen for tree models: drops a column only when nothing hints
      it is useful, alone or paired with another column, and the data are
      enough to rule out a useful effect. Keeps redundant copies.
  recursive_ic_score(df, feature_cols, target_col)
      Greedy ranking by what each feature adds beyond the ones already picked.
      A short, non-redundant list for explanation or linear models. It drops
      real columns, so it can cost a tree model accuracy.
"""

from __future__ import annotations

import functools
import math
import warnings
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np
import polars as pl

__all__ = ["nonlinear_corr", "recursive_ic_score", "screen_features"]

_MIN_N = 10
_MAX_BINS = 32
_EQUIV_BINS = 5     # bins per numeric axis in the equivalence test's table
_BLOCK_COLS = 64    # numeric columns scored together (see _corr_rows)
_BLOCK_CELLS = 4_000_000   # ... and at most this many values per block


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _default_bins(n: int) -> int:
    # ~n^(1/3) equal-frequency bins per axis -> ~n^(1/3) rows per 2-D cell,
    # which keeps bias-corrected MI stable. Capped so huge n stays fast.
    return int(min(_MAX_BINS, max(2, round(n ** (1 / 3)))))


def _chi2_sf(x: float, k: float) -> float:
    """Upper tail of chi-square(k), Wilson-Hilferty approximation (no scipy)."""
    if k <= 0 or x <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _ncx2_cdf(x: float, k: float, lam: float) -> float:
    """P(X <= x) for noncentral chi-square(k > 0, lam). k = 1 is exact,
    X = (Z + sqrt(lam))^2; otherwise Patnaik's two-moment fit,
    X ~ c * chi-square(f) (CDF within 0.02 of exact for k >= 2)."""
    if k == 1:
        if x <= 0:
            return 0.0
        a, b = math.sqrt(x), math.sqrt(lam)
        return 0.5 * (math.erfc((b - a) / math.sqrt(2)) - math.erfc((a + b) / math.sqrt(2)))
    c = (k + 2 * lam) / (k + lam)
    f = (k + lam) ** 2 / (k + 2 * lam)
    return 1.0 - _chi2_sf(x / c, f)


def _gamma_sf(x: float, a: float) -> float:
    """P(Gamma(a) > x), the regularized upper incomplete gamma Q(a, x).
    Series / Lentz continued fraction (Numerical Recipes 6.2) for a <= 50,
    Wilson-Hilferty above (no scipy)."""
    if x <= 0:
        return 1.0
    if a > 50:
        return _chi2_sf(2 * x, 2 * a)
    lg = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1:                                       # series for P(a, x)
        term = s = 1.0 / a
        for m in range(1, 1000):
            term *= x / (a + m)
            s += term
            if term < s * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(lg))
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > tiny else tiny)
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return min(1.0, math.exp(lg) * h)


# Skewness x sqrt(G - 1) of the k-sample Cramer-von Mises law sum_j chi2_{G-1} / (j pi)^2:
# 2 sqrt(2) zeta(6) / zeta(4)^1.5 (see _xi_tied_pvalue).
_CVM_SKEW = 2 * math.sqrt(2) * (math.pi**6 / 945) / (math.pi**4 / 90) ** 1.5


def _xi_tie_var(g: np.ndarray, t: dict) -> float:
    """Permutation variance (up to a constant factor) of the tie-averaged xi
    numerator for tie groups of sizes g, in x order.

    The numerator is sum over row pairs of w_ab |r_a - r_b|, w = 2/g inside a
    group of size g and 1/(g h) between neighbouring groups: a quadratic
    assignment statistic. Mantel's (1967) permutation variance, written with
    centred weights and distances so nothing cancels (t["xi_b"] holds the
    target's half)."""
    n = t["n"]
    if "xi_b" not in t:      # the target's half: centred |r_i - r_j| sums (once per target)
        u = t["rs"]
        i = np.arange(1, n + 1, dtype=np.float64)
        v = np.cumsum(u)
        P = v - u
        d = (i - 1) * u - P + (v[-1] - P - u) - (n - i) * u   # sum_j |r_i - r_j|
        B2 = 2 * n * float(((u - u.mean()) ** 2).sum()) - float(d.sum()) ** 2 / (n * (n - 1.0))
        t["xi_b"] = (B2, float(((d - d.mean()) ** 2).sum()))
    g = g.astype(np.float64)
    nb = np.full(g.size, 2.0)
    nb[[0, -1]] = 1.0                                   # neighbouring groups
    s = (2 * (g - 1) + nb) / g                          # each row's total weight
    A2 = 2 * ((2 * (g - 1) / g).sum() + (1 / (g[:-1] * g[1:])).sum()) - 4 * (n - 1) / n
    A3 = float((g * (s - 2 * (n - 1) / n) ** 2).sum())
    B2, B3 = t["xi_b"]
    n2, n3, n4 = n * (n - 1.0), n * (n - 1.0) * (n - 2), n * (n - 1.0) * (n - 2) * (n - 3)
    return 2 * A2 * B2 / n2 + 4 * (A3 - A2) * (B3 - B2) / n3 + (2 * A2 - 4 * A3) * (2 * B2 - 4 * B3) / n4


def _xi_tied_pvalue(xi: float, sd: float, g: np.ndarray, t: dict) -> float:
    """One-sided p-value of the tie-averaged xi (ties in x; groups of sizes g).

    sd: the usual (no ties in x) null sd, rescaled by the exact permutation
    variance ratio. Shape: with G large groups the null of the averaged xi
    tends to the k-sample Cramer-von Mises law, a weighted chi-square that is
    skewed (2.56 / sqrt(G - 1)); a Pearson type III (shifted gamma) with that
    skewness gives its tail. Many small groups are less skewed than that, so
    the tail errs conservative there."""
    v = _xi_tie_var(g, t)
    if v <= 0:
        return 1.0
    if "xi_v1" not in t:     # the same variance without ties in x (once per target)
        t["xi_v1"] = _xi_tie_var(np.ones(t["n"], dtype=np.int64), t)
    z = xi / (sd * math.sqrt(v / t["xi_v1"]))
    gam = _CVM_SKEW / math.sqrt(g.size - 1)
    k = 4 / gam**2
    return _gamma_sf(k + 2 * z / gam, k)


def _pi0(p: np.ndarray, lam: float = 0.5) -> float:
    """Storey's estimate of the share of true nulls among p-values."""
    return min(1.0, float((p > lam).mean()) / (1 - lam)) if p.size else 1.0


def _qvalues(p: np.ndarray, pi0: float) -> np.ndarray:
    """Storey q-values (pi0 = 1 gives Benjamini-Hochberg adjusted p-values)."""
    m = p.size
    if not m:
        return p
    order = np.argsort(p, kind="stable")
    q = pi0 * m * p[order] / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def _sample_rows(df: pl.DataFrame, m: int | None, seed: int) -> pl.DataFrame:
    """At most m rows, drawn at random without replacement, kept in their
    original order (the row-order flag depends on it)."""
    if m is None or df.height <= m:
        return df
    idx = np.sort(np.random.default_rng(seed).choice(df.height, m, replace=False))
    return df[idx]


def _clean(name: str, dtype: pl.DataType) -> tuple[pl.Expr, bool]:
    """Numeric expression for a column, plus whether its order is meaningful."""
    e = pl.col(name)
    if dtype == pl.Null:                                 # every value missing
        return e.cast(pl.Float64), True
    if dtype.is_float() or isinstance(dtype, pl.Decimal):
        return e.cast(pl.Float64).fill_nan(None), True
    if dtype == pl.Boolean:
        return e.cast(pl.Int8), True
    if dtype.is_numeric() or dtype.is_temporal():
        return e.to_physical(), True
    if dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum)):
        # Integer codes: valid as MI bins, meaningless as an order. Codes in
        # string order, not polars' physical codes: those change between
        # processes (and queries), which reorders ties between categories.
        return e.cast(pl.String).rank("dense").cast(pl.Int64), False
    raise TypeError(f"Column {name!r} has unsupported dtype {dtype}")


def _supported(dtype: pl.DataType) -> bool:
    return (dtype == pl.Null or dtype.is_numeric() or dtype.is_temporal()
            or isinstance(dtype, (pl.Decimal, pl.Categorical, pl.Enum))
            or dtype in (pl.Boolean, pl.String))


def _ranks(a: np.ndarray, order: np.ndarray | None = None):
    """Average ranks (1-based), sort order, tie-group info. One argsort, none
    when a sort order of `a` is given (ties in any order)."""
    n = a.size
    if order is None:
        order = np.argsort(a)
    s = a[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(s[1:], s[:-1], out=new[1:])
    gid = np.cumsum(new) - 1                    # tie-group id, in sorted order
    starts = np.flatnonzero(new)                # 0-based first index per group
    counts = np.diff(np.append(starts, n))
    avg = np.empty(n)
    avg[order] = (starts + (counts + 1) / 2)[gid]
    return avg, order, gid, starts, counts


def _seg_w(v: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """W per consecutive segment of v (each segment sorted ascending): the sum
    of |v_a - v_b| over its pairs, = sum (2i - g - 1) v_i (i 1-based). Exact
    for integer v while the sums stay below 2^53 (n up to ~200k rows)."""
    st = np.cumsum(sizes) - sizes
    S = np.add.reduceat(v, st)
    return 2 * (np.add.reduceat(np.arange(v.size) * v, st) - st * S) - (sizes - 1) * S


def _xi_num(lab: np.ndarray, G: int, t: dict) -> float:
    """xi's numerator, sum |r_(i+1) - r_(i)| along the x order, averaged over
    every way of breaking ties in x (exact, no randomness).

    lab: each row's tie group in x order (0..G-1). t: target stats with "ro"
    (row order sorting r) and "rs" (= r[ro]).
    Inside a tie group G of size g, a random order adds (2/g) W(G) on average,
    W(G) = sum of |r_a - r_b| over pairs in G. Between neighbouring groups G
    and H it adds C(G,H) / (g h), C = W(G u H) - W(G) - W(H). Rows taken in r
    order and stably sorted by group come out sorted inside each group, so W of
    every group (and of every union of two neighbours) needs one sort of small
    integers (numpy's radix sort up to 16 bits). Two groups need no sort.
    """
    ro, rs = t["ro"], t["rs"]
    n = rs.size
    L = lab[ro]                                         # group of each row, in r order
    g = np.bincount(L, minlength=G)
    if G == 2:                                          # 0/1: in-group positions by cumsum
        m = L.astype(bool)
        c1 = np.cumsum(m)
        i0 = np.where(m, c1 - 1, np.arange(n) - c1)     # 0-based position inside its group
        T = np.bincount(L, weights=2.0 * i0 * rs, minlength=2)
        S = np.bincount(L, weights=rs, minlength=2)
        W = T - (g - 1) * S
        W_all = 2.0 * float(np.arange(n) @ rs) - (n - 1) * float(rs.sum())
        return float(2 * W[0] / g[0] + 2 * W[1] / g[1] + (W_all - W[0] - W[1]) / (g[0] * g[1]))
    if G < 255:                                         # labels + 1 must fit too
        L, sort = L.astype(np.uint8), (lambda k: np.argsort(k, kind="stable"))
    elif G < 65535:
        L, sort = L.astype(np.uint16), (lambda k: np.argsort(k, kind="stable"))
    else:                                               # unique keys: any sort is stable
        pos = np.arange(n, dtype=np.int64)
        sort = lambda k: np.argsort(k.astype(np.int64) * n + pos)
    W = _seg_w(rs[sort(L)], g)
    total = float((2.0 * W / g).sum())
    for p in (0, 1):                                    # unions (0,1),(2,3).. then (1,2),(3,4)..
        gp = np.zeros(G + p + 1 + (G + p + 1) % 2, dtype=g.dtype)
        gp[p:p + G] = g
        sizes = gp[0::2] + gp[1::2]                     # union u = groups 2u - p, 2u - p + 1
        u = np.flatnonzero(sizes)
        Wu = _seg_w(rs[sort((L + p) >> 1)], sizes[u])
        first = 2 * u - p
        both = (first >= 0) & (first + 1 < G)
        f = first[both]
        total += float(((Wu[both] - W[f] - W[f + 1]) / (g[f] * g[f + 1])).sum())
    return total


def _entropy(counts: np.ndarray, n: int) -> float:
    p = counts[counts > 0] / n
    return float(-(p * np.log(p)).sum())


def _codes(a: np.ndarray):
    """Map arbitrary integer codes to 0..k-1 (in sorted order)."""
    if a.dtype.kind in "iu" and a.size and a.min() >= 0 and a.max() <= 4 * a.size + 1024:
        return (np.cumsum(np.bincount(a) > 0) - 1)[a]    # no sort (np.unique's result)
    _, inv = np.unique(a, return_inverse=True)
    return inv.astype(np.int64)


def _bins_from_ranks(avg: np.ndarray, n: int, B: int) -> np.ndarray:
    # Equal-frequency bins from average ranks: tied values share one bin.
    return np.minimum(((avg - 0.5) * (B / n)).astype(np.int64), B - 1)


_MIN_ENCODE_ROWS = 10   # categories smaller than this are pooled before encoding
_NATIVE_COVER = 0.9     # share of rows the "large" categories must cover to bin natively


def _ordered_encode(codes: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ordered target encoding (the CatBoost trick). Rows are put in a random
    order; each row gets the average target of the EARLIER rows of its category.
    A row never sees its own target, so the encoding cannot leak."""
    n = codes.size
    perm = rng.permutation(n)
    key = codes[perm]
    if key.max() < 65536:                                  # numpy's radix sort: same order
        key = key.astype(np.uint8 if key.max() < 256 else np.uint16)
    order = perm[np.argsort(key, kind="stable")]           # by category, random inside
    c, v = codes[order], target[order]
    first = np.empty(n, dtype=bool)
    first[0] = True
    np.not_equal(c[1:], c[:-1], out=first[1:])
    start = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    cs = np.cumsum(v)
    before = cs - v - np.concatenate(([0.0], cs[start[1:] - 1]))[gid]   # sum of earlier rows
    out = np.empty(n)
    out[order] = before / (np.arange(n) - start[gid] + 1.0)             # 1 pseudo-row at 0
    return out


def _cat_bins(codes: np.ndarray, target, B: int, max_levels: int,
              rng: np.random.Generator):
    """Bins for a categorical feature (codes 0..L-1). Returns (bins, as_number).

    Few, well-filled categories -> one bin per category (as_number=False).
        "Well-filled" = at least 5*B rows, so the count tables stay valid;
        smaller categories are pooled into one "other" bin.
    Otherwise -> ordered target encoding, rank-binned like a numeric feature
        (as_number=True). Categories under 10 rows are pooled first.
        Needs a numeric target; without one, small categories are just pooled.
    """
    n = codes.size
    cnt = np.bincount(codes)
    big = cnt >= 5 * B
    native = big.sum() <= max_levels and cnt[big].sum() >= _NATIVE_COVER * n
    if native or target is None:
        keep = np.flatnonzero(big)
        keep = keep[np.argsort(-cnt[keep], kind="stable")][:max_levels]
        if keep.size == cnt.size:
            return codes, False
        lut = np.full(cnt.size, keep.size, dtype=np.int64)
        lut[keep] = np.arange(keep.size)
        return lut[codes], False
    pooled = _codes(np.where(cnt[codes] >= _MIN_ENCODE_ROWS, codes, cnt.size))
    if pooled.max() == 0:                                # e.g. an ID column: nothing left
        return pooled, False
    enc = _ordered_encode(pooled, target, rng)
    return _bins_from_ranks(_ranks(enc)[0], n, B), True


def _margin(b: np.ndarray, n: int) -> dict:
    """Marginal stats of one binned axis (reused across many MI computations)."""
    cnt = np.bincount(b)
    nz = cnt[cnt > 0]
    return {"k": int(nz.size), "H": _entropy(nz, n), "inv": float((1.0 / nz).sum()),
            "width": int(cnt.size)}


def _ic_from_bins(bx: np.ndarray, mx: dict, by: np.ndarray, my: dict, n: int):
    """(ic, p-value, G, dof) from two binned axes: Miller-Madow MI + Williams
    G-test. G = 2 n MI (raw, uncorrected) for equivalence tests."""
    kx, ky = mx["k"], my["k"]
    if kx < 2 or ky < 2:                         # constant column
        return 0.0, 1.0, 0.0, 0
    cell = bx.astype(np.int64) * my["width"] + by
    if mx["width"] * my["width"] <= 4 * n:
        cxy = np.bincount(cell)
    else:                                        # high-cardinality categorical
        cxy = np.unique(cell, return_counts=True)[1]
    cells = int((cxy > 0).sum())
    mi_raw = mx["H"] + my["H"] - _entropy(cxy, n)
    mi = mi_raw - (cells - kx - ky + 1) / (2 * n)               # Miller-Madow
    ic = math.sqrt(1 - math.exp(-2 * max(mi, 0.0)))
    dof = (kx - 1) * (ky - 1)
    q = 1 + (n * mx["inv"] - 1) * (n * my["inv"] - 1) / (6 * n * dof)  # Williams
    return ic, _chi2_sf(2 * n * mi_raw / q, dof), max(2 * n * mi_raw, 0.0), dof


# --------------------------------------------------------------------------- #
# Target-side statistics (computed once, shared by every feature without nulls)
# --------------------------------------------------------------------------- #
def _target_stats(y: np.ndarray, ordered: bool, bins: int | None,
                  order: np.ndarray | None = None, codes: np.ndarray | None = None) -> dict:
    """order (ordered y): a sort order of y, if known; codes (class labels):
    y's codes 0..K-1, if known. Either saves a sort."""
    n = y.size
    B = bins or _default_bins(n)
    t: dict = {"n": n, "B": B, "ordered": ordered}
    if ordered:
        avg, order, gid, starts, counts = _ranks(y, order)
        by = _bins_from_ranks(avg, n, B)
        # r_i = #{y_j <= y_i},  l_i = #{y_j >= y_i}
        u = (starts + counts)[gid].astype(np.float64)     # r in y order: sorted
        r = np.empty(n)
        r[order] = u
        l = (n - starts)[gid].astype(np.float64)          # in y order; integer sums,
        den = float((l * (n - l)).sum())                  # exact in any order to ~200k rows
        # Null variance tau^2: Chatterjee (2021) eq. 2.2 (= scipy.chatterjeexi).
        i = np.arange(1, n + 1, dtype=np.float64)
        v = np.cumsum(u)
        an = ((2 * n - 2 * i + 1) * u**2).sum() / n**4
        bn = ((v + (n - i) * u) ** 2).sum() / n**5
        cn = ((2 * n - 2 * i + 1) * u).sum() / n**3
        dn = den / n**3
        t.update(avg=avg, r=r, den=den, tau2=(an - 2 * bn + cn**2) / dn**2 if dn else 0.0,
                 enc_target=avg - avg.mean(), ro=order, rs=u)
    else:
        by = _codes(y) if codes is None else codes
    # coarse target bins for the equivalence test (see _EQUIV_BINS)
    Be = min(B, _EQUIV_BINS)
    by_e = by * Be // B if ordered else by
    t.update(by=by, my=_margin(by, n), Be=Be, by_e=by_e, my_e=_margin(by_e, n))
    return t


# --------------------------------------------------------------------------- #
# One feature
# --------------------------------------------------------------------------- #
def _score(x: np.ndarray | None, x_ord: bool, t: dict, rng: np.random.Generator | None,
           max_levels: int = 32, avg: np.ndarray | None = None) -> dict:
    """One feature's statistics. Ordered x may come as its average ranks `avg`
    (1-based, ties share their mean rank) instead of values. A categorical
    binned one bin per category returns those bins as "cat_bins"."""
    n = t["n"]
    out = {"n": n, "ic": None, "ic_pvalue": None, "xi": None, "xi_pvalue": None,
           "spearman": None, "untested": False, "const": True,
           "sp_p": None, "coarse_p": None}
    if n < _MIN_N:
        return out

    if x_ord:
        if avg is None:
            avg = _ranks(x)[0]
        bx = _bins_from_ranks(avg, n, t["B"])
        # Average ranks of different tie groups differ by >= 1, so their
        # integer parts (1..n) order the groups: no sort (as in _score_block).
        key = avg.astype(np.intp)
        cnt = np.bincount(key, minlength=n + 1)
        has = cnt > 0
        counts = cnt[has]                                # tie-group sizes, in x order
        x_const = counts.size < 2
        x_num = True
    else:
        codes = _codes(x)
        bx, x_num = _cat_bins(codes, t.get("enc_target"), t["B"], max_levels, rng)
        x_const = bx.max() == 0
        # 2+ categories, but every one was too small to keep apart: the score
        # below is 0 because nothing could be tested, not because nothing is there
        out["untested"] = bool(x_const and codes.max() > 0)
        if not x_num:
            out["cat_bins"] = bx
    out["const"] = bool(x_const)

    # ---- ic: bias-corrected mutual information on rank bins --------------
    out["ic"], out["ic_pvalue"], _, _ = _ic_from_bins(bx, _margin(bx, n), t["by"], t["my"], n)

    # ---- G on a coarse table, for screen_features' equivalence test --------
    # Few degrees of freedom = power to rule out a smooth effect. Also the rows
    # outside the most common bin and the target bins: a chi-square
    # approximation needs ~5 expected rows per cell, so a mostly-constant
    # column below that can't be judged by it.
    bxe = bx * t["Be"] // t["B"] if x_num else bx
    _, out["coarse_p"], out["g"], out["dof"] = _ic_from_bins(
        bxe, _margin(bxe, n), t["by_e"], t["my_e"], n)
    out["n_rest"], out["ky"] = n - int(np.bincount(bxe).max()), t["my_e"]["k"]
    if out["n_rest"] < 5 * out["ky"]:                  # chi-square p-value not valid
        out["coarse_p"] = None

    # ---- xi (tie-aware) + spearman ----------------------------------------
    if x_ord and t["ordered"] and t["den"] > 0:
        if x_const:
            out.update(xi=0.0, xi_pvalue=1.0)
        else:
            tied = counts.size < n
            if tied:                 # ties in x: exact average over every tie-break
                lab = (np.cumsum(has) - 1)[key]          # tie group of each row
                num = _xi_num(lab, counts.size, t)
            else:
                r_x = np.empty(n + 1)
                r_x[key] = t["r"]                        # y's ranks in x order
                num = float(np.abs(np.diff(r_x[1:])).sum())
            xi = 1 - n * num / (2 * t["den"])
            sd = math.sqrt(max(t["tau2"], 1e-300) / n)
            out.update(xi=xi, xi_pvalue=_xi_tied_pvalue(xi, sd, counts, t) if tied
                       else 0.5 * math.erfc(xi / sd / math.sqrt(2)))
            sp = np.corrcoef(avg, t["avg"])[0, 1]
            out["spearman"] = None if np.isnan(sp) else float(sp)
            if out["spearman"] is not None:              # null: N(0, 1/(n-1))
                out["sp_p"] = math.erfc(abs(sp) * math.sqrt((n - 1) / 2))
    return out


def _count_tables(codes: np.ndarray, base: np.ndarray, T: int, which=None,
                  base_off: np.ndarray | None = None) -> np.ndarray:
    """Count tables [rows, T]: each row of codes (or codes[which]) added to the
    per-row cell offsets `base` (cells < T), counted. Many rows per bincount,
    through a reused per-thread index buffer (see _BC_CELLS)."""
    n = codes.shape[1]
    m_all = codes.shape[0] if which is None else len(which)
    per = max(1, min(m_all, _BC_CELLS // n))
    if base_off is None or base_off.shape[0] < per:
        base_off = base[None, :] + (np.arange(per, dtype=np.intp) * T)[:, None]
    buf = getattr(_tls, "buf", None)
    if buf is None or buf.size < per * n:
        buf = _tls.buf = np.empty(per * n, dtype=np.intp)
    t = np.empty((m_all, T), dtype=np.intp)
    for i in range(0, m_all, per):
        m = min(per, m_all - i)
        rows = codes[i:i + m] if which is None else codes[which[i:i + m]]
        idx = buf[:m * n]
        np.add(base_off[:m], rows, out=idx.reshape(m, n))
        t[i:i + m] = np.bincount(idx, minlength=m * T).reshape(m, T)
    return t


def _block_tables(codes: np.ndarray, by: np.ndarray, X: int, n: int):
    """(ic, p, G, dof, n_sx) for each row of codes [k, n] (bins 0..X-1) against
    y bins `by`."""
    k = codes.shape[0]
    Y = int(by.max()) + 1
    t = _count_tables(codes, by.astype(np.intp) * X, Y * X).reshape(k, 1, Y, X)
    ic, p, g, dof = _table_ic(t, n, full=True)
    return ic, p, g, dof, t.sum(2)[:, 0, :]


def _score_block(A: np.ndarray, t: dict):
    """_score for many numeric columns without nulls at once.

    A: [k, n] average ranks (1-based, tied values share their mean rank), one
    row per column, all scored against the shared target stats `t`.
    Returns (rows, fine bins [k, n] uint8). Same statistics as _score.
    """
    k, n = A.shape
    B, Be = t["B"], t["Be"]
    # Row-by-row through reused buffers: block-sized temporaries page-fault.
    bx = np.empty((k, n), dtype=np.uint8)            # rank bins
    f = np.empty(n)
    for j in range(k):
        np.subtract(A[j], 0.5, out=f)
        f *= B / n                                   # (A - 0.5) B / n < B: no clipping
        bx[j] = f                                    # truncates, like astype
    const = A.min(1) == A.max(1)
    m = A.mean(1)
    ss = np.einsum("ij,ij->i", A, A) - n * m * m         # sum (A - m)^2, no [k, n] temporary
    pos = np.arange(n) - (n - 1) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        row_rho = (A @ pos) / np.sqrt(ss * (pos @ pos))   # rank correlation with row order
    row_rho[const] = 0.0
    ic, ic_p, _, _, _ = _block_tables(bx, t["by"], B, n)
    to_e = (np.arange(B) * Be // B).astype(np.uint8)
    _, coarse_p, g, dof, n_sx = _block_tables(to_e[bx], t["by_e"], Be, n)
    n_rest = n - n_sx.max(1)
    ky = t["my_e"]["k"]

    xi = xi_p = sp = sp_p = None
    if t["ordered"] and t["den"] > 0:
        # No sort: average ranks of different tie groups differ by >= 1, so
        # their integer parts (1..n) order the rows; counting them gives the
        # tie groups, and without ties they place y's ranks in x order.
        num = np.zeros(k)
        sizes = {}
        key, seen = np.empty(n, dtype=np.intp), np.empty(n + 1, dtype=bool)
        r_x, d = np.empty(n + 1), np.empty(n - 1)
        for j in np.flatnonzero(~const):
            np.copyto(key, A[j], casting="unsafe")       # integer part, 1..n
            seen[:] = False
            seen[key] = True
            if np.count_nonzero(seen) == n:              # no ties
                r_x[key] = t["r"]                        # y's ranks in x order
                np.subtract(r_x[2:], r_x[1:-1], out=d)
                np.abs(d, out=d)
                num[j] = d.sum()
            else:                    # ties: exact average over tie-breaks
                cnt = np.bincount(key, minlength=n + 1)
                has = cnt > 0
                lab = (np.cumsum(has) - 1)[key]          # tie group, in x order
                sizes[j] = cnt[has]
                num[j] = _xi_num(lab, sizes[j].size, t)
        xi = 1 - n * num / (2 * t["den"])
        sd = math.sqrt(max(t["tau2"], 1e-300) / n)
        xi_p = 0.5 * _erfc(xi / sd / math.sqrt(2)).astype(np.float64)
        for j, gj in sizes.items():
            xi_p[j] = _xi_tied_pvalue(float(xi[j]), sd, gj, t)
        xi[const], xi_p[const] = 0.0, 1.0
        yc = t["avg"] - t["avg"].mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            sp = (A @ yc - m * yc.sum()) / np.sqrt(ss * (yc @ yc))
        sp[const] = np.nan           # ss is a rounding residue there, not 0
        sp_p = _erfc(np.abs(np.nan_to_num(sp)) * math.sqrt((n - 1) / 2)).astype(np.float64)

    rows = []
    for j in range(k):
        s_j = None if sp is None or not np.isfinite(sp[j]) else float(sp[j])
        rows.append({
            "n": n, "ic": float(ic[j]), "ic_pvalue": float(ic_p[j]),
            "xi": None if xi is None else float(xi[j]),
            "xi_pvalue": None if xi_p is None else float(xi_p[j]),
            "spearman": s_j, "sp_p": None if s_j is None else float(sp_p[j]),
            "untested": False, "const": bool(const[j]),
            "coarse_p": float(coarse_p[j]) if n_rest[j] >= 5 * ky else None,
            "g": float(g[j]), "dof": int(dof[j]), "n_rest": int(n_rest[j]), "ky": ky,
            "row_rho": float(row_rho[j])})
    return rows, bx


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _inputs(df, feature_cols, target_col):
    """(polars DataFrame / LazyFrame, feature names, target name) from polars or
    pandas input. feature_cols None = every column except the target."""
    if target_col is None:
        raise TypeError("target_col is required, e.g. screen_features(df, target_col='y'); "
                        "feature_cols defaults to every other column")
    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        names = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
    elif hasattr(df, "iloc") and hasattr(df, "columns"):            # pandas
        names = [str(c) for c in df.columns]
        if len(set(names)) < len(names):
            raise ValueError("duplicate column names")
        try:
            out = pl.from_pandas(df.reset_index(drop=True))
        except Exception:                                           # e.g. no pyarrow
            out = pl.DataFrame({n: df.iloc[:, i].to_numpy() for i, n in enumerate(names)})
        out.columns = names
        df, target_col = out, str(target_col)
        feature_cols = None if feature_cols is None else (
            [feature_cols] if isinstance(feature_cols, str) else [str(c) for c in feature_cols])
    else:
        raise TypeError(f"df must be a polars DataFrame / LazyFrame or a pandas DataFrame, "
                        f"not {type(df).__name__}. For numpy X, y use treescreen.TreeScreen, or "
                        f"pl.DataFrame(X).with_columns(y=pl.Series(y)).")
    if target_col not in names:
        raise KeyError(f"target column {target_col!r} not found")
    schema = df.collect_schema()
    if not _supported(schema[target_col]):
        raise TypeError(f"target column {target_col!r} has unsupported dtype {schema[target_col]}")
    default = feature_cols is None
    if default:
        feature_cols = [c for c in names if c != target_col]
    elif isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    bad = [c for c in feature_cols if c in schema and not _supported(schema[c])]
    if bad:
        what = ", ".join(f"{c} ({schema[c]})" for c in bad)
        if not default:
            raise TypeError(f"unsupported column types (lists, structs, binary, objects): {what}. "
                            f"Leave them out of feature_cols.")
        warnings.warn(f"left out, unsupported types: {what}", stacklevel=3)
        feature_cols = [c for c in feature_cols if c not in set(bad)]
    return df, list(feature_cols), target_col


def nonlinear_corr(
    df,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    *,
    bins: int | None = None,
    max_levels: int = 32,
    seed: int = 0,
    n_threads: int | None = None,
) -> pl.DataFrame:
    """Score every feature in `feature_cols` against `target_col`.

    bins:      equal-frequency bins per axis for `ic`. Default ~n^(1/3),
               clipped to [2, 32]. Raise it for huge n with very wiggly shapes.
    max_levels: string / categorical features with at most this many
               well-filled categories get one bin per category. Above that
               they are target-encoded (each row sees only the average target
               of earlier rows of its category) and scored like a number.
    seed:      seed for the categorical encoding order (results are
               deterministic).
    n_threads: worker threads (default: all cores).
    """
    df, feature_cols, target_col = _inputs(df, feature_cols, target_col)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    rows = _corr_rows(lf, feature_cols, target_col, bins, max_levels, seed, n_threads)
    schema_out = {"feature": pl.String, "n": pl.Int64, "ic": pl.Float64,
                  "ic_pvalue": pl.Float64, "xi": pl.Float64, "xi_pvalue": pl.Float64,
                  "spearman": pl.Float64}
    rows = [{k: r.get(k) for k in schema_out} for r in rows]
    return (pl.DataFrame(rows, schema=schema_out, strict=False)
            .sort("ic", descending=True, nulls_last=True))


def _corr_rows(lf: pl.LazyFrame, feature_cols, target_col, bins, max_levels, seed,
               n_threads) -> list[dict]:
    """nonlinear_corr's rows, unsorted, plus the internal `untested` field."""
    schema = lf.collect_schema()
    feature_cols = [c for c in dict.fromkeys(feature_cols) if c != target_col]
    missing = [c for c in [*feature_cols, target_col] if c not in schema]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    y_expr, y_ord = _clean(target_col, schema[target_col])
    exprs, ordered = [y_expr.alias("__y__")], {}
    for c in feature_cols:
        e, ordered[c] = _clean(c, schema[c])
        exprs.append(e.alias(c))
    clean = lf.select(exprs).filter(pl.col("__y__").is_not_null()).collect()

    y = clean["__y__"].to_numpy()
    if y.size < _MIN_N:
        raise ValueError(f"only {y.size} rows have a target value; at least {_MIN_N} are needed")
    if y.min() == y.max():
        raise ValueError(f"the target {target_col!r} has a single value: nothing to screen against")
    shared = _target_stats(y, y_ord, bins)

    n = y.size
    B = shared["B"]
    cache: dict = {}          # null pattern -> target stats (columns often share one)
    if y_ord:
        ro = shared["ro"]
    else:
        n_cls = int(shared["by"].max()) + 1

    def target_for(keep: np.ndarray) -> dict:
        key = hash(keep.tobytes())
        hit = cache.get(key)
        if hit is not None and np.array_equal(hit[0], keep):
            return hit[1]
        # the subset's sort order (or codes) from the full target's: no sort
        if y_ord:
            st = _target_stats(y[keep], True, bins, order=(np.cumsum(keep) - 1)[ro[keep[ro]]])
        else:
            by = shared["by"][keep]
            lut = np.cumsum(np.bincount(by, minlength=n_cls) > 0) - 1
            st = _target_stats(y[keep], False, bins, codes=lut[by])
        cache[key] = (keep, st)
        return st

    def with_nulls(c: str, keep: np.ndarray, x, rng, avg=None) -> dict:
        """Score on the rows where c has a value, plus the null indicator test."""
        m = int(keep.sum())
        out = ({"n": m} if m < _MIN_N else
               _score(x, ordered[c], target_for(keep), rng, max_levels, avg))
        # does being missing predict y? (null indicator vs coarse y bins; needs
        # ~5 expected rows per cell in both groups for the chi-square p-value)
        if min(m, n - m) >= 5 * shared["my_e"]["k"]:
            na = (~keep).astype(np.int64)
            out["na_p"] = _ic_from_bins(na, _margin(na, n), shared["by_e"],
                                        shared["my_e"], n)[1]
        return out

    def work(job):                     # categoricals, one at a time
        k, c = job
        s = clean[c]
        rng = np.random.default_rng([seed, k])
        if s.null_count() == 0:
            out = _score(s.to_numpy(), ordered[c], shared, rng, max_levels)
            b = out.pop("cat_bins", None)
            if b is not None:          # one bin per category: reused by the pair step
                out["pair_bins"] = (_compact(b), False)
            return out
        keep = s.is_not_null().to_numpy()
        out = with_nulls(c, keep, s.drop_nulls().to_numpy(), rng)   # integer codes
        out.pop("cat_bins", None)      # nulls dropped here, a category there
        return out

    # Numeric columns are ranked by polars in parallel, a block at a time.
    # Without nulls each statistic is one array operation per block.
    num = [c for c in feature_cols if ordered[c]] if n >= _MIN_N else []
    block = [c for c in num if clean[c].null_count() == 0]
    in_block = set(block)
    gappy = [c for c in num if c not in in_block]
    threads = n_threads or os.cpu_count() or 1
    size = max(1, min(_BLOCK_COLS, _BLOCK_CELLS // max(n, 1)))
    chunks = [(True, block[i:i + size]) for i in range(0, len(block), size)]
    # columns with nulls are scored one by one after ranking: spread them out
    g_size = max(1, min(size, -(-len(gappy) // threads)))
    chunks += [(False, gappy[i:i + g_size]) for i in range(0, len(gappy), g_size)]

    def block_work(job):
        full, cols = job
        A = clean.select([pl.col(c).rank("average").cast(pl.Float64) for c in cols])
        A = np.ascontiguousarray(A.to_numpy().T)       # nulls: NaN
        if full:
            rows, bx = _score_block(A, shared)
            for r, b in zip(rows, bx):
                r["bins"] = b              # fine bins, reused by screen_features' pair step
            return dict(zip(cols, rows))
        out = {}
        for c, a in zip(cols, A):
            keep = ~np.isnan(a)
            avg = a[keep]
            out[c] = r = with_nulls(c, keep, None, None, avg)
            # the pair step's bins: B rank bins of the values, nulls -> B
            pb = np.full(n, B, dtype=np.uint8)
            if avg.size:
                pb[keep] = _bins_from_ranks(avg, avg.size, B)
            r["pair_bins"] = (pb, True)
        return out

    in_num = set(num)
    rest = [(k, c) for k, c in enumerate(feature_cols) if c not in in_num]
    with ThreadPoolExecutor(threads) as ex:          # everything queued at once
        parts = [ex.submit(block_work, j) for j in chunks]
        singles = [ex.submit(work, j) for j in rest]
        scored = {c: r for f in parts for c, r in f.result().items()}
        scored.update((c, f.result()) for (_, c), f in zip(rest, singles))

    return [{"feature": c, **scored[c]} for c in feature_cols]



# --------------------------------------------------------------------------- #
# Recursive ic score: greedy, redundancy-aware feature ranking
# --------------------------------------------------------------------------- #
_NEIGHBORS = 3      # picked features re-checked per candidate, on each side
_DUP_SHARE = 0.5    # if less than this share of a candidate's information survives
                    # inside a picked feature's bins, the p-value comes from that test
_TAIL_P = 0.5       # a pick whose adjusted p exceeds this looks like noise
_TAIL_PATIENCE = 3  # after this many noise-level picks in a row, the rest is
                    # ranked in one shot instead of step by step


def _lump(codes: np.ndarray, k: int) -> np.ndarray:
    """Keep the k most frequent codes (renumbered 0..k-1); pool the rest as k."""
    cnt = np.bincount(codes)
    if cnt.size <= k + 1:
        return codes
    lut = np.full(cnt.size, k, dtype=np.int64)
    lut[np.argsort(-cnt, kind="stable")[:k]] = np.arange(k)
    return lut[codes]


def _cat_codes(s: pl.Series) -> np.ndarray:
    """Category codes 0..L-1; null is its own category."""
    a = s.to_numpy()
    if not s.null_count():
        return _codes(a)
    ok = ~s.is_null().to_numpy()
    out = np.zeros(s.len(), dtype=np.int64)
    if ok.any():
        out[ok] = _codes(s.drop_nulls().to_numpy())     # integers, not floats with NaN
        out[~ok] = out[ok].max() + 1
    return out


def _bin_feature(s: pl.Series, ordered: bool, n_bins: int) -> np.ndarray:
    """Integer bins for one feature. Ordered: equal-frequency rank bins
    0..n_bins-1, nulls -> n_bins. Unordered: one bin per category (null too)."""
    if not ordered:
        return _cat_codes(s)
    n = s.len()
    if not s.null_count():
        return _bins_from_ranks(_ranks(s.to_numpy())[0], n, n_bins)
    ok = ~s.is_null().to_numpy()
    out = np.full(n, n_bins, dtype=np.int64)
    if ok.any():
        v = s.to_numpy()[ok]
        out[ok] = _bins_from_ranks(_ranks(v)[0], v.size, n_bins)
    return out


def _compact(b: np.ndarray) -> np.ndarray:
    hi = int(b.max())
    return b.astype(np.uint8 if hi < 256 else np.uint16 if hi < 65536 else np.int32)


def _oof_bin_means(r: np.ndarray, b: np.ndarray, folds: np.ndarray, K: int,
                   shrink: float) -> np.ndarray:
    """Fitted effect E[r | bin] - E[r], computed out-of-fold, tiny bins shrunk
    to zero. Only the deviation from the mean is fitted: fitting a per-fold
    intercept as well would amplify fold-to-fold drift a little at every step."""
    W = int(b.max()) + 1
    idx = folds * W + b
    s = np.bincount(idx, weights=r, minlength=K * W).reshape(K, W)
    c = np.bincount(idx, minlength=K * W).reshape(K, W).astype(np.float64)
    s_out, c_out = s.sum(0) - s, c.sum(0) - c            # each fold sees other folds only
    mu = s_out.sum(1, keepdims=True) / np.maximum(c_out.sum(1, keepdims=True), 1.0)
    shrink = max(shrink, 1e-9)                           # avoid 0/0 in empty bins
    pred = (s_out + shrink * mu) / (c_out + shrink) - mu
    return pred[folds, b]


_erfc = np.frompyfunc(math.erfc, 1, 1)


def _chi2_sf_many(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Vectorized `_chi2_sf`."""
    out = np.ones(x.shape)
    ok = (k > 0) & (x > 0)
    kk = k[ok].astype(np.float64)
    z = ((x[ok] / kk) ** (1 / 3) - (1 - 2 / (9 * kk))) / np.sqrt(2 / (9 * kk))
    out[ok] = 0.5 * _erfc(z / math.sqrt(2)).astype(np.float64)
    return out


@functools.lru_cache(maxsize=8)
def _xlogx(n: int) -> np.ndarray:
    """i ln i for i = 0..n (0 ln 0 = 0): G-statistics by lookup."""
    i = np.arange(n + 1, dtype=np.float64)
    i[0] = 1.0
    return i * np.log(i)


def _table_ic(t: np.ndarray, n: int, full: bool = False, n_sy: np.ndarray | None = None):
    """ic and p-value of X vs Y inside strata, for a stack of count tables.

    t (integer counts) has shape [table, stratum, y, x]; one stratum = plain
    (unconditional) ic. Miller-Madow bias correction; G-test with Williams'
    correction per stratum. n_sy: the [stratum, y] margins when every table
    shares them (saves recomputing). Returns (ic, p), one entry per table;
    full=True adds raw G and its dof.
    G = 2 (sum t ln t - sum n_sx ln n_sx - sum n_sy ln n_sy + n_s ln n_s), by
    lookup: no per-cell log or division.
    """
    k, S, Y, X = t.shape
    f = _xlogx(n)
    n_sx = t[:, :, 0, :].copy()                  # a loop: numpy's middle-axis sum is slow
    for i in range(1, Y):
        n_sx += t[:, :, i, :]
    if n_sy is None:
        n_sy = t.sum(3)
    n_s = n_sy.sum(-1)
    kx, ky = np.count_nonzero(n_sx, axis=-1), np.count_nonzero(n_sy, axis=-1)
    dof_s = np.maximum(kx - 1, 0) * np.maximum(ky - 1, 0)
    dof = dof_s.sum(1)
    g = 2 * (f[t.reshape(k, S, Y * X)].sum(2) - f[n_sx].sum(2) - f[n_sy].sum(-1) + f[n_s])
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_x = np.where(n_sx > 0, 1.0 / n_sx, 0.0).sum(-1)
        inv_y = np.where(n_sy > 0, 1.0 / n_sy, 0.0).sum(-1)
        q = 1 + (n_s * inv_x - 1) * (n_s * inv_y - 1) / (6 * n_s * dof_s)
        g_adj = np.where(dof_s > 0, g / q, 0.0).sum(1)
    cells = np.count_nonzero(t.reshape(k, -1), axis=1)
    rest = kx.sum(1) + ky.sum(-1) - np.count_nonzero(n_s, axis=-1)
    mi = g.sum(1) / (2 * n) - (cells - rest) / (2 * n)
    ic = np.sqrt(1 - np.exp(-2 * np.maximum(mi, 0.0)))
    ic[dof == 0] = 0.0
    if full:
        return ic, _chi2_sf_many(g_adj, dof), np.maximum(g.sum(1), 0.0), dof
    return ic, _chi2_sf_many(g_adj, dof)


# _score_many sizes (measured on 12 logical cores, Windows): numpy temporaries
# over ~0.5 MB cost page faults on every allocation, and more than ~4 threads
# of short numpy calls only contend for the GIL.
_TASK_CELLS = 1 << 16       # table cells scored per _table_ic call
_BC_CELLS = 1 << 18         # rows x candidates counted per bincount (reused buffer)
_MAX_TASKS = 4
_tls = threading.local()    # per-thread index buffer


def _score_many(codes: np.ndarray, js: list, base: np.ndarray, S: int, Y: int, X: int,
                n: int, pool, collapse: bool = False) -> dict:
    """{j: (ic, p)} for feature rows codes[j] (bins 0..X-1) against the
    [stratum, y] layout in `base` (= (stratum * Y + y) * X per row). Candidates
    are counted many per bincount, into a reused per-thread buffer.
    collapse=True appends the ic of the same table with strata merged, i.e.
    the plain ic at identical bin resolution: {j: (ic, p, ic_plain)}."""
    T = S * Y * X
    ja = np.asarray(js, dtype=np.intp)
    # every candidate counts every row once, so the [stratum, y] margins are shared
    n_sy = np.bincount(base // X, minlength=S * Y).reshape(S, Y) if ja.size >= 8 else None
    tasks = max(1, min(_MAX_TASKS, ja.size * n >> 19))
    per_tab = max(1, _TASK_CELLS // T)
    per_bc = max(1, min(per_tab, _BC_CELLS // n, -(-ja.size // tasks)))
    base_off = base[None, :] + (np.arange(per_bc, dtype=np.intp) * T)[:, None]

    def run(part):
        out = []
        for lo in range(0, part.size, per_tab):
            a = part[lo:lo + per_tab]
            t = _count_tables(codes, base, T, a, base_off).reshape(a.size, S, Y, X)
            cols = [a.tolist(), *(v.tolist() for v in _table_ic(t, n, n_sy=n_sy))]
            if collapse:
                cols.append(_table_ic(t.sum(1, keepdims=True), n)[0].tolist())
            out.extend(zip(*cols))
        return out

    return {r[0]: r[1:] for part in pool.map(run, np.array_split(ja, tasks)) for r in part}


# Product-pair test (screen_features), rows subsampled above either cap: leftovers x
# columns x rows x target scores (the matrix product, ~15 ms here) and columns x rows
# (everything else; ~30 ms in all at 5k rows x 1,000 columns).
_PRODUCT_BUDGET = 5e9
_PRODUCT_CELLS = 5e6
_PRODUCT_MIN_ROWS = 500     # fewer rows after subsampling: skip the test
_PRODUCT_KURT = 1.0         # columns whose bins have heavier tails than this (excess
                            # kurtosis: rare values, e.g. a 0/1 column under ~17% ones)
                            # stay out: their products have long-tailed nulls
_PRODUCT_SCORES = 8         # class-label targets: one-vs-rest scores for this many classes


def _gamma_pq(x: np.ndarray, a: np.ndarray):
    """(P, Q) = lower and upper regularized incomplete gamma of arrays x, a > 0:
    _gamma_sf vectorized (series where x < a + 1, Lentz continued fraction
    above, Wilson-Hilferty for a > 50), each tail computed directly."""
    x, a = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(a, np.float64))
    P, Q = np.zeros(x.shape), np.ones(x.shape)
    pos = x > 0
    big = pos & (a > 50)
    if big.any():
        k, v = 2 * a[big], 2 * x[big]
        z = ((v / k) ** (1 / 3) - (1 - 2 / (9 * k))) / np.sqrt(2 / (9 * k))
        Q[big] = 0.5 * _erfc(z / math.sqrt(2))
        P[big] = 0.5 * _erfc(-z / math.sqrt(2))
    small = pos & ~big
    if not small.any():
        return P, Q
    xs, as_ = x[small], a[small]
    lg = as_ * np.log(xs) - xs - np.array([math.lgamma(v) for v in as_])
    ser = xs < as_ + 1
    ps, qs = np.zeros(xs.size), np.ones(xs.size)
    if ser.any():                                       # series for P(a, x)
        xa, aa = xs[ser], as_[ser]
        term = 1.0 / aa
        s = term.copy()
        for m in range(1, 1000):
            term = term * (xa / (aa + m))
            s += term
            if (term < s * 1e-15).all():
                break
        pv = np.minimum(1.0, s * np.exp(lg[ser]))
        ps[ser], qs[ser] = pv, np.maximum(0.0, 1.0 - pv)
    cf = ~ser
    if cf.any():                                        # continued fraction for Q(a, x)
        tiny = 1e-300
        xa, aa = xs[cf], as_[cf]
        b = xa + 1 - aa
        c = np.full(xa.size, 1 / tiny)
        d = 1 / b
        h = d.copy()
        for i in range(1, 1000):
            an = -i * (i - aa)
            b = b + 2
            d = an * d + b
            d = 1 / np.where(np.abs(d) > tiny, d, tiny)
            c = b + an / c
            c = np.where(np.abs(c) > tiny, c, tiny)
            h *= d * c
            if (np.abs(d * c - 1) < 1e-15).all():
                break
        qv = np.minimum(1.0, np.exp(lg[cf]) * h)
        qs[cf], ps[cf] = qv, np.maximum(0.0, 1.0 - qv)
    P[small], Q[small] = ps, qs
    return P, Q


def _tail_p(z: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Two-sided p-values of standardized statistics z with skewness g: Pearson
    type III tails (shifted gamma), never thinner than the normal ones."""
    z = np.abs(np.asarray(z, np.float64))
    g = np.asarray(g, np.float64)
    p = _erfc(z / math.sqrt(2)).astype(np.float64)
    sel = np.flatnonzero(np.abs(g) > 1e-3)
    if sel.size:
        zs, gs = z[sel], np.abs(g[sel])
        k = 4 / gs**2
        pn = 0.5 * p[sel]
        # X = 2 Gamma(k) / |g| - 2 / |g| (skewness |g|): the long tail is gamma's upper
        # tail, the short one its lower tail (0 beyond the support)
        long_t = _gamma_pq(k + 2 * zs / gs, k)[1]
        x = k - 2 * zs / gs
        short_t = np.where(x > 0, _gamma_pq(np.maximum(x, 1e-300), k)[0], 0.0)
        p[sel] = np.maximum(pn, long_t) + np.maximum(pn, short_t)
    return np.minimum(p, 1.0)


def _perm_z(s, A2, A3, B2, B3, n: int):
    """z and skewness of S = sum_k a_k b_pi(k) over uniform random permutations
    pi, from the centred power sums of a and b (A2 = sum (a - mean a)^2, A3 the
    cubes; B2, B3 likewise). The mean is n * mean(a) * mean(b), which is 0 here:
    b is centred. Variance A2 B2 / (n - 1); third cumulant n A3 B3 / ((n - 1)(n - 2))."""
    v = A2 * B2 / (n - 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(v > 0, s / np.sqrt(v), 0.0)
        g = np.where(v > 0, n * A3 * B3 / ((n - 1.0) * (n - 2.0)) / v**1.5, 0.0)
    return z, g


def _product_pairs(B: np.ndarray, L: int, ys: np.ndarray, pool=None):
    """Bilinear pair test: for each of the first L rows of B (fine bins, one row
    per column, leftovers first), its best partner among all rows by
    S = sum over rows of y * R_i * R_j (R: centred bins, y: a centred target score,
    one column of ys per score).

    A pure pair (a product, an XOR of signs) has no signal alone and none in a
    coarse table worth its degrees of freedom, but S sees it with 1 dof; one
    matrix product covers every pair. p-value: exact permutation mean, variance
    and skewness of S under three nulls (permute column j: j is independent of
    y and i; permute i; permute y: y is independent of both), Pearson III tail,
    the largest of the three.

    Returns raw p [L] (1 where no pair), partner row [L] (-1), number of
    partners searched per candidate (the Bonferroni count), tested [L]."""
    P, n = B.shape
    m = ys.shape[1]
    Y = ys - ys.mean(0)
    Yf = np.ascontiguousarray(Y, dtype=np.float32)
    one = np.ones((n, 1), dtype=np.float32)
    Y2f = np.hstack([Yf * Yf, one])
    Y3f = np.hstack([Y2f[:, :m] * Yf, one])
    R = np.empty((P, n), dtype=np.float32)         # centred bins
    W = np.empty((P, n), dtype=np.float32)         # workspace: R^2, R^3, then y * R
    M1, M2, M3 = np.empty((P, m)), np.empty((P, m + 1)), np.empty((P, m + 1))
    R4 = np.empty(P)
    parts = np.array_split(np.arange(P), min(_MAX_TASKS, P))

    def moments(part):     # sums of a = y * R, a^2, a^3 per score; R^2, R^3, R^4
        a, b = part[0], part[-1] + 1
        r, w = R[a:b], W[a:b]
        np.copyto(r, B[a:b])
        r -= r.mean(1, keepdims=True)
        M1[a:b] = r @ Yf
        np.multiply(r, r, out=w)
        R4[a:b] = np.einsum("ij,ij->i", w, w)
        M2[a:b] = w @ Y2f
        w *= r
        M3[a:b] = w @ Y3f

    run = map if pool is None else pool.map
    list(run(moments, parts))
    B2, B3 = M2[:, m], M3[:, m]
    with np.errstate(divide="ignore", invalid="ignore"):
        kurt = n * R4 / B2**2 - 3
    ok = (B2 > 0) & (kurt <= _PRODUCT_KURT)
    p_best, i_best = np.ones(L), np.full(L, -1)
    P_ok = int(ok.sum())
    if P_ok < 2 or not ok[:L].any():
        return p_best, i_best, 0, np.zeros(L, dtype=bool)
    inv_b = np.where(ok, 1 / np.where(B2 > 0, B2, 1), 0).astype(np.float32)
    rows = max(1, (1 << 16) // n)                  # pairs per chunk in the permute-y step
    for k in range(m):
        y = Yf[:, k]
        Y2, Y3 = float((Y[:, k] ** 2).sum()), float((Y[:, k] ** 3).sum())

        def weigh(part):
            np.multiply(R[part[0]:part[-1] + 1], y, out=W[part[0]:part[-1] + 1])

        list(run(weigh, parts))
        S = R[:L] @ W.T                                    # [L, P]
        mu = M1[:, k] / n                                  # a = y * R_i, centred
        A2 = M2[:, k] - n * mu * mu
        A3 = M3[:, k] - 3 * mu * M2[:, k] + 2 * n * mu**3
        inv_a = np.where(ok & (A2 > 0), 1 / np.where(A2 > 0, A2, 1), 0).astype(np.float32)
        # each candidate's partner: largest z^2 under "permute the candidate"
        z2 = S * S
        z2 *= inv_b[:L, None]
        z2 *= inv_a
        z2[np.arange(L), np.arange(L)] = 0                 # no self-pairs
        i = z2.argmax(1)
        js = np.flatnonzero(ok[:L] & (z2[np.arange(L), i] > 0))
        if not js.size:
            continue
        i = i[js]
        s = S[js, i].astype(np.float64)
        # permute j / permute i: S = sum a * b with a = y * R_partner, b = R_self
        nulls = [_perm_z(s, A2[i], A3[i], B2[js], B3[js], n),
                 _perm_z(s, A2[js], A3[js], B2[i], B3[i], n)]
        # permute y: S = sum y * u with u = R_j * R_i; power sums of u, in chunks
        U = np.empty((js.size, 3))

        def usums(part):
            u, v = np.empty((rows, n), dtype=np.float32), np.empty((rows, n), dtype=np.float32)
            for lo in range(part[0], part[-1] + 1, rows):
                c = slice(lo, min(lo + rows, part[-1] + 1))
                h = c.stop - c.start
                np.take(R, js[c], axis=0, out=u[:h])
                np.take(R, i[c], axis=0, out=v[:h])
                u[:h] *= v[:h]
                U[c, 0] = u[:h].sum(1)
                U[c, 1] = np.einsum("ij,ij->i", u[:h], u[:h])
                np.multiply(u[:h], u[:h], out=v[:h])
                U[c, 2] = np.einsum("ij,ij->i", v[:h], u[:h])

        list(run(usums, np.array_split(np.arange(js.size), min(_MAX_TASKS, js.size))))
        ub = U[:, 0] / n
        U2 = U[:, 1] - n * ub * ub
        U3 = U[:, 2] - 3 * ub * U[:, 1] + 2 * n * ub**3
        nulls.append(_perm_z(s, Y2, Y3, U2, U3, n))
        p = np.max([_tail_p(z, g) for z, g in nulls], axis=0)
        better = p < p_best[js]
        p_best[js[better]], i_best[js[better]] = p[better], i[better]
    return p_best, i_best, (P_ok - 1) * m, ok[:L]


# Masked test (screen_features): a leftover's own effect against y with the kept
# columns' additive effects removed (y as normal scores of its ranks: an additive y
# stays additive, outliers are tamed). Kept columns in the residual model: the
# strongest (by ic) up to _MASKED_FIT_COLS and n / (_MASKED_ROWS_PER_BIN * bins); rows
# subsampled above _MASKED_ROWS or leftovers x rows x target scores > _MASKED_CELLS.
_MASKED_FIT_COLS = 32
_MASKED_ROWS_PER_BIN = 8
_MASKED_PASSES = 2
_MASKED_CELLS = 1e7
_MASKED_ROWS = 50_000
# Weighted Bonferroni over the rescue families that ran for a leftover (share =
# weight / sum of the weights that ran): table 4, product 1 -> 0.8 / 0.2. The table
# weight is scaled by the share of its partners the work cap allowed (see screen_features).
_RESCUE_WEIGHT = {"table": 4.0, "product": 1.0, "masked": 1.0}


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Standard normal quantiles of p in (0, 1): Acklam's rational approximation
    (relative error < 1.2e-9; no scipy)."""
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p = np.asarray(p, np.float64)
    out = np.empty(p.shape)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    q = p[mid] - 0.5
    t = q * q
    out[mid] = (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * q /                (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1)
    for m, pp, sgn in ((lo, p[lo], 1.0), (hi, 1 - p[hi], -1.0)):
        q = np.sqrt(-2 * np.log(pp))
        out[m] = sgn * (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /                  ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return out


def _residual(ys: np.ndarray, fit_bins: list, fit_ranks: list, B: int,
              passes: int = _MASKED_PASSES) -> np.ndarray:
    """[scores, n] residuals: ys (n x scores) centred, minus additive piecewise-linear
    effects of the given columns (per fine bin: a mean and a slope in the column's
    rank), by backfitting (`passes` sweeps)."""
    r = np.ascontiguousarray((ys - ys.mean(0)).T, dtype=np.float64)
    pre = []
    for b, a in zip(fit_bins, fit_ranks):
        b = b.astype(np.intp)
        inv = 1 / np.maximum(np.bincount(b, minlength=B), 1)
        dev = a - (np.bincount(b, a, minlength=B) * inv)[b]       # rank within its bin, centred
        ss = np.bincount(b, dev * dev, minlength=B)
        pre.append((b, inv, dev, np.where(ss > 0, 1 / np.where(ss > 0, ss, 1), 0.0)))
    for _ in range(passes):
        for b, inv, dev, iss in pre:
            for k in range(r.shape[0]):
                r[k] -= (np.bincount(b, r[k], minlength=B) * inv)[b]
                r[k] -= (np.bincount(b, r[k] * dev, minlength=B) * iss)[b] * dev
    return r


def _masked_tests(Bl: np.ndarray, r: np.ndarray, B: int, pool=None):
    """Linear and quadratic trend of each leftover (row of Bl: fine bins) against
    each residual score (row of r): S = sum over rows of c(bin) * r, c an
    orthogonal polynomial contrast of the bin index. p-value: exact permutation
    variance and skewness of S (permute the column; r does not depend on it),
    Pearson III tail. Returns the smallest p per leftover and the number of
    tests behind it (Bonferroni; no quadratic for a column with 2 distinct bins)."""
    L, n = Bl.shape
    K = r.shape[0]
    r = r - r.mean(1, keepdims=True)
    R2, R3 = (r * r).sum(1), (r * r * r).sum(1)
    C = np.empty((L, B))
    T = np.empty((K, L, B))
    rows = max(1, (1 << 16) // n)                  # leftovers per bincount
    off = (np.arange(rows, dtype=np.intp) * B)[:, None]

    def count(part):
        idx = np.empty((rows, n), dtype=np.intp)
        w = [np.tile(r[k], rows) for k in range(K)]
        for lo in range(part[0], part[-1] + 1, rows):
            h = min(rows, part[-1] + 1 - lo)
            np.add(Bl[lo:lo + h], off[:h], out=idx[:h])
            flat = idx[:h].ravel()
            C[lo:lo + h] = np.bincount(flat, minlength=h * B).reshape(h, B)
            for k in range(K):
                T[k, lo:lo + h] = np.bincount(flat, w[k][:h * n], minlength=h * B).reshape(h, B)

    run = map if pool is None else pool.map
    list(run(count, np.array_split(np.arange(L), min(_MAX_TASKS, L))))
    v = np.arange(B, dtype=np.float64)
    a = v - (C @ v / n)[:, None]                   # linear contrast, centred
    A2 = (C * a * a).sum(1)
    q = a * a - (A2 / n)[:, None]                  # quadratic, centred, then
    Q0 = (C * q * q).sum(1)                        # orthogonal to the linear one
    with np.errstate(divide="ignore", invalid="ignore"):
        q -= np.where(A2 > 0, (C * q * a).sum(1) / A2, 0)[:, None] * a
    Q2 = (C * q * q).sum(1)
    Q2 = np.where(Q2 > 1e-9 * Q0, Q2, 0.0)         # 2 distinct bins: no quadratic
    A3, Q3 = (C * a**3).sum(1), (C * q**3).sum(1)
    p = np.ones(L)
    for k in range(K):
        for c, c2, c3 in ((a, A2, A3), (q, Q2, Q3)):
            z, g = _perm_z((c * T[k]).sum(1), c2, c3, R2[k], R3[k], n)
            p = np.minimum(p, _tail_p(z, g))
    return p, K * (1 + (Q2 > 0))


_MAX_CLASSES = 10   # recursive_ic_score: one-vs-rest passes for the most frequent classes


def _recursive_by_class(lf: pl.LazyFrame, feats: list, target_col: str, *,
                        alpha, max_features, **kw) -> pl.DataFrame:
    """recursive_ic_score for a class-label target: one class-vs-rest pass per
    class, merged into one ranking (see recursive_ic_score)."""
    lab = pl.col(target_col).cast(pl.String)
    counts = (lf.select(lab.alias("c")).drop_nulls().group_by("c").len()
                .sort(["len", "c"], descending=[True, False]).collect())
    classes = counts["c"].to_list()
    if len(classes) < 2:
        raise ValueError("the target has fewer than 2 classes")
    top = classes[:_MAX_CLASSES]
    passes = [lab == c for c in (top[:1] if len(classes) == 2 else top)]
    if len(classes) > _MAX_CLASSES:
        passes.append(~lab.is_in(top))                    # every other class, pooled
    K = len(passes)
    tmp = "__class_vs_rest__"
    while tmp in feats:
        tmp += "_"

    best: dict = {}                                       # feature -> (sort key, row)
    p_min: dict = {}
    for expr in passes:
        out = recursive_ic_score(lf.with_columns(expr.alias(tmp)), feats, tmp,
                                 alpha=None if alpha is None else alpha / K,
                                 max_features=max_features, **kw)
        for r in out.iter_rows(named=True):
            f = r["feature"]
            key = (r["step"] is None, r["step"] or 0, -(r["ic"] or 0.0))
            if f not in best or key < best[f][0]:
                best[f] = (key, r)
            if r["ic_pvalue"] is not None:
                p_min[f] = min(p_min.get(f, 1.0), r["ic_pvalue"])

    rows, step = [], 0
    for key, r in sorted(best.values(), key=lambda kv: kv[0]):
        picked = r["step"] is not None and (max_features is None or step < max_features)
        step += picked
        p = p_min.get(r["feature"])
        rows.append({"step": step if picked else None, "feature": r["feature"], "ic": r["ic"],
                     "ic_pvalue": None if p is None else min(1.0, p * K),
                     "ic_marginal": r["ic_marginal"], "delta_r2": None, "cumulative_r2": None})
    return pl.DataFrame(rows, schema={
        "step": pl.Int64, "feature": pl.String, "ic": pl.Float64, "ic_pvalue": pl.Float64,
        "ic_marginal": pl.Float64, "delta_r2": pl.Float64, "cumulative_r2": pl.Float64})


def _lo(a, b):      # the (ic, p) pair with the lower ic; ties -> larger p
    return a if (a[0], -a[1]) <= (b[0], -b[1]) else b


def _hi(a, b):      # the (ic, p) pair with the higher ic; ties -> smaller p
    return a if (a[0], -a[1]) >= (b[0], -b[1]) else b


def recursive_ic_score(
    df,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    *,
    alpha: float | None = None,
    max_features: int | None = None,
    bins: int | None = None,
    fit_bins: int = 255,
    max_levels: int = 32,
    n_folds: int = 5,
    shrink: float = 1.0,
    seed: int = 0,
    n_threads: int | None = None,
) -> pl.DataFrame:
    """Greedy feature ranking: pick the best feature, account for it, repeat.

    Step 1 picks the feature with the highest `ic` against the target.
    Every later step picks the candidate that adds the most beyond the
    features already picked. A candidate's score is the LOWEST of:

      * joint check: its ic against the residual, y minus every picked
        feature's fitted effect. Kills features that are redundant with a
        COMBINATION of picked features (e.g. a + c once a and c are picked).
        If the residual alone shows nothing, the candidate is re-scored inside
        bins of the picked features that make it most informative, so an
        interaction partner is not lost.
      * duplicate check: the joint score times the share of the candidate's
        information that survives inside the bins of its closest picked
        features. Kills copies of a picked feature, including copies of a
        feature that only acts through an interaction.

    A fitted effect is E[y | feature]: a binned average of y (up to `fit_bins`
    bins, like a GBDT histogram; nulls get their own bin), computed
    out-of-fold (`n_folds`) so a feature can't "explain" its own noise.
    No model is trained.

    Stopping: `alpha` stops when the best candidate's adjusted p-value exceeds
    it; `max_features` stops after that many picks. Unpicked features are
    listed last with step = null.

    Categorical / string features: with at most `max_levels` well-filled
    categories, every category is its own bin. Above that the feature is
    target-encoded (each row sees only the average target of earlier rows of
    its category) and scored like a number. Either way its fitted effect is
    the category's out-of-fold mean of y, over all categories.

    Output, one row per feature in pick order:
        step           1 = picked first. Null if never picked.
        feature
        ic             what it adds beyond the features above it (0..1).
        ic_pvalue      p-value of that ic, Bonferroni-adjusted for picking the
                       best of the remaining features.
        ic_marginal    ic vs. the original target, on its own.
        delta_r2       out-of-fold R^2 gained by removing its fitted effect.
        cumulative_r2  out-of-fold R^2 of the additive fit so far.

    Once three picks in a row look like pure noise (adjusted p > 0.5), the
    rest are ranked by their current score in one shot and their
    delta_r2 / cumulative_r2 are null.

    Class-label targets (strings, categoricals; multiclass): one pass per
    class, class vs rest (the 10 most frequent classes; the rest pooled as one
    more), each stopped at alpha / passes. The passes are merged: features
    picked at step 1 in any pass first (by ic), then step 2, and so on.
    ic_pvalue is the smallest over passes times the number of passes;
    delta_r2 / cumulative_r2 are null. A two-class label needs one pass.

    Limits:
      * Missed (measured): perfectly balanced sign-flip interactions
        (sign(x1*x2), 50/50 XOR, parity). Each feature alone says nothing
        about y, so nothing gets the search started.
      * Each candidate is re-checked against at most 6 picked features (the 3
        that leave it least, the 3 that reveal most), one at a time. A feature
        that is redundant only given 2+ picked features AND only through an
        interaction would slip through (by design; not tested).
      * Categories with fewer than 10 rows are pooled. A target-encoded
        feature is scored as one number, so an effect that differs by
        category of a high-cardinality feature is not seen.
      * Integer codes are treated as numbers. Cast IDs to String/Categorical.
      * Fitted effects are averages of raw y: unlike `ic` itself, they are
        not immune to extreme outliers in the target.
    """
    df, feature_cols, target_col = _inputs(df, feature_cols, target_col)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    schema = lf.collect_schema()
    feats = [c for c in dict.fromkeys(feature_cols) if c != target_col]
    missing = [c for c in [*feats, target_col] if c not in schema]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    y_expr, y_ord = _clean(target_col, schema[target_col])
    if not y_ord:
        return _recursive_by_class(
            lf, feats, target_col, alpha=alpha, max_features=max_features, bins=bins,
            fit_bins=fit_bins, max_levels=max_levels, n_folds=n_folds, shrink=shrink,
            seed=seed, n_threads=n_threads)

    exprs, ordered = [y_expr.cast(pl.Float64).alias("__y__")], []
    for c in feats:
        e, o = _clean(c, schema[c])
        exprs.append(e.alias(c))
        ordered.append(o)
    clean = (lf.select(exprs)
             .filter(pl.col("__y__").is_not_null() & pl.col("__y__").is_finite())
             .collect())
    y = clean["__y__"].to_numpy()
    n, P = y.size, len(feats)
    if n < _MIN_N:
        raise ValueError(f"only {n} rows have a usable target value; at least {_MIN_N} are needed")
    if y.min() == y.max():
        raise ValueError(f"the target {target_col!r} has a single value: nothing to rank against")

    # Bin counts. Plain ic uses B x B tables. Tables inside the Bs strata of a
    # picked feature use coarser B3 x B3 bins, to keep ~15 rows per cell.
    B = bins or _default_bins(n)
    Bs = int(min(32, B))
    B3 = int(min(16, B, max(2, round(math.sqrt(n / (16 * Bs))))))
    G = int(min(fit_bins, max(2, n // 50)))             # >= ~50 rows per fit bin

    rng = np.random.default_rng(seed)
    folds = rng.integers(0, n_folds, n)
    y0 = y - y.mean()
    resid = y0.copy()
    sst = float(y0 @ y0)

    def rank_bins(v: np.ndarray, k: int) -> np.ndarray:
        return _bins_from_ranks(_ranks(v)[0], n, k)

    with ThreadPoolExecutor(n_threads or os.cpu_count() or 1) as pool:
        # ---- bin every feature once -----------------------------------
        enc_target = _ranks(y)[0]
        enc_target -= enc_target.mean()

        def prep(j: int):
            s = clean[feats[j]]
            if ordered[j]:
                fine, as_number = _bin_feature(s, True, B), True
            else:
                # same seed for every feature: identical categoricals get
                # identical encodings, so they are recognised as duplicates
                fine, as_number = _cat_bins(_cat_codes(s), enc_target, B, max_levels,
                                            np.random.default_rng(seed))
            # coarse bins: numbers shrink B -> B3 (null bin B -> B3);
            # categories keep their 2*B3 most frequent, the rest become "other"
            coarse = fine * B3 // B if as_number else _lump(fine, 2 * B3)
            return _compact(fine), _compact(coarse), as_number

        binned = list(pool.map(prep, range(P)))
        as_number = [b[2] for b in binned]
        fine = np.stack([b[0] for b in binned]) if P else np.zeros((0, n), np.uint8)
        coarse = np.stack([b[1] for b in binned]) if P else np.zeros((0, n), np.uint8)
        del binned
        W2 = int(fine.max()) + 1 if P else 1
        W3 = int(coarse.max()) + 1 if P else 1

        def strata_of(j: int):
            """Bins of picked feature j used as strata, and how many there are."""
            if as_number[j]:
                st = fine[j].astype(np.intp) * Bs // B   # null bin B -> Bs
            else:
                st = _lump(fine[j].astype(np.int64), Bs - 1).astype(np.intp)
            return st, int(st.max()) + 1

        remaining = list(range(P))
        picked: list[int] = []
        marginal: dict[int, float] = {}
        # Per candidate, from tables of (candidate vs. y) inside each pick's bins:
        low: dict[int, list] = {j: [] for j in remaining}    # picks that leave it least
        high: dict[int, list] = {j: [] for j in remaining}   # picks that reveal the most
        floor: dict[int, tuple] = {}                         # lowest such (ic, p)
        rows, sse = [], sst
        limit = P if max_features is None else min(P, max_features)
        stopped_by_rule, noise_streak = False, 0

        while remaining and len(picked) < limit:
            # ---- score every candidate against the current residual ------
            by = rank_bins(resid, B)
            m = _score_many(fine, remaining, by * W2, 1, B, W2, n, pool)
            if not marginal:
                marginal = {j: v[0] for j, v in m.items()}
            by3 = by * B3 // B
            cache: dict[int, tuple] = {}

            def resid_within(j: int, s: int):
                if s not in cache:
                    st, S = strata_of(s)
                    cache[s] = ((st * B3 + by3) * W3, S)
                base3, S = cache[s]
                return _score_many(coarse, [j], base3, S, B3, W3, n, pool, True)[j]

            # Upper bound on each score; exact scores only for the contenders.
            upper = {j: _hi(m[j], floor[j]) if picked else m[j] for j in remaining}

            def exact(j: int):
                if not picked:
                    return m[j]
                cr = {s: resid_within(j, s) for s in {t[2] for t in low[j] + high[j]}}
                joint = m[j]                             # joint check (+ partner rescue)
                for _, _, s, _ in high[j]:
                    joint = _hi(joint, cr[s][:2])
                keep, p_keep = 1.0, 0.0                  # duplicate check
                for v, p, s, kept_y in low[j]:
                    ic_r, p_r, plain_r = cr[s]
                    kept_r = ic_r / plain_r if plain_r > 0 else 1.0
                    k, pk = (kept_r, p_r) if kept_r >= kept_y else (kept_y, p)
                    if (min(k, 1.0), -pk) < (keep, -p_keep) or p_keep == 0.0:
                        keep, p_keep = min(k, 1.0), pk
                p = max(joint[1], p_keep) if keep < _DUP_SHARE else joint[1]
                return _lo((joint[0] * keep, p), upper[j])

            order = sorted(remaining, key=lambda j: (upper[j][0], -upper[j][1]), reverse=True)
            best_j, best, score = None, (-1.0, 1.0), dict(upper)
            for j in order:
                if best_j is not None and best[0] >= upper[j][0]:
                    break
                score[j] = exact(j)
                if (score[j][0], -score[j][1]) > (best[0], -best[1]):
                    best_j, best = j, score[j]
            p_adj = min(1.0, best[1] * len(remaining))   # Bonferroni: best of k

            if alpha is not None and p_adj > alpha:
                stopped_by_rule = True
                break
            noise_streak = noise_streak + 1 if p_adj > _TAIL_P else 0
            if noise_streak >= _TAIL_PATIENCE:           # nothing but noise left
                break

            # ---- pick it, remove its effect ------------------------------
            j = best_j
            fb = _bin_feature(clean[feats[j]], ordered[j], G)
            resid = resid - _oof_bin_means(resid, fb, folds, n_folds, shrink)
            new_sse = float(resid @ resid)
            rows.append({"step": len(picked) + 1, "feature": feats[j], "ic": best[0],
                         "ic_pvalue": p_adj, "ic_marginal": marginal[j],
                         "delta_r2": (sse - new_sse) / sst if sst else 0.0,
                         "cumulative_r2": 1 - new_sse / sst if sst else 0.0})
            sse = new_sse
            picked.append(j)
            remaining.remove(j)
            if not remaining:
                break

            # ---- every candidate vs. y inside bins of the new pick -------
            # y has only this pick's own effect removed: that can never hide
            # a candidate, and stops a near-copy leaking through coarse bins.
            st, S = strata_of(j)
            own = y0 - _oof_bin_means(y0, fb, folds, n_folds, shrink)
            base3 = (st * B3 + rank_bins(own, B3)) * W3
            got = _score_many(coarse, remaining, base3, S, B3, W3, n, pool, True)
            for c, (v, p, plain) in got.items():
                kept = v / plain if plain > 0 else 1.0   # share of its ic that survives
                low[c] = sorted(low[c] + [(v, p, j, kept)], key=lambda t: (t[3], t[0]))[:_NEIGHBORS]
                high[c] = sorted(high[c] + [(v, p, j, kept)], key=lambda t: -t[0])[:_NEIGHBORS]
                floor[c] = _lo(floor[c], (v, p)) if c in floor else (v, p)
        else:
            stopped_by_rule = True                       # max_features reached / all ranked

    # ---- whatever is left: rank by current score, no further fitting -----
    if remaining:
        if not picked and not marginal:                  # max_features == 0
            score = {j: (0.0, 1.0) for j in remaining}
            marginal = {j: None for j in remaining}
        rest = sorted(remaining, key=lambda j: (score[j][0], -score[j][1]), reverse=True)
        for k, j in enumerate(rest):
            rows.append({"step": None if stopped_by_rule else len(picked) + k + 1,
                         "feature": feats[j], "ic": score[j][0],
                         "ic_pvalue": min(1.0, score[j][1] * len(remaining)),
                         "ic_marginal": marginal[j], "delta_r2": None,
                         "cumulative_r2": None})

    return pl.DataFrame(rows, schema={
        "step": pl.Int64, "feature": pl.String, "ic": pl.Float64,
        "ic_pvalue": pl.Float64, "ic_marginal": pl.Float64,
        "delta_r2": pl.Float64, "cumulative_r2": pl.Float64})


# --------------------------------------------------------------------------- #
# Screening: drop only what shows no hint of being useful
# --------------------------------------------------------------------------- #
_ID_UNIQUE = 0.95     # share of distinct values above which a column is "id_like"
_ROW_ORDER = 0.95     # |rank correlation with row position| above which "row_order"
_EQUIV_ALPHA = 0.05   # one-sided level for "the data rule out ic >= min_ic"
# p-values that count as a hint: fine-table ic (any shape), xi (y a function of x),
# Spearman (monotone, 1 dof), coarse 5 x 5 table (smooth shapes, few dof), and
# whether the column being missing predicts y (columns with nulls only)
_HINT_TESTS = ("ic_pvalue", "xi_pvalue", "sp_p", "coarse_p", "na_p")
# Work cap for the table pair test: leftovers x partners x rows (~30 ms here). Binds on
# wide or tall data; there leftover-vs-leftover partners are a lottery, and the product
# and masked tests try every column. 1e8 cost ~100 ms more per screen; on the 73 real
# datasets it changed nothing; it mainly found pure U x U pairs among ~100 columns at
# 20k+ rows (pure_pairs.py squares, 100 columns, 20k rows: 0.3 / 0.9 -> 0.0 / 0.2).
_PAIR_BUDGET = 3e7


def screen_features(
    df,
    feature_cols: Sequence[str] | None = None,
    target_col: str | None = None,
    *,
    keep_p: float = 0.2,
    keep_q: float = 0.5,
    min_ic: float = 0.1,
    pair_alpha: float = 0.05,
    pair_partners: int = 64,
    bins: int | None = None,
    max_levels: int = 32,
    max_rows: int | None = 200_000,
    seed: int = 0,
    n_threads: int | None = None,
) -> pl.DataFrame:
    """Pre-fit screen for tree models: drop a column only if nothing hints it
    is useful. Kept junk costs a tree model little; a dropped real column is
    gone for good, so every doubt is resolved in favour of keeping.

    A column is KEPT if any holds:
      * alone:   any of its p-values is below `keep_p` (uncorrected): ic (any
        shape), xi (y a function of x), Spearman (monotone trend), ic on a
        coarse 5 x 5 table (smooth shapes), or, for a column with nulls,
        whether being missing predicts y (scored on all rows; the others use
        the non-null rows). And its q-value is below `keep_q`:
        Storey's adaptive false discovery rate, from the smallest p-value times
        the number of tests, estimates the share of junk among columns with
        at least this much evidence. Where most columns carry signal it is
        small and changes nothing; where most look like junk it tightens the
        cut. keep_q=1 turns it off.
      * in a pair: it failed the test above, but it is informative inside
        the bins of another column (kept or not), at `pair_alpha` after
        correcting for the number of pairs tried, and its Benjamini-Hochberg
        q-value across the leftovers is below `keep_q`. This catches columns
        that only matter together (a difference, an XOR). Two tests: a
        coarse table inside each partner's bins (any shape; up to
        `pair_partners` partners), and a product test against every numeric
        null-free column at once (y vs rank(a) x rank(b), 1 dof: a product,
        an XOR of signs; not U x U shapes). Columns with rare values (excess
        kurtosis of their bins over 1) stay out of the product test.
      * masked: it failed the alone test, but its own effect shows once the
        kept columns' effects are taken out of y: linear and quadratic trend
        of its bins against y (normal scores of y's ranks; class targets:
        one-vs-rest indicators) minus an additive piecewise-linear fit of the
        strongest kept columns (up to 32, at most n / (8 x bins)). A weak real
        column next to strong ones (a seasonal month next to size and grade)
        hides inside their variance on its own. Same cut as a pair:
        `pair_alpha` and BH q < `keep_q`. Numeric null-free columns only.
        Pair and masked tests share `pair_alpha` (weighted Bonferroni over the
        tests that ran for a column: table 4, product 1, masked 1).
      * untested: a categorical with 2+ categories, every one too small to
        test on its own (under 10 rows, or under ~5 * n^(1/3) rows with a
        categorical target), so pooling left nothing to score. Kept unless
        flagged id_like. A tree model's own encoding may still use it.
      * inconclusive: no hint of signal, but too little data to rule out an
        ic of `min_ic` or more (equivalence test: one-sided 5% upper bound of
        its ic on a coarse 5 x 5 table, from the noncentral chi-square of its
        G statistic; coarse = few degrees of freedom = the power to rule out
        a smooth effect).
        On small data this keeps nearly everything; as rows grow, junk is
        ruled out and dropped. A mostly-constant column with fewer than 5
        rows per target bin outside its most common value can't be judged
        this way and is kept too. Constant columns are dropped.
        min_ic=0 turns this off.

    Redundancy is ignored on purpose: correlated copies are all kept.
    Use recursive_ic_score when you want a short, non-redundant list.

    pair_partners: in the table test, each leftover column is tried against at
        most this many other columns (the ones with the most evidence on their
        own, half from the kept columns, half from the leftovers). Every column
        is tried when there are no more than that. Fewer (at least 2) when
        leftovers x partners x rows would pass 3e7 (wide or tall data: e.g. 6
        partners at 1,000 columns x 5k rows). The product test always tries every
        column, on a row sample when leftovers x columns x rows passes 5e9 or
        columns x rows 5e6. 0 turns the rescue step (pair and masked) off.

    The target may be numeric, boolean or categorical (multiclass works).

    max_rows: screen a random sample of this many rows (order kept) when the
        data are larger. At 1M x 100 the full screen took 4.5 s, a 200k sample
        0.9 s, keeping the same real columns. None = all rows.

    Output, one row per feature, kept first:
        feature, n
        keep            True = pass it to the model
        reason          "alone", "pair", "masked", "untested", "inconclusive"
                        or null (dropped)
        ic, ic_pvalue   standalone score (same as nonlinear_corr)
        xi_pvalue       null for categorical features / targets
        pair_pvalue     corrected p-value of its best rescue (pair or masked
                        test); null if not tried
        pair_with       the other column of that pair (null when masked won)
        flag            "id_like": integer/string column with a different
                        value on (nearly) every row. "row_order": tracks row
                        position. Either can score well through a leak (e.g.
                        rows sorted by label). Flags never change `keep`.
    """
    df, feature_cols, target_col = _inputs(df, feature_cols, target_col)
    have = set(df.collect_schema().names())
    missing = [c for c in feature_cols if c not in have]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    if max_rows is not None:
        height = df.height if isinstance(df, pl.DataFrame) else df.select(pl.len()).collect().item()
        if height > max_rows:
            cols = list(dict.fromkeys([*feature_cols, target_col]))
            df = _sample_rows((df.select(cols) if isinstance(df, pl.DataFrame)
                               else df.select(cols).collect()), max_rows, seed)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    schema = lf.collect_schema()
    feats = [c for c in dict.fromkeys(feature_cols) if c != target_col]

    base = _corr_rows(lf, feats, target_col, bins, max_levels, seed, n_threads)
    rec = {r["feature"]: {"ic": None, "ic_pvalue": None, "xi_pvalue": None, **r,
                          "reason": None, "pair_pvalue": None, "pair_with": None,
                          "flag": None} for r in base}
    # ---- alone: a hint on any test, beyond the junk expected at that level --
    # constant columns are left out (their p = 1 would inflate the junk share),
    # unless their nulls carry a test: "1 or missing" is a real column
    scored = [r for r in rec.values()
              if (r["ic_pvalue"] is not None and not r.get("const")) or r.get("na_p") is not None]
    hints = [[r[k] for k in _HINT_TESTS if r.get(k) is not None] for r in scored]
    if keep_q < 1 and scored:
        p_any = np.array([min(1.0, len(h) * min(h)) for h in hints])   # Bonferroni
        # share of junk: each test's estimate is biased up by its blind spots,
        # so take the lowest (each is uniform on junk)
        pi0 = min(_pi0(np.array([r[k] for r in scored if r.get(k) is not None]))
                  for k in _HINT_TESTS)
        q = _qvalues(p_any, pi0)
    else:
        q = np.zeros(len(scored))
    for r, h, qi in zip(scored, hints, q):
        if min(h) < keep_p and qi < keep_q:
            r["reason"] = "alone"

    # ---- flags: identifier-like and row-order columns -------------------
    id_cols = [c for c in feats if schema[c].is_integer() or schema[c] == pl.String
               or isinstance(schema[c], (pl.Categorical, pl.Enum))]
    if id_cols:
        uniq = lf.select(pl.len().alias("__n__"), *[pl.col(c).n_unique() for c in id_cols]) \
                 .collect().row(0, named=True)
        for c in id_cols:
            if uniq["__n__"] and uniq[c] / uniq["__n__"] >= _ID_UNIQUE:
                rec[c]["flag"] = "id_like"
    num_cols = [c for c in feats if schema[c].is_numeric() or schema[c].is_temporal()]
    for c in num_cols:                       # block-scored columns: from their ranks
        v = rec[c].get("row_rho")
        if v is not None and np.isfinite(v) and abs(v) >= _ROW_ORDER and rec[c]["flag"] is None:
            rec[c]["flag"] = "row_order"
    num_cols = [c for c in num_cols if rec[c].get("row_rho") is None]
    if num_cols:
        # rank correlation with row position, on ~2000 evenly spaced rows
        n_all = lf.select(pl.len()).collect().item()
        # average ranks: tied values share a rank, so a mostly-constant column
        # can't inherit row order from how the ties are broken. Missing = lowest.
        rk = (lf.select([pl.col(c).to_physical().cast(pl.Float64).fill_nan(None)
                         .fill_null(float("-inf")) for c in num_cols])
                .gather_every(max(1, n_all // 2000))
                .select(pl.all().rank("average").cast(pl.Float64)).collect().to_numpy())
        if rk.shape[0] >= _MIN_N:
            rk = rk - rk.mean(0)
            sub = rk
            pos = np.arange(sub.shape[0]) - (sub.shape[0] - 1) / 2
            with np.errstate(divide="ignore", invalid="ignore"):
                rho = (rk * pos[:, None]).sum(0) / np.sqrt((rk**2).sum(0) * (pos**2).sum())
            for c, v in zip(num_cols, rho):
                if np.isfinite(v) and abs(v) >= _ROW_ORDER and rec[c]["flag"] is None:
                    rec[c]["flag"] = "row_order"

    # ---- categoricals the tests could not see: keep unless they are IDs -
    for r in rec.values():
        if r["reason"] is None and r.get("untested") and r["flag"] != "id_like":
            r["reason"] = "untested"

    # ---- pair rescue for the leftovers ----------------------------------
    def own_p(c: str) -> float:
        r = rec[c]
        return min(p for p in (*(r.get(k) for k in _HINT_TESTS), 1.0) if p is not None)

    usable = [c for c in feats if (rec[c]["n"] or 0) >= _MIN_N]
    left = [c for c in usable if rec[c]["reason"] is None]
    kept = [c for c in usable if rec[c]["reason"] is not None]
    if left and len(usable) >= 2 and pair_partners > 0:
        # Partners: every column when few enough; otherwise the kept columns and
        # the leftovers with the most evidence on their own, half the budget each.
        n_est = max(rec[c]["n"] for c in usable)
        n_partners = min(pair_partners, max(2, int(_PAIR_BUDGET // (len(left) * n_est))))
        # the table test's share of pair_alpha shrinks with the share of the
        # partners it was meant to try that the work cap let it try
        weight = dict(_RESCUE_WEIGHT)
        weight["table"] *= min(1.0, n_partners / max(1, min(pair_partners, len(usable) - 1)))
        if len(usable) <= n_partners + 1:
            partner_names = usable
        else:
            half = n_partners // 2
            partner_names = (sorted(kept, key=own_p)[:half]
                             + sorted(left, key=own_p)[:n_partners - min(half, len(kept))])
        names = left + [c for c in partner_names if c not in set(left)]
        idx = {c: i for i, c in enumerate(names)}
        L = len(left)                                    # candidates are names[:L]

        y_expr, y_ord = _clean(target_col, schema[target_col])
        y_ok = pl.col("__y__").is_not_null()
        clean = lf.select(y_expr.alias("__y__")).filter(y_ok).collect()
        n = clean.height

        def pre_bins(c: str):                     # fine bins from the scoring: same rows, same B
            b = rec[c].get("bins")
            return b if b is not None and b.size == n else None

        def pair_bins(c: str):    # (fine bins, numeric) from the scoring, for columns
            pb = rec[c].get("pair_bins")          # with nulls or categories: all rows
            return pb if pb is not None and pb[0].size == n else None

        # only columns without reusable bins need their values
        need = [c for c in names if pre_bins(c) is None and pair_bins(c) is None]
        exprs = {c: _clean(c, schema[c]) for c in need}
        if need and n >= _MIN_N:
            clean = (lf.select(y_expr.alias("__y__"), *(e.alias(c) for c, (e, _) in exprs.items()))
                       .filter(y_ok).collect())
        if n >= _MIN_N:
            B = bins or _default_bins(n)
            Bs = int(min(32, B))
            B3 = int(min(16, B, max(2, round(math.sqrt(n / (16 * Bs))))))
            if y_ord:
                avg = _ranks(clean["__y__"].to_numpy())[0]
                by3 = _bins_from_ranks(avg, n, B3)
                enc_target = avg - avg.mean()
            else:
                by3 = _lump(_cat_codes(clean["__y__"]), 2 * B3)
                enc_target = None
            Y = int(by3.max()) + 1

            with ThreadPoolExecutor(n_threads or os.cpu_count() or 1) as pool:
                to_coarse = (np.arange(B + 1) * B3 // B).astype(np.uint8)   # fine bin -> coarse
                to_strata = (np.arange(B + 1) * Bs // B).astype(np.uint8)
                partners = set(partner_names)

                def prep(j: int):                # (coarse bins, strata bins if a partner)
                    c = names[j]
                    pre = pre_bins(c)
                    if pre is not None:
                        return to_coarse[pre], to_strata[pre] if c in partners else None
                    pb = pair_bins(c)
                    if pb is not None:
                        fine, num = pb
                    elif exprs[c][1]:
                        fine, num = _bin_feature(clean[c], True, B), True
                    else:
                        fine, num = _cat_bins(_cat_codes(clean[c]), enc_target, B, max_levels,
                                              np.random.default_rng(seed))
                    if num:                              # fine bins 0..B (B = null)
                        return to_coarse[fine], to_strata[fine]
                    return _compact(_lump(fine, 2 * B3)), _compact(_lump(fine, Bs - 1))

                binned = list(pool.map(prep, range(len(names))))
                coarse = np.stack([b[0] for b in binned])
                W3 = int(coarse.max()) + 1

                best = {j: (1.0, None) for j in range(L)}       # raw p, other column
                tests = np.zeros(L, dtype=int)                  # pairs each leftover was in
                # A hit for candidate j inside partner s's bins shows j matters. It
                # shows s matters only if j has no signal of its own; otherwise any
                # junk partner of a weak real column would be "rescued" with it.
                no_hint = [own_p(c) >= keep_p for c in left]
                for c in partner_names:
                    s = idx[c]
                    st = binned[s][1].astype(np.intp)
                    S = int(st.max()) + 1
                    js = [j for j in range(L) if j != s]
                    if not js:
                        continue
                    got = _score_many(coarse, js, (st * Y + by3) * W3, S, Y, W3, n, pool)
                    tests[js] += 1
                    if s < L:
                        tests[s] += len(js)
                    for j, (_, p) in got.items():
                        if p < best[j][0]:
                            best[j] = (p, s)
                        if s < L and no_hint[j] and p < best[s][0]:   # rescues both leftovers
                            best[s] = (p, j)

                # Product pairs: each leftover against every column with reusable bins,
                # one matrix product (_product_pairs). Rows subsampled on wide data.
                g_left = [c for c in left if pre_bins(c) is not None]
                g_set = set(g_left)
                g_cols = g_left + [c for c in usable if c not in g_set and pre_bins(c) is not None]
                prod, prod_tested = {}, set()
                if g_left:
                    if y_ord:
                        ys = enc_target[:, None]
                    else:
                        y_codes = _cat_codes(clean["__y__"])
                        top = np.argsort(-np.bincount(y_codes), kind="stable")
                        top = top[:1 if top.size <= 2 else _PRODUCT_SCORES]
                        ys = (y_codes[:, None] == top[None, :]).astype(np.float64)
                if g_left and len(g_cols) >= 2:
                    n_g = min(n, int(_PRODUCT_BUDGET / (len(g_left) * len(g_cols) * ys.shape[1])),
                              int(_PRODUCT_CELLS / len(g_cols)))
                    if n_g >= min(n, _PRODUCT_MIN_ROWS):
                        rows = (None if n_g >= n else
                                np.sort(np.random.default_rng(seed).choice(n, n_g, replace=False)))
                        Bm = np.stack([pre_bins(c) if rows is None else pre_bins(c)[rows] for c in g_cols])
                        p_g, i_g, M, tested = _product_pairs(
                            Bm, len(g_left), ys if rows is None else ys[rows], pool)
                        prod_tested = {c for c, t in zip(g_left, tested) if t}
                        for c, p, i in zip(g_left, p_g, i_g):
                            if i >= 0:
                                prod[c] = (min(1.0, p * M), g_cols[i])

                # Masked: each leftover against y minus the strongest kept columns'
                # additive effects (their fine bins), where a weak effect of its own
                # no longer drowns in theirs. The residual never uses the leftover.
                masked = {}
                fit = sorted((c for c in kept if pre_bins(c) is not None),
                             key=lambda c: -(rec[c]["ic"] or 0.0))
                fit = fit[:min(_MASKED_FIT_COLS, max(1, n // (_MASKED_ROWS_PER_BIN * B)))]
                if g_left and fit:
                    n_m = min(n, _MASKED_ROWS, int(_MASKED_CELLS / (len(g_left) * ys.shape[1])))
                    if n_m >= min(n, _PRODUCT_MIN_ROWS):
                        rows = (None if n_m >= n else np.sort(
                            np.random.default_rng(seed + 1).choice(n, n_m, replace=False)))

                        def sub(a):
                            return a if rows is None else a[rows]

                        fr = lf.select(*(_clean(c, schema[c])[0].alias(c) for c in fit),
                                       y_expr.alias("__y__")).filter(y_ok).drop("__y__")
                        fr = fr.collect() if rows is None else fr.collect()[rows]
                        ranks = list(fr.select(pl.all().rank("average").cast(pl.Float64))
                                       .to_numpy().T)
                        ym = sub(ys)
                        if y_ord:                         # normal scores of y's ranks
                            ym = _norm_ppf((_ranks(ym[:, 0])[0] - 0.5) / ym.shape[0])[:, None]
                        res = _residual(ym, [sub(pre_bins(c)) for c in fit], ranks, B)
                        Bl = np.stack([sub(pre_bins(c)) for c in g_left])
                        p_m, n_t = _masked_tests(Bl, res, B, pool)
                        masked = {c: min(1.0, p * t) for c, p, t in zip(g_left, p_m, n_t)}

            p_adj = np.ones(L)
            why = ["pair"] * L
            for j, c in enumerate(left):
                p, other = best[j]
                ran = {"table": bool(tests[j]), "product": c in prod_tested, "masked": c in masked}
                tot = sum(weight[f] for f, r in ran.items() if r)
                share = {f: weight[f] / tot for f, r in ran.items() if r}
                found = ([] if other is None else
                         [(min(1.0, p * int(tests[j]) / share["table"]), names[other], "pair")])
                if c in prod:
                    pp, w = prod[c]
                    found.append((min(1.0, pp / share["product"]), w, "pair"))
                if c in masked:
                    found.append((min(1.0, masked[c] / share["masked"]), None, "masked"))
                if not found:
                    continue
                p_adj[j], w, why[j] = min(found, key=lambda f: f[0])
                rec[c]["pair_pvalue"], rec[c]["pair_with"] = p_adj[j], w
            q_pair = _qvalues(p_adj, 1.0) if keep_q < 1 else np.zeros(L)
            for j, c in enumerate(left):
                if p_adj[j] < pair_alpha and q_pair[j] < keep_q:
                    rec[c]["reason"] = why[j]

    # ---- no hint, but too little data to rule out a useful effect: keep --
    # Applied last, so it only ever adds keeps: the pair step above sees the
    # same leftovers (and partners) whatever min_ic is.
    if min_ic > 0:
        mi_floor = -0.5 * math.log1p(-min(min_ic, 0.999) ** 2)   # ic -> MI
        for r in rec.values():
            if r["reason"] is None and r.get("dof") is not None and not r["const"]:
                if (r["dof"] == 0 or r["n_rest"] < 5 * r["ky"]        # too sparse to judge
                        or _ncx2_cdf(r["g"], r["dof"], 2 * r["n"] * mi_floor) >= _EQUIV_ALPHA):
                    r["reason"] = "inconclusive"

    rows = [{"feature": c, "n": r["n"], "keep": r["reason"] is not None, "reason": r["reason"],
             "ic": r["ic"], "ic_pvalue": r["ic_pvalue"], "xi_pvalue": r["xi_pvalue"],
             "pair_pvalue": r["pair_pvalue"], "pair_with": r["pair_with"], "flag": r["flag"]}
            for c, r in rec.items()]
    out = pl.DataFrame(rows, schema={
        "feature": pl.String, "n": pl.Int64, "keep": pl.Boolean, "reason": pl.String,
        "ic": pl.Float64, "ic_pvalue": pl.Float64, "xi_pvalue": pl.Float64,
        "pair_pvalue": pl.Float64, "pair_with": pl.String, "flag": pl.String})
    return out.sort(["keep", "ic"], descending=True, nulls_last=True)
