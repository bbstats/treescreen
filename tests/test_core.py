import numpy as np
import polars as pl
import pytest

from treescreen import nonlinear_corr, recursive_ic_score, screen_features

N = 6000


def rng():
    return np.random.default_rng(0)


def test_xi_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    r = rng(); x = r.normal(size=N); y = x**2 + 0.3 * r.normal(size=N)
    ours = nonlinear_corr(pl.DataFrame({"f": x, "t": y}), ["f"], "t")["xi"][0]
    assert abs(ours - stats.chatterjeexi(x, y).statistic) < 1e-9


def _xi_num_of(x, r):
    from treescreen.core import _xi_num
    vals, lab = np.unique(x, return_inverse=True)
    ro = np.argsort(r, kind="stable")
    return _xi_num(lab, vals.size, {"ro": ro, "rs": r[ro].astype(float)})


def test_xi_numerator_is_the_exact_average_over_tie_breaks():
    import itertools
    x = np.array([2, 1, 2, 3, 3, 3, 0, 1])                 # tie groups of size 1, 2, 2, 3
    r = np.array([5.0, 1, 8, 2, 7, 3, 6, 4])
    groups = [np.flatnonzero(x == v) for v in np.unique(x)]
    sums = [np.abs(np.diff(r[np.concatenate(p)])).sum()    # every within-group order
            for p in itertools.product(*[list(itertools.permutations(g)) for g in groups])]
    assert abs(_xi_num_of(x, r) - np.mean(sums)) < 1e-9
    assert _xi_num_of(np.array([0.3, 0.1, 0.2]), np.array([3.0, 1, 2])) == 2.0  # no ties
    xb = np.array([1, 0, 1, 1, 0])                           # two groups (no-sort path)
    rb = np.array([4.0, 2, 2, 5, 1])
    sb = [np.abs(np.diff(rb[np.concatenate(p)])).sum() for p in itertools.product(
        *[list(itertools.permutations(np.flatnonzero(xb == v))) for v in (0, 1)])]
    assert abs(_xi_num_of(xb, rb) - np.mean(sb)) < 1e-9


def test_xi_numerator_matches_the_direct_formula_for_any_group_count():
    # 2 groups (no sort), < 255 (uint8 radix sort), < 65535 (uint16), more (int64 keys)
    r = rng()
    for n, x in [(500, r.integers(0, 2, 500)), (3000, r.integers(0, 40, 3000)),
                 (3000, r.integers(0, 1500, 3000)), (70000, np.arange(70000))]:
        if n == 70000:
            x[1::50] = x[::50]                              # 68,600 groups of size 1 or 2
        y = np.round(r.normal(size=n), 1)                   # ties in r too
        rv = np.searchsorted(np.sort(y), y, side="right").astype(float)  # r_i = #{y_j <= y_i}
        o = np.lexsort((rv, x))
        gs = np.split(rv[o], np.flatnonzero(np.diff(x[o])) + 1)
        pair = lambda a, b: np.abs(a[:, None] - b[None, :]).sum()
        want = sum(pair(a, a) / a.size for a in gs) + \
            sum(pair(a, b) / (a.size * b.size) for a, b in zip(gs, gs[1:]))
        assert abs(_xi_num_of(x, rv) - want) <= 1e-9 * want


def test_xi_tie_variance_is_the_exact_permutation_variance():
    import itertools
    from treescreen.core import _xi_num, _xi_tie_var, _target_stats
    x = np.array([0, 0, 1, 2, 2, 2, 3])                      # groups of 2, 1, 3, 1
    t = _target_stats(np.array([3.0, 1, 1, 4, 5, 5, 2]), True, None)  # ties in y too
    lab, g = x, np.bincount(x)
    nums = []
    for p in itertools.permutations(range(7)):
        rp = t["r"][list(p)]
        ro = np.argsort(rp, kind="stable")
        nums.append(_xi_num(lab, 4, {"ro": ro, "rs": rp[ro]}))
    assert abs(_xi_tie_var(g, t) / 4 - np.var(nums)) < 1e-9 * np.var(nums)


def test_gamma_sf_matches_scipy():
    special = pytest.importorskip("scipy.special")
    from treescreen.core import _gamma_sf
    for a in (0.3, 0.61, 1.0, 2.5, 12.0, 49.0):
        for x in (0.01, 0.5, 1.0, 3.0, 10.0, 40.0, 80.0):
            want = special.gammaincc(a, x)
            assert abs(_gamma_sf(x, a) - want) <= 1e-10 + 1e-8 * want, (a, x)
    assert abs(_gamma_sf(130.0, 120.0) - special.gammaincc(120.0, 130.0)) < 2e-3  # Wilson-Hilferty


def test_xi_on_tied_columns_is_deterministic_and_calibrated():
    r = rng(); n = 1000
    X = np.column_stack([np.where(r.random(n) < 0.7, 0, r.integers(1, k, n))  # 0/1, few levels,
                         for k in (2, 6, 50) for _ in range(150)]).astype(float)  # zero-inflated
    df = pl.DataFrame(X, schema=[f"f{i}" for i in range(X.shape[1])]).with_columns(t=pl.Series(r.normal(size=n)))
    a = nonlinear_corr(df, df.columns[:-1], "t", seed=1)
    b = nonlinear_corr(df, df.columns[:-1], "t", seed=2)
    assert a.equals(b)                                       # no random tie-breaking
    p = a["xi_pvalue"].to_numpy()
    assert 0.03 < (p < 0.05).mean() < 0.08 and 0.15 < (p < 0.2).mean() < 0.25


def test_ic_tracks_pearson_and_sees_nonmonotone():
    r = rng(); x, e = r.normal(size=(2, N)); th = r.uniform(0, 2 * np.pi, N)
    lin = nonlinear_corr(pl.DataFrame({"f": x, "t": 0.5 * x + np.sqrt(0.75) * e}), ["f"], "t")["ic"][0]
    circ = nonlinear_corr(pl.DataFrame({"f": np.cos(th), "t": np.sin(th)}), ["f"], "t")["ic"][0]
    assert 0.42 < lin < 0.58 and circ > 0.9


def test_invariant_to_monotone_transforms():
    r = rng(); x = r.normal(size=N); y = x + r.normal(size=N)
    a = nonlinear_corr(pl.DataFrame({"f": x, "t": y}), ["f"], "t").row(0)[1:]
    b = nonlinear_corr(pl.DataFrame({"f": np.exp(x), "t": y**3}), ["f"], "t").row(0)[1:]
    assert a == b


def test_edge_dtypes_and_id_column_scores_zero():
    r = rng(); x = r.normal(size=N)
    df = pl.DataFrame({
        "good": x, "nan": np.where(r.random(N) < 0.1, np.nan, x), "const": np.ones(N),
        "all_null": pl.Series([None] * N, dtype=pl.Float64), "flag": x > 0,
        "cat": np.where(x > 0, "hi", "lo"), "row_id": [f"id{i}" for i in range(N)],
        "t": x + r.normal(size=N)})
    out = nonlinear_corr(df, [c for c in df.columns if c != "t"], "t")
    ic = dict(zip(out["feature"], out["ic"]))
    assert ic["good"] > 0.5 and ic["const"] == 0 and ic["all_null"] is None and ic["row_id"] < 0.1


def test_noise_pvalues_are_calibrated():
    r = rng(); X = r.normal(size=(2000, 200))
    df = pl.DataFrame(X, schema=[f"f{i}" for i in range(200)]).with_columns(t=pl.Series(r.normal(size=2000)))
    out = nonlinear_corr(df, df.columns[:-1], "t")
    assert (out["ic_pvalue"] < 0.05).mean() < 0.12 and (out["xi_pvalue"] < 0.05).mean() < 0.12


def test_screen_hint_pvalues_are_calibrated():
    from treescreen.core import _corr_rows
    r = rng(); X = r.normal(size=(2000, 200))
    df = pl.DataFrame(X, schema=[f"f{i}" for i in range(200)]).with_columns(t=pl.Series(r.normal(size=2000)))
    rows = _corr_rows(df.lazy(), df.columns[:-1], "t", None, 32, 0, None)
    for k in ("sp_p", "coarse_p"):
        assert np.mean([row[k] < 0.05 for row in rows]) < 0.09


def test_block_scoring_matches_per_column_scoring():
    from treescreen.core import _score, _score_block, _target_stats, _ranks
    r = rng(); n = 3000; y = r.normal(size=n)
    cols = [r.normal(size=n), y + r.normal(size=n), np.round(r.exponential(size=n)),  # ties
            (r.random(n) < 0.03).astype(float), np.ones(n), np.sin(3 * y) + r.normal(size=n)]
    t = _target_stats(y, True, None)
    A = np.stack([_ranks(c)[0] for c in cols])
    rows, _ = _score_block(A, t)
    for j, (c, b) in enumerate(zip(cols, rows)):
        a = _score(c, True, t, np.random.default_rng(1))
        for k in ("ic", "ic_pvalue", "coarse_p", "g", "sp_p", "spearman", "xi", "xi_pvalue"):
            assert (a[k] is None and b[k] is None) or abs(a[k] - b[k]) < 1e-9, (j, k)
        assert (a["dof"], a["n_rest"], a["const"]) == (b["dof"], b["n_rest"], b["const"])


def test_columns_with_nulls_match_scoring_their_rows_from_scratch():
    # the null path (polars ranks, the subset's target stats from the full
    # target's sort order or codes) equals scoring the non-null rows directly
    from treescreen.core import _score, _target_stats
    r = rng(); n = 3000; y = r.normal(size=n)
    xs = [y + r.normal(size=n), np.round(r.exponential(size=n)), r.normal(size=n),
          np.round(np.sin(2 * y) + r.normal(size=n), 1)]
    holes = [r.random(n) < share for share in (0.02, 0.3, 0.5, 0.3)]   # own pattern each
    for target in (y, np.round(y), (y > 0.5).astype(float),
                   np.digitize(y, [-1, 0, 1]).astype(str)):
        cls = target.dtype.kind == "U"
        df = pl.DataFrame({f"x{j}": pl.Series(x).scatter(np.flatnonzero(h), None)
                           for j, (x, h) in enumerate(zip(xs, holes))}).with_columns(t=target)
        got = {row["feature"]: row for row in nonlinear_corr(df, None, "t").iter_rows(named=True)}
        yv = pl.Series(target).rank("dense").to_numpy() if cls else target
        for j, (x, h) in enumerate(zip(xs, holes)):
            want = _score(x[~h], True, _target_stats(yv[~h], not cls, None), None)
            for k in ("ic", "ic_pvalue", "xi", "xi_pvalue", "spearman"):
                a, b = got[f"x{j}"][k], want[k]
                assert (a is None and b is None) or abs(a - b) < 1e-12, (target.dtype, j, k)


def test_category_codes_follow_string_order_not_polars_codes():
    # polars' physical codes change between processes; codes in string order don't
    from treescreen.core import _clean
    pl.Series(["zz", "yy", "b"]).cast(pl.Categorical)            # claims codes first
    for dtype in (pl.String, pl.Categorical, pl.Enum(["zz", "b", "yy"])):
        s = pl.DataFrame({"c": pl.Series(["b", "zz", None, "yy", "b"]).cast(dtype)})
        e, ordered = _clean("c", s.schema["c"])
        assert s.select(e)["c"].to_list() == [1, 3, None, 2, 1] and not ordered


def test_recursive_demotes_copies_and_sums_keeps_weak():
    r = rng(); a, c, w, e = r.normal(size=(4, N))
    df = pl.DataFrame({"a": a, "c": c, "a_copy": a + 0.05 * r.normal(size=N), "a_plus_c": a + c, "w": w,
                       **{f"z{i}": r.normal(size=N) for i in range(10)},
                       "y": 2 * np.sin(2 * a) + c + 0.3 * w + 0.5 * e})
    out = recursive_ic_score(df, [x for x in df.columns if x != "y"], "y", alpha=0.05)
    kept = out.filter(pl.col("step").is_not_null())["feature"].to_list()
    assert "w" in kept and "c" in kept and len({"a", "a_copy"} & set(kept)) == 1
    assert not any(k.startswith("z") for k in kept)


def test_recursive_keeps_hidden_partner_and_drops_interaction_copies():
    r = rng(); x1, x2, e = r.normal(size=(3, N))
    df = pl.DataFrame({"x1": x1, "x2": x2, "x1_copy": x1 + 0.05 * r.normal(size=N),
                       **{f"z{i}": r.normal(size=N) for i in range(10)}, "y": x1 * x2 + 0.5 * e})
    kept = recursive_ic_score(df, [x for x in df.columns if x != "y"], "y", alpha=0.05) \
        .filter(pl.col("step").is_not_null())["feature"].to_list()
    assert "x2" in kept and len({"x1", "x1_copy"} & set(kept)) == 1


def test_recursive_multiclass_target():
    r = rng(); a, b = r.normal(size=(2, N))
    y = np.where(a > 0.8, "A", np.where(b > 0.8, "B", np.where(a + b < -1, "C", "rest")))
    df = pl.DataFrame({"a": a, "b": b, **{f"z{i}": r.normal(size=N) for i in range(10)}, "y": y})
    out = recursive_ic_score(df, [c for c in df.columns if c != "y"], "y", alpha=0.05)
    kept = out.filter(pl.col("step").is_not_null())["feature"].to_list()
    assert set(kept) == {"a", "b"} and out["step"].drop_nulls().to_list() == [1, 2]
    two = recursive_ic_score(df.with_columns(y=pl.Series(np.where(a > 0, "hi", "lo"))),
                             ["a", "z0"], "y", alpha=0.05)
    assert two.filter(pl.col("step").is_not_null())["feature"].to_list() == ["a"]


def test_screen_keeps_copies_rescues_pairs_flags_ids():
    r = rng(); a, x1, x2, e = r.normal(size=(4, N)); pm = r.integers(0, 60, N); dur = r.integers(1, 30, N)
    df = pl.DataFrame({"pickup": pm, "dropoff": (pm + dur) % 60, "xor1": x1, "xor2": x2,
                       "a": a, "a_copy": a + 0.3 * r.normal(size=N), "line_no": np.arange(N),
                       **{f"z{i}": r.normal(size=N) for i in range(40)},
                       "y": dur + 3 * np.sign(x1 * x2) + 3 * a + 4 * e})
    out = screen_features(df, [c for c in df.columns if c != "y"], "y")
    keep = dict(zip(out["feature"], out["keep"])); flag = dict(zip(out["feature"], out["flag"]))
    assert all(keep[c] for c in ("pickup", "dropoff", "xor1", "xor2", "a", "a_copy"))
    assert flag["line_no"] == "id_like"
    assert sum(keep[f"z{i}"] for i in range(40)) < 28          # most junk goes


def test_row_order_flag_ignores_sparse_columns():
    r = rng()
    df = pl.DataFrame({"sparse": (r.random(N) < 0.01).astype(float),
                       "ordered": np.arange(N) + r.normal(size=N), "y": r.normal(size=N)})
    flag = dict(screen_features(df, ["sparse", "ordered"], "y").select("feature", "flag").iter_rows())
    assert flag["sparse"] is None and flag["ordered"] == "row_order"


def test_screen_keeps_untestable_categoricals_but_not_ids():
    r = rng(); lvl = r.permutation(np.arange(N) % 1000)        # 6 rows per category
    effect = r.normal(size=1000)[lvl]
    df = pl.DataFrame({"zipish": [f"z{v}" for v in lvl], "row_id": [f"id{i}" for i in range(N)],
                       "one_level": ["a"] * N, "y": effect + r.normal(size=N)})
    out = screen_features(df, ["zipish", "row_id", "one_level"], "y")
    reason = dict(zip(out["feature"], out["reason"]))
    assert reason == {"zipish": "untested", "row_id": None, "one_level": None}
    # small data, categorical target: no category reaches the size a table test needs
    r = rng(); n = 140; sp = r.integers(0, 20, n)
    df = pl.DataFrame({"sp": [f"s{v}" for v in sp], "y": [f"c{v % 5}" for v in sp]})
    assert screen_features(df, ["sp"], "y")["reason"][0] == "untested"


def test_noncentral_chi2_cdf_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    from treescreen.core import _ncx2_cdf
    for k in (1, 2, 8, 64, 256):
        for lam in (0.5, 8, 100):
            for q in (0.01, 0.05, 0.5):
                x = stats.ncx2.ppf(q, k, lam)
                assert abs(_ncx2_cdf(x, k, lam) - q) < (1e-9 if k == 1 else 0.02)


def test_screen_keeps_weak_columns_on_small_data_drops_junk_on_large():
    def dropped(n, min_ic):
        r = np.random.default_rng(n); y = r.normal(size=n)
        cols = {**{f"z{i}": r.normal(size=n) for i in range(60)},
                **{f"w{i}": 0.1 * y + np.sqrt(0.99) * r.normal(size=n) for i in range(60)}}
        out = screen_features(pl.DataFrame(cols).with_columns(y=y), list(cols), "y",
                              min_ic=min_ic, keep_q=1.0, pair_partners=0)
        keep = dict(zip(out["feature"], out["keep"]))
        return (np.mean([not keep[f"w{i}"] for i in range(60)]),
                np.mean([not keep[f"z{i}"] for i in range(60)]))
    weak_on, junk_small = dropped(1000, 0.1)
    _, junk_large = dropped(15000, 0.1)
    assert weak_on < 0.1                          # ic = 0.1 columns survive small data
    assert junk_small < 0.3 and junk_large > 0.3  # junk still goes once n can rule it out


def test_mostly_constant_columns_are_kept_constant_ones_dropped():
    r = rng(); n = 2000; sparse = np.zeros(n); sparse[r.choice(n, 3, replace=False)] = 1.0
    df = pl.DataFrame({"sparse": sparse, "const": np.ones(n), "y": r.normal(size=n)})
    keep = dict(screen_features(df, ["sparse", "const"], "y").select("feature", "keep").iter_rows())
    assert keep == {"sparse": True, "const": False}


def test_min_ic_only_adds_keeps():
    r = rng(); n = 3000; pm = r.integers(0, 60, n); dur = r.integers(1, 30, n)
    df = pl.DataFrame({"pickup": pm, "dropoff": (pm + dur) % 60,
                       **{f"z{i}": r.normal(size=n) for i in range(100)},
                       "y": dur + 4 * r.normal(size=n)})
    cols = [c for c in df.columns if c != "y"]
    kept = {m: set(screen_features(df, cols, "y", min_ic=m).filter("keep")["feature"])
            for m in (0.0, 0.1)}
    assert kept[0.0] < kept[0.1]               # 102 columns: partner limit is active


def test_screen_sees_informative_missingness():
    r = rng(); y = r.normal(size=N)
    flag_or_null = np.where(y + r.normal(size=N) > 1, 1.0, np.nan)    # "1 or missing"
    noise_masked = np.where(y > 0.5, np.nan, r.normal(size=N))          # value junk, nulls not
    random_nulls = np.where(r.random(N) < 0.3, np.nan, r.normal(size=N))
    df = pl.DataFrame({"flag": flag_or_null, "masked": noise_masked, "mcar": random_nulls, "y": y})
    out = screen_features(df, ["flag", "masked", "mcar"], "y", min_ic=0.0, pair_partners=0)
    keep = dict(zip(out["feature"], out["keep"]))
    assert keep["flag"] and keep["masked"]
    from treescreen.core import _corr_rows                # missing at random: calibrated
    X = np.where(r.random((N, 150)) < 0.3, np.nan, r.normal(size=(N, 150)))
    d = pl.DataFrame(X, schema=[f"f{i}" for i in range(150)]).with_columns(t=pl.Series(y))
    na_p = [row["na_p"] for row in _corr_rows(d.lazy(), d.columns[:-1], "t", None, 32, 0, None)]
    assert np.mean(np.array(na_p) < 0.05) < 0.1


def test_screen_samples_large_data_and_keeps_row_order():
    from treescreen.core import _sample_rows
    df = pl.DataFrame({"i": np.arange(1000)})
    s = _sample_rows(df, 100, 0)["i"].to_numpy()
    assert s.size == 100 and np.all(np.diff(s) > 0)
    r = rng(); n = 3000; a = r.normal(size=n)
    d = pl.DataFrame({"a": a, "z": r.normal(size=n), "pos": np.arange(n) + 0.5,   # float: not an id
                      "y": a + r.normal(size=n)})
    out = screen_features(d, ["a", "z", "pos"], "y", max_rows=500)
    assert out["n"].max() == 500
    assert dict(zip(out["feature"], out["flag"]))["pos"] == "row_order"


def test_screen_multiclass_target():
    r = rng(); a = r.normal(size=N)
    df = pl.DataFrame({"a": a, "z": r.normal(size=N), "y": np.where(a > 0.5, "hi", np.where(a < -0.5, "lo", "mid"))})
    out = screen_features(df, ["a", "z"], "y")
    assert dict(zip(out["feature"], out["keep"]))["a"]


def test_product_pair_permutation_moments_are_exact():
    import itertools
    from treescreen.core import _perm_z
    r = rng(); a = r.normal(size=7) ** 2; b = r.normal(size=7); b -= b.mean()
    S = np.array([a @ b[list(p)] for p in itertools.permutations(range(7))])
    ac = a - a.mean()
    z, g = _perm_z(1.0, (ac**2).sum(), (ac**3).sum(), (b**2).sum(), (b**3).sum(), 7)
    assert abs(S.mean()) < 1e-12
    assert abs(1 / z**2 - S.var()) < 1e-9 * S.var()
    assert abs(g - ((S - S.mean()) ** 3).mean() / S.var() ** 1.5) < 1e-9


def test_tail_p_matches_pearson3():
    stats = pytest.importorskip("scipy.stats")
    from treescreen.core import _tail_p
    z = np.array([0.5, 2.0, 4.0, 2.0, 3.0]); g = np.array([0.0, 0.3, 0.3, -0.8, 1.5])
    ref = [max(stats.norm.sf(v), stats.pearson3.sf(v, s)) + max(stats.norm.sf(v), stats.pearson3.cdf(-v, s))
           for v, s in zip(z, g)]
    assert np.allclose(_tail_p(z, g), ref, rtol=1e-6, atol=1e-12)


def test_screen_finds_a_pure_product_pair_among_many_columns():
    r = rng(); n, p = 4000, 300; X = r.uniform(size=(n, p))
    y = 0.3 * (X[:, 0] - 0.5) * (X[:, 1] - 0.5) * 12 + r.normal(size=n)   # 8% of var(y)
    names = [f"c{i}" for i in range(p)]
    out = screen_features(pl.DataFrame(X, schema=names).with_columns(y=pl.Series(y)), names, "y")
    keep = dict(zip(out["feature"], out["keep"])); w = dict(zip(out["feature"], out["pair_with"]))
    assert keep["c0"] and keep["c1"]
    assert w["c0"] == "c1" or w["c1"] == "c0"                     # found as a pair
    assert out["keep"].sum() < 40                                 # junk still goes


def test_screen_finds_a_masked_column():
    # c3's effect is ~0.03% of var(y): invisible next to c0..c2 on its own, clear
    # once their (additive) effects are taken out of y
    r = np.random.default_rng(6); n, p = 2000, 40; X = r.uniform(size=(n, p))
    y = 2 * (X[:, 0] + X[:, 1] + X[:, 2]) + 0.03 * X[:, 3] + 0.03 * r.normal(size=n)
    names = [f"c{i}" for i in range(p)]
    out = screen_features(pl.DataFrame(X, schema=names).with_columns(y=pl.Series(y)), names, "y")
    reason = dict(zip(out["feature"], out["reason"]))
    assert reason["c3"] == "masked"
    assert out["keep"].sum() < 20                                 # junk still goes


def test_masked_tests_null_and_normal_quantiles():
    from treescreen.core import _masked_tests, _norm_ppf, _residual, _bins_from_ranks, _ranks
    p = np.array([1e-10, 0.01, 0.3, 0.5, 0.9, 0.999])
    ref = [-6.361340902404056, -2.3263478740408408, -0.5244005127080407, 0.0,
           1.2815515655446004, 3.090232306167813]
    assert np.allclose(_norm_ppf(p), ref, rtol=2e-9, atol=1e-12)
    r = rng(); n, B = 600, 8
    X = r.uniform(size=(n, 2)); y = X[:, 0] + r.normal(size=n) * 0.3
    fit = [_bins_from_ranks(_ranks(X[:, 0])[0], n, B)]
    res = _residual(y[:, None], fit, [_ranks(X[:, 0])[0]], B)
    assert abs(np.corrcoef(res[0], X[:, 0])[0, 1]) < 0.02             # its effect is gone
    junk = np.stack([_bins_from_ranks(_ranks(r.uniform(size=n))[0], n, B) for _ in range(400)])
    pv, t = _masked_tests(junk.astype(np.uint8), res, B)
    assert (t == 2).all() and 0.02 < np.mean(pv * t < 0.1) < 0.18      # calibrated


def test_pandas_input_and_default_feature_list():
    pd = pytest.importorskip("pandas")
    r = rng(); n = 2000
    pdf = pd.DataFrame({"a": r.normal(size=n), "cat": pd.Categorical(r.choice(list("xyz"), n)),
                        "when": pd.date_range("2020-01-01", periods=n, freq="h"), 0: r.normal(size=n)})
    pdf["y"] = 2 * pdf["a"] + (pdf["cat"] == "x") + r.normal(size=n)
    out = screen_features(pdf, target_col="y")
    assert set(out["feature"]) == {"a", "cat", "when", "0"}
    pl_out = screen_features(pl.from_pandas(pdf.rename(columns=str)), ["a", "cat", "when", "0"], "y")
    assert out.sort("feature").equals(pl_out.sort("feature"))
    assert nonlinear_corr(pdf, ["a"], "y")["feature"].to_list() == ["a"]
    with pytest.raises(TypeError, match="target_col is required"):
        screen_features(pdf)
    with pytest.raises(TypeError, match="TreeScreen"):
        screen_features(pdf.to_numpy(), target_col="y")
    with pytest.raises(KeyError):
        screen_features(pdf, ["a", "zz"], "y")


def test_command_line(tmp_path, capsys):
    from treescreen.__main__ import main
    r = rng(); n = 3000
    X = r.uniform(size=(n, 12))
    y = 3 * X[:, 0] + np.sin(6 * X[:, 1]) + r.normal(size=n) * 0.3
    df = pl.DataFrame(X, schema=[f"f{i}" for i in range(12)]).with_columns(
        row_id=pl.int_range(n), y=pl.Series(y))
    path = tmp_path / "d.csv"
    df.write_csv(path)
    assert main([str(path), "-t", "y", "-o", str(tmp_path / "s.csv")]) == 0
    text = capsys.readouterr().out
    assert "KEEP" in text and "DROP" in text and "row_id: id-like" in text
    table = pl.read_csv(tmp_path / "s.csv")
    assert table.height == 13 and {"f0", "f1"} <= set(table.filter("keep")["feature"])
    assert main([str(path), "-t", "y", "-x", "row_id", "--print-keep"]) == 0
    names = capsys.readouterr().out.split()
    assert "f0" in names and "row_id" not in names
    with pytest.raises(SystemExit):
        main([str(path), "-t", "nope"])


def test_unusable_inputs_give_clear_errors():
    r = rng(); n = 500
    df = pl.DataFrame({"x": r.normal(size=n), "empty": [None] * n, "lst": [[1, 2]] * n,
                       "y": r.normal(size=n)})
    with pytest.warns(UserWarning, match="lst"):
        out = screen_features(df, target_col="y")                 # default list: skipped
    assert set(out["feature"]) == {"x", "empty"}
    assert not out.filter(pl.col("feature") == "empty")["keep"][0]
    with pytest.raises(TypeError, match="lst"):
        screen_features(df, ["x", "lst"], "y")                    # asked for: an error
    for bad, msg in [(df.with_columns(y=pl.lit(1.0)), "single value"),
                     (df.head(8), "at least 10")]:
        for fn in (screen_features, nonlinear_corr, recursive_ic_score):
            with pytest.raises(ValueError, match=msg):
                fn(bad, ["x"], "y")
