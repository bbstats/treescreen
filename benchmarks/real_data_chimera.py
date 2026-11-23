"""Real-data verdict for screen_features: ChimeraBoost on Grinsztajn (59) + high-card (14).

Same datasets, row caps, train/test splits and 25%-rows shrink as the external
report (docs/gbdt_findings_2026-11-20.md), loaded through ChimeraBoost's own
benchmark harness. Per (dataset, seed, size) three fits:

    base    every column
    screen  only the columns screen_features keeps (screen runs on training rows only)
    random  the same number of columns dropped at random (control)

Dropping a column before the fit = blocking it from every split and every cross
feature built from it. Error: test RMSE (regression), Brier (classification).

    python benchmarks/real_data_chimera.py --jobs 5 --threads 2
    python benchmarks/real_data_chimera.py --summarize benchmarks/results/<file>.jsonl

Results stream to benchmarks/results/real_data_<stamp>.jsonl (one line per task);
--resume <file> skips tasks already in it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
CB_BENCH = os.environ.get("CHIMERA_BENCH",
                          os.path.join(os.path.expanduser("~"), "Documents", "chimeraboost", "benchmarks"))
sys.path.insert(0, CB_BENCH)

import numpy as np  # noqa: E402

SIZES = (1.0, 0.25)
SEEDS = (0, 1, 2)

_RB = None
_COLS: dict = {}


def _rb():
    """ChimeraBoost's harness, with dataset registries filled and column names captured."""
    global _RB
    if _RB is None:
        import run_benchmarks as rb
        orig = rb._frame_to_dataset

        def capture(X_df, y, cats, task):
            _COLS["names"] = [str(c) for c in X_df.columns]
            return orig(X_df, y, cats, task)

        rb._frame_to_dataset = capture
        rb._add_grinsztajn_datasets()
        rb._add_highcard_datasets()
        _RB = rb
    return _RB


def dataset_keys():
    rb = _rb()
    return sorted(k for k in rb.DATASETS if k.startswith(("gr:", "hc:")) and "@" not in k)


def run_one(key: str, seed: int, frac: float, threads: int) -> dict:
    import polars as pl
    from sklearn.model_selection import train_test_split
    from treescreen import screen_features

    rb = _rb()
    _COLS.clear()
    X, y, cat, task = rb.DATASETS[key](1, np.random.default_rng(1000 + seed))
    p = X.shape[1]
    names = _COLS.get("names") or [f"c{i}" for i in range(p)]
    strat = y if task != "regression" else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed,
                                          stratify=strat)
    if frac < 1:
        Xtr, ytr = rb._subsample_train(Xtr, ytr, frac, task)
    cat_set = set(cat or [])

    # ---- screen on training rows -------------------------------------------
    cols = {}
    for i in range(p):
        v = Xtr[:, i]
        cols[f"c{i}"] = pl.Series(v.astype(str) if i in cat_set else v.astype(np.float64))
    yser = pl.Series(ytr.astype(str)) if task == "multiclass" else pl.Series(ytr)
    df = pl.DataFrame(cols).with_columns(__y__=yser)
    t0 = time.perf_counter()
    sc = screen_features(df, list(cols), "__y__", n_threads=threads)
    screen_s = time.perf_counter() - t0
    keep = sorted(int(f[1:]) for f in sc.filter("keep")["feature"])
    if not keep:                                    # never fit on zero columns
        keep = [int(sc["feature"][0][1:])]

    def fit(idx):
        Xa, Xb = Xtr[:, idx], Xte[:, idx]
        sub_cat = [j for j, i in enumerate(idx) if i in cat_set] or None
        if sub_cat is None and Xa.dtype == object:
            Xa, Xb = Xa.astype(np.float64), Xb.astype(np.float64)
        m, fit_s, _, best = rb._run_chimera(task, Xa, ytr, Xb, yte, sub_cat, threads)
        return {"err": m["rmse"] if task == "regression" else m["brier"],
                "fit_s": fit_s, "best_iter": best}

    base = fit(list(range(p)))
    if len(keep) == p:
        screen = rand = dict(base)
    else:
        screen = fit(keep)
        r = np.random.default_rng([zlib.crc32(key.encode()), seed, int(frac * 100)])
        rand = fit(sorted(r.choice(p, len(keep), replace=False).tolist()))

    def nm(f):
        return names[int(f[1:])] if f and f.startswith("c") else f

    table = [{"feature": nm(r["feature"]), "keep": r["keep"], "reason": r["reason"],
              "ic": r["ic"], "ic_p": r["ic_pvalue"], "xi_p": r["xi_pvalue"],
              "pair_p": r["pair_pvalue"], "pair_with": nm(r["pair_with"]), "flag": r["flag"]}
             for r in sc.iter_rows(named=True)]
    return {"dataset": key, "suite": key.split(":")[0], "task": task, "seed": seed,
            "frac": frac, "n_train": int(len(ytr)), "p": p, "n_keep": len(keep),
            "screen_s": screen_s, "base": base, "screen": screen, "random": rand,
            "screen_table": table}


def _job(args):
    try:
        return run_one(*args)
    except Exception as e:                           # record, don't abort the run
        import traceback
        return {"dataset": args[0], "seed": args[1], "frac": args[2],
                "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(path: str) -> None:
    recs = [json.loads(s) for s in open(path, encoding="utf-8") if s.strip()]
    errs = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r and r.get("dataset")]
    for e in errs:
        print(f"  [error] {e['dataset']} seed {e['seed']} frac {e['frac']}: {e['error']}")

    groups = defaultdict(list)
    for r in recs:
        groups[(r["suite"], r["frac"], r["dataset"])].append(r)

    print(f"\n{len(recs)} tasks from {path}")
    print("\nWins vs losses per dataset (error averaged over seeds; ties left out; "
          "two-sided sign test)")
    hdr = f"{'stratum':22s} {'datasets':>8s} {'dropped any':>11s} " \
          f"{'screen vs all':>20s} {'random vs all':>20s} {'screen vs random':>20s} " \
          f"{'median change':>13s} {'cols dropped':>12s} {'time':>6s}"
    print(hdr)
    print("-" * len(hdr))
    detail = {}
    for suite in ("gr", "hc"):
        for frac in SIZES:
            ds = {k[2]: v for k, v in groups.items() if k[0] == suite and k[1] == frac}
            if not ds:
                continue
            cnt = {"sa": [0, 0, 0], "ra": [0, 0, 0], "sr": [0, 0, 0]}
            changes, dropped_share, t_new, t_old, rows = [], [], 0.0, 0.0, []
            any_drop = 0
            for name, rs in ds.items():
                b = np.mean([r["base"]["err"] for r in rs])
                s = np.mean([r["screen"]["err"] for r in rs])
                q = np.mean([r["random"]["err"] for r in rs])
                drop = np.mean([1 - r["n_keep"] / r["p"] for r in rs])
                for key, (x, z) in (("sa", (s, b)), ("ra", (q, b)), ("sr", (s, q))):
                    cnt[key][0 if x < z else 1 if x > z else 2] += 1
                if drop > 0:
                    any_drop += 1
                    changes.append(100 * (s / b - 1) if b else 0.0)
                    dropped_share.append(drop)
                    t_new += sum(r["screen_s"] + r["screen"]["fit_s"] for r in rs)
                    t_old += sum(r["base"]["fit_s"] for r in rs)
                rows.append((name, 100 * (s / b - 1) if b else 0.0, drop, rs))
            label = f"{'Grinsztajn' if suite == 'gr' else 'High-card'}, " \
                    f"{'full rows' if frac == 1 else '25% rows'}"

            def cell(c):
                return f"{c[0]} vs {c[1]} (p={sign_p(c[0], c[1]):.2g})"
            med = f"{np.median(changes):+.2f}%" if changes else "-"
            dsh = f"{100 * np.median(dropped_share):.0f}%" if dropped_share else "-"
            tim = f"{t_new / t_old:.2f}x" if t_old else "-"
            print(f"{label:22s} {len(ds):8d} {any_drop:11d} {cell(cnt['sa']):>20s} "
                  f"{cell(cnt['ra']):>20s} {cell(cnt['sr']):>20s} {med:>13s} {dsh:>12s} {tim:>6s}")
            detail[label] = rows

    print("\n'median change' = screen error vs all columns, over datasets where the screen "
          "dropped something (negative = better).\n'cols dropped' = median share of columns "
          "dropped on those datasets. 'time' = (screen + screened fit) / all-columns fit, "
          "same datasets.")

    for label, rows in detail.items():
        hurt = sorted((r for r in rows if r[2] > 0), key=lambda r: -r[1])[:5]
        shown = {r[0] for r in hurt}
        helped = sorted((r for r in rows if r[2] > 0 and r[0] not in shown),
                        key=lambda r: r[1])[:3]
        print(f"\n{label}: worst 5 / best 3 (error change, share of columns dropped)")
        for name, ch, drop, rs in hurt + helped:
            dropped = sorted({t["feature"] for r in rs for t in r["screen_table"] if not t["keep"]})
            show = ", ".join(dropped[:8]) + (f" ... (+{len(dropped) - 8})" if len(dropped) > 8 else "")
            print(f"  {ch:+7.2f}%  {100 * drop:4.0f}%  {name:52s} {show}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--datasets", nargs="*", help="substring filter on dataset keys")
    ap.add_argument("--sizes", nargs="*", type=float, default=list(SIZES))
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    ap.add_argument("--resume", help="append to this JSONL, skipping finished tasks")
    ap.add_argument("--summarize", help="only print the summary of this JSONL")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
        return

    os.environ["POLARS_MAX_THREADS"] = str(a.threads)   # inherited by worker processes
    keys = dataset_keys()
    if a.datasets:
        keys = [k for k in keys if any(s in k for s in a.datasets)]
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = a.resume or os.path.join(out_dir, time.strftime("real_data_%Y%m%d-%H%M%S.jsonl"))
    done = set()
    if a.resume and os.path.exists(path):
        for s in open(path, encoding="utf-8"):
            r = json.loads(s)
            if "error" not in r and r.get("dataset"):
                done.add((r["dataset"], r["seed"], r["frac"]))

    tasks = [(k, s, f, a.threads) for f in a.sizes for k in keys for s in range(a.seeds)
             if (k, s, f) not in done]
    import chimeraboost
    core = os.path.join(HERE, "..", "src", "treescreen", "core.py")
    header = {"header": True, "chimeraboost": chimeraboost.__version__,
              "chimeraboost_file": chimeraboost.__file__,
              "treescreen_core_sha1": hashlib.sha1(open(core, "rb").read()).hexdigest()[:12],
              "jobs": a.jobs, "threads": a.threads, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"{len(tasks)} tasks -> {path}\n{header}", flush=True)
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        with ProcessPoolExecutor(a.jobs) as ex:
            futs = [ex.submit(_job, t) for t in tasks]
            for i, fu in enumerate(as_completed(futs), 1):
                r = fu.result()
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                if "error" in r:
                    msg = f"ERROR {r['error']}"
                else:
                    ch = 100 * (r["screen"]["err"] / r["base"]["err"] - 1) if r["base"]["err"] else 0
                    msg = f"kept {r['n_keep']}/{r['p']}  screen {r['screen_s']:.1f}s  error {ch:+.2f}%"
                print(f"[{i}/{len(tasks)} {time.time() - t0:6.0f}s] {r['dataset']} seed {r['seed']} "
                      f"frac {r['frac']}: {msg}", flush=True)
    summarize(path)


if __name__ == "__main__":
    main()
