"""Command line: which columns to keep for a tree model.

    python -m treescreen DATA --target COL [--exclude COL ...] [--out TABLE] [--print-keep]

DATA: .csv, .tsv, .parquet, .feather / .arrow / .ipc or .ndjson / .jsonl (read with polars).
Prints a short summary; --out writes the full table (.csv, .parquet or .json);
--print-keep prints only the kept column names, one per line (for scripts).
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time

import polars as pl

from .core import _supported, screen_features

_READERS = {
    ".csv": lambda p: pl.scan_csv(p, infer_schema_length=10_000),
    ".tsv": lambda p: pl.scan_csv(p, separator="\t", infer_schema_length=10_000),
    ".parquet": pl.scan_parquet,
    ".feather": pl.scan_ipc, ".arrow": pl.scan_ipc, ".ipc": pl.scan_ipc,
    ".ndjson": pl.scan_ndjson, ".jsonl": pl.scan_ndjson,
}


def _read(path: str) -> pl.LazyFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext not in _READERS:
        raise SystemExit(f"treescreen: can't read {ext or 'files without an extension'}; "
                         f"use one of {', '.join(_READERS)}")
    return _READERS[ext](path)


def _write(table: pl.DataFrame, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        table.write_parquet(path)
    elif ext == ".json":
        table.write_json(path)
    else:
        table.write_csv(path)


def _names(cols: list, cap: int = 80, width: int = 88) -> str:
    more = f" ... (+{len(cols) - cap} more)" if len(cols) > cap else ""
    return textwrap.fill(", ".join(cols[:cap]) + more, width, initial_indent="    ",
                         subsequent_indent="    ", break_on_hyphens=False, break_long_words=False)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m treescreen",
        description="Model-free screen: which columns to keep for a tree model (GBDT). "
                    "Drops only columns with no hint of signal, alone, in a pair, or once "
                    "stronger columns are accounted for.")
    ap.add_argument("data", help="table to screen: .csv .tsv .parquet .feather .ndjson")
    ap.add_argument("-t", "--target", required=True, help="target column")
    ap.add_argument("-x", "--exclude", nargs="+", default=[], metavar="COL",
                    help="columns to leave out of the screen (IDs, leaks, other targets)")
    ap.add_argument("--classes", action="store_true",
                    help="treat a numeric target as class labels (e.g. 0 / 1 / 2 coded classes)")
    ap.add_argument("--max-rows", type=int, default=200_000,
                    help="screen a random sample of this many rows (default 200000; 0 = all)")
    ap.add_argument("-o", "--out", help="write the full table here (.csv, .parquet or .json)")
    ap.add_argument("--print-keep", action="store_true",
                    help="print only the kept column names, one per line")
    a = ap.parse_args(argv)

    lf = _read(a.data)
    names = lf.collect_schema().names()
    if a.target not in names:
        raise SystemExit(f"treescreen: target column {a.target!r} not found; columns: "
                         f"{', '.join(names[:20])}{' ...' if len(names) > 20 else ''}")
    unknown = [c for c in a.exclude if c not in names]
    if unknown:
        raise SystemExit(f"treescreen: --exclude columns not found: {', '.join(unknown)}")
    feats = [c for c in names if c != a.target and c not in set(a.exclude)]
    if not feats:
        raise SystemExit("treescreen: no feature columns left to screen")
    if a.classes:
        lf = lf.with_columns(pl.col(a.target).cast(pl.String))

    schema = lf.collect_schema()
    bad = [c for c in feats if not _supported(schema[c])]
    if bad:
        print(f"treescreen: left out, unsupported types: "
              f"{', '.join(f'{c} ({schema[c]})' for c in bad)}", file=sys.stderr)
        feats = [c for c in feats if c not in set(bad)]
    t0 = time.perf_counter()
    try:
        df = lf.select([*feats, a.target]).collect()
    except pl.exceptions.PolarsError as e:
        raise SystemExit(f"treescreen: could not read {a.data}: {e}")
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    try:
        out = screen_features(df, feats, a.target, max_rows=a.max_rows or None)
    except (ValueError, TypeError) as e:
        raise SystemExit(f"treescreen: {e}")
    t_screen = time.perf_counter() - t0

    kept = out.filter("keep")
    if a.out:
        _write(out, a.out)
    if a.print_keep:
        print("\n".join(kept["feature"].to_list()))
        return 0

    y = df[a.target]
    kind = ("binary" if y.drop_nulls().n_unique() == 2 else
            "class labels" if y.dtype == pl.String or isinstance(y.dtype, (pl.Categorical, pl.Enum))
            else "numeric")
    rows = min(df.height, a.max_rows) if a.max_rows else df.height
    print(f"treescreen: {a.data}: {df.height:,} rows x {len(feats)} feature columns, target "
          f"{a.target!r} ({kind}); screened {rows:,} rows in {t_screen:.2f} s (read {t_read:.2f} s)")
    by = dict(kept.group_by("reason").len().iter_rows())
    order = ["alone", "pair", "masked", "inconclusive", "untested"]
    print(f"\nKEEP {kept.height} of {len(feats)}: "
          + ", ".join(f"{by[r]} {r}" for r in order if by.get(r)))
    strong = kept.filter((pl.col("reason") == "alone") & (pl.col("ic") > 0)).head(8)
    if strong.height:
        print("  strongest (ic): " + ", ".join(f"{f} {v:.2f}" for f, v in
                                                strong.select("feature", "ic").iter_rows()))
    for r in order:
        cols = kept.filter(pl.col("reason") == r)["feature"].to_list()
        if cols:
            print(f"  {r}:")
            print(_names(cols))
    drop = out.filter(~pl.col("keep"))["feature"].to_list()
    print(f"\nDROP {len(drop)} of {len(feats)}" + (":" if drop else ""))
    if drop:
        print(_names(drop))
    flagged = out.filter(pl.col("flag").is_not_null())
    if flagged.height:
        print("\nCHECK (may score well through a leak, e.g. rows sorted by the target):")
        for f, flag, k in flagged.select("feature", "flag", "keep").iter_rows():
            print(f"    {f}: {flag.replace('_', '-')}, {'kept' if k else 'dropped'}")
    print("\nreasons: alone = signal on its own; pair = signal together with another column;\n"
          "masked = signal once stronger columns are accounted for; inconclusive = too few rows\n"
          "to rule out a useful effect; untested = categories too small to test."
          + ("" if a.out else "\nFull table with p-values: add --out screen.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
