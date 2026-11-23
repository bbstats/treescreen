"""
sklearn-compatible feature selectors.

    from treescreen import TreeScreen, RecursiveICSelector

    pipe = make_pipeline(TreeScreen(), LGBMRegressor())
    cross_val_score(pipe, X, y, cv=5)

TreeScreen             Pre-fit screen for tree models. Drops a column only when
                     nothing hints it is useful; keeps correlated copies.
RecursiveICSelector  Short, non-redundant list (for explanation or linear
                     models). Drops real columns, so it can cost a tree model
                     accuracy.

Put either INSIDE the pipeline. Each CV fold then picks its features from that
fold's training rows only, so the held-out rows never influence the choice.

X can be a pandas DataFrame, a polars DataFrame or a numpy array.
transform() returns the same kind of object it was given.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .core import _sample_rows, recursive_ic_score, screen_features

__all__ = ["TreeScreen", "RecursiveICSelector"]

_TARGET = "__target__"


def _check_2d(arr: np.ndarray) -> None:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D array instead. Reshape your data "
                         "either using array.reshape(-1, 1) if your data has a single feature "
                         "or array.reshape(1, -1) if it contains a single sample.")
    if np.iscomplexobj(arr):
        raise ValueError("Complex data not supported")


def _object_column(name: str, vals: np.ndarray) -> pl.Series:
    """A numpy object column: numbers stay numbers, anything else becomes text."""
    try:
        s = pl.Series(name, vals.tolist())
        if s.dtype != pl.Object:
            return s
    except Exception:
        pass
    return pl.Series(name, [None if v is None else str(v) for v in vals.tolist()])


def _to_polars(X) -> tuple[pl.DataFrame, list]:
    """Any supported X -> (polars frame with string column names, original names)."""
    if hasattr(X, "tocsr") or type(X).__module__.startswith("scipy.sparse"):
        raise TypeError("Sparse data was passed for X, but dense data is required. "
                        "Use '.toarray()' to convert to a dense numpy array.")
    if isinstance(X, pl.DataFrame):
        return X, list(X.columns)
    if hasattr(X, "iloc") and hasattr(X, "columns"):                 # pandas
        names = list(X.columns)
        try:
            out = pl.from_pandas(X.reset_index(drop=True))
        except Exception:                                            # e.g. no pyarrow
            out = pl.DataFrame({str(c): X[c].to_numpy() for c in names})
        out.columns = [str(c) for c in names]
        return out, names
    arr = np.asarray(X)
    _check_2d(arr)
    names = [f"x{i}" for i in range(arr.shape[1])]
    if arr.dtype == object:
        return pl.DataFrame([_object_column(c, arr[:, i]) for i, c in enumerate(names)]), names
    return pl.DataFrame(arr, schema=names, orient="row"), names


class _ColumnSelector(TransformerMixin, BaseEstimator):
    """Shared transform / support logic. Subclasses set self._keep_idx in fit."""

    def _store(self, names: list, cols: list, kept: list) -> None:
        pos = {c: i for i, c in enumerate(cols)}
        self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(names)
        self._keep_idx = np.array(sorted(pos[c] for c in kept), dtype=int)   # input order
        self.selected_features_ = [names[pos[c]] for c in kept]              # as ranked
        self.support_ = np.zeros(len(names), dtype=bool)
        self.support_[self._keep_idx] = True

    def transform(self, X):
        check_is_fitted(self, "support_")
        if isinstance(X, pl.DataFrame):
            self._check_width(X.width)
            return X.select([X.columns[i] for i in self._keep_idx])
        if hasattr(X, "iloc"):                                           # pandas
            self._check_width(X.shape[1])
            return X.iloc[:, self._keep_idx]
        if hasattr(X, "tocsr") or type(X).__module__.startswith("scipy.sparse"):
            raise TypeError("Sparse data was passed for X, but dense data is required. "
                            "Use '.toarray()' to convert to a dense numpy array.")
        arr = np.asarray(X)
        _check_2d(arr)
        self._check_width(arr.shape[1])
        return arr[:, self._keep_idx]

    def _check_width(self, width: int) -> None:
        if width != self.n_features_in_:
            raise ValueError(f"X has {width} features, but {type(self).__name__} is "
                             f"expecting {self.n_features_in_} features as input.")

    def get_support(self, indices: bool = False):
        check_is_fitted(self, "support_")
        return self._keep_idx.copy() if indices else self.support_.copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features, dtype=object)
        return names[self._keep_idx]

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = True
        tags.input_tags.string = True
        tags.input_tags.categorical = True
        return tags


def _frame_with_target(X, y, max_rows, seed):
    if y is None:
        raise ValueError("This selector requires y to be passed, but the target y is None.")
    df, names = _to_polars(X)
    if df.width == 0:
        raise ValueError(f"Found array with 0 feature(s) (shape=({df.height}, 0)) while a "
                         "minimum of 1 is required.")
    if df.height < 2:
        raise ValueError(f"Found array with {df.height} sample(s) (shape=({df.height}, "
                         f"{df.width})) while a minimum of 2 is required.")
    y_arr = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y).ravel()
    if np.iscomplexobj(y_arr):
        raise ValueError("Complex data not supported")
    if y_arr.shape[0] != df.height:
        raise ValueError(f"X has {df.height} rows but y has {y_arr.shape[0]}")
    cols = list(df.columns)
    if _TARGET in cols:
        raise ValueError(f"X may not contain a column named {_TARGET!r}")
    if y_arr.dtype.kind in "OUS":
        y_arr = y_arr.astype(str)
    data = _sample_rows(df.with_columns(pl.Series(_TARGET, y_arr)), max_rows, seed)
    return data, names, cols, y_arr


class TreeScreen(_ColumnSelector):
    """Pre-fit screen for tree models, built on screen_features.

    Drops a column only when it shows no hint of signal, alone or paired with
    another column, and there are enough rows to rule out a useful effect.
    Correlated copies are kept. y may be numeric, boolean or class labels
    (multiclass and string labels work).

    Parameters
    ----------
    keep_p : float, default 0.2
        A column is kept if any of its p-values (ic, xi, Spearman, coarse ic)
        is below this (uncorrected). Higher = keeps more.
    keep_q : float, default 0.5
        ...and its q-value (adaptive false discovery rate across the columns)
        is below this. Only bites when most columns look like junk. 1 = off.
    min_ic : float, default 0.1
        A column without a hint of signal is still kept unless the data rule
        out an ic this large (one-sided 5%). Higher = drops more on small
        data. 0 = off.
    pair_alpha : float, default 0.05
        Level for rescuing a column that only matters paired with another.
    pair_partners : int, default 64
        How many other columns each leftover is tried against. 0 = no pairs.
    drop_flagged : bool, default False
        Also drop columns flagged "id_like" or "row_order" (possible leaks).
    min_features : int, default 1
        Always keep at least this many (highest ic first).
    max_rows : int or None, default 200_000
        Screen on a random sample of this many rows (order kept) when X is
        larger. None = all rows.
    max_levels, bins, n_threads : passed to screen_features.
    random_state : int, default 0

    Attributes (after fit)
    ----------------------
    report_ : polars DataFrame, the screen_features table (see its `flag` column).
    selected_features_, support_, feature_names_in_, n_features_in_
    """

    def __init__(self, keep_p=0.2, keep_q=0.5, min_ic=0.1, pair_alpha=0.05, pair_partners=64,
                 drop_flagged=False, min_features=1, max_rows=200_000, max_levels=32, bins=None,
                 n_threads=None, random_state=0):
        self.keep_p = keep_p
        self.keep_q = keep_q
        self.min_ic = min_ic
        self.pair_alpha = pair_alpha
        self.pair_partners = pair_partners
        self.drop_flagged = drop_flagged
        self.min_features = min_features
        self.max_rows = max_rows
        self.max_levels = max_levels
        self.bins = bins
        self.n_threads = n_threads
        self.random_state = random_state

    def fit(self, X, y):
        seed = 0 if self.random_state is None else int(self.random_state)
        data, names, cols, _ = _frame_with_target(X, y, self.max_rows, seed)
        rep = screen_features(data, cols, _TARGET, keep_p=self.keep_p, keep_q=self.keep_q,
                              min_ic=self.min_ic, max_rows=None,       # sampled above

                              pair_alpha=self.pair_alpha, pair_partners=self.pair_partners,
                              bins=self.bins, max_levels=self.max_levels, seed=seed,
                              n_threads=self.n_threads)
        ok = pl.col("keep")
        if self.drop_flagged:
            ok = ok & pl.col("flag").is_null()
        kept = rep.filter(ok)["feature"].to_list()
        floor = min(int(self.min_features), len(cols))
        if len(kept) < floor:                              # top up with the highest ic
            extra = [c for c in rep.sort("ic", descending=True, nulls_last=True)["feature"]
                     if c not in set(kept)]
            kept += extra[:floor - len(kept)]
        self.report_ = rep
        self._store(names, cols, kept)
        return self


class RecursiveICSelector(_ColumnSelector):
    """Keep the features recursive_ic_score picks: a short, non-redundant list.
    For a tree model prefer TreeScreen; this one drops real columns that add
    little beyond the ones already picked. y may be numeric, boolean or class
    labels (strings: one class-vs-rest pass per class). Integer labels are
    treated as numbers; pass them as strings for an unordered multiclass target.

    Parameters
    ----------
    alpha : float or None, default 0.05
        Keep picking until the best remaining feature's adjusted p-value
        exceeds alpha. None = no significance stop.
    max_features : int or None, default None
        Keep at most this many features. With alpha=None, keeps exactly the
        top `max_features`.
    min_features : int, default 1
        Always keep at least this many (the top-ranked ones), so the next
        pipeline step never receives zero columns.
    max_rows : int or None, default None
        Rank on a random sample of this many rows (faster on big data).
        transform() always returns every row.
    max_levels, bins, fit_bins, n_folds, shrink, n_threads :
        Passed to recursive_ic_score.
    random_state : int, default 0
        Seed for the row sample and for recursive_ic_score.

    Attributes (after fit)
    ----------------------
    ranking_ : polars DataFrame, the full recursive_ic_score table.
    selected_features_ : names of the kept features, best first.
    support_ : boolean mask over the input columns.
    feature_names_in_, n_features_in_
    """

    def __init__(self, alpha=0.05, max_features=None, min_features=1, max_rows=None,
                 max_levels=32, bins=None, fit_bins=255, n_folds=5, shrink=1.0,
                 n_threads=None, random_state=0):
        self.alpha = alpha
        self.max_features = max_features
        self.min_features = min_features
        self.max_rows = max_rows
        self.max_levels = max_levels
        self.bins = bins
        self.fit_bins = fit_bins
        self.n_folds = n_folds
        self.shrink = shrink
        self.n_threads = n_threads
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        seed = 0 if self.random_state is None else int(self.random_state)
        data, names, cols, y_arr = _frame_with_target(X, y, self.max_rows, seed)
        limit = len(cols) if self.max_features is None else int(self.max_features)
        floor = min(int(self.min_features), len(cols))
        ranking = recursive_ic_score(
            data, cols, _TARGET, alpha=self.alpha, max_features=max(limit, floor),
            bins=self.bins, fit_bins=self.fit_bins, max_levels=self.max_levels,
            n_folds=self.n_folds, shrink=self.shrink, seed=seed, n_threads=self.n_threads)

        order = ranking["feature"].to_list()               # best first, unpicked last
        n_keep = int(ranking["step"].is_not_null().sum())
        if self.alpha is None and self.max_features is None:
            n_keep = len(order)                            # no rule given: keep everything
        n_keep = max(floor, min(n_keep, limit)) if limit else floor
        kept = order[:n_keep]

        self.ranking_ = ranking
        self._store(names, cols, kept)
        return self
