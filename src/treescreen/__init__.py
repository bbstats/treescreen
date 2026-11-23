"""Model-free feature screening and ranking for polars / pandas / numpy data."""
from .core import nonlinear_corr, recursive_ic_score, screen_features

__all__ = ["nonlinear_corr", "recursive_ic_score", "screen_features", "TreeScreen", "RecursiveICSelector"]


def __getattr__(name):          # sklearn is optional: import the wrappers lazily
    if name in ("TreeScreen", "RecursiveICSelector"):
        from . import sklearn as _sk
        return getattr(_sk, name)
    raise AttributeError(name)
