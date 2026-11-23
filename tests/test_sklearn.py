import numpy as np
import pytest

pytest.importorskip("sklearn")
pd = pytest.importorskip("pandas")
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from treescreen import TreeScreen, RecursiveICSelector


def data(n=3000):
    r = np.random.default_rng(0); a, b, e = r.normal(size=(3, n))
    X = pd.DataFrame({"a": a, "b": b, "team": r.choice(list("ABCD"), n),
                      **{f"z{i}": r.normal(size=n) for i in range(15)}})
    return X, 2 * a + b + e


@pytest.mark.parametrize("cls", [TreeScreen, RecursiveICSelector])
def test_fit_transform_types_and_clone(cls):
    X, y = data(); s = cls().fit(X, y)
    assert {"a", "b"} <= set(s.selected_features_)
    assert isinstance(s.transform(X), pd.DataFrame)
    Xn = X.drop(columns="team").to_numpy()
    assert isinstance(cls().fit(Xn, y).transform(Xn), np.ndarray)
    assert clone(s).get_params() == s.get_params()
    with pytest.raises(ValueError):
        s.transform(X.iloc[:, :3])


@pytest.mark.parametrize("cls", [TreeScreen, RecursiveICSelector])
def test_passes_sklearn_estimator_checks(cls):
    from sklearn.utils.estimator_checks import check_estimator
    failed = [(r["check_name"], str(r["exception"])[:200])
              for r in check_estimator(cls(), on_fail=None) if r["status"] == "failed"]
    assert not failed


def test_runs_inside_cv_pipeline():
    X, y = data(); Xn = X.drop(columns="team")
    assert cross_val_score(make_pipeline(TreeScreen(), Ridge()), Xn, y, cv=3, scoring="r2").mean() > 0.7


def test_screen_accepts_string_labels_and_min_features():
    X, y = data()
    assert "a" in TreeScreen().fit(X, np.where(y > 0, "pos", "neg")).selected_features_
    Xn = X.drop(columns="team")
    out = TreeScreen(keep_p=0.0, min_ic=0.0, pair_partners=0).fit(
        Xn, np.random.default_rng(1).normal(size=len(Xn)))
    assert out.transform(Xn).shape[1] == 1
