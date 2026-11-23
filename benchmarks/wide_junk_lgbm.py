import sys, time, warnings; warnings.filterwarnings("ignore")
# optional screen_features kwargs: python wide_junk_lgbm.py keep_q=0.5 min_ic=0.1
KW = {k: float(v) for k, v in (a.split("=") for a in sys.argv[1:])}
import numpy as np, polars as pl, lightgbm as lgb
from treescreen import screen_features, recursive_ic_score, nonlinear_corr
def make(n, seed, p=1000):
    rng = np.random.default_rng(seed); X = rng.uniform(size=(n, p))
    f = (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-.5)**2 + 10*X[:,3] + 5*X[:,4]      # Friedman
         + 8*(X[:,5]-.5)*(X[:,6]-.5)                                                     # product interaction
         + sum(4*0.75**k*X[:,7+k] for k in range(7))                                     # decaying linear
         + 2*(X[:,14] > .5) + 1.5*(X[:,15] > .7) + 2*np.sin(6*X[:,16]) + np.sin(9*X[:,17])
         + 0.8*X[:,18] + 0.6*X[:,19])
    return X, f + rng.normal(size=n)*2.0
def run(n, task, seeds=(0, 1)):
    rows = {k: [] for k in ("default", "old rule", "screen", "perfect")}
    info = []
    for seed in seeds:
        X, y = make(n + 5000, seed)
        if task == "binary": y = (y > np.median(y)).astype(int)
        Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]
        names = [f"c{i}" for i in range(X.shape[1])]
        df = pl.DataFrame(Xtr, schema=names).with_columns(y=pl.Series(ytr))
        def fit(cols, t_sel=0.0):
            t = time.perf_counter()
            M = lgb.LGBMRegressor if task == "reg" else lgb.LGBMClassifier
            m = M(n_estimators=300, learning_rate=.05, num_leaves=31, verbose=-1, n_jobs=2).fit(Xtr[:, cols], ytr)
            dt = time.perf_counter() - t + t_sel
            if task == "reg": err = np.sqrt(np.mean((m.predict(Xte[:, cols]) - yte)**2))
            else: err = np.mean((m.predict_proba(Xte[:, cols])[:, 1] - yte)**2)
            return dt, err
        rows["default"].append(fit(list(range(1000))))
        t = time.perf_counter()
        nc = nonlinear_corr(df, names, "y"); k = len(names)
        marg = set(nc.filter((pl.col("ic_pvalue")*k < .05) | (pl.col("xi_pvalue")*k < .05))["feature"])
        rec = set(recursive_ic_score(df, names, "y", alpha=.05).filter(pl.col("step").is_not_null())["feature"])
        old = sorted(int(c[1:]) for c in marg | rec); t_old = time.perf_counter() - t
        rows["old rule"].append(fit(old, t_old))
        t = time.perf_counter(); s = screen_features(df, names, "y", **KW); t_s = time.perf_counter() - t
        new = sorted(int(c[1:]) for c in s.filter("keep")["feature"])
        rows["screen"].append(fit(new, t_s))
        rows["perfect"].append(fit(list(range(20))))
        info.append((sum(c < 20 for c in old), len(old), sum(c < 20 for c in new), len(new), t_s,
                     s.filter(pl.col("reason") == "pair")["feature"].to_list()))
    d = np.mean(rows["default"], axis=0)
    print(f"{task}, {n:,} rows: default fit {d[0]:.1f}s")
    for k in ("old rule", "screen", "perfect"):
        v = np.mean(rows[k], axis=0); print(f"   {k:9s}: time {v[0]/d[0]:.2f}x, error {100*(v[1]/d[1]-1):+.1f}%")
    for i in info: print(f"   old rule kept {i[0]}/20 real ({i[1]} total) | screen kept {i[2]}/20 real ({i[3]} total), screen took {i[4]:.1f}s, pair rescues: {i[5]}")
for n, task in [(5000, "reg"), (20000, "reg"), (5000, "binary")]: run(n, task)
