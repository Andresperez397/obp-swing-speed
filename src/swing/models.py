"""Model pipelines and grouped cross-validation (ANALYSIS_PLAN.md sections 3-4)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GBM_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=it, min_samples_leaf=leaf)
            for d, lr, it, leaf in itertools.product([2, 3], [0.03, 0.1], [100, 300], [10, 30])]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column at its training 1st/99th percentiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def preprocessing():
    """Per-fold preprocessing for the linear models: impute, winsorize, standardize."""
    return make_pipeline(SimpleImputer(strategy="median"), Winsorizer(), StandardScaler())


def gbm(params: dict):
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(early_stopping=False, random_state=0, **params))


def inner_folds(X, y, groups, n_splits: int = 5):
    return list(GroupKFold(n_splits=n_splits).split(X, y, groups))


def fit_enet(X_tr, y_tr, g_tr):
    prep = preprocessing().fit(X_tr)
    Xp = prep.transform(X_tr)
    net = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], alphas=50, cv=inner_folds(Xp, y_tr, g_tr),
                       max_iter=20000).fit(Xp, y_tr)
    return prep, net


def tune_gbm(X_tr, y_tr, g_tr) -> dict:
    """Pick GBM settings by inner GroupKFold on the training hitters only."""
    folds = inner_folds(X_tr, y_tr, g_tr)
    best, best_mse = None, np.inf
    for params in GBM_GRID:
        mse = np.mean([np.mean((gbm(params).fit(X_tr[a], y_tr[a]).predict(X_tr[b]) - y_tr[b]) ** 2)
                       for a, b in folds])
        if mse < best_mse:
            best, best_mse = params, mse
    return best


def fit_predict(kind: str, X_tr, y_tr, g_tr, X_te) -> np.ndarray:
    """Fit on the training fold only (including all tuning) and predict the held-out fold."""
    if kind == "mean":
        return DummyRegressor(strategy="mean").fit(X_tr, y_tr).predict(X_te)
    if kind == "ols":
        return make_pipeline(preprocessing(), LinearRegression()).fit(X_tr, y_tr).predict(X_te)
    if kind == "enet":
        prep, net = fit_enet(X_tr, y_tr, g_tr)
        return net.predict(prep.transform(X_te))
    if kind == "gbm":
        return gbm(tune_gbm(X_tr, y_tr, g_tr)).fit(X_tr, y_tr).predict(X_te)
    raise ValueError(f"unknown model kind: {kind}")


def shuffled_group_folds(groups: np.ndarray, n_splits: int, seed: int):
    """GroupKFold with a random hitter-to-fold assignment (GroupKFold itself is deterministic)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.unique(groups))
    fold_of = {g: i % n_splits for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in groups])
    for k in range(n_splits):
        yield np.where(f != k)[0], np.where(f == k)[0]


def oof_predictions(df: pd.DataFrame, features: list[str], kind: str, outcome: str, group: str,
                    seed: int, scheme: str = "group", n_splits: int = 10) -> np.ndarray:
    X = df[features].to_numpy(dtype=float) if features else np.zeros((len(df), 1))
    y = df[outcome].to_numpy(dtype=float)
    g = df[group].to_numpy()
    pred = np.full(len(df), np.nan)
    if scheme == "group":
        splits = shuffled_group_folds(g, n_splits, seed)
    elif scheme == "random":
        splits = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X)
    else:
        raise ValueError(scheme)
    for tr, te in splits:
        pred[te] = fit_predict(kind, X[tr], y[tr], g[tr], X[te])
    return pred


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2_vs(y, p, p0) -> float:
    """R2 relative to a reference prediction (the out-of-fold training mean, B0)."""
    y, p, p0 = (np.asarray(a, dtype=float) for a in (y, p, p0))
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - p0) ** 2))
