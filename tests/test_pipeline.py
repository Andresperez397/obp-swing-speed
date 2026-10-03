import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from swing import data, models, stats  # noqa: E402

HAVE_DATA = (data.RAW / "poi_metrics.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run scripts/fetch_data.py first")


@pytest.fixture(scope="module")
def df():
    return data.load()


@needs_data
def test_every_column_has_exactly_one_role():
    roles = data.column_roles()
    assert "UNASSIGNED" not in roles.values()
    assert sum(v == "outcome" for v in roles.values()) == 1


@needs_data
def test_join_keeps_every_swing_and_one_session_per_hitter(df):
    assert len(df) == len(pd.read_csv(data.RAW / "poi_metrics.csv"))
    assert df["session_swing"].is_unique
    assert (df.groupby(data.GROUP)["session"].nunique() == 1).all()


@needs_data
def test_blocks_are_nested_and_free_of_leakage_and_duplicates():
    b = data.blocks()
    assert set(b["B1"]) < set(b["B2"]) < set(b["B3"]) < set(b["B4"])
    for name, cols in b.items():
        assert data.OUTCOME not in cols, name
        assert not set(cols) & set(data.LEAKAGE), name
        assert not set(cols) & set(data.DROP_DUPLICATES), name
        assert len(cols) == len(set(cols)), name


@needs_data
def test_no_remaining_duplicate_predictors(df):
    corr = np.array(df[data.blocks()["B4"]].astype(float).corr().abs())
    np.fill_diagonal(corr, 0)
    assert np.nanmax(corr) < 0.995


@needs_data
def test_cleaning_rules(df):
    assert (df["pitch_speed_mph"].dropna() > 0).all()           # zero pitch speeds set to missing
    ss = np.sqrt(sum(df[f"sweet_spot_velo_mph_contact_{a}"] ** 2 for a in "xyz"))
    assert np.allclose(df["sweet_spot_speed_mph"], ss)
    assert df["mass_kg"].between(55, 130).all() and df["height_m"].between(1.5, 2.1).all()


def test_grouped_folds_never_share_a_hitter():
    g = np.repeat(np.arange(30), 4)
    seen = []
    for tr, te in models.shuffled_group_folds(g, 10, seed=3):
        assert not set(g[tr]) & set(g[te])
        seen.extend(te)
    assert sorted(seen) == list(range(len(g)))


def test_oof_mean_uses_only_training_folds():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 3)
    d = pd.DataFrame({"y": rng.normal(70, 5, len(g)), "g": g})
    pred = models.oof_predictions(d, [], "mean", "y", "g", seed=1, n_splits=5)
    for tr, te in models.shuffled_group_folds(g, 5, seed=1):
        assert np.allclose(pred[te], d["y"].iloc[tr].mean())


@pytest.mark.parametrize("kind", ["ols", "enet", "gbm"])
def test_held_out_hitter_outcome_cannot_affect_its_own_prediction(kind):
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(40), 3)
    X = rng.normal(size=(len(g), 4))
    y = X @ np.array([3.0, -2.0, 0.0, 1.0]) + rng.normal(0, 1, len(g))
    d = pd.DataFrame(X, columns=list("abcd")).assign(y=y, g=g)
    before = models.oof_predictions(d, list("abcd"), kind, "y", "g", seed=0, n_splits=5)
    bad = d.copy()
    bad.loc[bad["g"] == 7, "y"] = 1e6
    after = models.oof_predictions(bad, list("abcd"), kind, "y", "g", seed=0, n_splits=5)
    rows = (d["g"] == 7).to_numpy()
    assert np.allclose(before[rows], after[rows])
    assert not np.allclose(before[~rows], after[~rows])


def test_gbm_has_no_hidden_early_stopping():
    hgb = models.gbm(models.GBM_GRID[0]).steps[-1][1]
    assert hgb.get_params()["early_stopping"] is False
    assert len(models.GBM_GRID) == 16


def test_collision_q_recovers_known_value():
    rng = np.random.default_rng(4)
    n, q = 4000, 0.21
    v_bat, v_pitch = rng.normal(70, 5, n), rng.normal(60, 3, n)
    ev_max = q * v_pitch + (1 + q) * v_bat
    ev = ev_max - np.abs(rng.normal(0, 8, n)) + rng.normal(0, 0.3, n)  # mis-hits only lose speed
    # Near the top of the distribution the relation is the squared-up physics.
    assert stats.collision_q(ev, v_bat, v_pitch, tau=0.99) == pytest.approx(q, abs=0.03)


def test_cluster_bootstrap_resamples_whole_clusters():
    sizes = {1: 1, 2: 2, 3: 3, 4: 5}
    d = pd.DataFrame({"g": np.repeat(list(sizes), list(sizes.values()))})

    def whole(sub):
        c = sub["g"].value_counts()
        return {"ok": all(c[k] % sizes[k] == 0 for k in c.index)}

    assert stats.cluster_bootstrap(d, "g", whole, n_boot=200, seed=0)["ok"].all()


def test_icc_recovers_realized_value():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(200), 5)
    u = rng.normal(0, 2, 200)
    y = u[g] + rng.normal(0, 1, len(g))
    realized = np.var(u, ddof=1) / (np.var(u, ddof=1) + 1)
    est = stats.icc(pd.DataFrame({"y": y, "g": g}), "y", "g", n_boot=100)
    assert est["ci_low"] < realized < est["ci_high"]
