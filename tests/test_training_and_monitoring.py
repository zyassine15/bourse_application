import json

import joblib
import numpy as np
import pytest

from app.features import FEATURE_NAMES, prepare_features
from app.model import ModelRegistry
from app.monitoring import DriftMonitor, psi
from tests.conftest import make_prices
from training.train import (
    build_dataset,
    evaluate,
    feature_reference,
    new_estimator,
    walk_forward_splits,
)


def _dataset():
    frames = {f"T{i}": make_prices(700, drift=0.0004, seed=i) for i in range(6)}
    return build_dataset(frames)


def test_walk_forward_has_no_leakage():
    dates = np.sort(np.repeat(np.arange(1000), 3))
    horizon = 21
    splits = list(walk_forward_splits(dates, horizon, n_splits=5))
    assert len(splits) == 5
    for train_mask, test_mask in splits:
        assert dates[train_mask].max() + horizon <= dates[test_mask].min()
        assert not (train_mask & test_mask).any()


def test_dataset_labels_use_future_only():
    data = _dataset()
    assert {"fwd_court", "fwd_moyen", "fwd_long"} <= set(data.columns)
    # Les dernières lignes de chaque ticker n'ont pas (encore) de rendement futur.
    assert data["fwd_long"].isna().any()


def test_evaluate_returns_metrics():
    metrics = evaluate(_dataset(), "moyen", n_splits=3)
    assert 0.3 < metrics["auc_mean"] < 0.7  # marche aléatoire : pas de signal exploitable
    assert metrics["n_samples"] > 1000


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    train = rng.normal(0, 1, 5000)
    edges = np.quantile(train, np.linspace(0, 1, 11)[1:-1])
    counts, _ = np.histogram(train, bins=np.concatenate([[-np.inf], edges, [np.inf]]))
    frac = counts / counts.sum()
    assert psi(edges, frac, rng.normal(0, 1, 2000)) < 0.1
    assert psi(edges, frac, rng.normal(1.5, 1, 2000)) > 0.25


def test_drift_monitor_status():
    data = _dataset()
    monitor = DriftMonitor(feature_reference(data))
    assert monitor.report()["status"] == "insufficient_data"
    for _, row in data.sample(200, random_state=0)[FEATURE_NAMES].iterrows():
        monitor.observe(row)
    assert monitor.report()["status"] == "ok"
    shifted = data[FEATURE_NAMES].iloc[:200].copy()
    shifted["rsi_14"] = 95.0
    monitor = DriftMonitor(feature_reference(data))
    for _, row in shifted.iterrows():
        monitor.observe(row)
    assert monitor.report()["status"] == "alert"


def test_registry_loads_model_and_scores(tmp_path):
    data = _dataset()
    labelled = data.dropna(subset=["fwd_moyen"])
    y = (labelled["fwd_moyen"] > 0).astype(int)
    joblib.dump(new_estimator().fit(labelled[FEATURE_NAMES], y), tmp_path / "m.joblib")
    meta = {
        "version": "test-1",
        "features": FEATURE_NAMES,
        "horizons": {"moyen": {"file": "m.joblib", "base_rate": float(y.mean())}},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))

    registry = ModelRegistry(tmp_path)
    assert registry.version == "test-1"
    feats = prepare_features(make_prices())
    assert registry.ml_score("court", feats) is None
    scores = registry.ml_score("moyen", feats)
    assert scores.between(0, 100).all()


def test_registry_rejects_incompatible_features(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"version": "x", "features": ["a"], "horizons": {}}))
    assert ModelRegistry(tmp_path).version == "rules-only"


def test_validation_status_from_backtest(tmp_path):
    def status(ic_mean, tstat):
        report = {"horizons": {"moyen": {"ic_mean": ic_mean, "ic_tstat": tstat}}}
        (tmp_path / "backtest.json").write_text(json.dumps(report))
        return ModelRegistry(tmp_path).validation()["status"]

    assert status(0.0, 0.1) == "not_validated"
    assert status(-0.05, 3.0) == "not_validated"
    assert status(0.05, 2.5) == "validated"


def test_analyze_uses_ml_when_model_available(tmp_path):
    from app.instruments import Instrument
    from app.service import analyze

    data = _dataset()
    meta = {"version": "test-2", "features": FEATURE_NAMES, "horizons": {}}
    for name in ("court", "moyen", "long"):
        labelled = data.dropna(subset=[f"fwd_{name}"])
        y = (labelled[f"fwd_{name}"] > 0).astype(int)
        joblib.dump(new_estimator().fit(labelled[FEATURE_NAMES], y), tmp_path / f"{name}.joblib")
        meta["horizons"][name] = {"file": f"{name}.joblib", "base_rate": float(y.mean())}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))

    ma = Instrument("TEST", "Société test", 600, None, None)
    result, _ = analyze(ma, make_prices(), ModelRegistry(tmp_path), ml_weight=0.6, history_days=60)
    assert result["model_version"] == "test-2"
    for h in result["horizons"].values():
        assert h["ml_score"] is not None
        assert h["score"] == pytest.approx(0.6 * h["ml_score"] + 0.4 * h["rule_score"], abs=0.2)
    assert len(result["history"]["dates"]) == 60


def test_validation_flags_inverted_signal(tmp_path):
    report = {"horizons": {"court": {"ic_mean": -0.09, "ic_tstat": -4.2}, "long": {"ic_mean": 0.07, "ic_tstat": 1.6}}}
    (tmp_path / "backtest.json").write_text(json.dumps(report))
    validation = ModelRegistry(tmp_path).validation()
    assert validation["status"] == "inverted"
    assert "court terme" in validation["message"]


def test_liquid_only_filters_illiquid_names():
    from training.train import liquid_only

    liquid = make_prices(400)
    illiquid = make_prices(400)
    illiquid.iloc[:, illiquid.columns.get_loc("Volume")] = np.where(np.arange(400) % 4 == 0, 1000, 0)
    assert set(liquid_only({"A": liquid, "B": illiquid})) == {"A"}


def test_ml_model_not_applied_to_international_market(tmp_path):
    from app.instruments import Instrument
    from app.service import analyze

    data = _dataset()
    labelled = data.dropna(subset=["fwd_court"])
    y = (labelled["fwd_court"] > 0).astype(int)
    joblib.dump(new_estimator().fit(labelled[FEATURE_NAMES], y), tmp_path / "c.joblib")
    meta = {"version": "v9", "features": FEATURE_NAMES, "horizons": {"court": {"file": "c.joblib", "base_rate": float(y.mean())}}}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    registry = ModelRegistry(tmp_path)
    prices = make_prices()

    ma, _ = analyze(Instrument("A", "A", 600, None, None), prices, registry, 0.3, 60)
    intl, _ = analyze(Instrument("B", "B", 600, None, None, market="INTL", currency="EUR", kind="ETF"), prices, registry, 0.3, 60)
    assert ma["model_version"] == "v9" and ma["horizons"]["court"]["ml_score"] is not None
    assert intl["model_version"] == "rules-only"
    assert intl["horizons"]["court"]["ml_score"] is None
    assert intl["horizons"]["court"]["score"] == intl["horizons"]["court"]["rule_score"]


def test_training_universe_excludes_international_instruments():
    from app.instruments import load_instruments
    from training.train import default_universe

    universe = default_universe()
    assert len(universe) > 50
    instruments = load_instruments()
    assert {instruments[s].market for s in universe} == {"MA"}
    assert not {"CW8.PA", "STLAP.PA", "SPCX"} & set(universe)
