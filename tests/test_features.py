import numpy as np

from app.features import FEATURE_NAMES, compute_features, prepare_features
from tests.conftest import make_prices


def test_prepare_features_complete(prices):
    feats = prepare_features(prices)
    assert list(feats.columns) == FEATURE_NAMES
    assert not feats.isna().any().any()
    assert len(feats) > 300


def test_rsi_bounds(prices):
    rsi = compute_features(prices)["rsi_14"].dropna()
    assert rsi.between(0, 100).all()


def test_no_lookahead(prices):
    """Les features à la date t ne doivent dépendre que du passé."""
    full = compute_features(prices)
    cut = 450
    partial = compute_features(prices.iloc[:cut])
    np.testing.assert_allclose(full.iloc[cut - 1].to_numpy(), partial.iloc[-1].to_numpy(), equal_nan=True)


def test_uptrend_vs_downtrend():
    up = prepare_features(make_prices(drift=0.003, vol=0.003)).iloc[-1]
    down = prepare_features(make_prices(drift=-0.003, vol=0.003)).iloc[-1]
    assert up["px_vs_sma200"] > 0 > down["px_vs_sma200"]
    assert up["ret_126"] > 0 > down["ret_126"]
