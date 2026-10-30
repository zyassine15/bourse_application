import math

import numpy as np
import pytest

from app.features import prepare_features
from app.scoring import (
    COMPONENT_WEIGHTS,
    HORIZON_WEIGHTS,
    HORIZONS,
    blend,
    ml_to_score,
    rule_score,
    verdict,
)
from tests.conftest import make_prices


def test_weights_sum_to_one():
    assert math.isclose(sum(HORIZON_WEIGHTS.values()), 1.0)
    for weights in COMPONENT_WEIGHTS.values():
        assert math.isclose(sum(weights.values()), 1.0)


@pytest.mark.parametrize("horizon", list(HORIZONS))
def test_rule_score_in_range(prices, horizon):
    score = rule_score(prepare_features(prices), horizon)
    assert score.between(0, 100).all()


@pytest.mark.parametrize("horizon", list(HORIZONS))
def test_uptrend_scores_higher_than_downtrend(horizon):
    up = rule_score(prepare_features(make_prices(drift=0.002, vol=0.004)), horizon).iloc[-1]
    down = rule_score(prepare_features(make_prices(drift=-0.002, vol=0.004)), horizon).iloc[-1]
    assert up > down


def test_verdict_thresholds():
    assert verdict(80) == "Favorable"
    assert verdict(65) == "Favorable"
    assert verdict(50) == "Neutre"
    assert verdict(39.9) == "Défavorable"


def test_ml_to_score_centered_on_base_rate():
    out = ml_to_score(np.array([0.55, 0.75, 0.2]), base_rate=0.55)
    assert out[0] == pytest.approx(50)
    assert out[1] == pytest.approx(100)
    assert out[2] == 0


def test_blend_without_ml_is_rule(prices):
    rule = rule_score(prepare_features(prices), "moyen")
    assert blend(rule, None, 0.6).equals(rule)
