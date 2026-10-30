"""Score de 0 à 100 par horizon : règles techniques + (optionnel) modèle ML."""

import numpy as np
import pandas as pd

HORIZONS: dict[str, dict] = {
    # `days` = horizon de prédiction (jours de bourse) utilisé comme cible du modèle ML.
    "court": {"label": "Court terme (~1 semaine)", "days": 5},
    "moyen": {"label": "Moyen terme (~1 mois)", "days": 21},
    "long": {"label": "Long terme (~3 mois)", "days": 63},
}

# Pondération globale des horizons.
HORIZON_WEIGHTS = {"court": 0.25, "moyen": 0.40, "long": 0.35}

# Poids des composantes du score à règles, par horizon.
COMPONENT_WEIGHTS: dict[str, dict[str, float]] = {
    "court": {"trend": 0.15, "momentum": 0.25, "rsi": 0.20, "macd": 0.25, "risk": 0.15},
    "moyen": {"trend": 0.30, "momentum": 0.30, "rsi": 0.10, "macd": 0.15, "risk": 0.15},
    "long": {"trend": 0.40, "momentum": 0.30, "rsi": 0.05, "macd": 0.05, "risk": 0.20},
}

BUY_THRESHOLD = 65.0
AVOID_THRESHOLD = 40.0


def verdict(score: float) -> str:
    if score >= BUY_THRESHOLD:
        return "Favorable"
    if score < AVOID_THRESHOLD:
        return "Défavorable"
    return "Neutre"


def _squash(x: pd.Series, scale: float) -> pd.Series:
    """Projette une valeur centrée sur 0 vers [0, 100] (50 = neutre)."""
    return 50 + 50 * np.tanh(x / scale)


def components(feats: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Composantes 0-100 (plus haut = plus favorable), alignées sur `feats`."""
    mom_col = {"court": "ret_5", "moyen": "ret_21", "long": "ret_126"}[horizon]
    mom_scale = {"court": 0.04, "moyen": 0.10, "long": 0.25}[horizon]
    mid_col = {"court": "ret_21", "moyen": "ret_63", "long": "ret_63"}[horizon]

    trend = (
        _squash(feats["px_vs_sma50"], 0.08) * 0.35
        + _squash(feats["px_vs_sma200"], 0.15) * 0.35
        + _squash(feats["sma50_vs_sma200"], 0.10) * 0.30
    )
    momentum = _squash(feats[mom_col], mom_scale) * 0.6 + _squash(feats[mid_col], mom_scale * 2) * 0.4
    # RSI : zone saine autour de 58, pénalise surachat (>75) et survente prononcée.
    rsi = 100 * np.exp(-(((feats["rsi_14"] - 58) / 22) ** 2))
    macd = _squash(feats["macd_hist"], 0.01)
    risk = (100 - (feats["vol_21"] / 0.6).clip(0, 1) * 100) * 0.5 + (100 - (-feats["drawdown_252"] / 0.4).clip(0, 1) * 100) * 0.5
    return pd.DataFrame(
        {"trend": trend, "momentum": momentum, "rsi": rsi, "macd": macd, "risk": risk},
        index=feats.index,
    )


def rule_score(feats: pd.DataFrame, horizon: str) -> pd.Series:
    comps = components(feats, horizon)
    weights = COMPONENT_WEIGHTS[horizon]
    score = sum(comps[name] * w for name, w in weights.items())
    return score.clip(0, 100)


def ml_to_score(proba: np.ndarray, base_rate: float) -> np.ndarray:
    """Probabilité de hausse -> score 0-100, centrée sur le taux de hausse historique."""
    return np.clip(50 + 250 * (np.asarray(proba) - base_rate), 0, 100)


def blend(rule: pd.Series, ml: pd.Series | None, ml_weight: float) -> pd.Series:
    if ml is None:
        return rule
    return (ml_weight * ml + (1 - ml_weight) * rule).clip(0, 100)


def global_score(per_horizon: dict[str, float]) -> float:
    return float(sum(per_horizon[h] * w for h, w in HORIZON_WEIGHTS.items()))
