import numpy as np
import pandas as pd

from app.features import MIN_ROWS, prepare_features
from app.model import ModelRegistry
from app.scoring import (
    HORIZON_WEIGHTS,
    HORIZONS,
    blend,
    components,
    rule_score,
    verdict,
)

DISCLAIMER = (
    "Indicateur statistique à but informatif uniquement, ne constitue pas un conseil en "
    "investissement. Les performances passées ne préjugent pas des performances futures."
)


class InsufficientHistoryError(Exception):
    pass


def _clean(series: pd.Series) -> list[float | None]:
    return [None if pd.isna(v) else round(float(v), 1) for v in series]


def analyze(
    ticker: str,
    prices: pd.DataFrame,
    registry: ModelRegistry,
    ml_weight: float,
    history_days: int,
) -> tuple[dict, pd.Series]:
    """Retourne (réponse, dernière ligne de features) pour l'API et le monitoring."""
    if len(prices) < MIN_ROWS:
        raise InsufficientHistoryError(f"Historique insuffisant pour {ticker} ({len(prices)} séances, {MIN_ROWS} requises)")
    feats = prepare_features(prices)
    if feats.empty:
        raise InsufficientHistoryError(f"Indicateurs non calculables pour {ticker}")

    scores: dict[str, pd.Series] = {}
    horizons_out: dict[str, dict] = {}
    for name, cfg in HORIZONS.items():
        rule = rule_score(feats, name)
        ml = registry.ml_score(name, feats)
        final = blend(rule, ml, ml_weight)
        scores[name] = final
        last = final.iloc[-1]
        horizons_out[name] = {
            "label": cfg["label"],
            "days": cfg["days"],
            "score": round(float(last), 1),
            "verdict": verdict(float(last)),
            "rule_score": round(float(rule.iloc[-1]), 1),
            "ml_score": None if ml is None else round(float(ml.iloc[-1]), 1),
            "components": {k: round(float(v), 1) for k, v in components(feats.iloc[[-1]], name).iloc[0].items()},
        }

    global_series = sum(scores[h] * w for h, w in HORIZON_WEIGHTS.items())
    global_last = float(global_series.iloc[-1])

    tail = feats.index[-history_days:]
    history = {
        "dates": [d.strftime("%Y-%m-%d") for d in tail],
        "series": {"global": _clean(global_series.loc[tail]), **{h: _clean(s.loc[tail]) for h, s in scores.items()}},
    }

    close = prices["Close"]
    change = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else None
    last_feats = feats.iloc[-1]
    response = {
        "ticker": ticker,
        "as_of": prices.index[-1].strftime("%Y-%m-%d"),
        "last_price": round(float(close.iloc[-1]), 2),
        "change_1d": None if change is None or np.isnan(change) else round(change, 4),
        "model_version": registry.version,
        "validation": registry.validation(),
        "global": {"score": round(global_last, 1), "verdict": verdict(global_last)},
        "horizons": horizons_out,
        "indicators": {k: round(float(v), 4) for k, v in last_feats.items()},
        "history": history,
        "disclaimer": DISCLAIMER,
    }
    return response, last_feats
