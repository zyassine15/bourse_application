import numpy as np
import pandas as pd

from app.features import MIN_ROWS, prepare_features
from app.instruments import Instrument
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


def liquidity(prices: pd.DataFrame, window: int = 60) -> dict:
    """Part des dernières séances avec des échanges : beaucoup de valeurs marocaines cotent peu."""
    ratio = float((prices["Volume"].tail(window) > 0).mean()) if "Volume" in prices else 0.0
    level = "faible" if ratio < 0.5 else "moyenne" if ratio < 0.8 else "bonne"
    warning = None
    if level == "faible":
        warning = (
            f"Valeur peu liquide : échanges sur seulement {ratio:.0%} des {window} dernières séances. "
            "Le score est peu fiable et l'achat/la revente peuvent être difficiles."
        )
    return {"traded_days_ratio": round(ratio, 3), "level": level, "warning": warning}


def _clean(series: pd.Series, digits: int = 1) -> list[float | None]:
    return [None if pd.isna(v) else round(float(v), digits) for v in series]


def leverage_notice(leverage: int) -> str | None:
    if leverage <= 1:
        return None
    return (
        f"ETF à effet de levier ×{leverage} quotidien : sur plusieurs jours sa performance ne vaut pas "
        f"{leverage}× celle de l'indice (capitalisation quotidienne) et ses variations sont très fortes. "
        "Produit risqué, conçu pour un horizon très court."
    )


def price_history(prices: pd.DataFrame) -> dict:
    close = prices["Close"].astype(float)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in close.index],
        "close": [round(float(v), 2) for v in close],
        "sma50": _clean(close.rolling(50).mean(), 2),
        "sma200": _clean(close.rolling(200).mean(), 2),
    }


def price_summary(instrument: Instrument, prices: pd.DataFrame) -> dict:
    """Champs communs à l'analyse complète et à la vue « cours seul »."""
    close = prices["Close"].astype(float)
    change = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) > 1 else None
    return {
        "ticker": instrument.symbol,
        "name": instrument.name,
        "currency": instrument.currency,
        "market": instrument.market,
        "kind": instrument.kind,
        "leverage": instrument.leverage,
        "notice": leverage_notice(instrument.leverage),
        "as_of": prices.index[-1].strftime("%Y-%m-%d"),
        "last_price": round(float(close.iloc[-1]), 2),
        "change_1d": None if change is None or np.isnan(change) else round(change, 4),
        "prices": price_history(prices),
    }


def analyze(
    instrument: Instrument,
    prices: pd.DataFrame,
    registry: ModelRegistry,
    ml_weight: float,
    history_days: int,
) -> tuple[dict, pd.Series]:
    """Retourne (réponse, dernière ligne de features) pour l'API et le monitoring."""
    ticker = instrument.symbol
    if len(prices) < MIN_ROWS:
        raise InsufficientHistoryError(f"Historique insuffisant pour {ticker} ({len(prices)} séances, {MIN_ROWS} requises)")
    feats = prepare_features(prices)
    if feats.empty:
        raise InsufficientHistoryError(f"Indicateurs non calculables pour {ticker}")

    scores: dict[str, pd.Series] = {}
    horizons_out: dict[str, dict] = {}
    for name, cfg in HORIZONS.items():
        rule = rule_score(feats, name)
        ml = registry.ml_score(name, feats) if registry.applies_to(instrument.market) else None
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

    last_feats = feats.iloc[-1]
    response = {
        **price_summary(instrument, prices),
        "model_version": registry.version if registry.applies_to(instrument.market) else "rules-only",
        "validation": registry.validation(instrument.market),
        "liquidity": liquidity(prices),
        "global": {"score": round(global_last, 1), "verdict": verdict(global_last)},
        "horizons": horizons_out,
        "history": history,
        "disclaimer": DISCLAIMER,
    }
    return response, last_feats
