"""Rubrique « Indicateurs détaillés » : 7 indicateurs fondamentaux + 3 indicateurs techniques."""

from collections.abc import Callable
from datetime import date

import pandas as pd

from app.data import DataUnavailableError
from app.features import rsi
from app.fundamentals import (
    NOT_APPLICABLE,
    UNAVAILABLE,
    compute_fundamentals,
    item,
    value,
)
from app.instruments import ML_MARKET, Instrument

FUNDAMENTAL_LABELS = [
    ("per", "PER"),
    ("peg", "PEG"),
    ("fcf", "Free Cash Flow (FCF)"),
    ("roe_roic", "ROE / ROIC"),
    ("net_debt_ebitda", "Dette nette / EBITDA"),
    ("operating_margin", "Marge opérationnelle"),
    ("payout", "Payout Ratio"),
]
NO_SOURCE_MA = (
    "Non disponible : la Bourse de Casablanca ne publie les états financiers des sociétés que sous forme de documents "
    "PDF, sans données exploitables."
)
NOT_APPLICABLE_ETF = "Sans objet : un ETF est un panier de titres, il n'a pas de PER, de FCF ou de marges propres."
SOURCE_UNAVAILABLE = "Données fondamentales momentanément indisponibles (source Yahoo Finance) : réessayez plus tard."
NO_DATA = "Aucune donnée financière publiée pour cette valeur."
SOURCE_NOTE = "Source : Yahoo Finance, derniers états financiers publiés. Montants dans la devise de publication des comptes."


def _placeholder_items(status: str) -> list[dict]:
    return [item(i, label, "", [value(label, status=status)]) for i, label in FUNDAMENTAL_LABELS]


def fundamentals_group(instrument: Instrument, fetch_raw: Callable[[str], dict] | None, today: date | None = None) -> dict:
    group = {"id": "fundamental", "label": "Fondamentaux"}
    if instrument.market == ML_MARKET:
        return {**group, "status": UNAVAILABLE, "note": NO_SOURCE_MA, "items": _placeholder_items(UNAVAILABLE)}
    if instrument.kind == "ETF":
        return {**group, "status": NOT_APPLICABLE, "note": NOT_APPLICABLE_ETF, "items": _placeholder_items(NOT_APPLICABLE)}
    try:
        raw = fetch_raw(instrument.symbol) if fetch_raw else {}
    except DataUnavailableError:
        return {**group, "status": UNAVAILABLE, "note": SOURCE_UNAVAILABLE, "items": _placeholder_items(UNAVAILABLE)}
    if not any(raw.values()):
        return {**group, "status": UNAVAILABLE, "note": NO_DATA, "items": _placeholder_items(UNAVAILABLE)}
    items = compute_fundamentals(raw, today, currency=instrument.currency)
    return {**group, "status": "ok", "note": SOURCE_NOTE, "items": items}


def _pct(a: float, b: float) -> float:
    return a / b - 1


def moving_averages(close: pd.Series, currency: str) -> dict:
    help_text = "Moyennes des cours sur 50 et 200 séances. MM50 au-dessus de MM200 : configuration haussière."
    last = float(close.iloc[-1])
    sma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    sma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

    def ma(label: str, avg: float | None, needed: int) -> list[dict]:
        if avg is None:
            return [value(label, status=UNAVAILABLE, note=f"Historique insuffisant ({len(close)} séances sur {needed})")]
        return [
            value(label, avg, "price", currency=currency),
            value(f"Cours vs {label}", _pct(last, avg), "pct"),
        ]

    values = ma("MM50", sma50, 50) + ma("MM200", sma200, 200)
    reading = None
    if sma50 is not None and sma200 is not None:
        reading = (
            "MM50 au-dessus de la MM200 : configuration haussière"
            if sma50 > sma200
            else "MM50 sous la MM200 : configuration baissière"
        )
    return item("moving_averages", "Moyennes Mobiles (MM50 / MM200)", help_text, values, reading)


def rsi_indicator(close: pd.Series) -> dict:
    help_text = "Indice de force relative sur 14 séances (0-100) : au-dessus de 70 le titre est jugé suracheté, sous 30 survendu."
    series = rsi(close).dropna()
    if series.empty:
        v = value("RSI (14)", status=UNAVAILABLE, note="Historique insuffisant")
        return item("rsi", "RSI", help_text, [v])
    last = float(series.iloc[-1])
    reading = "Surachat (RSI > 70)" if last > 70 else "Survente (RSI < 30)" if last < 30 else "Zone neutre (RSI entre 30 et 70)"
    return item("rsi", "RSI", help_text, [value("RSI (14)", last, "points")], reading)


def volumes(volume: pd.Series) -> dict:
    help_text = "Titres échangés. Un volume supérieur à sa moyenne habituelle confirme un mouvement de prix."
    if volume.tail(60).sum() <= 0:
        return item(
            "volumes", "Volumes", help_text, [value("Volume", status=UNAVAILABLE, note="Volumes non fournis par la source")]
        )
    avg5, avg20, avg60 = (float(volume.tail(n).mean()) for n in (5, 20, 60))
    ratio = avg5 / avg60 if avg60 > 0 else None
    values = [value("Dernière séance", volume.iloc[-1], "number"), value("Moyenne 20 séances", avg20, "number")]
    reading = None
    if ratio is not None:
        values.append(value("Ratio 5 j / 60 j", ratio, "ratio"))
        reading = (
            "Volume récent nettement supérieur à la moyenne 60 j"
            if ratio >= 1.2
            else "Volume récent nettement inférieur à la moyenne 60 j"
            if ratio <= 0.8
            else "Volume récent proche de la moyenne 60 j"
        )
    return item("volumes", "Volumes", help_text, values, reading)


def technical_group(prices: pd.DataFrame, currency: str) -> dict:
    close = prices["Close"].astype(float)
    volume = prices["Volume"].astype(float) if "Volume" in prices else pd.Series(0.0, index=close.index)
    items = [moving_averages(close, currency), rsi_indicator(close), volumes(volume)]
    return {"id": "technical", "label": "Technique", "status": "ok", "note": None, "items": items}


def build_indicators(
    instrument: Instrument,
    prices: pd.DataFrame,
    fetch_raw: Callable[[str], dict] | None = None,
    today: date | None = None,
) -> dict:
    return {
        "ticker": instrument.symbol,
        "as_of": prices.index[-1].strftime("%Y-%m-%d"),
        "currency": instrument.currency,
        "groups": [
            fundamentals_group(instrument, fetch_raw, today),
            technical_group(prices, instrument.currency),
        ],
    }
