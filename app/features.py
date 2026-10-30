"""Calcul vectorisé des indicateurs techniques.

Utilisé à l'identique par l'entraînement et par l'inférence pour éviter tout
training/serving skew : une seule définition des features.
"""

import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    "ret_5",
    "ret_21",
    "ret_63",
    "ret_126",
    "rsi_14",
    "macd_hist",
    "px_vs_sma20",
    "px_vs_sma50",
    "px_vs_sma200",
    "sma50_vs_sma200",
    "vol_21",
    "drawdown_252",
    "bb_pctb",
    "volume_ratio",
]

# Historique minimal pour que tous les indicateurs (SMA200, drawdown 252j) soient définis.
MIN_ROWS = 260


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.where(loss != 0, 100.0)


def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """prices : colonnes `Close` (et `Volume` optionnel), index = dates croissantes."""
    close = prices["Close"].astype(float)
    volume = prices["Volume"].astype(float) if "Volume" in prices else pd.Series(np.nan, index=close.index)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    std20 = close.rolling(20).std()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_hist = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    vol_ratio = volume.rolling(5).mean() / volume.rolling(60).mean().replace(0, np.nan)

    feats = pd.DataFrame(
        {
            "ret_5": close.pct_change(5),
            "ret_21": close.pct_change(21),
            "ret_63": close.pct_change(63),
            "ret_126": close.pct_change(126),
            "rsi_14": _rsi(close),
            "macd_hist": macd_hist,
            "px_vs_sma20": close / sma20 - 1,
            "px_vs_sma50": close / sma50 - 1,
            "px_vs_sma200": close / sma200 - 1,
            "sma50_vs_sma200": sma50 / sma200 - 1,
            "vol_21": close.pct_change().rolling(21).std() * np.sqrt(252),
            "drawdown_252": close / close.rolling(252, min_periods=60).max() - 1,
            "bb_pctb": (close - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan),
            "volume_ratio": vol_ratio.fillna(1.0),
        },
        index=prices.index,
    )
    return feats[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)


def prepare_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Features prêtes à l'emploi : lignes incomplètes retirées, %B neutre si bandes plates."""
    feats = compute_features(prices)
    feats = feats.dropna(subset=[c for c in FEATURE_NAMES if c != "bb_pctb"]).copy()
    feats["bb_pctb"] = feats["bb_pctb"].fillna(0.5)
    return feats
