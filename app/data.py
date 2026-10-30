import re
import threading
import time
from urllib.parse import quote

import httpx
import pandas as pd

from app.config import get_settings

TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^=]{1,15}$")


class DataUnavailableError(Exception):
    pass


def normalize_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not TICKER_RE.match(ticker):
        raise ValueError("Symbole invalide")
    return ticker


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0 (compatible; bourse-score/1.0)"


def _download_chart_api(ticker: str, period: str) -> pd.DataFrame:
    """API chart Yahoo directe : pas de cookie/crumb, donc plus fiable que yfinance depuis un serveur."""
    resp = httpx.get(
        YAHOO_CHART_URL.format(symbol=quote(ticker, safe="")),
        params={"range": period, "interval": "1d", "events": "div,splits"},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    if resp.status_code == 404:  # symbole inconnu : inutile d'essayer une autre source
        raise DataUnavailableError(f"Symbole introuvable : {ticker}")
    resp.raise_for_status()
    result = (resp.json().get("chart", {}).get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise DataUnavailableError(f"Aucune donnée pour {ticker}")
    quote_data = result["indicators"]["quote"][0]
    adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    offset = result.get("meta", {}).get("gmtoffset", 0)
    index = pd.to_datetime(pd.Series(result["timestamp"]) + offset, unit="s").dt.normalize()
    df = pd.DataFrame({"Close": adj or quote_data["close"], "Volume": quote_data.get("volume")}, index=pd.DatetimeIndex(index))
    return df


def _download_yfinance(ticker: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df is None or df.empty or "Close" not in df:
        raise DataUnavailableError(f"Aucune donnée pour {ticker}")
    df = df[["Close", "Volume"]].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    return df


def download_prices(ticker: str, period: str = "3y") -> pd.DataFrame:
    errors = []
    for fetch in (_download_chart_api, _download_yfinance):
        try:
            df = fetch(ticker, period)
        except DataUnavailableError:
            raise
        except Exception as exc:  # réseau, parsing, rate-limit : on essaie la source suivante
            errors.append(f"{fetch.__name__}: {type(exc).__name__}")
            continue
        df = df.dropna(subset=["Close"])
        if not df.empty:
            return df[~df.index.duplicated(keep="last")].sort_index()
    raise DataUnavailableError(f"Aucune donnée pour {ticker} ({'; '.join(errors)})"[:300])


class PriceCache:
    """Cache mémoire TTL, thread-safe : évite de marteler la source de données."""

    def __init__(self, ttl_seconds: int | None = None):
        self._ttl = ttl_seconds if ttl_seconds is not None else get_settings().cache_ttl_seconds
        self._store: dict[str, tuple[float, pd.DataFrame]] = {}
        self._lock = threading.Lock()

    def get(self, ticker: str, loader=download_prices) -> pd.DataFrame:
        now = time.monotonic()
        with self._lock:
            hit = self._store.get(ticker)
            if hit and now - hit[0] < self._ttl:
                return hit[1]
        df = loader(ticker)
        with self._lock:
            self._store[ticker] = (now, df)
            if len(self._store) > 256:  # borne mémoire
                oldest = min(self._store, key=lambda k: self._store[k][0])
                self._store.pop(oldest, None)
        return df
