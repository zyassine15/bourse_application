import re
import threading
import time

import pandas as pd

from app.config import get_settings

TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


class DataUnavailableError(Exception):
    pass


def normalize_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not TICKER_RE.match(ticker):
        raise ValueError("Symbole invalide")
    return ticker


def download_prices(ticker: str) -> pd.DataFrame:
    """Cours journaliers de la valeur, depuis la source de son marché."""
    from app.instruments import load_instruments

    instrument = load_instruments().get(ticker)
    if instrument is None:
        raise DataUnavailableError(f"Valeur inconnue : {ticker}")
    # Imports tardifs : ces modules dépendent de celui-ci (DataUnavailableError).
    if instrument.market == "MA":
        from app.casablanca import fetch_history
    else:
        from app.yahoo import fetch_history
    return fetch_history(ticker)


class TTLCache:
    """Cache mémoire TTL, thread-safe : évite de solliciter les sources de données à chaque requête."""

    def __init__(self, ttl_seconds: int | None = None):
        self._ttl = ttl_seconds if ttl_seconds is not None else get_settings().cache_ttl_seconds
        self._store: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, ticker: str, loader):
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


class PriceCache(TTLCache):
    """Cache des cours : charge par défaut depuis la source du marché de la valeur."""

    def get(self, ticker: str, loader=download_prices) -> pd.DataFrame:
        return super().get(ticker, loader)
