"""Client de l'API « chart » de Yahoo Finance pour les actions et ETF internationaux (cours journaliers).

Cette API n'est pas officielle : elle peut limiter le débit (429) ou bloquer certaines IP de datacenter.
On réessaie donc avec un délai croissant en alternant entre les deux hôtes publics de Yahoo.
"""

import logging
import threading
import time
from urllib.parse import quote

import httpx
import pandas as pd

from app.data import DataUnavailableError

logger = logging.getLogger(__name__)

HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
USER_AGENT = "Mozilla/5.0 (compatible; bourse-score/1.0; +https://github.com/zyassine15/bourse_application)"
HISTORY_RANGE = "5y"
ATTEMPTS = 3

_client: httpx.Client | None = None
_client_lock = threading.Lock()


def get_client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=httpx.Timeout(15.0),
                follow_redirects=True,
            )
        return _client


def get_json(client: httpx.Client, path: str, params: dict, symbol: str) -> dict:
    """GET JSON sur Yahoo (réessais, bascule d'hôte). `path` commence par « / »."""
    last_status: int | str = "?"
    for attempt in range(ATTEMPTS):
        url = f"https://{HOSTS[attempt % len(HOSTS)]}{path}"
        try:
            resp = client.get(url, params=params)
        except httpx.TransportError as exc:
            last_status = type(exc).__name__
        else:
            if resp.status_code == 404:
                raise DataUnavailableError(f"Symbole introuvable chez le fournisseur de données : {symbol}")
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    last_status = "réponse illisible"
            elif resp.status_code == 429 or resp.status_code >= 500:
                last_status = resp.status_code
            else:
                raise DataUnavailableError(f"Données refusées par le fournisseur ({resp.status_code}) : {symbol}")
        if attempt < ATTEMPTS - 1:
            time.sleep(1.0 * (attempt + 1))
    logger.warning("Yahoo indisponible pour %s (%s)", symbol, last_status)
    raise DataUnavailableError("Source de données internationale indisponible (limite de débit ou panne)")


def fetch_chart(symbol: str, history_range: str = HISTORY_RANGE, client: httpx.Client | None = None) -> tuple[pd.DataFrame, dict]:
    """Retourne (prix, meta) : prix = colonnes `Close` (ajusté des dividendes) et `Volume`, index = dates."""
    payload = get_json(
        client or get_client(), f"/v8/finance/chart/{quote(symbol, safe='')}", {"range": history_range, "interval": "1d"}, symbol
    )
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise DataUnavailableError(f"Aucune donnée de cotation pour {symbol}")
    meta = result.get("meta", {})
    quote_data = result["indicators"]["quote"][0]
    adjusted = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    # Les horodatages sont en UTC : on applique le décalage de la place pour obtenir la date de séance locale.
    stamps = pd.Series(result["timestamp"]) + meta.get("gmtoffset", 0)
    index = pd.DatetimeIndex(pd.to_datetime(stamps, unit="s").dt.normalize())
    volume = quote_data.get("volume") or [None] * len(index)
    df = pd.DataFrame(
        {
            "Close": pd.to_numeric(pd.Series(adjusted or quote_data["close"]), errors="coerce").to_numpy(),
            "Volume": pd.to_numeric(pd.Series(volume), errors="coerce").fillna(0).to_numpy(),
        },
        index=index,
    )
    # La barre du jour arrive parfois sans cours (séance non consolidée) : on la complète avec le dernier prix coté.
    last_price, last_time = meta.get("regularMarketPrice"), meta.get("regularMarketTime")
    if last_price and last_time and pd.isna(df["Close"].iloc[-1]):
        quote_day = pd.Timestamp(last_time + meta.get("gmtoffset", 0), unit="s").normalize()
        if quote_day == df.index[-1]:
            df.iloc[-1, df.columns.get_loc("Close")] = float(last_price)
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        raise DataUnavailableError(f"Aucune donnée de cotation pour {symbol}")
    return df, meta


def fetch_history(symbol: str) -> pd.DataFrame:
    return fetch_chart(symbol)[0]
