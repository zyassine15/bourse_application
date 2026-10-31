"""Client de l'API publique de la Bourse de Casablanca (historique journalier des actions).

Le serveur casablanca-bourse.com n'envoie pas son certificat intermédiaire (Sectigo DV R36) :
les navigateurs le récupèrent via l'extension AIA, mais pas Python. Plutôt que de désactiver la
vérification TLS, on ajoute cet intermédiaire public (`certs/`) au magasin de confiance certifi.
"""

import logging
import ssl
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import certifi
import httpx
import pandas as pd

from app.data import DataUnavailableError

logger = logging.getLogger(__name__)

API_URL = "https://www.casablanca-bourse.com/api/boursenova/stock-historical"
LISTING_URL = "https://www.casablanca-bourse.com/market-data/cours"
INTERMEDIATE_CA = Path(__file__).parent / "certs" / "sectigo-dv-r36.pem"
USER_AGENT = "Mozilla/5.0 (compatible; bourse-score/1.0; +https://github.com/zyassine15/bourse_application)"

# L'API ne renvoie rien avant la mise en service de la plateforme Boursenova.
HISTORY_START = date(2023, 12, 1)
# L'API renvoie 0 ligne au-delà de ~2 ans par requête : on découpe en tranches d'un an.
CHUNK_DAYS = 360

_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context(cafile=certifi.where())
    ctx.load_verify_locations(cafile=str(INTERMEDIATE_CA))
    return ctx


def get_client() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = httpx.Client(
                verify=_ssl_context(),
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                timeout=httpx.Timeout(20.0),
                follow_redirects=True,
            )
        return _client


def _get_json(client: httpx.Client, url: str, params: dict, retries: int = 2) -> dict:
    for attempt in range(retries + 1):
        try:
            resp = client.get(url, params=params)
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError("erreur serveur", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp.json()
        except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
            if attempt == retries:
                raise DataUnavailableError("Source de données de la Bourse de Casablanca indisponible") from exc
            time.sleep(0.5 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_history(symbol: str, start: date = HISTORY_START, end: date | None = None) -> pd.DataFrame:
    """Cours de clôture (`Close`) et titres échangés (`Volume`) par séance, index = dates croissantes."""
    end = end or date.today()
    client = get_client()
    rows: list[dict] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end)
        payload = _get_json(
            client,
            API_URL,
            {
                "instrument": symbol,
                "market": "comptant",
                "type": "actions",
                "startDate": cursor.isoformat(),
                "endDate": chunk_end.isoformat(),
                "pageNumber": 1,
                "pageSize": 1000,
            },
        )
        rows.extend(payload.get("items", []))
        cursor = chunk_end + timedelta(days=1)

    if not rows:
        raise DataUnavailableError(f"Aucune donnée de cotation pour {symbol}")
    df = pd.DataFrame(
        {
            "Close": [r.get("dernierCours") for r in rows],
            "Volume": [r.get("titresEchanges") or 0 for r in rows],
        },
        index=pd.to_datetime([r["seance"] for r in rows], format="%d/%m/%Y"),
    )
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        raise DataUnavailableError(f"Aucune donnée de cotation pour {symbol}")
    return df
