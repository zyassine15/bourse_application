"""Régénère `app/instruments.json` : liste des actions cotées à Casablanca + profondeur d'historique.

Usage : python -m training.refresh_instruments
La liste est extraite de la page « Cours » du site officiel ; l'historique est mesuré via l'API.
"""

import json
import logging
import re
from datetime import UTC, datetime, timedelta

from app.casablanca import LISTING_URL, fetch_history, get_client
from app.data import DataUnavailableError
from app.instruments import INSTRUMENTS_FILE

logger = logging.getLogger("instruments")

STALE_DAYS = 30

# La page embarque la liste sous forme d'objets {"symbol": "IAM  ", "label": {"fr": "..."}}.
ENTRY_RE = re.compile(r'\{\s*"symbol"\s*:\s*"[^"]*"\s*,\s*"label"\s*:\s*\{[^{}]*\}\s*\}')


def parse_listing(html: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for raw in ENTRY_RE.findall(html):
        entry = json.loads(raw)
        symbol = entry["symbol"].strip().upper()
        if symbol:
            found.setdefault(symbol, " ".join((entry["label"].get("fr") or entry["label"].get("en") or symbol).split()))
    return found


def main() -> None:
    resp = get_client().get(LISTING_URL)
    resp.raise_for_status()
    listing = parse_listing(resp.text)
    if len(listing) < 50:
        raise SystemExit(f"Liste suspecte ({len(listing)} valeurs) : le format de la page a probablement changé")
    logger.info("%d valeurs trouvées", len(listing))

    instruments = []
    for symbol, name in sorted(listing.items()):
        try:
            df = fetch_history(symbol)
            item = {
                "symbol": symbol,
                "name": name,
                "sessions": int(len(df)),
                "first_date": df.index[0].strftime("%Y-%m-%d"),
                "last_date": df.index[-1].strftime("%Y-%m-%d"),
            }
        except DataUnavailableError:
            item = {"symbol": symbol, "name": name, "sessions": 0, "first_date": None, "last_date": None}
        logger.info("%-6s %-32s %4d séances", symbol, name[:32], item["sessions"])
        instruments.append(item)

    # La page liste aussi des droits d'attribution et des obligations (aucune cotation en actions)
    # ainsi que des valeurs suspendues/radiées (dernière séance ancienne) : ce ne sont pas des actions investissables.
    cutoff = (datetime.now(UTC) - timedelta(days=STALE_DAYS)).strftime("%Y-%m-%d")
    kept = [i for i in instruments if i["sessions"] > 0 and i["last_date"] >= cutoff]
    logger.info("%d instruments écartés (droits, obligations, valeurs radiées)", len(instruments) - len(kept))
    instruments = kept

    payload = {
        "source": "Bourse de Casablanca",
        "currency": "MAD",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "instruments": instruments,
    }
    INSTRUMENTS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    logger.info("%s écrit (%d valeurs)", INSTRUMENTS_FILE, len(instruments))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
