"""Régénère `app/international.json` : actions et ETF américains/français suivis par l'application.

Usage : python -m training.refresh_international
Pour ajouter une valeur : l'ajouter à `UNIVERSE` ci-dessous (symbole Yahoo Finance), puis relancer ce script
(il vérifie que le symbole existe et mesure l'historique et la devise).
"""

import json
import logging
from datetime import UTC, datetime

from app.data import DataUnavailableError
from app.instruments import INTERNATIONAL_FILE
from app.yahoo import fetch_chart

logger = logging.getLogger("international")

# symbole Yahoo, nom affiché, type, effet de levier (ETF ×2 quotidien = 2)
UNIVERSE: list[dict] = [
    {"symbol": "STLAP.PA", "name": "Stellantis", "kind": "Action"},
    {"symbol": "NXI.PA", "name": "Nexity", "kind": "Action"},
    {"symbol": "STMPA.PA", "name": "STMicroelectronics", "kind": "Action"},
    {"symbol": "SPCX", "name": "SpaceX (Space Exploration Technologies)", "kind": "Action"},
    {"symbol": "LVE.PA", "name": "Amundi EURO STOXX 50 Daily (2x) Leveraged UCITS ETF Acc", "kind": "ETF", "leverage": 2},
    {"symbol": "CW8.PA", "name": "Amundi MSCI World UCITS ETF Acc", "kind": "ETF"},
    {"symbol": "PUST.PA", "name": "Amundi PEA Nasdaq-100 UCITS ETF Acc", "kind": "ETF"},
    {"symbol": "SEMI.AS", "name": "iShares MSCI Global Semiconductors UCITS ETF USD Acc", "kind": "ETF"},
]


def main() -> None:
    instruments = []
    for spec in UNIVERSE:
        try:
            df, meta = fetch_chart(spec["symbol"])
        except DataUnavailableError as exc:
            raise SystemExit(f"{spec['symbol']} : {exc}") from exc
        item = {
            **spec,
            "sessions": int(len(df)),
            "first_date": df.index[0].strftime("%Y-%m-%d"),
            "last_date": df.index[-1].strftime("%Y-%m-%d"),
            "currency": meta.get("currency", "USD"),
            "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
        }
        logger.info(
            "%-9s %-4s %-3s %4d séances depuis %s",
            item["symbol"],
            item["kind"],
            item["currency"],
            item["sessions"],
            item["first_date"],
        )
        instruments.append(item)
    payload = {
        "source": "Yahoo Finance",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "instruments": instruments,
    }
    INTERNATIONAL_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    logger.info("%s écrit (%d valeurs)", INTERNATIONAL_FILE, len(instruments))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
