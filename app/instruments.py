"""Univers d'instruments, en deux marchés :

- ``MA`` : actions de la Bourse de Casablanca (``instruments.json``, généré par
  ``python -m training.refresh_instruments``) ;
- ``INTL`` : actions et ETF américains/français choisis à la main (``international.json``, généré par
  ``python -m training.refresh_international`` à partir de la liste curatée de ce script).
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DIR = Path(__file__).parent
INSTRUMENTS_FILE = DIR / "instruments.json"
INTERNATIONAL_FILE = DIR / "international.json"

MARKETS: dict[str, str] = {"MA": "Maroc", "INTL": "International"}
# Le modèle ML est entraîné sur la Bourse de Casablanca : il ne s'applique qu'à ce marché.
ML_MARKET = "MA"

# Valeurs phares affichées en raccourci, par marché (filtrées sur l'univers réellement chargé).
POPULAR: dict[str, list[str]] = {
    "MA": ["IAM", "ATW", "BCP", "BOA", "CIH", "LHM", "MNG", "TQM", "ADH", "SNP"],
    "INTL": ["STLAP.PA", "STMPA.PA", "NXI.PA", "SPCX", "CW8.PA", "PUST.PA", "SEMI.AS", "LVE.PA"],
}


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    sessions: int
    first_date: str | None
    last_date: str | None
    market: str = "MA"
    currency: str = "MAD"
    kind: str = "Action"  # « Action » ou « ETF »
    leverage: int = 1
    exchange: str | None = None


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


@lru_cache
def load_instruments() -> dict[str, Instrument]:
    out: dict[str, Instrument] = {}
    for path, market in ((INSTRUMENTS_FILE, "MA"), (INTERNATIONAL_FILE, "INTL")):
        for item in _read(path).get("instruments", []):
            item = {"market": market, **item, "name": " ".join(item["name"].split())}
            out[item["symbol"]] = Instrument(**item)
    return out


def instruments_meta() -> dict:
    """Métadonnées de génération (date, source) de chaque marché."""
    return {
        market: {k: v for k, v in _read(path).items() if k != "instruments"}
        for market, path in (("MA", INSTRUMENTS_FILE), ("INTL", INTERNATIONAL_FILE))
    }
