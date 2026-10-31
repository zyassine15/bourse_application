from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.data import DataUnavailableError


def make_prices(n: int = 600, drift: float = 0.0005, vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({"Close": close, "Volume": rng.integers(1_000_000, 2_000_000, n)}, index=idx)


def make_fundamentals(today: date | None = None) -> dict:
    """États financiers synthétiques d'une société rentable (dates relatives à aujourd'hui pour la fraîcheur du PER)."""
    today = today or date.today()
    fresh = today.isoformat()

    def pt(d: str, v: float, cur: str | None = "EUR"):
        return (d, v, cur)

    return {
        "trailingPeRatio": [pt(fresh, 18.5, None)],
        "trailingPegRatio": [pt(fresh, 1.4, None)],
        "trailingFreeCashFlow": [pt("2026-06-30", 800.0)],
        "trailingTotalRevenue": [pt("2026-06-30", 10_000.0)],
        "trailingOperatingIncome": [pt("2026-06-30", 1_500.0)],
        "annualFreeCashFlow": [pt("2025-12-31", 700.0)],
        "annualTotalRevenue": [pt("2025-12-31", 9_500.0)],
        "annualOperatingIncome": [pt("2025-12-31", 1_300.0)],
        "annualNetIncome": [pt("2025-12-31", 1_000.0)],
        "annualStockholdersEquity": [pt("2025-12-31", 8_000.0)],
        "annualInvestedCapital": [pt("2025-12-31", 12_000.0)],
        "annualTaxRateForCalcs": [pt("2025-12-31", 0.25)],
        "annualTotalDebt": [pt("2025-12-31", 3_000.0)],
        "annualCashCashEquivalentsAndShortTermInvestments": [pt("2025-12-31", 1_000.0)],
        "annualEBITDA": [pt("2025-12-31", 2_000.0)],
        "annualCashDividendsPaid": [pt("2025-12-31", -400.0)],
    }


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Client API sans réseau ni modèle : source de prix synthétique, models/ vide."""
    monkeypatch.setenv("BOURSE_MODELS_DIR", str(tmp_path))
    get_settings.cache_clear()
    from app.main import app

    def loader(ticker: str) -> pd.DataFrame:
        if ticker in ("ATH", "SPCX"):  # valeurs à historique trop court (introduction en bourse récente)
            return make_prices(100)
        if ticker == "SNP":  # valeur peu liquide : échanges un jour sur cinq
            df = make_prices(seed=5)
            df.loc[df.index[np.arange(len(df)) % 5 != 0], "Volume"] = 0
            return df
        if ticker == "IMO":  # source indisponible
            from app.data import DataUnavailableError

            raise DataUnavailableError("Source de données de la Bourse de Casablanca indisponible")
        return make_prices(seed=sum(map(ord, ticker)))

    def fundamentals_loader(ticker: str) -> dict:
        if ticker == "NXI.PA":  # source de fondamentaux en panne
            raise DataUnavailableError("panne")
        if ticker == "SPCX":  # aucune donnée publiée
            return {}
        return make_fundamentals()

    with TestClient(app) as c:
        app.state.price_loader = loader
        app.state.fundamentals_loader = fundamentals_loader
        yield c
    get_settings.cache_clear()
