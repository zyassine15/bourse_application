import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


def make_prices(n: int = 600, drift: float = 0.0005, vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.DataFrame({"Close": close, "Volume": rng.integers(1_000_000, 2_000_000, n)}, index=idx)


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
        if ticker == "UNKNOWN":
            from app.data import DataUnavailableError

            raise DataUnavailableError("Aucune donnée pour UNKNOWN")
        if ticker == "SHORT":
            return make_prices(100)
        return make_prices(seed=abs(hash(ticker)) % 1000)

    with TestClient(app) as c:
        app.state.price_loader = loader
        yield c
    get_settings.cache_clear()
