import json
from datetime import date

import httpx
import pytest

from app import casablanca
from app.data import DataUnavailableError, normalize_ticker
from app.instruments import load_instruments
from training.refresh_instruments import parse_listing


def _item(day: str, close: float, shares: int = 100) -> dict:
    return {"seance": day, "dernierCours": close, "titresEchanges": shares}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_history_chunks_and_parses(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        n = len(calls)
        items = [_item(f"0{n}/01/2024", 100 + n), _item(f"0{n}/01/2024", 100 + n)]  # doublon volontaire
        return httpx.Response(200, json={"items": items})

    monkeypatch.setattr(casablanca, "get_client", lambda: _client(handler))
    df = casablanca.fetch_history("IAM", start=date(2023, 12, 1), end=date(2026, 1, 1))
    assert len(calls) == 3  # tranches de ~1 an
    assert all(c["instrument"] == "IAM" and c["type"] == "actions" for c in calls)
    assert list(df.columns) == ["Close", "Volume"]
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert len(df) == 3


def test_fetch_history_empty_raises(monkeypatch):
    monkeypatch.setattr(casablanca, "get_client", lambda: _client(lambda r: httpx.Response(200, json={"items": []})))
    with pytest.raises(DataUnavailableError):
        casablanca.fetch_history("ZZZ", start=date(2025, 1, 1), end=date(2025, 6, 1))


def test_fetch_history_server_error_raises(monkeypatch):
    monkeypatch.setattr(casablanca.time, "sleep", lambda s: None)
    monkeypatch.setattr(casablanca, "get_client", lambda: _client(lambda r: httpx.Response(503)))
    with pytest.raises(DataUnavailableError, match="indisponible"):
        casablanca.fetch_history("IAM", start=date(2025, 1, 1), end=date(2025, 6, 1))


def test_tls_context_trusts_bundled_intermediate():
    assert casablanca.INTERMEDIATE_CA.exists()
    ctx = casablanca._ssl_context()
    assert ctx.verify_mode.name == "CERT_REQUIRED" and ctx.check_hostname


def test_parse_listing():
    entry = {"symbol": "IAM  ", "label": {"fr": "ITISSALAT AL-MAGHRIB", "en": "ITISSALAT"}}
    html = f"<script>{json.dumps([entry, entry, {'symbol': 'ATW', 'label': {'fr': 'ATTIJARIWAFA BANK'}}])}</script>"
    assert parse_listing(html) == {"IAM": "ITISSALAT AL-MAGHRIB", "ATW": "ATTIJARIWAFA BANK"}


def test_parse_listing_normalizes_whitespace():
    entry = {"symbol": "TGC", "label": {"fr": "TGCC S.A                       "}}
    assert parse_listing(json.dumps([entry])) == {"TGC": "TGCC S.A"}


def test_instrument_names_are_clean():
    assert all(i.name == " ".join(i.name.split()) for i in load_instruments().values())


def test_universe_is_moroccan_equities_only():
    instruments = load_instruments()
    assert {"IAM", "ATW", "BCP", "CIH"} <= set(instruments)
    # Ni droits d'attribution (DA ...) ni obligations (OC...) dans l'univers.
    assert not [i for i in instruments.values() if i.name.startswith("DA ") or i.symbol.startswith("OC")]
    assert all(i.sessions > 0 for i in instruments.values())


@pytest.mark.parametrize("raw,expected", [("iam", "IAM"), (" stlap.pa ", "STLAP.PA"), ("SEMI.AS", "SEMI.AS"), ("BRK-B", "BRK-B")])
def test_normalize_accepts_moroccan_and_yahoo_symbols(raw, expected):
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["AA$PL", "", "A" * 16, "iam;DROP", ".PA", "-X", "A B", "../etc"])
def test_normalize_rejects_bad_symbols(raw):
    with pytest.raises(ValueError):
        normalize_ticker(raw)
