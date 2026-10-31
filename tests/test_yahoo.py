import httpx
import pytest

from app import data, yahoo
from app.data import DataUnavailableError
from app.instruments import load_instruments

DAY = 86_400
T0 = 1_790_000_000  # une date arbitraire (UTC), ~9h00 heure de Paris après décalage


def chart(closes, adj=None, volume=None, regular=None, offset=3600):
    stamps = [T0 + i * DAY for i in range(len(closes))]
    indicators = {"quote": [{"close": closes, "volume": volume if volume is not None else [1000] * len(closes)}]}
    if adj is not None:
        indicators["adjclose"] = [{"adjclose": adj}]
    meta = {"currency": "EUR", "gmtoffset": offset, **(regular or {})}
    return {"chart": {"result": [{"meta": meta, "timestamp": stamps, "indicators": indicators}]}}


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(yahoo.time, "sleep", lambda s: None)


def test_parses_adjusted_close_and_volume():
    payload = chart([10.0, 11.0, None, 12.0], adj=[9.0, 10.0, None, 11.0], volume=[5, None, 7, 8])
    df, meta = yahoo.fetch_chart("CW8.PA", client=client_for(lambda r: httpx.Response(200, json=payload)))
    assert meta["currency"] == "EUR"
    assert list(df["Close"]) == [9.0, 10.0, 11.0]  # cours ajusté, jour sans cours écarté
    assert list(df["Volume"]) == [5, 0, 8]
    assert df.index.is_monotonic_increasing and df.index.is_unique


def test_falls_back_to_plain_close_without_adjclose():
    df, _ = yahoo.fetch_chart("X", client=client_for(lambda r: httpx.Response(200, json=chart([3.0, 4.0]))))
    assert list(df["Close"]) == [3.0, 4.0]


def test_missing_volume_series_is_zero():
    payload = chart([3.0, 4.0])
    payload["chart"]["result"][0]["indicators"]["quote"][0]["volume"] = None
    df, _ = yahoo.fetch_chart("X", client=client_for(lambda r: httpx.Response(200, json=payload)))
    assert list(df["Volume"]) == [0, 0]


def test_unsettled_last_bar_filled_with_last_traded_price():
    last_stamp = T0 + 2 * DAY
    payload = chart([3.0, 4.0, None], regular={"regularMarketPrice": 4.5, "regularMarketTime": last_stamp + 3600})
    df, _ = yahoo.fetch_chart("X", client=client_for(lambda r: httpx.Response(200, json=payload)))
    assert list(df["Close"]) == [3.0, 4.0, 4.5]


def test_last_bar_not_filled_when_quote_is_from_another_day():
    payload = chart([3.0, 4.0, None], regular={"regularMarketPrice": 9.9, "regularMarketTime": T0})
    df, _ = yahoo.fetch_chart("X", client=client_for(lambda r: httpx.Response(200, json=payload)))
    assert list(df["Close"]) == [3.0, 4.0]


def test_404_is_unknown_symbol_without_retry():
    calls = []
    client = client_for(lambda r: calls.append(r) or httpx.Response(404, json={}))
    with pytest.raises(DataUnavailableError, match="introuvable"):
        yahoo.fetch_chart("ZZZ", client=client)
    assert len(calls) == 1


def test_rate_limit_retries_on_alternate_host_then_succeeds():
    hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if len(hosts) < 3:
            return httpx.Response(429)
        return httpx.Response(200, json=chart([1.0, 2.0]))

    df, _ = yahoo.fetch_chart("X", client=client_for(handler))
    assert len(df) == 2
    assert hosts[:2] == list(yahoo.HOSTS)  # alternance query1 / query2


def test_persistent_rate_limit_raises_unavailable():
    with pytest.raises(DataUnavailableError, match="indisponible"):
        yahoo.fetch_chart("X", client=client_for(lambda r: httpx.Response(429)))


def test_empty_result_raises():
    payload = {"chart": {"result": None, "error": {"code": "Not Found"}}}
    with pytest.raises(DataUnavailableError):
        yahoo.fetch_chart("X", client=client_for(lambda r: httpx.Response(200, json=payload)))


def test_symbol_is_url_encoded():
    seen = []
    client = client_for(lambda r: seen.append(str(r.url)) or httpx.Response(200, json=chart([1.0])))
    yahoo.fetch_chart("A/B?x", client=client)
    assert "/chart/A%2FB%3Fx" in seen[0]


def test_download_prices_routes_by_market(monkeypatch):
    calls = []
    monkeypatch.setattr("app.casablanca.fetch_history", lambda s: calls.append(("casablanca", s)) or "MA-DF")
    monkeypatch.setattr("app.yahoo.fetch_history", lambda s: calls.append(("yahoo", s)) or "INTL-DF")
    assert data.download_prices("IAM") == "MA-DF"
    assert data.download_prices("CW8.PA") == "INTL-DF"
    assert calls == [("casablanca", "IAM"), ("yahoo", "CW8.PA")]
    with pytest.raises(DataUnavailableError):
        data.download_prices("AAPL")


def test_symbol_namespaces_do_not_collide():
    markets = {}
    for symbol, instrument in load_instruments().items():
        assert markets.setdefault(symbol, instrument.market) == instrument.market
    assert {i.market for i in load_instruments().values()} == {"MA", "INTL"}
