def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_analysis_ok(client):
    r = client.get("/api/analysis/iam")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "IAM"
    assert body["name"] == "ITISSALAT AL-MAGHRIB"
    assert body["currency"] == "MAD"
    assert 0 <= body["global"]["score"] <= 100
    assert set(body["horizons"]) == {"court", "moyen", "long"}
    assert body["model_version"] == "rules-only"
    hist = body["history"]
    assert len(hist["dates"]) == len(hist["series"]["global"]) == 120
    assert list(hist["series"]) == ["global", "court", "moyen", "long"]
    assert "conseil" in body["disclaimer"]
    assert body["validation"]["status"] == "unknown"  # pas de backtest dans le répertoire de test
    prices = body["prices"]
    assert len(prices["dates"]) == len(prices["close"]) == len(prices["sma50"]) == len(prices["sma200"]) == 600
    assert prices["sma50"][48] is None and prices["sma50"][49] is not None
    assert prices["sma200"][198] is None and prices["sma200"][199] is not None
    assert prices["dates"] == sorted(prices["dates"])
    assert prices["close"][-1] == body["last_price"]
    assert body["liquidity"]["level"] == "bonne"
    assert body["liquidity"]["warning"] is None


def test_illiquid_stock_gets_warning(client):
    liq = client.get("/api/analysis/SNP").json()["liquidity"]
    assert liq["level"] == "faible"
    assert "peu liquide" in liq["warning"]


def test_invalid_ticker(client):
    assert client.get("/api/analysis/AA$PL").status_code == 422
    assert client.get("/api/analysis/" + "A" * 40).status_code == 422


def test_unlisted_ticker_rejected(client):
    r = client.get("/api/analysis/AAPL")  # ni dans l'univers marocain, ni dans la liste internationale
    assert r.status_code == 404
    assert "inconnue" in r.json()["detail"]


def test_source_unavailable_404(client):
    assert client.get("/api/analysis/IMO").status_code == 404


def test_insufficient_history_422(client):
    r = client.get("/api/analysis/ATH")
    assert r.status_code == 422
    assert "insuffisant" in r.json()["detail"]


def test_instruments_endpoint(client):
    body = client.get("/api/instruments").json()
    by_symbol = {i["symbol"]: i for i in body["instruments"]}
    assert [m["id"] for m in body["markets"]] == ["MA", "INTL"]
    counts = {m["id"]: m["count"] for m in body["markets"]}
    assert counts["MA"] > 60 and counts["INTL"] == 8
    assert counts["MA"] + counts["INTL"] == len(by_symbol)
    assert {"IAM", "ATW", "BCP"} <= {s for s, i in by_symbol.items() if i["market"] == "MA"}
    assert {"AAPL", "MSFT"}.isdisjoint(by_symbol)
    assert all(i["currency"] == "MAD" for i in by_symbol.values() if i["market"] == "MA")
    for market, symbols in body["popular"].items():
        assert set(symbols) <= {s for s, i in by_symbol.items() if i["market"] == market}
    assert len(body["popular"]["MA"]) == 10 and len(body["popular"]["INTL"]) == 8
    assert all(set(i) == {"symbol", "name", "market", "currency", "kind", "leverage", "analyzable"} for i in by_symbol.values())


def test_international_universe_content(client):
    by_symbol = {i["symbol"]: i for i in client.get("/api/instruments").json()["instruments"]}
    expected = {
        "STLAP.PA": ("Action", "EUR"),
        "NXI.PA": ("Action", "EUR"),
        "STMPA.PA": ("Action", "EUR"),
        "SPCX": ("Action", "USD"),
        "LVE.PA": ("ETF", "EUR"),
        "CW8.PA": ("ETF", "EUR"),
        "PUST.PA": ("ETF", "EUR"),
        "SEMI.AS": ("ETF", "USD"),
    }
    for symbol, (kind, currency) in expected.items():
        assert (by_symbol[symbol]["kind"], by_symbol[symbol]["currency"], by_symbol[symbol]["market"]) == (kind, currency, "INTL")
    assert by_symbol["LVE.PA"]["leverage"] == 2
    assert by_symbol["SPCX"]["analyzable"] is False  # cotée depuis juin 2026 seulement


def test_international_analysis_is_rules_only(client):
    body = client.get("/api/analysis/cw8.pa").json()
    assert body["ticker"] == "CW8.PA" and body["market"] == "INTL"
    assert body["currency"] == "EUR" and body["kind"] == "ETF"
    assert body["model_version"] == "rules-only"
    assert all(h["ml_score"] is None for h in body["horizons"].values())
    assert body["validation"]["status"] == "unknown" and "internationales" in body["validation"]["message"]
    assert body["notice"] is None and body["leverage"] == 1


def test_leveraged_etf_has_risk_notice(client):
    body = client.get("/api/analysis/LVE.PA").json()
    assert body["leverage"] == 2
    assert "effet de levier ×2" in body["notice"]


def test_new_listing_returns_422_but_prices_available(client):
    r = client.get("/api/analysis/SPCX")
    assert r.status_code == 422 and "insuffisant" in r.json()["detail"]
    p = client.get("/api/prices/SPCX").json()
    assert p["currency"] == "USD" and p["market"] == "INTL"
    assert len(p["prices"]["close"]) == 100 and "insuffisant" in p["reason"]


def test_prices_endpoint_for_analyzable_stock(client):
    p = client.get("/api/prices/IAM").json()
    assert p["reason"] is None and p["currency"] == "MAD" and len(p["prices"]["dates"]) == 600
    assert client.get("/api/prices/AAPL").status_code == 404
    assert client.get("/api/prices/IMO").status_code == 404
    assert client.get("/api/prices/AA$PL").status_code == 422


def test_international_stocks_do_not_feed_drift_monitor(client):
    client.get("/api/analysis/NXI.PA")
    assert client.app.state.drift.report()["samples"] == 0
    client.get("/api/analysis/IAM")
    assert client.app.state.drift.report()["samples"] == 1


def test_model_info_without_model(client):
    body = client.get("/api/model").json()
    assert body["ml_enabled"] is False


def test_metrics_and_drift(client):
    client.get("/api/analysis/ATW")
    metrics = client.get("/metrics").text
    assert 'bourse_analyses_total{status="ok"}' in metrics
    assert client.get("/api/drift").json()["status"] == "no_reference"


def test_pwa_assets_served(client):
    assert client.get("/").status_code == 200
    manifest = client.get("/manifest.webmanifest").json()
    assert manifest["display"] == "standalone"
    assert client.get("/sw.js").status_code == 200
    assert client.get("/icons/icon-512.png").headers["content-type"] == "image/png"
    assert client.get("/icons/apple-touch-icon.png").status_code == 200
    assert client.get("/").headers["x-content-type-options"] == "nosniff"


def test_static_assets_always_revalidated(client):
    assert client.get("/app.js").headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/api/instruments").headers


def test_ranking_feature_removed(client):
    assert client.get("/api/ranking").status_code == 404
    html = client.get("/").text
    assert "Classement" not in html and "rk-list" not in html
    assert "ranking" not in client.get("/app.js").text.lower()


def test_analysis_no_longer_returns_old_indicator_table(client):
    assert "indicators" not in client.get("/api/analysis/IAM").json()


def test_indicators_endpoint_moroccan_stock(client):
    body = client.get("/api/indicators/IAM").json()
    assert body["ticker"] == "IAM" and body["currency"] == "MAD"
    fundamental, technical = body["groups"]
    assert fundamental["status"] == "unavailable" and len(fundamental["items"]) == 7
    assert [i["id"] for i in technical["items"]] == ["moving_averages", "rsi", "volumes"]


def test_indicators_endpoint_international_stock_uses_cache(client):
    calls = []
    original = client.app.state.fundamentals_loader
    client.app.state.fundamentals_loader = lambda s: calls.append(s) or original(s)
    first = client.get("/api/indicators/STMPA.PA").json()
    second = client.get("/api/indicators/STMPA.PA").json()
    assert first == second and calls == ["STMPA.PA"]  # un seul appel : cache de fondamentaux
    fundamental = first["groups"][0]
    assert fundamental["status"] == "ok"
    per = fundamental["items"][0]["values"][0]
    assert per["label"] == "PER" and per["value"] == 18.5 and per["unit"] == "ratio"


def test_indicators_endpoint_etf_and_outage_and_new_listing(client):
    assert client.get("/api/indicators/CW8.PA").json()["groups"][0]["status"] == "not_applicable"
    outage = client.get("/api/indicators/NXI.PA")  # source de fondamentaux en panne : pas d'erreur HTTP
    assert outage.status_code == 200 and outage.json()["groups"][0]["status"] == "unavailable"
    assert outage.json()["groups"][1]["status"] == "ok"
    spcx = client.get("/api/indicators/SPCX")  # historique trop court pour un score, indicateurs quand même servis
    assert spcx.status_code == 200
    mm200 = spcx.json()["groups"][1]["items"][0]["values"][-1]
    assert mm200["label"] == "MM200" and mm200["status"] == "unavailable"


def test_indicators_endpoint_errors(client):
    assert client.get("/api/indicators/AAPL").status_code == 404
    assert client.get("/api/indicators/AA$PL").status_code == 422
    assert client.get("/api/indicators/IMO").status_code == 404  # source de cours indisponible
