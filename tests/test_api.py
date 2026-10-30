def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_analysis_ok(client):
    r = client.get("/api/analysis/aapl")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert 0 <= body["global"]["score"] <= 100
    assert set(body["horizons"]) == {"court", "moyen", "long"}
    assert body["model_version"] == "rules-only"
    hist = body["history"]
    assert len(hist["dates"]) == len(hist["series"]["global"]) == 120
    assert set(hist["series"]) == {"court", "moyen", "long", "global"}
    assert "conseil" in body["disclaimer"]
    assert body["validation"]["status"] == "unknown"  # pas de backtest dans le répertoire de test


def test_invalid_ticker(client):
    assert client.get("/api/analysis/AA$PL").status_code == 422
    assert client.get("/api/analysis/" + "A" * 40).status_code == 422


def test_unknown_ticker_404(client):
    assert client.get("/api/analysis/UNKNOWN").status_code == 404


def test_insufficient_history_422(client):
    r = client.get("/api/analysis/SHORT")
    assert r.status_code == 422
    assert "insuffisant" in r.json()["detail"]


def test_model_info_without_model(client):
    body = client.get("/api/model").json()
    assert body["ml_enabled"] is False


def test_metrics_and_drift(client):
    client.get("/api/analysis/MSFT")
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
