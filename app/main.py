import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import monitoring
from app.config import get_settings
from app.data import DataUnavailableError, PriceCache, TTLCache, download_prices, normalize_ticker
from app.fundamentals import fetch_fundamentals
from app.indicators import build_indicators
from app.instruments import MARKETS, ML_MARKET, POPULAR, instruments_meta, load_instruments
from app.model import ModelRegistry
from app.schemas import AnalysisResponse, IndicatorsResponse, PricesResponse
from app.service import InsufficientHistoryError, analyze, price_summary

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger("bourse")


def _load_registry(app: FastAPI) -> None:
    registry = ModelRegistry(get_settings().models_dir)
    app.state.registry = registry
    app.state.drift.set_reference(registry.metadata.get("feature_reference"))
    monitoring.MODEL_LOADED.clear()
    monitoring.MODEL_LOADED.labels(version=registry.version).set(1 if registry.models else 0)


def _run_analysis(symbol: str) -> tuple[dict, pd.Series]:
    """Analyse complète d'une valeur de l'univers."""
    settings = get_settings()
    instrument = load_instruments()[symbol]
    prices = app.state.cache.get(symbol, app.state.price_loader)
    return analyze(instrument, prices, app.state.registry, settings.ml_weight, settings.history_days)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=get_settings().log_level, format="%(message)s")
    app.state.cache = PriceCache()
    app.state.price_loader = download_prices
    app.state.fund_cache = TTLCache(get_settings().fundamentals_ttl_seconds)
    app.state.fundamentals_loader = fetch_fundamentals
    app.state.drift = monitoring.DriftMonitor()
    _load_registry(app)
    yield


app = FastAPI(title="Bourse Score", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    if not request.url.path.startswith(("/api", "/metrics", "/health")):
        # Assets statiques : toujours revalider (304 si inchangé) pour qu'un déploiement soit visible immédiatement.
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/instruments", tags=["analysis"])
def instruments() -> dict:
    from app.features import MIN_ROWS

    items = load_instruments()
    return {
        "markets": [
            {"id": m, "label": label, "count": sum(1 for i in items.values() if i.market == m)} for m, label in MARKETS.items()
        ],
        "sources": instruments_meta(),
        "popular": {m: [s for s in symbols if s in items] for m, symbols in POPULAR.items()},
        "instruments": [
            {
                "symbol": i.symbol,
                "name": i.name,
                "market": i.market,
                "currency": i.currency,
                "kind": i.kind,
                "leverage": i.leverage,
                "analyzable": i.sessions >= MIN_ROWS,
            }
            for i in items.values()
        ],
    }


@app.get("/api/model", tags=["ops"])
def model_info() -> dict:
    registry: ModelRegistry = app.state.registry
    meta = {k: v for k, v in registry.metadata.items() if k != "feature_reference"}
    return {
        "version": registry.version,
        "ml_enabled": bool(registry.models),
        "validation": registry.validation(),
        "backtest": registry.backtest,
        "metadata": meta,
    }


@app.get("/api/drift", tags=["ops"])
def drift() -> dict:
    return app.state.drift.report()


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics() -> Response:
    app.state.drift.report()  # rafraîchit la jauge PSI avant le scrape
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get(
    "/api/analysis/{ticker}",
    response_model=AnalysisResponse,
    response_model_by_alias=True,
    tags=["analysis"],
)
def analysis(ticker: str) -> dict:
    start = time.perf_counter()
    try:
        symbol = normalize_ticker(ticker)
    except ValueError as exc:
        monitoring.REQUESTS.labels(status="invalid").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    instrument = load_instruments().get(symbol)
    if instrument is None:
        monitoring.REQUESTS.labels(status="unknown_symbol").inc()
        raise HTTPException(status_code=404, detail=f"Valeur inconnue : {symbol}")

    try:
        result, last_feats = _run_analysis(symbol)
    except DataUnavailableError as exc:
        monitoring.REQUESTS.labels(status="no_data").inc()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InsufficientHistoryError as exc:
        monitoring.REQUESTS.labels(status="insufficient_history").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    elapsed = time.perf_counter() - start
    monitoring.REQUESTS.labels(status="ok").inc()
    monitoring.LATENCY.observe(elapsed)
    monitoring.SCORE.observe(result["global"]["score"])
    if instrument.market == ML_MARKET:  # la référence de dérive est celle du marché d'entraînement
        app.state.drift.observe(last_feats)
    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "ticker": symbol,
                "model_version": result["model_version"],
                "global_score": result["global"]["score"],
                "scores": {h: v["score"] for h, v in result["horizons"].items()},
                "latency_ms": round(elapsed * 1000),
            }
        )
    )
    return result


@app.get("/api/indicators/{ticker}", response_model=IndicatorsResponse, tags=["analysis"])
def indicators_detail(ticker: str) -> dict:
    """Indicateurs détaillés (fondamentaux + techniques). Séparé de l'analyse : une panne de la source
    de fondamentaux ne doit jamais empêcher d'afficher le score."""
    try:
        symbol = normalize_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    instrument = load_instruments().get(symbol)
    if instrument is None:
        raise HTTPException(status_code=404, detail=f"Valeur inconnue : {symbol}")
    try:
        prices = app.state.cache.get(symbol, app.state.price_loader)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return build_indicators(instrument, prices, lambda s: app.state.fund_cache.get(s, app.state.fundamentals_loader))


@app.get("/api/prices/{ticker}", response_model=PricesResponse, tags=["analysis"])
def prices_only(ticker: str) -> dict:
    """Cours seul d'une valeur (utile quand l'historique est trop court pour calculer un score)."""
    try:
        symbol = normalize_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    instrument = load_instruments().get(symbol)
    if instrument is None:
        raise HTTPException(status_code=404, detail=f"Valeur inconnue : {symbol}")
    try:
        prices = app.state.cache.get(symbol, app.state.price_loader)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    from app.features import MIN_ROWS

    reason = (
        f"Historique insuffisant pour calculer un score ({len(prices)} séances, {MIN_ROWS} requises) : seul le cours est affiché."
        if len(prices) < MIN_ROWS
        else None
    )
    return {**price_summary(instrument, prices), "reason": reason}


# Doit rester en dernier : sert la PWA (index.html, manifest, service worker, icônes).
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
