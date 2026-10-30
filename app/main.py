import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import monitoring
from app.config import get_settings
from app.data import DataUnavailableError, PriceCache, download_prices, normalize_ticker
from app.model import ModelRegistry
from app.schemas import AnalysisResponse
from app.service import InsufficientHistoryError, analyze

STATIC_DIR = Path(__file__).parent / "static"
logger = logging.getLogger("bourse")


def _load_registry(app: FastAPI) -> None:
    registry = ModelRegistry(get_settings().models_dir)
    app.state.registry = registry
    app.state.drift.set_reference(registry.metadata.get("feature_reference"))
    monitoring.MODEL_LOADED.clear()
    monitoring.MODEL_LOADED.labels(version=registry.version).set(1 if registry.models else 0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=get_settings().log_level, format="%(message)s")
    app.state.cache = PriceCache()
    app.state.price_loader = download_prices
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
    if request.url.path in ("/sw.js", "/") or request.url.path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


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

    settings = get_settings()
    try:
        prices = app.state.cache.get(symbol, app.state.price_loader)
        result, last_feats = analyze(symbol, prices, app.state.registry, settings.ml_weight, settings.history_days)
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


# Doit rester en dernier : sert la PWA (index.html, manifest, service worker, icônes).
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
