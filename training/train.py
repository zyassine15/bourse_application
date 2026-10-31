"""Pipeline d'entraînement : données -> features -> CV walk-forward -> gate qualité -> artefacts.

Usage :
    python -m training.train                 # entraîne et écrit dans models/
    python -m training.train --min-auc 0.53  # gate qualité plus strict

Tracking MLflow (params, métriques, artefacts) dans ./mlruns, ou MLFLOW_TRACKING_URI.
Code de sortie 2 si aucun horizon ne passe le gate (le modèle en production est conservé).
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from app.data import DataUnavailableError, download_prices
from app.features import FEATURE_NAMES, prepare_features
from app.instruments import ML_MARKET, load_instruments
from app.scoring import HORIZONS

logger = logging.getLogger("training")

MIN_SESSIONS = 300  # historique minimal d'une valeur pour entrer dans l'entraînement


# Les valeurs peu liquides ont des prix figés (rendement nul) : un modèle y apprend « pas de hausse = illiquide »
# (AUC ≈ 0,55 avec la seule illiquidité, mesuré) au lieu d'un signal exploitable. On les exclut.
MIN_TRADED_RATIO = 0.9
LIQUIDITY_WINDOW = 250


def liquid_only(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    kept = {t: f for t, f in frames.items() if (f["Volume"].tail(LIQUIDITY_WINDOW) > 0).mean() >= MIN_TRADED_RATIO}
    logger.info("Univers liquide : %d/%d valeurs", len(kept), len(frames))
    return kept


def default_universe() -> list[str]:
    """Actions du marché d'entraînement (Bourse de Casablanca) avec assez d'historique.

    Les valeurs internationales sont exclues : le modèle ne leur est jamais appliqué.
    """
    return sorted(s for s, i in load_instruments().items() if i.market == ML_MARKET and i.sessions >= MIN_SESSIONS)


MODEL_PARAMS = dict(
    max_depth=3,
    learning_rate=0.05,
    max_iter=200,
    l2_regularization=1.0,
    min_samples_leaf=200,
    random_state=42,
)


def new_estimator() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**MODEL_PARAMS)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def build_dataset(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une ligne par (date, ticker) avec features + rendement futur pour chaque horizon."""
    parts = []
    for ticker, prices in price_frames.items():
        feats = prepare_features(prices)
        close = prices["Close"].reindex(feats.index)
        full_close = prices["Close"]
        for name, cfg in HORIZONS.items():
            fwd = full_close.shift(-cfg["days"]) / full_close - 1
            feats[f"fwd_{name}"] = fwd.reindex(feats.index)
        feats["ticker"] = ticker
        feats["close"] = close
        parts.append(feats)
    data = pd.concat(parts)
    data.index.name = "date"
    return data.sort_index()


def walk_forward_splits(dates: np.ndarray, horizon_days: int, n_splits: int = 5, min_train_frac: float = 0.4):
    """Splits expansifs dans le temps avec embargo = horizon (pas de fuite du futur)."""
    unique = np.unique(dates)
    start = int(len(unique) * min_train_frac)
    bounds = np.linspace(start, len(unique), n_splits + 1, dtype=int)
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        train_end = unique[max(a - horizon_days, 0)]
        yield dates < train_end, (dates >= unique[a]) & (dates <= unique[b - 1])


def evaluate(data: pd.DataFrame, horizon: str, n_splits: int = 5) -> dict:
    days = HORIZONS[horizon]["days"]
    labelled = data.dropna(subset=[f"fwd_{horizon}"])
    X = labelled[FEATURE_NAMES]
    y = (labelled[f"fwd_{horizon}"] > 0).astype(int)
    dates = labelled.index.to_numpy()

    aucs, briers, accs, baselines = [], [], [], []
    for train_mask, test_mask in walk_forward_splits(dates, days, n_splits):
        if train_mask.sum() < 1000 or y[test_mask].nunique() < 2:
            continue
        model = new_estimator().fit(X[train_mask], y[train_mask])
        proba = model.predict_proba(X[test_mask])[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[test_mask], proba))
        briers.append(brier_score_loss(y[test_mask], proba))
        accs.append(accuracy_score(y[test_mask], pred))
        # Baseline naïf : toujours prédire la classe majoritaire du train.
        baselines.append(accuracy_score(y[test_mask], np.full(test_mask.sum(), int(y[train_mask].mean() >= 0.5))))
    if not aucs:
        raise RuntimeError(f"Pas assez de données pour évaluer l'horizon {horizon}")
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_folds": [round(a, 4) for a in aucs],
        "brier_mean": float(np.mean(briers)),
        "accuracy_mean": float(np.mean(accs)),
        "baseline_accuracy_mean": float(np.mean(baselines)),
        "n_samples": int(len(labelled)),
    }


def feature_reference(data: pd.DataFrame, bins: int = 10) -> dict:
    """Bins quantiles de chaque feature (référence pour le PSI de monitoring)."""
    ref = {}
    for name in FEATURE_NAMES:
        values = data[name].to_numpy()
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
        counts, _ = np.histogram(values, bins=np.concatenate([[-np.inf], edges, [np.inf]]))
        ref[name] = {"edges": edges.tolist(), "fractions": (counts / counts.sum()).tolist()}
    return ref


def download_universe(tickers: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for t in tickers:
        try:
            frames[t] = download_prices(t)
        except DataUnavailableError as exc:
            logger.warning("Ticker ignoré %s : %s", t, exc)
    return frames


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    tickers = args.tickers or default_universe()
    frames = liquid_only(download_universe(tickers))
    if len(frames) < 5:
        logger.error("Trop peu de tickers téléchargés (%d)", len(frames))
        return 1
    data = build_dataset(frames)
    logger.info("Dataset : %d lignes, %d tickers", len(data), len(frames))

    old_meta = {}
    if (out_dir / "metadata.json").exists():
        old_meta = json.loads((out_dir / "metadata.json").read_text())

    version = f"{datetime.now(UTC):%Y%m%d}-{git_sha()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import mlflow
    except ImportError:
        mlflow = None
        logger.warning("mlflow non installé : tracking désactivé")
    if mlflow:
        mlflow.set_experiment("bourse-score")
        run_ctx = mlflow.start_run(run_name=version)
    else:
        from contextlib import nullcontext

        run_ctx = nullcontext()

    horizons_meta: dict[str, dict] = {}
    with run_ctx:
        if mlflow:
            mlflow.log_params({**MODEL_PARAMS, "n_tickers": len(frames), "min_auc": args.min_auc})
        for name in HORIZONS:
            metrics = evaluate(data, name, args.splits)
            champion = old_meta.get("horizons", {}).get(name, {}).get("metrics", {}).get("auc_mean")
            passed = metrics["auc_mean"] >= args.min_auc
            beats_champion = champion is None or metrics["auc_mean"] >= champion - args.tolerance
            logger.info(
                "[%s] AUC=%.4f±%.4f acc=%.3f (baseline %.3f) gate=%s champion=%s",
                name, metrics["auc_mean"], metrics["auc_std"], metrics["accuracy_mean"],
                metrics["baseline_accuracy_mean"], passed, beats_champion,
            )  # fmt: skip
            if mlflow:
                mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items() if isinstance(v, float)})
            if not (passed and beats_champion):
                if name in old_meta.get("horizons", {}) and not passed:
                    logger.warning("[%s] gate non passé : modèle précédent conservé", name)
                if name in old_meta.get("horizons", {}) and old_meta.get("features") == FEATURE_NAMES:
                    horizons_meta[name] = old_meta["horizons"][name]
                continue

            labelled = data.dropna(subset=[f"fwd_{name}"])
            y = (labelled[f"fwd_{name}"] > 0).astype(int)
            final = new_estimator().fit(labelled[FEATURE_NAMES], y)
            filename = f"model_{name}.joblib"
            joblib.dump(final, out_dir / filename, compress=3)
            horizons_meta[name] = {
                "file": filename,
                "days": HORIZONS[name]["days"],
                "base_rate": float(y.mean()),
                "metrics": metrics,
            }

        if not horizons_meta:
            logger.error("Aucun modèle n'a passé le gate qualité : rien n'est écrit")
            return 2

        meta = {
            "version": version,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "data_source": "Bourse de Casablanca",
            "universe_filter": f"échanges sur >= {MIN_TRADED_RATIO:.0%} des {LIQUIDITY_WINDOW} dernières séances",
            "data_start": str(data.index.min().date()),
            "data_end": str(data.index.max().date()),
            "n_tickers": len(frames),
            "tickers": sorted(frames),
            "features": FEATURE_NAMES,
            "model_params": MODEL_PARAMS,
            "horizons": horizons_meta,
            "feature_reference": feature_reference(data),
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        if mlflow:
            mlflow.log_artifacts(str(out_dir), artifact_path="model")
        logger.info("Modèle %s écrit dans %s", version, out_dir)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="models")
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument(
        "--min-auc", type=float, default=0.55, help="AUC walk-forward minimale pour publier un modèle (> benchmark illiquidité)"
    )
    p.add_argument("--tolerance", type=float, default=0.01, help="Régression d'AUC tolérée vs modèle en production")
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(run(parse_args()))
