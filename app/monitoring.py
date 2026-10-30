"""Observabilité : métriques Prometheus + détection de dérive des features (PSI)."""

import threading
from collections import deque

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram

from app.features import FEATURE_NAMES

REQUESTS = Counter("bourse_analyses_total", "Analyses demandées", ["status"])
LATENCY = Histogram("bourse_analysis_seconds", "Durée d'une analyse")
SCORE = Histogram("bourse_global_score", "Distribution des scores globaux servis", buckets=list(range(0, 101, 10)))
MODEL_LOADED = Gauge("bourse_model_loaded", "1 si un modèle ML est chargé", ["version"])
DRIFT_PSI = Gauge("bourse_feature_psi_max", "PSI maximal sur les features (live vs entraînement)")

PSI_WARN = 0.10
PSI_ALERT = 0.25
MIN_SAMPLES = 50


def psi(expected_edges: np.ndarray, expected_frac: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index entre la distribution d'entraînement (bins) et le live."""
    edges = np.concatenate([[-np.inf], expected_edges, [np.inf]])
    counts, _ = np.histogram(actual, bins=edges)
    actual_frac = np.clip(counts / max(counts.sum(), 1), 1e-4, None)
    expected = np.clip(np.asarray(expected_frac), 1e-4, None)
    return float(np.sum((actual_frac - expected) * np.log(actual_frac / expected)))


class DriftMonitor:
    """Fenêtre glissante des features servies, comparée aux bins d'entraînement."""

    def __init__(self, reference: dict | None = None, window: int = 2000):
        self.reference = reference or {}
        self._rows: deque[dict[str, float]] = deque(maxlen=window)
        self._lock = threading.Lock()

    def set_reference(self, reference: dict | None) -> None:
        self.reference = reference or {}

    def observe(self, feats_row: pd.Series) -> None:
        with self._lock:
            self._rows.append({k: float(feats_row[k]) for k in FEATURE_NAMES})

    def report(self) -> dict:
        with self._lock:
            rows = list(self._rows)
        if not self.reference:
            return {"status": "no_reference", "samples": len(rows), "features": {}}
        if len(rows) < MIN_SAMPLES:
            return {"status": "insufficient_data", "samples": len(rows), "features": {}}
        df = pd.DataFrame(rows)
        result = {}
        for name in FEATURE_NAMES:
            ref = self.reference.get(name)
            if ref:
                result[name] = round(psi(np.array(ref["edges"]), np.array(ref["fractions"]), df[name].dropna().to_numpy()), 4)
        worst = max(result.values(), default=0.0)
        DRIFT_PSI.set(worst)
        status = "alert" if worst >= PSI_ALERT else "warning" if worst >= PSI_WARN else "ok"
        return {"status": status, "samples": len(rows), "max_psi": worst, "features": result}
