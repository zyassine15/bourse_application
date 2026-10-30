"""Chargement des modèles entraînés (artefacts versionnés dans `models/`)."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.features import FEATURE_NAMES
from app.scoring import ml_to_score

logger = logging.getLogger(__name__)


@dataclass
class HorizonModel:
    estimator: object
    base_rate: float

    def score(self, feats: pd.DataFrame) -> pd.Series:
        proba = self.estimator.predict_proba(feats[FEATURE_NAMES])[:, 1]
        return pd.Series(ml_to_score(proba, self.base_rate), index=feats.index)


class ModelRegistry:
    """Lit `models/metadata.json` + un `.joblib` par horizon.

    Si absent ou incohérent (features différentes), l'application retombe proprement
    sur le score à règles seul : le service reste disponible.
    """

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.metadata: dict = {}
        self.backtest: dict = {}
        self.models: dict[str, HorizonModel] = {}
        self.load()

    def load(self) -> None:
        meta_path = self.models_dir / "metadata.json"
        self.metadata, self.models = {}, {}
        backtest_path = self.models_dir / "backtest.json"
        self.backtest = json.loads(backtest_path.read_text()) if backtest_path.exists() else {}
        if not meta_path.exists():
            logger.warning("Aucun modèle trouvé dans %s : score à règles uniquement", self.models_dir)
            return
        meta = json.loads(meta_path.read_text())
        if meta.get("features") != FEATURE_NAMES:
            logger.error("Features du modèle incompatibles avec le code : modèle ignoré")
            return
        for horizon, info in meta.get("horizons", {}).items():
            path = self.models_dir / info["file"]
            if not path.exists():
                continue
            self.models[horizon] = HorizonModel(joblib.load(path), float(info["base_rate"]))
        self.metadata = meta
        logger.info("Modèle %s chargé (%s)", meta.get("version"), ", ".join(self.models))

    @property
    def version(self) -> str:
        return self.metadata.get("version", "rules-only") if self.models else "rules-only"

    def validation(self) -> dict:
        """Le score a-t-il montré un pouvoir prédictif en backtest ? (IC de Spearman, t-stat >= 2)"""
        horizons = self.backtest.get("horizons")
        if not horizons:
            return {"status": "unknown", "message": "Aucun backtest disponible : score purement descriptif."}
        validated = [h for h, r in horizons.items() if r["ic_mean"] > 0 and r["ic_tstat"] >= 2]
        if not validated:
            return {
                "status": "not_validated",
                "message": "Le backtest ne démontre pas de pouvoir prédictif : ce score décrit la situation "
                "technique de l'action, il ne prédit pas son évolution.",
            }
        status = "validated" if len(validated) == len(horizons) else "partial"
        return {"status": status, "message": f"Pouvoir prédictif démontré en backtest sur : {', '.join(validated)}."}

    def ml_score(self, horizon: str, feats: pd.DataFrame) -> pd.Series | None:
        model = self.models.get(horizon)
        if model is None:
            return None
        valid = feats.dropna()
        out = pd.Series(np.nan, index=feats.index)
        if not valid.empty:
            out.loc[valid.index] = model.score(valid)
        return out
