"""Backtest du score à règles : le score servi est-il corrélé aux rendements futurs ?

Usage : python -m training.backtest   -> écrit models/backtest.json (exposé par /api/model)

Mesures par horizon (actions de la Bourse de Casablanca, historique disponible depuis déc. 2023) :
- IC de Spearman score/rendement futur, par date puis moyenné (+ t-stat approximatif) ;
- rendement futur moyen et taux de hausse par verdict (Favorable / Neutre / Défavorable).
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app.scoring import HORIZONS, rule_score, verdict
from training.train import build_dataset, default_universe, download_universe, liquid_only

logger = logging.getLogger("backtest")


def backtest(data: pd.DataFrame) -> dict:
    out = {}
    for name, cfg in HORIZONS.items():
        d = data.dropna(subset=[f"fwd_{name}"]).copy()
        d["score"] = rule_score(d, name)
        d["verdict"] = d["score"].map(verdict)
        d["fwd"] = d[f"fwd_{name}"]

        # IC en coupe transversale (par date), échantillonné tous les `days` jours pour limiter le chevauchement.
        dates = np.unique(d.index.to_numpy())[:: cfg["days"]]
        ics = []
        for dt in dates:
            day = d.loc[dt]
            if len(day) >= 10 and day["score"].nunique() > 1:
                ics.append(day["score"].corr(day["fwd"], method="spearman"))
        ics = np.array(ics)
        by_verdict = (
            d.groupby("verdict")["fwd"]
            .agg(mean_return="mean", hit_rate=lambda s: (s > 0).mean(), n="count")
            .round(4)
            .to_dict("index")
        )
        out[name] = {
            "days": cfg["days"],
            "ic_mean": round(float(ics.mean()), 4),
            "ic_tstat": round(float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), 2),
            "n_periods": int(len(ics)),
            "by_verdict": by_verdict,
            "unconditional_mean_return": round(float(d["fwd"].mean()), 4),
            "unconditional_hit_rate": round(float((d["fwd"] > 0).mean()), 4),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="models")
    args = p.parse_args()

    frames = liquid_only(download_universe(default_universe()))
    data = build_dataset(frames)
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_tickers": len(frames),
        "horizons": backtest(data),
    }
    path = Path(args.out)
    path.mkdir(parents=True, exist_ok=True)
    (path / "backtest.json").write_text(json.dumps(report, indent=2))
    logger.info(json.dumps(report["horizons"], indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
