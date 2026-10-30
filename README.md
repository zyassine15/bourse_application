# Bourse Score

Application mobile (PWA installable sur **Android et iOS**) qui calcule un **score de 0 à 100** indiquant si la situation d'une action est favorable, **à trois horizons** (court ~1 semaine, moyen ~1 mois, long ~3 mois) et **dans le temps** (courbe d'évolution du score sur ~6 mois). Hébergée sur **Render**, avec une chaîne MLOps complète.

> ⚠️ Indicateur statistique informatif, **pas un conseil en investissement**.

## Ce que vaut le score (backtest)

Le backtest ([`models/backtest.json`](models/backtest.json), régénéré par `make backtest`) mesure la corrélation (IC de Spearman) entre le score à règles et les rendements futurs sur 34 titres / 10 ans : **IC ≈ 0, non significatif** aux trois horizons. De même, le modèle ML (gradient boosting) obtient une AUC walk-forward ≈ 0,50-0,51 : **le gate qualité refuse de le publier**. Conséquence :

- le score servi = **règles techniques** (tendance, momentum, RSI, MACD, risque) : il **décrit** l'état technique d'une action ;
- l'API/UI affichent ce statut (`validation`) tant qu'aucun pouvoir prédictif n'est démontré ;
- dès qu'un réentraînement passera le gate (`--min-auc`), le modèle sera publié automatiquement et mélangé au score (`BOURSE_ML_WEIGHT`).

## Architecture

```
app/            FastAPI + PWA (static/)     training/   entraînement, backtest
  features.py   features (partagées train/serve, pas de skew)
  scoring.py    score à règles par horizon  models/     artefacts versionnés (git)
  model.py      registre de modèles         tests/      pytest (sans réseau)
  monitoring.py Prometheus + dérive (PSI)   .github/    CI + réentraînement hebdo
  data.py       API chart Yahoo (+ yfinance en secours), cache TTL
```

API : `GET /api/analysis/{ticker}` · `GET /api/model` (version, métriques, backtest) · `GET /api/drift` · `GET /metrics` (Prometheus) · `GET /health` · docs Swagger sur `/docs`.

## MLOps

| Pratique | Mise en œuvre |
|---|---|
| Reproductibilité | dépendances figées (`requirements*.txt`), seeds fixes, version du modèle = date + SHA git |
| Pas de training/serving skew | une seule définition des features (`app/features.py`) |
| Validation temporelle | CV **walk-forward** expansive avec embargo = horizon (aucune fuite du futur) |
| Tracking | MLflow (params, métriques, artefacts), store local (`mlflow.db`/`mlruns`) ou `MLFLOW_TRACKING_URI` |
| Gate qualité / champion-challenger | publication seulement si AUC ≥ `--min-auc` et pas de régression vs modèle en prod |
| Versionnement des artefacts | `models/` (joblib + `metadata.json` avec métriques, features, bins de référence) |
| Dégradation maîtrisée | modèle absent/incompatible ⇒ fallback sur le score à règles, le service reste up |
| Monitoring | métriques Prometheus, logs JSON de chaque prédiction, **dérive des features (PSI)** vs entraînement |
| CI/CD | GitHub Actions : lint, tests, build Docker + smoke test ; Render ne déploie que si la CI est verte |
| Entraînement continu | workflow hebdo : réentraîne + backteste + ouvre une PR avec les artefacts ; le merge redéploie |

## Développement

```bash
make install          # venv + dépendances
make run              # http://localhost:8000
make test lint
make train            # entraînement (MLflow : `mlflow ui`)
make backtest
```

## Déploiement sur Render

1. Pousser le dépôt sur GitHub.
2. Render → **New → Blueprint** → sélectionner le dépôt (lit [`render.yaml`](render.yaml), service Docker, health check `/health`).
3. Chaque push sur `main` avec CI verte redéploie. Le plan gratuit met le service en veille après inactivité (premier chargement ~30-60 s).

Variables : `BOURSE_ML_WEIGHT` (0.6), `BOURSE_CACHE_TTL_SECONDS` (900), `BOURSE_LOG_LEVEL`.
Pour le workflow de réentraînement : Settings → Actions → autoriser « Read and write permissions » et « Allow GitHub Actions to create pull requests ».

## Installer sur le téléphone

- **Android (Chrome)** : ouvrir l'URL Render → menu ⋮ → *Installer l'application*.
- **iOS (Safari)** : ouvrir l'URL → Partager → *Sur l'écran d'accueil*.

L'app s'ouvre alors en plein écran avec son icône ; la coque est mise en cache (démarrage rapide), les scores restent toujours chargés en direct.