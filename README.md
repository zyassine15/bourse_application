# Bourse Score

Application mobile (PWA installable sur **Android et iOS**) qui calcule un **score de 0 à 100** pour **deux marchés** sélectionnables par un filtre (**Maroc** : 78 actions de la Bourse de Casablanca, cours en MAD ; **International** : 8 actions et ETF américains/français, cours en EUR/USD), **à trois horizons** (court ~1 semaine, moyen ~1 mois, long ~3 mois) et **dans le temps** (courbe d'évolution du score sur ~6 mois), avec, pour chaque action, le **graphique du cours** (clôture, MM50/MM200, périodes 3M/6M/1A/Tout, lecture au toucher). Hébergée sur **Render**, avec une chaîne MLOps complète.

> ⚠️ Indicateur statistique informatif, **pas un conseil en investissement**.

## Marchés

| Marché | Contenu | Source | Modèle ML |
|---|---|---|---|
| **Maroc** (`MA`) | 78 actions de la cote de Casablanca | API de la Bourse de Casablanca | oui (court terme) |
| **International** (`INTL`) | Stellantis (`STLAP.PA`), Nexity (`NXI.PA`), STMicroelectronics (`STMPA.PA`), SpaceX (`SPCX`), Amundi EURO STOXX 50 Daily (2x) Leveraged (`LVE.PA`), Amundi MSCI World Swap UCITS ETF Acc (`CW8.PA`), Amundi PEA Nasdaq-100 (`PUST.PA`), iShares MSCI Global Semiconductors USD Acc (`SEMI.AS`) | Yahoo Finance (cours ajustés des dividendes, ~5 ans) | non : règles techniques seules |

- Le filtre (Maroc / International / Tous) s'applique à la recherche et aux raccourcis. Le choix est mémorisé.
- Pour ajouter une valeur internationale : l'ajouter à `UNIVERSE` dans [`training/refresh_international.py`](training/refresh_international.py) (symbole Yahoo Finance), puis `python -m training.refresh_international`.
- **SpaceX** est cotée depuis juin 2026 (~98 séances) : il en faut 260 pour calculer un score. Seul son cours est affiché (il sera scoré automatiquement une fois l'historique suffisant). Même traitement pour CAP, GTM et T2S au Maroc.
- L'ETF `LVE.PA` (levier ×2 quotidien) affiche un avertissement de risque. L'ETF MSCI World est `CW8.PA` (Amundi MSCI World **Swap**, depuis 2021) ; `MWRD.PA` (Amundi Core, depuis 2024) est l'alternative. Le semi-conducteurs est coté en USD sur Euronext Amsterdam (`SEMI.AS`) ; `SEME.PA` (Paris, EUR) existe aussi.
- Yahoo Finance n'a pas d'API officielle : elle peut limiter le débit ou bloquer des IP de serveur (le client réessaie 3 fois sur deux hôtes). Si les valeurs internationales échouent sur Render, l'analyse de la valeur renvoie une erreur « source indisponible », sans affecter le Maroc.

## Indicateurs détaillés

La rubrique affiche 10 indicateurs, servis par `GET /api/indicators/{symbole}` (séparé de l'analyse : une panne de la source de fondamentaux n'empêche jamais d'afficher le score) :

| Indicateur | Actions internationales | ETF | Actions marocaines |
|---|---|---|---|
| PER, PEG, Free Cash Flow, ROE / ROIC, Dette nette / EBITDA, Marge opérationnelle, Payout Ratio | oui (Yahoo Finance) | sans objet | **non disponible** |
| Moyennes mobiles MM50 / MM200, RSI, Volumes | oui | oui | oui |

- **Fondamentaux marocains non disponibles** : la Bourse de Casablanca ne publie les états financiers que sous forme de PDF, et aucune source gratuite exploitable n'a été trouvée. L'interface l'indique (« N/D ») plutôt que d'afficher des valeurs inventées.
- Les ratios sont recalculés à partir des états financiers publiés, avec des règles explicites ([`app/fundamentals.py`](app/fundamentals.py)) : jamais deux périodes différentes dans un même ratio ; un ratio sans sens (résultat net négatif, EBITDA négatif…) est marqué « N/S » (non significatif) au lieu d'être calculé ; une donnée absente est « N/D ». Les montants sont dans la **devise de publication des comptes** (ex. STMicroelectronics publie en USD, même si le titre cote en EUR).
- Définitions : ROIC = résultat opérationnel × (1 − impôt) / capital investi (impôt publié, sinon 25 % supposé et indiqué) ; Dette nette = valeur publiée par la source, sinon dette totale − trésorerie et placements (baux inclus) ; Payout = dividendes versés / résultat net de l'exercice.
- PER et PEG viennent de Yahoo (le PEG repose sur des prévisions d'analystes) ; ils ne sont affichés que s'ils datent de moins de 60 jours.
- **Ces indicateurs sont informatifs : ils n'entrent pas dans le calcul du score**, qui reste purement technique. Fondamentaux mis en cache 24 h (`BOURSE_FUNDAMENTALS_TTL_SECONDS`).

## Données Maroc

- Source : API publique de la [Bourse de Casablanca](https://www.casablanca-bourse.com) (cours journaliers). **Historique disponible depuis le 1er déc. 2023 seulement** (~700 séances) ; 75 valeurs sur 78 ont assez d'historique pour être analysées (les introductions récentes CAP, GTM, T2S sont listées mais non analysables).
- L'univers (`app/instruments.json`) est généré par `python -m training.refresh_instruments` (exclut droits d'attribution, obligations et valeurs radiées).
- Cours non ajustés des dividendes. Beaucoup de valeurs cotent peu : l'application affiche un **avertissement de liquidité** (échanges sur < 50 % des 60 dernières séances).
- Le serveur du site n'envoie pas son certificat intermédiaire : [`app/certs/sectigo-dv-r36.pem`](app/certs/sectigo-dv-r36.pem) (CA publique) est ajouté au magasin de confiance, **la vérification TLS reste active**. Cette API n'est pas documentée officiellement : elle peut changer sans préavis.

## Ce que vaut le score (backtest)

> Le backtest ci-dessous porte **uniquement sur le marché marocain**. Aucun backtest n'existe pour les valeurs internationales : leur score est purement descriptif, ce que l'application indique.

[`models/backtest.json`](models/backtest.json) (régénéré par `make backtest`) mesure la corrélation (IC de Spearman) entre le score à règles et les rendements futurs, sur les 52 valeurs liquides :

| Horizon | IC | t-stat | Périodes indépendantes | Lecture |
|---|---|---|---|---|
| Court (5 j) | **-0,09** | **-4,2** | 103 | **significativement inversé** : un score élevé a plutôt précédé de moins bons rendements (retour à la moyenne) |
| Moyen (21 j) | -0,01 | -0,3 | 24 | aucun signal |
| Long (63 j) | +0,07 | +1,6 | 8 | prometteur mais non significatif (8 périodes, marché haussier) |

Le modèle ML (gradient boosting, valeurs liquides uniquement) : AUC walk-forward 0,558 au court terme (publié), 0,544 au moyen terme et 0,497 au long terme (refusés par le gate à 0,55). **Pourquoi 0,55** : sur ce marché, l'illiquidité seule (prix figés = « pas de hausse ») donne déjà une AUC de 0,549, ce qui avait fait croire à tort à un signal de 0,61 avant filtrage des valeurs peu liquides.

Conséquences :
- l'API/UI affichent le statut `validation` (ici `inverted`) : **n'utilisez pas ce score comme signal d'achat** ;
- le ML ne pèse que 30 % du score (`BOURSE_ML_WEIGHT`) et seulement sur l'horizon court ;
- avec ~2,9 ans d'historique, ces résultats sont fragiles ; ils seront recalculés à chaque réentraînement.

## Architecture

```
app/            FastAPI + PWA (static/)     training/   entraînement, backtest, univers
  features.py   features (partagées train/serve, pas de skew)
  scoring.py    score à règles par horizon  models/     artefacts versionnés (git)
  model.py      registre de modèles         tests/      pytest (sans réseau)
  monitoring.py Prometheus + dérive (PSI)   .github/    CI + réentraînement hebdo
  casablanca.py client API Bourse de Casablanca (TLS vérifié, découpage par tranches)
  yahoo.py      client Yahoo Finance (actions/ETF internationaux), réessais, bascule d'hôte
  fundamentals.py / indicators.py  ratios fondamentaux + indicateurs techniques de la rubrique « Indicateurs détaillés »
  instruments.py  univers à deux marchés (instruments.json = Maroc, international.json) · data.py  routage par marché + cache TTL
```

API : `GET /api/analysis/{symbole}` · `GET /api/instruments` (marchés + valeurs) · `GET /api/indicators/{symbole}` (indicateurs détaillés) · `GET /api/prices/{symbole}` (cours seul) · `GET /api/model` (version, métriques, backtest) · `GET /api/drift` · `GET /metrics` (Prometheus) · `GET /health` · docs Swagger sur `/docs`.

## MLOps

| Pratique | Mise en œuvre |
|---|---|
| Reproductibilité | dépendances figées (`requirements*.txt`), seeds fixes, version du modèle = date + SHA git |
| Pas de training/serving skew | une seule définition des features (`app/features.py`) |
| Validation temporelle | CV **walk-forward** expansive avec embargo = horizon (aucune fuite du futur) |
| Tracking | MLflow (params, métriques, artefacts), store local (`mlflow.db`/`mlruns`) ou `MLFLOW_TRACKING_URI` |
| Gate qualité / champion-challenger | publication seulement si AUC ≥ `--min-auc` (0,55) et pas de régression vs modèle en prod ; entraînement sur valeurs liquides uniquement |
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
python -m training.refresh_instruments   # régénère la liste des valeurs marocaines
python -m training.refresh_international  # régénère la liste internationale (Yahoo)
```

## Déploiement sur Render

1. Pousser le dépôt sur GitHub.
2. Render → **New → Blueprint** → sélectionner le dépôt (lit [`render.yaml`](render.yaml), service Docker, health check `/health`).
3. Chaque push sur `main` avec CI verte redéploie. Le plan gratuit met le service en veille après inactivité (premier chargement ~30-60 s).

Variables : `BOURSE_ML_WEIGHT` (0.3), `BOURSE_CACHE_TTL_SECONDS` (900), `BOURSE_LOG_LEVEL`, `BOURSE_FUNDAMENTALS_TTL_SECONDS` (86400).
Pour le workflow de réentraînement : Settings → Actions → autoriser « Read and write permissions » et « Allow GitHub Actions to create pull requests ».

## Installer sur le téléphone

- **Android (Chrome)** : ouvrir l'URL Render → menu ⋮ → *Installer l'application*.
- **iOS (Safari)** : ouvrir l'URL → Partager → *Sur l'écran d'accueil*.

L'app s'ouvre alors en plein écran avec son icône ; la coque est mise en cache (démarrage rapide), les scores restent toujours chargés en direct.