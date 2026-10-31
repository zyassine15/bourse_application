from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOURSE_", env_file=".env", extra="ignore")

    models_dir: Path = ROOT / "models"
    cache_ttl_seconds: int = 900
    # Les états financiers changent au plus chaque trimestre : un cache d'une journée suffit.
    fundamentals_ttl_seconds: int = 86400
    history_days: int = 120
    # Poids du modèle ML dans le score final (le reste revient aux règles techniques).
    ml_weight: float = 0.3
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
