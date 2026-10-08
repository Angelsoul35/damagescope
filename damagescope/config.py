"""Application configuration, loaded from environment variables (prefix ``DS_``)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DS_", env_file=".env", extra="ignore")

    environment: str = "local"
    model_dir: str = "artifacts"
    image_size: int = 128
    residual_sigma: float = 3.0
    min_region_area: int = 20
    severity_medium: float = 0.03
    severity_high: float = 0.08


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
