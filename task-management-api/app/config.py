from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str
    gemini_api_key: str = ""
    app_name: str = "Task Management API"


@lru_cache
def get_settings() -> Settings:
    return Settings()
