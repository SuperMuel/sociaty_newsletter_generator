from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False, env_file=".env", env_file_encoding="utf-8"
    )

    mongodb_uri: str
    mongodb_database: str
    mongodb_articles_collection: str
    mongodb_clusters_collection: str
    mongodb_clusterings_sessions_collection: str
