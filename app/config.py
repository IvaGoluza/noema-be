from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_max_tokens: int = 200
    anthropic_timeout_s: float = 30.0
    anthropic_max_retries: int = 3

    google_translate_api_key: str = ""
    translate_timeout_s: float = 5.0

    model_binary_path: str = "./models/binary_detector"
    model_type_path: str = "./models/category_classifier"

    cors_allowed_origins: list[str] = ["http://localhost:5173"]

    log_level: str = "INFO"
    log_user_text: bool = False


def get_settings() -> Settings:
    return Settings()
