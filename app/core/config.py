from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "exterior_assistant"
    cors_origins: list[str] = ["*"]
    max_image_edge: int = 1280
    max_upload_mb: int = 12
    log_level: str = "INFO"
    sam3_model: str = "unsloth/sam3.1"  # ungated mirror of facebook/sam3
    torch_threads: int = 6
    edit_model: str = "kontext"  # which free Space edits the photo; see EDITORS in services/redesign/editor.py
    hf_token: str | None = None  # image editing runs on Hugging Face ZeroGPU Spaces; free accounts get a daily GPU quota


settings = Settings()
