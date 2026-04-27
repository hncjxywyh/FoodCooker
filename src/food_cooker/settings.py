from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider: "openai" or "dashscope"
    llm_provider: str = "dashscope"

    # OpenAI
    openai_api_key: str = ""
    openai_model_name: str = "gpt-4o-mini"

    # DashScope
    dashscope_api_key: str = ""
    dashscope_model_name: str = "qwen-plus"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Embedding provider: "huggingface" or "dashscope"
    embedding_provider: str = "dashscope"
    huggingface_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dashscope_embedding_model: str = "text-embedding-v3"

    chroma_db_path: Path = BASE_DIR / "data" / "chroma_db"
    user_profiles_path: Path = BASE_DIR / "data" / "user_profiles.json"
    recipes_data_path: Path = BASE_DIR / "data" / "recipes_raw.json"
    nutrition_db_path: Path = BASE_DIR / "data" / "nutrition_db.json"

    max_token_buffer: int = 2000

    # Database for Chainlit chat history (PostgreSQL required for SQLAlchemyDataLayer)
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/chainlit"

    # LangSmith observability (set API key to enable auto-tracing)
    langsmith_api_key: str = ""
    langsmith_project: str = "foodcooker"

    # JWT Auth
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Auth database (SQLite for dev, Postgres for prod)
    auth_db_url: str = f"sqlite+aiosqlite:///{BASE_DIR / 'data' / 'auth.db'}"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Cohere (optional reranker alternative)
    cohere_api_key: str = ""

    # Image generation
    openai_api_key_for_images: str = ""  # Uses openai_api_key if empty
    image_generation_model: str = "dall-e-3"

    # Logging
    log_level: str = "INFO"  # DEBUG/INFO/WARNING/ERROR
    log_file: Path | None = BASE_DIR / "data" / "logs" / "app.log"

settings = Settings()