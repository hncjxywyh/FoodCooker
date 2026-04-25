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

    max_token_buffer: int = 2000

settings = Settings()