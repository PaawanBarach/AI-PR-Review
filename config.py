from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "none")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    store_backend: str = os.getenv("STORE_BACKEND", "faiss")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    p95_target: float = float(os.getenv("P95_TARGET", "90"))
