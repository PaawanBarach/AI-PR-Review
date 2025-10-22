from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    store_dir: str = os.getenv("STORE_DIR", ".rag/faiss")
    top_k: int = int(os.getenv("TOP_K", "6"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    llm_tier: str = os.getenv("LLM_TIER", "balanced")
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "30"))
    enable_check_runs: bool = os.getenv("ENABLE_CHECK_RUNS", "true").lower() == "true"
    enable_sarif: bool = os.getenv("ENABLE_SARIF", "true").lower() == "true"
