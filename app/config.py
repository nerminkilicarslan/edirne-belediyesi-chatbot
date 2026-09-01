from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "Edirne Belediye Chatbot"
    VERSION: str = "2.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # ── Server ──────────────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # ── Ollama LLM ──────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_CONTEXT_WINDOW: int = 4096
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_TOP_P: float = 0.9
    OLLAMA_REPEAT_PENALTY: float = 1.1

    # ── Embeddings ──────────────────────────────────────────────────────────────
    # BGE-M3 supports Turkish natively — 8192 token context
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    # "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" as fallback
    EMBEDDING_DEVICE: str = "auto"
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MAX_LENGTH: int = 512
    EMBEDDING_NORMALIZE: bool = True

    # ── ChromaDB Vector Store ────────────────────────────────────────────────────
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "edirne_kb"

    # ── Retrieval ────────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_SCORE_THRESHOLD: float = 0.28
    # BM25 + semantic hybrid weights (must sum to 1.0)
    BM25_WEIGHT: float = 0.30
    SEMANTIC_WEIGHT: float = 0.70
    # After hybrid, rerank and keep top-N for LLM context
    RERANK_TOP_N: int = 3

    # ── Classifier Thresholds ────────────────────────────────────────────────────
    INTENT_HIGH_CONFIDENCE: float = 0.72
    INTENT_LOW_CONFIDENCE: float = 0.42
    CATEGORY_HIGH_CONFIDENCE: float = 0.62
    CATEGORY_LOW_CONFIDENCE: float = 0.32

    # ── Pipeline Behaviour ───────────────────────────────────────────────────────
    MAX_QUERY_LENGTH: int = 500
    # Confidence below this → ask for clarification instead of retrieving
    CLARIFICATION_THRESHOLD: float = 0.30
    FALLBACK_MESSAGE: str = (
        "Üzgünüm, bu konuda size kesin bir bilgi sunamıyorum. "
        "Daha fazla yardım için Edirne Belediyesi'nin 153 Beyaz Masa hattını "
        "arayabilir veya edirne.bel.tr adresini ziyaret edebilirsiniz."
    )

    # ── Confidence Thresholds (legacy keys kept for orchestrator) ────────────────
    CONFIDENCE_THRESHOLD_RAG: float = 0.45
    CONFIDENCE_THRESHOLD_DIRECT: float = 0.85

    # ── Logging ──────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    LOG_JSON: bool = False

    # ── Security ─────────────────────────────────────────────────────────────────
    API_KEY: Optional[str] = None
    CORS_ORIGINS: list = ["*"]

    model_config = {
        "case_sensitive": True,
        "env_file": ".env",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
