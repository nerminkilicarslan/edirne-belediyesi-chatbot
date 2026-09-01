"""
FastAPI Application Entry Point

Lifespan yönetimi:
  startup: classifier'ları başlat, BM25 indeksini kur, Ollama bağlantısını kontrol et
  shutdown: temizlik (gerekirse)

CORS:
  Production'da CORS_ORIGINS'i .env'den spesifik origin olarak ayarla.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.config import settings
from app.core.logger import setup_logging

# Logging en başta kurulur
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan: startup ve shutdown mantığı."""
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("{} v{} başlatılıyor…", settings.PROJECT_NAME, settings.VERSION)
    logger.info("Ortam: {} | Debug: {}", settings.ENVIRONMENT, settings.DEBUG)

    # Pipeline başlatma: embedding yükleme, classifier exemplar hesaplama, BM25 kurma
    from app.pipeline.orchestrator import decision_pipeline
    decision_pipeline.startup()

    # Ollama erişilebilirlik kontrolü
    from app.services.llm_service import llm_service
    if await llm_service.is_available():
        logger.info("Ollama bağlantısı OK | model: {}", settings.OLLAMA_MODEL)
    else:
        logger.warning(
            "Ollama şu an erişilemiyor ({})! Chat istekleri fallback döndürecek.",
            settings.OLLAMA_BASE_URL,
        )

    logger.info("{} hazır. Bekleniyor…", settings.PROJECT_NAME)
    logger.info("=" * 60)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("{} kapatılıyor.", settings.PROJECT_NAME)


# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Edirne Belediyesi Local RAG Chatbot — "
        "self-hosted, Ollama + BGE-M3 + ChromaDB"
    ),
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Router
app.include_router(router, prefix=settings.API_PREFIX)


# ── Development entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
