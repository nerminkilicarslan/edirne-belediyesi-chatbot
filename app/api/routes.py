"""
API Routes — chat, health ve admin endpoint'leri.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from loguru import logger

from app.api.models import (
    ChatRequest,
    ChatResponse,
    HealthDetail,
    IngestRequest,
    IngestResponse,
)
from app.config import settings
from app.pipeline.orchestrator import decision_pipeline

router = APIRouter()
_start_time = time.time()

# ── Opsiyonel API key güvenliği ────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _check_api_key(key: str | None = Security(api_key_header)) -> None:
    """API_KEY tanımlıysa header'da gelmesini zorunlu kıl."""
    if settings.API_KEY and key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Geçersiz veya eksik API anahtarı.",
        )


# ── Chat ───────────────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Belediye chatbot'una soru sor",
    dependencies=[Depends(_check_api_key)],
)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    logger.info("Chat isteği | session={}", request.session_id)
    try:
        return await decision_pipeline.process(
            query=request.query,
            session_id=request.session_id,
        )
    except Exception as exc:
        logger.exception("Chat endpoint hatası: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sunucu hatası. Lütfen daha sonra tekrar deneyin.",
        )


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthDetail, summary="Sistem sağlık durumu")
async def health_check() -> HealthDetail:
    from app.services.vector_db import vector_db
    from app.services.bm25_service import bm25_service

    service_status: dict[str, str] = {}

    # Ollama
    from app.services.llm_service import llm_service
    service_status["ollama"] = "ok" if await llm_service.is_available() else "unreachable"

    # ChromaDB
    try:
        doc_count = vector_db.doc_count
        service_status["chromadb"] = f"ok ({doc_count} döküman)"
    except Exception:
        service_status["chromadb"] = "error"

    # BM25
    service_status["bm25"] = f"ok ({bm25_service.doc_count} döküman)" if bm25_service.is_ready else "not_built"

    # Classifiers
    from app.pipeline.intent_classifier import intent_classifier
    from app.pipeline.category_classifier import category_classifier
    service_status["intent_classifier"] = "ok" if intent_classifier._initialized else "not_ready"
    service_status["category_classifier"] = "ok" if category_classifier._initialized else "not_ready"

    overall = "ok" if all("ok" in v for v in service_status.values()) else "degraded"
    return HealthDetail(status=overall, version=settings.VERSION, services=service_status)


# ── Admin ──────────────────────────────────────────────────────────────────────

@router.post(
    "/admin/ingest",
    response_model=IngestResponse,
    summary="Yeni doküman indeksle",
    dependencies=[Depends(_check_api_key)],
)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    """
    Metin veya dosya yoluyla yeni döküman ekle ve ChromaDB + BM25 indekslerini güncelle.
    """
    from app.ingestion.indexer import DocumentIndexer
    indexer = DocumentIndexer()

    try:
        if request.source_type == "text" and request.content:
            result = indexer.index_text(
                text=request.content,
                metadata=request.metadata,
            )
        elif request.source_type == "file" and request.source_path:
            result = indexer.index_file(
                file_path=request.source_path,
                metadata=request.metadata,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="source_type='text' için content, 'file' için source_path gereklidir.",
            )

        # BM25 indeksini yenile
        from app.services.vector_db import vector_db
        from app.services.bm25_service import bm25_service
        ids, texts, metas = vector_db.get_all_for_bm25()
        bm25_service.rebuild(ids, texts, metas)

        return IngestResponse(
            status="success",
            documents_indexed=result.get("documents_indexed", 0),
            chunks_created=result.get("chunks_created", 0),
            message="İndeksleme tamamlandı.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("İndeksleme hatası: {}", exc)
        raise HTTPException(status_code=500, detail=f"İndeksleme hatası: {exc}")


@router.delete(
    "/admin/collection",
    summary="Koleksiyonu sıfırla",
    dependencies=[Depends(_check_api_key)],
)
async def reset_collection() -> dict[str, Any]:
    """Tüm koleksiyonu sil ve yeniden oluştur. Dikkatli kullanın!"""
    from app.services.vector_db import vector_db
    from app.services.bm25_service import bm25_service
    try:
        vector_db.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        vector_db.collection = vector_db.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        bm25_service._is_built = False
        bm25_service._doc_ids = []
        bm25_service._doc_texts = []
        logger.warning("Koleksiyon sıfırlandı!")
        return {"status": "ok", "message": "Koleksiyon sıfırlandı."}
    except Exception as exc:
        logger.error("Sıfırlama hatası: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/admin/stats", summary="İndeks istatistikleri", dependencies=[Depends(_check_api_key)])
async def index_stats() -> dict[str, Any]:
    from app.services.vector_db import vector_db
    from app.services.bm25_service import bm25_service
    uptime = round(time.time() - _start_time, 1)
    return {
        "chromadb_docs": vector_db.doc_count,
        "bm25_docs": bm25_service.doc_count,
        "bm25_ready": bm25_service.is_ready,
        "uptime_seconds": uptime,
        "version": settings.VERSION,
    }
