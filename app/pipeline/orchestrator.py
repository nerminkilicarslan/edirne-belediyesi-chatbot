"""
Decision Pipeline Orchestrator

Gelen her sorgu için tam karar akışını yönetir:

  Query
    │
    ▼ 1. Preprocessing (normalize, truncate)
    │
    ▼ 2. Intent Classification (keyword → embedding hybrid)
    │
    ▼ 3. Category Classification (keyword → embedding + intent boost)
    │
    ▼ 4. Pre-retrieval Confidence Scoring
    │
    ▼ 5. Route Decision
    │     ├─ DIRECT_ANSWER  → canned response döndür (LLM/retrieval atla)
    │     ├─ EMERGENCY      → acil numaralar döndür
    │     ├─ OUT_OF_SCOPE   → polite redirect
    │     ├─ CLARIFICATION  → açıklama isteği
    │     ├─ FALLBACK       → çok düşük güven mesajı
    │     ├─ RETRIEVAL_FOCUSED → dar retrieval + LLM
    │     └─ RETRIEVAL_BROAD  → geniş retrieval + LLM
    │
    ▼ 6. Hybrid Retrieval (semantic + BM25)
    │
    ▼ 7. Post-retrieval Confidence Update
    │
    ▼ 8. LLM Generation (async Ollama)
    │
    ▼ 9. Response Assembly
"""
from __future__ import annotations

import time
import uuid

from loguru import logger

from app.api.models import (
    ChatResponse,
    DocumentSource,
    RouteDecision,
)
from app.config import settings
from app.pipeline.category_classifier import category_classifier
from app.pipeline.confidence import confidence_scorer
from app.pipeline.intent_classifier import intent_classifier
from app.pipeline.prompts import build_rag_user_message, select_system_prompt
from app.pipeline.router import hybrid_router
from app.services.llm_service import llm_service
from app.services.retrieval_service import retrieval_service


def _normalize_query(text: str) -> str:
    """Girişi temizle: çoklu boşluk, kenar boşlukları, uzunluk sınırı."""
    text = " ".join(text.split())
    return text[: settings.MAX_QUERY_LENGTH]


class DecisionPipeline:
    """
    Tüm pipeline adımlarını koordine eden ana sınıf.
    Tüm servisler startup'ta başlatılır; bu sınıf sadece koordinatördür.
    """

    # ── Startup ────────────────────────────────────────────────────────────────

    def startup(self) -> None:
        """
        FastAPI lifespan event'inde çağrılır.
        Classifier'ları embedding servisiyle başlatır ve BM25 indeksini kurar.
        """
        from app.services.embedding_service import embedding_service
        from app.services.bm25_service import bm25_service
        from app.services.vector_db import vector_db

        # Classifier'ları başlat (exemplar vektörleri hesapla)
        intent_classifier.initialize(embedding_service)
        category_classifier.initialize(embedding_service)

        # BM25 indeksini ChromaDB'den yükle
        ids, texts, metas = vector_db.get_all_for_bm25()
        if texts:
            bm25_service.build(ids, texts, metas)
            logger.info("BM25 indeksi {} dökümanla kuruldu.", len(texts))
        else:
            logger.warning("ChromaDB boş — BM25 indeksi oluşturulamadı. Lütfen veri yükleyin.")

        logger.info("Decision pipeline hazır.")

    # ── Ana işlem ─────────────────────────────────────────────────────────────

    async def process(self, query: str, session_id: str | None = None) -> ChatResponse:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        logger.info("[{}] Sorgu: '{}'", request_id, query[:80])

        # ── 1. Preprocessing ────────────────────────────────────────────────
        clean_query = _normalize_query(query)

        # ── 2. Intent Classification ─────────────────────────────────────────
        intent = intent_classifier.classify(clean_query)
        logger.info(
            "[{}] Intent: {} ({:.2f}) [{}]",
            request_id,
            intent.label,
            intent.confidence,
            intent.method,
        )

        # ── 3. Category Classification ───────────────────────────────────────
        category = category_classifier.classify(clean_query, intent_label=intent.label)
        logger.info(
            "[{}] Kategori: {} ({:.2f}) [{}]",
            request_id,
            category.label,
            category.confidence,
            category.method,
        )

        # ── 4. Pre-retrieval Confidence ──────────────────────────────────────
        confidence = confidence_scorer.score_pre_retrieval(
            intent.confidence, category.confidence
        )

        # ── 5. Route Decision ────────────────────────────────────────────────
        route, canned = hybrid_router.decide(intent, category, confidence)
        logger.info("[{}] Route: {}", request_id, route.value)

        # Canned response: retrieval ve LLM atlanır
        if canned:
            elapsed = (time.perf_counter() - start) * 1000
            return ChatResponse(
                answer=canned,
                intent=intent.label,
                category=category.label,
                confidence=round(confidence.overall, 3),
                route=route.value,
                sources=[],
                processing_time_ms=round(elapsed, 2),
                is_fallback=(route in {RouteDecision.FALLBACK, RouteDecision.OUT_OF_SCOPE}),
                request_id=request_id,
            )

        # ── 6. Hybrid Retrieval ──────────────────────────────────────────────
        retrieval_cfg = hybrid_router.get_retrieval_config(route, category)
        docs: list[DocumentSource] = retrieval_service.retrieve(
            query=clean_query,
            top_k=retrieval_cfg["top_k"],
            score_threshold=retrieval_cfg["score_threshold"],
            category_filter=retrieval_cfg.get("category_filter"),
            use_bm25=retrieval_cfg.get("use_bm25", True),
        )
        logger.info("[{}] Retrieved {} döküman", request_id, len(docs))

        # ── 7. Post-retrieval Confidence ─────────────────────────────────────
        retrieval_scores = [d.score for d in docs]
        confidence = confidence_scorer.score_post_retrieval(
            intent.confidence, category.confidence, retrieval_scores
        )

        # Döküman bulunamazsa fallback
        if not docs:
            logger.warning("[{}] Döküman bulunamadı → fallback", request_id)
            elapsed = (time.perf_counter() - start) * 1000
            return ChatResponse(
                answer=settings.FALLBACK_MESSAGE,
                intent=intent.label,
                category=category.label,
                confidence=round(confidence.overall, 3),
                route=RouteDecision.FALLBACK.value,
                sources=[],
                processing_time_ms=round(elapsed, 2),
                is_fallback=True,
                request_id=request_id,
            )

        # ── 8. LLM Generation ────────────────────────────────────────────────
        system_prompt = select_system_prompt(intent.label)
        user_message = build_rag_user_message(
            query=clean_query,
            docs=docs,
            intent_label=intent.label,
            category_label=category.label,
        )

        answer = await llm_service.chat(
            user_message=user_message,
            system_prompt=system_prompt,
        )

        # ── 9. Response Assembly ─────────────────────────────────────────────
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "[{}] Tamamlandı: {}ms | conf={:.2f} | route={}",
            request_id,
            round(elapsed),
            confidence.overall,
            route.value,
        )

        return ChatResponse(
            answer=answer,
            intent=intent.label,
            category=category.label,
            confidence=round(confidence.overall, 3),
            route=route.value,
            sources=docs,
            processing_time_ms=round(elapsed, 2),
            is_fallback=False,
            request_id=request_id,
        )


# Uygulama geneli singleton
decision_pipeline = DecisionPipeline()
