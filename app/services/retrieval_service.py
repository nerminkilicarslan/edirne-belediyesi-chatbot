"""
Hybrid Retrieval Service — semantic + BM25 kombinasyonu.

Hibrit yaklaşımın avantajı:
  Semantic alone zayıf: kısa / nadir terimler, telefon numaraları
  BM25 alone zayıf: anlam dönüşümü, farklı ifadeler
  Hibrit: her iki durumda da tutarlı performans

Birleştirme formülü (Reciprocal Rank Fusion benzeri ama daha basit):
  combined_score = semantic_weight × semantic_score + bm25_weight × bm25_score
  Sonra yeniden sıralama → top-N alınır
"""
from __future__ import annotations

from loguru import logger

from app.api.models import DocumentSource
from app.config import settings
from app.services.bm25_service import bm25_service
from app.services.vector_db import vector_db


class RetrievalService:
    """
    Tek arayüz ile semantic + BM25 hibrit retrieval.
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        category_filter: str | None = None,
        use_bm25: bool = True,
    ) -> list[DocumentSource]:
        """
        Hibrit geri çağırma → birleştirilmiş ve yeniden sıralanmış sonuçlar.
        """
        # ── Semantic search ──────────────────────────────────────────────────
        semantic_results = vector_db.search(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            category_filter=category_filter,
        )
        logger.debug("Semantic: {} sonuç", len(semantic_results))

        # ── BM25 search ──────────────────────────────────────────────────────
        bm25_results: list[dict] = []
        if use_bm25 and bm25_service.is_ready:
            bm25_results = bm25_service.search(query, top_k=top_k)
            logger.debug("BM25: {} sonuç", len(bm25_results))

        # ── Birleştir ────────────────────────────────────────────────────────
        combined = self._fuse(semantic_results, bm25_results)

        # ── Eşik filtresi ve top-N ────────────────────────────────────────────
        filtered = [d for d in combined if d.score >= score_threshold]
        top_n = filtered[: settings.RERANK_TOP_N]

        logger.info(
            "Retrieval: semantic={} bm25={} fused={} filtered={}",
            len(semantic_results),
            len(bm25_results),
            len(combined),
            len(top_n),
        )
        return top_n

    def _fuse(
        self,
        semantic_hits: list[dict],
        bm25_hits: list[dict],
    ) -> list[DocumentSource]:
        """
        ID bazlı birleştirme: aynı döküman her iki listede varsa
        ağırlıklı toplam alınır, sadece birinde varsa tek ağırlıkla eklenir.
        """
        sem_w = settings.SEMANTIC_WEIGHT
        bm25_w = settings.BM25_WEIGHT

        scores: dict[str, float] = {}
        meta_map: dict[str, dict] = {}
        content_map: dict[str, str] = {}
        method_map: dict[str, str] = {}

        for hit in semantic_hits:
            doc_id = hit["id"]
            scores[doc_id] = hit["score"] * sem_w
            meta_map[doc_id] = hit["metadata"]
            content_map[doc_id] = hit["content"]
            method_map[doc_id] = "semantic"

        for hit in bm25_hits:
            doc_id = hit["id"]
            bm25_contrib = hit["score"] * bm25_w
            if doc_id in scores:
                scores[doc_id] += bm25_contrib
                method_map[doc_id] = "hybrid"
            else:
                scores[doc_id] = bm25_contrib
                meta_map[doc_id] = hit["metadata"]
                content_map[doc_id] = hit["content"]
                method_map[doc_id] = "bm25"

        sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)
        return [
            DocumentSource(
                content=content_map[doc_id],
                metadata=meta_map[doc_id],
                score=round(scores[doc_id], 4),
                retrieval_method=method_map[doc_id],
            )
            for doc_id in sorted_ids
        ]


retrieval_service = RetrievalService()
