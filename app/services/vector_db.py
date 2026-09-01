"""
Vector DB Service — ChromaDB 1.x ile semantic search.

ChromaDB 1.x API notları:
  - PersistentClient: kalıcı disk depolaması
  - collection.query(): embedding ile arama
  - collection.upsert(): güvenli ekleme/güncelleme
  - metadata filtresi: where={"category": "..."}
  - cosine distance → benzerlik skoru: score = 1.0 - distance
"""
from __future__ import annotations

import uuid
from typing import Optional

import chromadb
from loguru import logger

from app.config import settings
from app.services.embedding_service import embedding_service


class VectorDBService:
    def __init__(self) -> None:
        logger.info("ChromaDB başlatılıyor → {}", settings.CHROMA_PERSIST_DIRECTORY)
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB hazır | koleksiyon: {} | {} döküman",
            settings.CHROMA_COLLECTION_NAME,
            self.collection.count(),
        )

    # ── Arama ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        category_filter: Optional[str] = None,
    ) -> list[dict]:
        """Vektörel semantic arama."""
        query_embedding = embedding_service.get_embedding(query)

        where: Optional[dict] = None
        if category_filter and category_filter != "GENEL":
            where = {"category": category_filter}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(self.collection.count(), 1)),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB arama hatası: {}", exc)
            return []

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        formatted = []
        for doc, meta, dist in zip(docs, metas, distances):
            score = round(1.0 - dist, 4)
            if score < score_threshold:
                continue
            formatted.append({
                "id": meta.get("chunk_id", str(uuid.uuid4())),
                "content": doc,
                "metadata": meta or {},
                "score": score,
                "retrieval_method": "semantic",
            })
        return formatted

    def get_all_for_bm25(self) -> tuple[list[str], list[str], list[dict]]:
        """BM25 indeksi için tüm dökümanları döndürür."""
        total = self.collection.count()
        if total == 0:
            return [], [], []

        result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metas = result.get("metadatas", []) or [{}] * len(ids)
        return ids, docs, metas

    # ── Ekleme / Güncelleme ───────────────────────────────────────────────────

    def upsert(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: Optional[list[str]] = None,
    ) -> int:
        """Döküman ekle / var olanı güncelle."""
        if not texts:
            return 0

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        embeddings = embedding_service.get_embeddings(texts)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("{} döküman ChromaDB'ye eklendi/güncellendi", len(texts))
        return len(texts)

    def delete_by_source(self, source: str) -> int:
        """Belirli kaynaktan gelen dökümanları sil (yeniden indeksleme için)."""
        try:
            results = self.collection.get(where={"source": source}, include=[])
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info("Silindi: {} döküman (kaynak: {})", len(ids_to_delete), source)
            return len(ids_to_delete)
        except Exception as exc:
            logger.error("Silme hatası: {}", exc)
            return 0

    @property
    def doc_count(self) -> int:
        return self.collection.count()


vector_db = VectorDBService()
