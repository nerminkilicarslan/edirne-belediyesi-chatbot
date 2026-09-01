"""
Embedding Service — BGE-M3 ile çok dilli embedding üretimi.

BGE-M3'ün seçilme nedenleri:
  - Türkçe semantiği çok güçlü (multilingual, 100+ dil)
  - 8192 token bağlam uzunluğu (uzun belgeler için ideal)
  - Normalize edilmiş çıktı → cosine similarity = dot product (hız optimizasyonu)
  - Apple Silicon MPS desteği ile hızlı çıkarım

Device otomatik tespiti: MPS → CUDA → CPU
"""
from __future__ import annotations

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.config import settings


def _detect_device() -> str:
    if settings.EMBEDDING_DEVICE != "auto":
        return settings.EMBEDDING_DEVICE
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS tespit edildi — embedding hızlandırması aktif")
        return "mps"
    if torch.cuda.is_available():
        logger.info("CUDA GPU tespit edildi")
        return "cuda"
    logger.info("GPU bulunamadı — CPU kullanılıyor")
    return "cpu"


class EmbeddingService:
    """
    Tek örnek (singleton) embedding servisi.
    Classifiers ve VectorDB bu servisi paylaşır → tek model yüklemesi.
    """

    def __init__(self) -> None:
        device = _detect_device()
        logger.info("Embedding modeli yükleniyor: {} | cihaz: {}", settings.EMBEDDING_MODEL_NAME, device)
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            device=device,
        )
        self._dim = self.model.get_embedding_dimension()
        logger.info("Embedding modeli hazır. Boyut: {}", self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def get_embedding(self, text: str) -> list[float]:
        vec = self.model.encode(
            text,
            normalize_embeddings=settings.EMBEDDING_NORMALIZE,
            show_progress_bar=False,
        )
        return vec.tolist()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Toplu encode — sınıflandırıcı örnek vektörü hesaplamasında kullanılır."""
        vecs = self.model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=settings.EMBEDDING_NORMALIZE,
            show_progress_bar=False,
        )
        return vecs.tolist()


embedding_service = EmbeddingService()
