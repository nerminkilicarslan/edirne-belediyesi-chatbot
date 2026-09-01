"""
BM25 Service — keyword tabanlı geri çağırma.

Neden BM25 de var?
  Semantic search tek başına zayıf kaldığı durumlar:
  - Kullanıcı tam kod/numara yazar: "153", "0284 225 11 22"
  - Nadir / teknik terimler: "imar aplikasyon krokisi"
  - Kısa sorular: "fatura" → semantic çok geniş eşleşir

  BM25 + semantic = hibrit → her iki sorgu türünde iyi performans

Dizin belleğe alınır (municipal corpus genellikle <50K chunk):
  - 10K chunk × ortalama 100 token ≈ ~5 MB
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from loguru import logger
from rank_bm25 import BM25Okapi


def _turkish_tokenize(text: str) -> list[str]:
    """
    Basit Türkçe tokenizasyon.
    spaCy/Zemberek gibi ağır bağımlılıklar yerine kural tabanlı kullanılır.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    # Noktalama kaldır, Türkçe karakterleri koru
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = text.split()
    # Çok kısa token'ları çıkar
    return [t for t in tokens if len(t) > 1]


class BM25Service:
    """
    ChromaDB'deki dökümanları belleğe alarak BM25 indeksi kurar.
    Yeni döküman eklendiğinde rebuild() çağrılmalıdır.
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: list[str] = []
        self._doc_texts: list[str] = []
        self._doc_metadatas: list[dict] = []
        self._is_built = False

    def build(
        self,
        doc_ids: list[str],
        doc_texts: list[str],
        doc_metadatas: list[dict],
    ) -> None:
        """ChromaDB verisiyle BM25 indeksini kur."""
        if not doc_texts:
            logger.warning("BM25 build: döküman bulunamadı, indeks boş.")
            return

        tokenized = [_turkish_tokenize(t) for t in doc_texts]
        self._bm25 = BM25Okapi(tokenized)
        self._doc_ids = doc_ids
        self._doc_texts = doc_texts
        self._doc_metadatas = doc_metadatas
        self._is_built = True
        logger.info("BM25 indeksi oluşturuldu: {} döküman", len(doc_texts))

    def rebuild(
        self,
        doc_ids: list[str],
        doc_texts: list[str],
        doc_metadatas: list[dict],
    ) -> None:
        """Yeni döküman eklendikten sonra indeksi yenile."""
        self.build(doc_ids, doc_texts, doc_metadatas)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        BM25 skoru ile üst-k dökümanı döndür.
        Normalize edilmiş skor [0, 1] aralığına dönüştürülür.
        """
        if not self._is_built or self._bm25 is None:
            return []

        tokens = _turkish_tokenize(query)
        if not tokens:
            return []

        raw_scores = self._bm25.get_scores(tokens)
        max_score = float(raw_scores.max()) if raw_scores.max() > 0 else 1.0

        scored = [
            (i, raw_scores[i] / max_score)
            for i in range(len(raw_scores))
            if raw_scores[i] > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, norm_score in scored[:top_k]:
            results.append({
                "id": self._doc_ids[idx],
                "content": self._doc_texts[idx],
                "metadata": self._doc_metadatas[idx],
                "score": round(norm_score, 4),
                "retrieval_method": "bm25",
            })
        return results

    @property
    def is_ready(self) -> bool:
        return self._is_built

    @property
    def doc_count(self) -> int:
        return len(self._doc_texts)


bm25_service = BM25Service()
