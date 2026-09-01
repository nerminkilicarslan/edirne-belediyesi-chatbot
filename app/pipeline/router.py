"""
Hybrid Routing System

Classifier çıktısı + confidence skoru alır, pipeline için
RouteDecision üretir. Karar ağacı:

  1. Acil durum → EMERGENCY (her zaman, confidence ne olursa olsun)
  2. Kapsam dışı → OUT_OF_SCOPE
  3. Doğrudan cevap (selamlama/veda/teşekkür) → DIRECT_ANSWER
  4. VERY_LOW confidence → FALLBACK
  5. LOW confidence → CLARIFICATION
  6. HIGH confidence → RETRIEVAL_FOCUSED (dar retrieval)
  7. MEDIUM/diğer → RETRIEVAL_BROAD (geniş retrieval)
"""
from __future__ import annotations

from app.api.models import RouteDecision
from app.pipeline.intent_classifier import IntentResult
from app.pipeline.category_classifier import CategoryResult
from app.pipeline.confidence import OverallConfidence
from app.pipeline.intent_classifier import DIRECT_INTENTS
from loguru import logger


# Doğrudan sabit cevap verilen niyetler
CANNED_RESPONSES: dict[str, str] = {
    "SELAMLASMA": (
        "Merhaba! Edirne Belediyesi Asistanına hoş geldiniz. "
        "Size nasıl yardımcı olabilirim?"
    ),
    "VEDALASMA": (
        "İyi günler dileriz! Edirne Belediyesi olarak her zaman yanınızdayız."
    ),
    "TESEKKUR": (
        "Rica ederim! Size yardımcı olabildiysem ne mutlu. "
        "Başka bir sorunuz varsa buradayım."
    ),
    "ACIL_DURUM": (
        "⚠️ ACİL DURUM İÇİN:\n"
        "• İtfaiye: 110\n"
        "• Ambulans: 112\n"
        "• Polis: 155\n"
        "• Belediye Acil: 153 (Beyaz Masa)\n\n"
        "Lütfen hemen ilgili numarayı arayın!"
    ),
    "KAPSAM_DISI": (
        "Bu konu belediye hizmetlerimizin kapsamı dışında kalıyor. "
        "Belediye hizmetleri, başvurular, şikayetler veya duyurular hakkında "
        "yardımcı olmaktan memnuniyet duyarım."
    ),
}

CLARIFICATION_MESSAGE = (
    "Sorunuzu tam olarak anlayabilmem için biraz daha detay verir misiniz? "
    "Örneğin; hangi hizmet veya konu hakkında bilgi almak istediğinizi belirtirseniz "
    "daha iyi yardımcı olabilirim."
)


class HybridRouter:
    """
    Pipeline routing kararı vericisi.

    Tasarım felsefesi:
      - Kural tabanlı (1-3. adımlar): hızlı, kesin, model bağımsız
      - Confidence tabanlı (4-7. adımlar): adaptif, bağlama göre değişir
    """

    def decide(
        self,
        intent: IntentResult,
        category: CategoryResult,
        confidence: OverallConfidence,
    ) -> tuple[RouteDecision, str]:
        """
        Returns:
            (RouteDecision, canned_response_or_empty_string)
            canned_response dolu ise LLM ve retrieval atlanır.
        """
        label = intent.label

        # ── Kural 1: Acil durum her şeyin önüne geçer ────────────────────────
        if label == "ACIL_DURUM":
            logger.warning("ACİL DURUM intent tespit edildi!")
            return RouteDecision.EMERGENCY, CANNED_RESPONSES["ACIL_DURUM"]

        # ── Kural 2: Kapsam dışı ─────────────────────────────────────────────
        if label == "KAPSAM_DISI":
            return RouteDecision.OUT_OF_SCOPE, CANNED_RESPONSES["KAPSAM_DISI"]

        # ── Kural 3: Doğrudan cevap verilecek niyetler ───────────────────────
        if label in DIRECT_INTENTS:
            canned = CANNED_RESPONSES.get(label, "")
            return RouteDecision.DIRECT_ANSWER, canned

        # ── Kural 4 & 5: Düşük/çok düşük güven → geniş retrieval dene ─────────
        # Belediye chatbot'u için her zaman retrieval önce denenmelidir.
        # Gerçek FALLBACK yalnızca retrieval sonucu boş döndüğünde tetiklenir.
        if confidence.tier in {"VERY_LOW", "LOW"}:
            logger.info("{} confidence → RETRIEVAL_BROAD (liberal)", confidence.tier)
            return RouteDecision.RETRIEVAL_BROAD, ""

        # ── Kural 6: Yüksek güven → odaklı retrieval ────────────────────────
        if confidence.tier == "HIGH":
            return RouteDecision.RETRIEVAL_FOCUSED, ""

        # ── Kural 7: Orta güven → geniş retrieval ───────────────────────────
        return RouteDecision.RETRIEVAL_BROAD, ""

    def get_retrieval_config(
        self,
        route: RouteDecision,
        category: CategoryResult,
    ) -> dict:
        """
        Retrieval ayarlarını route ve kategoriye göre döndürür.
        FOCUSED: daha az döküman, kategori filtreli
        BROAD:   daha fazla döküman, kategori filtresi yok
        """
        base_top_k = 5
        if route == RouteDecision.RETRIEVAL_FOCUSED:
            return {
                "top_k": base_top_k,
                "score_threshold": 0.28,
                "category_filter": category.label if category.confidence >= 0.72 else None,
                "use_bm25": True,
            }
        return {
            "top_k": base_top_k + 2,
            "score_threshold": 0.18,
            "category_filter": None,
            "use_bm25": True,
        }


hybrid_router = HybridRouter()
