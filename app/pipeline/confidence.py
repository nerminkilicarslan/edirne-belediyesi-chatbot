"""
Confidence Scoring System

Pipeline'daki tüm sinyalleri tek bir nihai güven skoruna dönüştürür:
  - intent_confidence  : intentin ne kadar emin sınıflandırıldığı
  - category_confidence: kategorinin ne kadar emin sınıflandırıldığı
  - retrieval_score    : en iyi dökümanın benzerlik skoru (0 = döküman yok)

Çıktı: OverallConfidence nesnesi + route kararı için eşik karşılaştırması.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings


@dataclass
class OverallConfidence:
    overall: float           # 0.0 – 1.0
    intent_confidence: float
    category_confidence: float
    retrieval_confidence: float
    tier: str                # "HIGH" | "MEDIUM" | "LOW" | "VERY_LOW"
    explanation: str


class ConfidenceScorer:
    """
    Çok boyutlu güven hesaplayıcı.

    Ağırlıklandırma mantığı:
      - Retrieval aşamasından önce (routing kararı): intent + category kullan
      - Retrieval sonrası (cevap kalitesi): tüm sinyalleri kullan

    Tier belirleme:
      HIGH     (≥0.65): Odaklı retrieval, güvenli cevap üretimi
      MEDIUM   (0.42–0.65): Geniş retrieval, LLM'e bağlam daha geniş verilir
      LOW      (0.28–0.42): Clarification isteği veya çok dikkatli cevap
      VERY_LOW (<0.28): Fallback veya kullanıcıdan detay iste
    """

    HIGH_THRESHOLD = 0.65
    MEDIUM_THRESHOLD = 0.42
    LOW_THRESHOLD = 0.28

    # Ağırlıklar: retrieval skoru en önemli sinyal (RAG doğruluğu yansıtır)
    WEIGHTS_NO_RETRIEVAL = {"intent": 0.55, "category": 0.45}
    WEIGHTS_WITH_RETRIEVAL = {"intent": 0.30, "category": 0.25, "retrieval": 0.45}

    def score_pre_retrieval(
        self,
        intent_confidence: float,
        category_confidence: float,
    ) -> OverallConfidence:
        """Retrieval öncesi routing kararı için skor."""
        w = self.WEIGHTS_NO_RETRIEVAL
        overall = (
            intent_confidence * w["intent"]
            + category_confidence * w["category"]
        )
        return self._build(overall, intent_confidence, category_confidence, 0.0)

    def score_post_retrieval(
        self,
        intent_confidence: float,
        category_confidence: float,
        retrieval_scores: list[float],
    ) -> OverallConfidence:
        """Retrieval sonrası cevap kalite skoru."""
        retrieval_conf = max(retrieval_scores) if retrieval_scores else 0.0
        w = self.WEIGHTS_WITH_RETRIEVAL
        overall = (
            intent_confidence * w["intent"]
            + category_confidence * w["category"]
            + retrieval_conf * w["retrieval"]
        )
        return self._build(overall, intent_confidence, category_confidence, retrieval_conf)

    def _build(
        self,
        overall: float,
        intent_conf: float,
        cat_conf: float,
        retrieval_conf: float,
    ) -> OverallConfidence:
        overall = max(0.0, min(overall, 1.0))
        tier, explanation = self._determine_tier(overall, retrieval_conf)
        return OverallConfidence(
            overall=round(overall, 4),
            intent_confidence=round(intent_conf, 4),
            category_confidence=round(cat_conf, 4),
            retrieval_confidence=round(retrieval_conf, 4),
            tier=tier,
            explanation=explanation,
        )

    def _determine_tier(self, overall: float, retrieval_conf: float) -> tuple[str, str]:
        if overall >= self.HIGH_THRESHOLD:
            return "HIGH", f"Yüksek güven ({overall:.2f}) — odaklı retrieval önerilir"
        if overall >= self.MEDIUM_THRESHOLD:
            return "MEDIUM", f"Orta güven ({overall:.2f}) — geniş retrieval önerilir"
        if overall >= self.LOW_THRESHOLD:
            # Retrieval skoru yüksekse bile overall düşükse medium'a çekebilir
            if retrieval_conf >= 0.55:
                return "MEDIUM", f"Düşük sınıflandırma ama iyi döküman ({retrieval_conf:.2f})"
            return "LOW", f"Düşük güven ({overall:.2f}) — açıklama isteği düşünülmeli"
        return "VERY_LOW", f"Çok düşük güven ({overall:.2f}) — fallback tetiklenmeli"


confidence_scorer = ConfidenceScorer()
