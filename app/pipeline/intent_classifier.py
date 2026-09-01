"""
Intent Classifier — iki katmanlı hibrit mimari:

  Katman 1 – Keyword/Regex (µs düzeyinde):
    Selamlama, veda, teşekkür, acil durum gibi net sinyaller
    regex + anahtar kelime ile %85+ güvenle yakalanır.

  Katman 2 – Embedding benzerliği (10–50 ms):
    Her intent için önceden hazırlanmış örnek cümle vektörleri
    ile cosine similarity karşılaştırması yapılır.
    Aynı BGE-M3 modeli kullanıldığından ek model yüklenmez.

  Neden LLM kullanılmıyor:
    - Her istek için ek 2-5 sn gecikme ekler
    - Tutarsız JSON çıktısı verme riski taşır
    - Embeddings zaten yüklenmiş olduğundan ücretsizdir
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger


# ── Intent tanımları: keyword, regex ve örnek cümleler ───────────────────────

INTENT_DEFINITIONS: dict[str, dict] = {
    "SELAMLASMA": {
        "keywords": ["merhaba", "selam", "günaydın", "iyi günler", "iyi akşamlar", "hey", "merhabalar", "nasılsın", "nasılsınız", "naber", "iyi misin"],
        "patterns": [r"^(merhaba|selam|günaydın|iyi\s+günler|iyi\s+akşamlar|hey|nasılsın|nasılsınız|naber)\b"],
        "exemplars": [
            "Merhaba!",
            "Selam, yardım alabilir miyim?",
            "İyi günler.",
            "Günaydın, sorum var.",
            "Nasılsın?",
            "Naber?",
        ],
        "keyword_confidence": 0.90,
    },
    "VEDALASMA": {
        "keywords": ["görüşürüz", "hoşçakal", "güle güle", "bye", "iyi günler dilerim"],
        "patterns": [r"^(görüşürüz|hoşçakal|güle\s+güle)\b"],
        "exemplars": [
            "Görüşürüz!",
            "Hoşçakal, iyi günler.",
            "Güle güle!",
        ],
        "keyword_confidence": 0.90,
    },
    "TESEKKUR": {
        "keywords": ["teşekkürler", "teşekkür ederim", "sağol", "sağolun", "eyvallah", "çok teşekkür"],
        "patterns": [r"(teşekkür|sağol|eyvallah)"],
        "exemplars": [
            "Teşekkür ederim, çok yardımcı oldunuz.",
            "Sağolun.",
            "Çok teşekkürler!",
        ],
        "keyword_confidence": 0.88,
    },
    "ACIL_DURUM": {
        "keywords": ["acil", "yangın", "su baskını", "sel", "deprem", "yardım edin", "tehlike", "kaza", "itfaiye"],
        "patterns": [r"(acil|yangın|itfaiye|sel bask|deprem|yardım\s+edin)"],
        "exemplars": [
            "Acil yardım lazım, su borusu patladı.",
            "Yangın var, hemen yardım edin!",
            "Elektrik direği devrildi, tehlikeli.",
        ],
        "keyword_confidence": 0.93,
    },
    "BILGI_TALEBI": {
        "keywords": ["nedir", "nasıl", "ne zaman", "hangi", "kaç", "bilgi", "hakkında", "açıkla", "anlat", "ne gerekiyor", "öğrenmek"],
        "exemplars": [
            "Çevre temizlik vergisi nedir?",
            "İmar durumu nasıl sorgulanır?",
            "Nikah işlemleri nasıl yapılır?",
            "Belediye hangi hizmetleri sunuyor?",
            "Su aboneliği için ne gerekiyor?",
            "İnşaat ruhsatı hakkında bilgi almak istiyorum.",
        ],
        "keyword_confidence": 0.72,
    },
    "BASVURU_ISLEM": {
        "keywords": ["başvuru", "başvurmak", "müracaat", "nasıl başvurabilirim", "başvuru formu",
                     "evrak", "belge", "izin", "ruhsat", "kayıt", "randevu", "talep etmek", "başvuruyorum"],
        "exemplars": [
            "İnşaat ruhsatı için nasıl başvurabilirim?",
            "Sosyal yardım başvurusu yapmak istiyorum.",
            "Nikah randevusu almak istiyorum.",
            "Su aboneliği açmak için başvurmak istiyorum.",
            "Esnaf ruhsatı için gerekli evraklar neler?",
        ],
        "keyword_confidence": 0.80,
    },
    "SIKAYET_BILDIRIM": {
        "keywords": ["şikayet", "şikâyet", "bildirmek istiyorum", "sorun var", "problem",
                     "çalışmıyor", "bozuk", "hasarlı", "tehlikeli", "pis", "gürültü", "rahatsız",
                     "şikayetim var", "bildirmek", "ihbar"],
        "exemplars": [
            "Sokak lambası bozuk, şikayet etmek istiyorum.",
            "Mahallemizde çöpler toplanmıyor.",
            "Park alanında hasar var, bildirmek istiyorum.",
            "Komşumdan gürültü şikayetim var.",
            "Yol çukurlu ve tehlikeli.",
        ],
        "keyword_confidence": 0.82,
    },
    "DUYURU_HABER": {
        "keywords": ["duyuru", "haber", "ilan", "son haberler", "etkinlik", "açıklama", "kampanya", "program", "duyurular"],
        "exemplars": [
            "Belediyenin son duyuruları neler?",
            "Bu ay hangi etkinlikler var?",
            "Yeni imar planı hakkında açıklama var mı?",
            "Belediye başkanının son açıklaması nedir?",
        ],
        "keyword_confidence": 0.78,
    },
    "ILETISIM_ULASIM": {
        "keywords": ["telefon", "numara", "adres", "iletişim", "e-posta", "mail",
                     "ulaşmak", "arayabilir miyim", "müdürlük", "birim", "153", "çağrı merkezi"],
        "exemplars": [
            "Belediyenin telefon numarası nedir?",
            "İmar müdürlüğüne nasıl ulaşabilirim?",
            "Şikayet hattı var mı?",
            "E-posta adresiniz nedir?",
            "Hangi birime başvurmalıyım?",
        ],
        "keyword_confidence": 0.80,
    },
    "YONERGE_KONUM": {
        "keywords": ["nerede", "nereye", "yol tarifi", "nasıl gidilir", "konum",
                     "adres nedir", "nerede bulunuyor", "tarif", "hangi katta", "hangi kat"],
        "exemplars": [
            "Edirne Belediyesi nerede?",
            "Nikah dairesi hangi katta?",
            "İmar müdürlüğüne nasıl gidilir?",
            "Sosyal hizmetler binası nerede?",
        ],
        "keyword_confidence": 0.78,
    },
    "KAPSAM_DISI": {
        "keywords": ["hava durumu", "futbol", "siyaset", "borsa", "yemek tarifi", "film", "müzik", "bitcoin"],
        "exemplars": [
            "Bugün hava nasıl olacak?",
            "Galatasaray kaç aldı?",
            "En iyi restoran hangisi?",
            "Bitcoin fiyatı nedir?",
            "Türkiye siyaseti hakkında ne düşünüyorsunuz?",
        ],
        "keyword_confidence": 0.75,
    },
}

# Doğrudan cevap verilecek niyetler (RAG / LLM gerekmez)
DIRECT_INTENTS = {"SELAMLASMA", "VEDALASMA", "TESEKKUR", "ACIL_DURUM", "KAPSAM_DISI"}


@dataclass
class IntentResult:
    label: str
    confidence: float
    method: str  # "keyword" | "embedding" | "hybrid"
    alternatives: list[dict] = field(default_factory=list)


class IntentClassifier:
    """
    Hibrit intent sınıflandırıcı.

    Önemli optimizasyon: Bu sınıf, zaten yüklenmiş olan embedding modelini
    paylaşır (EmbeddingService singleton). Ek model belleğe yüklenmez.
    Startup sırasında tüm örnek vektörler önceden hesaplanır.
    """

    def __init__(self) -> None:
        self._embedding_service: Optional[object] = None
        self._exemplar_embeddings: dict[str, np.ndarray] = {}
        self._initialized = False
        self._regex_patterns: dict[str, list[re.Pattern]] = {
            intent: [re.compile(p, re.IGNORECASE) for p in defn.get("patterns", [])]
            for intent, defn in INTENT_DEFINITIONS.items()
        }

    def initialize(self, embedding_service: object) -> None:
        """Embedding servisi hazır olduktan sonra çağrılır."""
        self._embedding_service = embedding_service
        logger.info("Intent classifier: örnek vektörler hesaplanıyor…")
        for intent, defn in INTENT_DEFINITIONS.items():
            exemplars = defn.get("exemplars", [])
            if exemplars:
                vecs = embedding_service.get_embeddings(exemplars)  # type: ignore[union-attr]
                self._exemplar_embeddings[intent] = np.array(vecs, dtype=np.float32)
        self._initialized = True
        logger.info("Intent classifier hazır. {} intent yüklendi.", len(self._exemplar_embeddings))

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str) -> IntentResult:
        text_norm = text.strip().lower()

        # 1) Regex / keyword: hızlı, yüksek güven
        kw_result = self._keyword_classify(text_norm)
        if kw_result and kw_result.confidence >= 0.80:
            return kw_result

        # 2) Embedding benzerliği: semantik anlama
        if self._initialized:
            emb_result = self._embedding_classify(text)
            # Hem keyword hem embedding sonucu varsa ağırlıklı birleştir
            if kw_result:
                return self._merge(kw_result, emb_result)
            return emb_result

        # 3) Sadece keyword çalıştıysa onu döndür
        if kw_result:
            return kw_result

        return IntentResult(label="BILGI_TALEBI", confidence=0.35, method="fallback")

    # ── Katman 1: Keyword / Regex ─────────────────────────────────────────────

    def _keyword_classify(self, text: str) -> Optional[IntentResult]:
        best_label: Optional[str] = None
        best_conf = 0.0

        for intent, defn in INTENT_DEFINITIONS.items():
            # Regex kontrolü (kesin eşleşme)
            for pattern in self._regex_patterns.get(intent, []):
                if pattern.search(text):
                    conf = defn.get("keyword_confidence", 0.85)
                    if conf > best_conf:
                        best_conf = conf
                        best_label = intent

            # Anahtar kelime sayımı
            if not best_label or best_conf < 0.80:
                kw_count = sum(1 for kw in defn.get("keywords", []) if kw in text)
                if kw_count > 0:
                    base_conf = defn.get("keyword_confidence", 0.72)
                    conf = min(base_conf + (kw_count - 1) * 0.04, 0.92)
                    if conf > best_conf:
                        best_conf = conf
                        best_label = intent

        if best_label:
            return IntentResult(label=best_label, confidence=best_conf, method="keyword")
        return None

    # ── Katman 2: Embedding benzerliği ───────────────────────────────────────

    def _embedding_classify(self, text: str) -> IntentResult:
        query_vec = np.array(
            self._embedding_service.get_embedding(text),  # type: ignore[union-attr]
            dtype=np.float32,
        )

        scores: dict[str, float] = {}
        for intent, exemplar_matrix in self._exemplar_embeddings.items():
            # Cosine similarity: vektörler normalize edilmiş → nokta çarpımı yeterli
            sims = exemplar_matrix @ query_vec
            scores[intent] = float(np.max(sims))

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_label, best_conf = sorted_scores[0]

        alternatives = [{"label": lbl, "confidence": round(sc, 4)} for lbl, sc in sorted_scores[1:4]]
        return IntentResult(label=best_label, confidence=best_conf, method="embedding", alternatives=alternatives)

    # ── Birleştirme ───────────────────────────────────────────────────────────

    @staticmethod
    def _merge(kw: IntentResult, emb: IntentResult) -> IntentResult:
        """
        Keyword ve embedding sonuçlarını birleştirir.
        Aynı label → ağırlıklı ortalama.
        Farklı label → yüksek güvenli olan kazanır.
        """
        if kw.label == emb.label:
            conf = kw.confidence * 0.45 + emb.confidence * 0.55
            return IntentResult(label=kw.label, confidence=min(conf, 0.97), method="hybrid")
        # Farklı label: keyword genellikle daha kesin, ancak düşük güven varsa embedding'e git
        winner, loser = (kw, emb) if kw.confidence >= emb.confidence else (emb, kw)
        return IntentResult(
            label=winner.label,
            confidence=winner.confidence,
            method="hybrid",
            alternatives=[{"label": loser.label, "confidence": round(loser.confidence, 4)}],
        )


# Uygulama genelinde paylaşılan tek örnek
intent_classifier = IntentClassifier()
