"""
Category Classifier — aynı hibrit mimari: keyword → embedding.

Kategoriler, belediye hizmet alanlarına göre tasarlanmıştır.
Intent ile ortaklaşa çalışır: intent=BASVURU_ISLEM ise kategori skorları
belirli alanlarda boosted edilir (örn. IMAR_INSAAT için ruhsat intentleri).

Neden ML model değil embedding?
- BGE-M3 zaten yüklü → sıfır ek bellek
- Türkçe semantiği çok iyi yakalar
- Yeni kategori eklemek için sadece örnek cümle eklemek yeterli
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger


CATEGORY_DEFINITIONS: dict[str, dict] = {
    "SU_ALTYAPI": {
        "keywords": ["su", "su faturası", "su kesintisi", "abonelik", "su sayacı",
                     "kanalizasyon", "boru", "içme suyu", "atık su", "su borcu",
                     "su aboneliği", "musluk"],
        "exemplars": [
            "Su faturasına itiraz etmek istiyorum.",
            "Su kesintisi ne zaman bitecek?",
            "Su aboneliği nasıl açılır?",
            "Kanalizasyon tıkandı.",
            "Musluktan sarı su geliyor.",
        ],
        "keyword_weight": 1.0,
    },
    "ULASIM_TRAFIK": {
        "keywords": ["otobüs", "dolmuş", "ulaşım", "hat", "sefer", "park",
                     "trafik", "kaldırım", "bisiklet yolu", "durak", "toplu taşıma"],
        "exemplars": [
            "Otobüs güzergahları nelerdir?",
            "Belediye otobüsü saatleri neler?",
            "Park yeri sorunum var.",
            "Bisiklet yolu nerede başlıyor?",
            "Durak kaldırıldı mı?",
        ],
        "keyword_weight": 1.0,
    },
    "IMAR_INSAAT": {
        "keywords": ["imar", "ruhsat", "yapı", "inşaat", "parsel", "kadastro",
                     "yapı kullanım izni", "yıkım", "tadilat", "imar durumu",
                     "imar planı", "iskân"],
        "exemplars": [
            "İmar durumu nasıl sorgulanır?",
            "İnşaat ruhsatı için başvuru nasıl yapılır?",
            "Yapı kullanım izni nasıl alınır?",
            "Tadilat için ruhsat gerekli mi?",
            "Parsel bilgisi nasıl öğrenilir?",
        ],
        "keyword_weight": 1.0,
    },
    "CEVRE_TEMIZLIK": {
        "keywords": ["çöp", "temizlik", "geri dönüşüm", "atık", "çevre",
                     "temizlik vergisi", "çöp toplama", "konteyner", "moloz",
                     "çevre kirliliği", "atık yönetimi"],
        "exemplars": [
            "Çöpler toplanmıyor şikayeti.",
            "Geri dönüşüm kutuları nerede?",
            "Moloz atma yerleri nerede?",
            "Çevre temizlik vergisi ödemesi.",
            "Sokak temizliği ne zaman yapılıyor?",
        ],
        "keyword_weight": 1.0,
    },
    "SOSYAL_HIZMETLER": {
        "keywords": ["sosyal yardım", "engelli", "yaşlı", "muhtaç", "kömür yardımı",
                     "gıda yardımı", "burs", "bakım", "kreş", "dar gelirli",
                     "sosyal destek", "yardım başvurusu"],
        "exemplars": [
            "Sosyal yardım başvurusu nasıl yapılır?",
            "Engelli vatandaş için destek var mı?",
            "Yaşlı bakım hizmetleri neler?",
            "Kömür yardımı başvurusu.",
            "Dar gelirli aile yardım başvurusu.",
        ],
        "keyword_weight": 1.0,
    },
    "KULTUR_SANAT": {
        "keywords": ["kültür", "sanat", "festival", "etkinlik", "tiyatro",
                     "sinema", "konser", "sergi", "tarihi", "müze", "turizm",
                     "kültür merkezi", "selimiye"],
        "exemplars": [
            "Bu ay hangi etkinlikler var?",
            "Edirne müzeleri hangileri?",
            "Selimiye Camii hakkında bilgi.",
            "Kültür merkezi programı ne zaman?",
            "Edirne turizm rehberi.",
        ],
        "keyword_weight": 1.0,
    },
    "SAGLIK": {
        "keywords": ["sağlık", "muayene", "aşı", "dezenfektan", "ilaç",
                     "ambulans", "veteriner", "haşere", "ilaçlama"],
        "exemplars": [
            "Ücretsiz aşı hizmetleri var mı?",
            "Belediye veteriner kliniği nerede?",
            "Haşere ilaçlama takvimi nedir?",
            "Sağlık hizmetleri hakkında bilgi.",
        ],
        "keyword_weight": 1.0,
    },
    "MALI_ISLER": {
        "keywords": ["vergi", "harç", "ödeme", "borç", "bütçe", "emlak vergisi",
                     "iş yeri vergisi", "ilan reklam", "ödeme planı",
                     "belediye borcu", "taksit"],
        "exemplars": [
            "Emlak vergisi nasıl ödenir?",
            "Belediye borcu sorgulama.",
            "İlan reklam vergisi nedir?",
            "Vergi borcu taksitlendirme.",
            "Ödeme planı nasıl yapılır?",
        ],
        "keyword_weight": 1.0,
    },
    "PARK_BAHCE": {
        "keywords": ["park", "bahçe", "yeşil alan", "ağaç", "çiçek",
                     "oyun alanı", "piknik", "çim", "yeşillendirme"],
        "exemplars": [
            "En yakın park nerede?",
            "Park bakımı şikayeti.",
            "Çocuk oyun alanı talebi.",
            "Ağaç budama isteği.",
            "Piknik alanları nerede?",
        ],
        "keyword_weight": 1.0,
    },
    "NIKAH_EVLENDIRME": {
        "keywords": ["nikah", "evlenme", "evlilik", "düğün", "nikah randevusu",
                     "evlilik cüzdanı", "aile cüzdanı", "nikah tarihi", "evlendirme"],
        "exemplars": [
            "Nikah randevusu almak istiyorum.",
            "Evlilik için gerekli belgeler neler?",
            "Nikah tarihi nasıl belirlenir?",
            "Evlilik cüzdanı nasıl çıkarılır?",
            "Nikah işlemleri nereden yapılır?",
            "Nikah için nereye başvurmam gerekiyor?",
            "Evlendirme memurluğu nerede?",
            "Belediyede nikah nasıl kıyılır?",
        ],
        "keyword_weight": 1.0,
    },
    "GENEL": {
        "keywords": [],
        "exemplars": [
            "Belediye hizmetleri hakkında bilgi almak istiyorum.",
            "Genel sorum var.",
            "Belediye neler yapıyor?",
        ],
        "keyword_weight": 0.5,
    },
}

# Intent → kategori boost tablosu
# Belirli bir intent varsa bu kategorilerin embedding skoru boosted edilir
INTENT_CATEGORY_BOOST: dict[str, dict[str, float]] = {
    "BASVURU_ISLEM": {
        "IMAR_INSAAT": 0.08,
        "SOSYAL_HIZMETLER": 0.06,
        "NIKAH_EVLENDIRME": 0.10,
        "MALI_ISLER": 0.06,
    },
    "SIKAYET_BILDIRIM": {
        "CEVRE_TEMIZLIK": 0.08,
        "ULASIM_TRAFIK": 0.06,
        "SU_ALTYAPI": 0.06,
        "PARK_BAHCE": 0.05,
    },
    "DUYURU_HABER": {
        "KULTUR_SANAT": 0.07,
    },
    "ILETISIM_ULASIM": {
        "GENEL": 0.05,
    },
}


@dataclass
class CategoryResult:
    label: str
    confidence: float
    method: str
    alternatives: list[dict] = field(default_factory=list)


class CategoryClassifier:
    """
    Belediye hizmet kategorisi sınıflandırıcısı.
    Intent bağlamını kullanarak daha isabetli sınıflandırma yapar.
    """

    def __init__(self) -> None:
        self._embedding_service: Optional[object] = None
        self._exemplar_embeddings: dict[str, np.ndarray] = {}
        self._initialized = False

    def initialize(self, embedding_service: object) -> None:
        self._embedding_service = embedding_service
        logger.info("Category classifier: örnek vektörler hesaplanıyor…")
        for cat, defn in CATEGORY_DEFINITIONS.items():
            exemplars = defn.get("exemplars", [])
            if exemplars:
                vecs = embedding_service.get_embeddings(exemplars)  # type: ignore[union-attr]
                self._exemplar_embeddings[cat] = np.array(vecs, dtype=np.float32)
        self._initialized = True
        logger.info("Category classifier hazır. {} kategori yüklendi.", len(self._exemplar_embeddings))

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, text: str, intent_label: str = "BILGI_TALEBI") -> CategoryResult:
        text_norm = text.strip().lower()

        # 1) Keyword ile hızlı kontrol
        kw_result = self._keyword_classify(text_norm)
        if kw_result and kw_result.confidence >= 0.65:
            return kw_result

        # 2) Embedding + intent boost
        if self._initialized:
            emb_result = self._embedding_classify(text, intent_label)
            if kw_result:
                return self._merge(kw_result, emb_result)
            return emb_result

        if kw_result:
            return kw_result

        return CategoryResult(label="GENEL", confidence=0.30, method="fallback")

    # ── Katman 1: Keyword ─────────────────────────────────────────────────────

    def _keyword_classify(self, text: str) -> Optional[CategoryResult]:
        best_label: Optional[str] = None
        best_conf = 0.0

        for cat, defn in CATEGORY_DEFINITIONS.items():
            kw_count = sum(1 for kw in defn.get("keywords", []) if kw in text)
            if kw_count > 0:
                weight = defn.get("keyword_weight", 1.0)
                conf = min(0.68 + (kw_count - 1) * 0.05, 0.90) * weight
                if conf > best_conf:
                    best_conf = conf
                    best_label = cat

        if best_label:
            return CategoryResult(label=best_label, confidence=best_conf, method="keyword")
        return None

    # ── Katman 2: Embedding + intent boost ───────────────────────────────────

    def _embedding_classify(self, text: str, intent_label: str) -> CategoryResult:
        query_vec = np.array(
            self._embedding_service.get_embedding(text),  # type: ignore[union-attr]
            dtype=np.float32,
        )

        boost_map = INTENT_CATEGORY_BOOST.get(intent_label, {})
        scores: dict[str, float] = {}
        for cat, exemplar_matrix in self._exemplar_embeddings.items():
            base_score = float(np.max(exemplar_matrix @ query_vec))
            scores[cat] = base_score + boost_map.get(cat, 0.0)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_label, best_conf = sorted_scores[0]
        alternatives = [{"label": lbl, "confidence": round(sc, 4)} for lbl, sc in sorted_scores[1:4]]
        return CategoryResult(label=best_label, confidence=best_conf, method="embedding", alternatives=alternatives)

    # ── Birleştirme ───────────────────────────────────────────────────────────

    @staticmethod
    def _merge(kw: CategoryResult, emb: CategoryResult) -> CategoryResult:
        if kw.label == emb.label:
            conf = kw.confidence * 0.40 + emb.confidence * 0.60
            return CategoryResult(label=kw.label, confidence=min(conf, 0.97), method="hybrid")
        winner, loser = (kw, emb) if kw.confidence >= emb.confidence else (emb, kw)
        return CategoryResult(
            label=winner.label,
            confidence=winner.confidence,
            method="hybrid",
            alternatives=[{"label": loser.label, "confidence": round(loser.confidence, 4)}],
        )


category_classifier = CategoryClassifier()
