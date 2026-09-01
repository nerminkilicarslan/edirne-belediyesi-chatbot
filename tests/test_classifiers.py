"""
Classifier testleri — embedding modeli gerektirmeden keyword katmanını test eder.
Embedding katmanı için integration test ayrıca yazılabilir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.intent_classifier import IntentClassifier, IntentResult
from app.pipeline.category_classifier import CategoryClassifier, CategoryResult
from app.pipeline.confidence import ConfidenceScorer
from app.pipeline.router import HybridRouter, CANNED_RESPONSES
from app.api.models import RouteDecision


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def intent_clf():
    return IntentClassifier()


@pytest.fixture
def category_clf():
    return CategoryClassifier()


@pytest.fixture
def scorer():
    return ConfidenceScorer()


@pytest.fixture
def router():
    return HybridRouter()


# ── Intent Classifier ─────────────────────────────────────────────────────────

class TestIntentClassifierKeyword:
    def test_greeting_merhaba(self, intent_clf):
        result = intent_clf._keyword_classify("merhaba")
        assert result is not None
        assert result.label == "SELAMLASMA"
        assert result.confidence >= 0.85

    def test_greeting_iyi_gunler(self, intent_clf):
        result = intent_clf._keyword_classify("iyi günler, yardım alabilir miyim")
        assert result is not None
        assert result.label == "SELAMLASMA"

    def test_farewell(self, intent_clf):
        result = intent_clf._keyword_classify("görüşürüz")
        assert result is not None
        assert result.label == "VEDALASMA"

    def test_thanks(self, intent_clf):
        result = intent_clf._keyword_classify("teşekkür ederim çok yardımcı oldunuz")
        assert result is not None
        assert result.label == "TESEKKUR"

    def test_emergency(self, intent_clf):
        result = intent_clf._keyword_classify("yangın var acil yardım")
        assert result is not None
        assert result.label == "ACIL_DURUM"
        assert result.confidence >= 0.90

    def test_complaint_sikayet(self, intent_clf):
        result = intent_clf._keyword_classify("çöpler toplanmıyor şikayetim var")
        assert result is not None
        assert result.label == "SIKAYET_BILDIRIM"

    def test_application(self, intent_clf):
        result = intent_clf._keyword_classify("ruhsat başvurusu yapmak istiyorum")
        assert result is not None
        assert result.label == "BASVURU_ISLEM"

    def test_contact(self, intent_clf):
        result = intent_clf._keyword_classify("belediyenin telefon numarası nedir")
        assert result is not None
        assert result.label == "ILETISIM_ULASIM"

    def test_location(self, intent_clf):
        result = intent_clf._keyword_classify("belediye nerede")
        assert result is not None
        assert result.label == "YONERGE_KONUM"

    def test_no_match_returns_none(self, intent_clf):
        # Hiçbir keyword eşleşmemeli
        result = intent_clf._keyword_classify("xyzmno")
        assert result is None

    def test_classify_returns_intent_result(self, intent_clf):
        """classify() her durumda IntentResult döndürmeli."""
        result = intent_clf.classify("belediyeye nasıl başvurabilirim")
        assert isinstance(result, IntentResult)
        assert 0.0 <= result.confidence <= 1.0
        assert result.label
        assert result.method in ("keyword", "embedding", "hybrid", "fallback")

    def test_emergency_always_high_confidence(self, intent_clf):
        result = intent_clf.classify("acil durum var su baskını")
        assert result.label == "ACIL_DURUM"
        assert result.confidence >= 0.85


# ── Category Classifier ───────────────────────────────────────────────────────

class TestCategoryClassifierKeyword:
    def test_water_su(self, category_clf):
        result = category_clf._keyword_classify("su faturası nasıl ödenir")
        assert result is not None
        assert result.label == "SU_ALTYAPI"

    def test_construction_imar(self, category_clf):
        result = category_clf._keyword_classify("imar durumu nasıl sorgulanır")
        assert result is not None
        assert result.label == "IMAR_INSAAT"

    def test_garbage_cevre(self, category_clf):
        result = category_clf._keyword_classify("çöpler toplanmıyor")
        assert result is not None
        assert result.label == "CEVRE_TEMIZLIK"

    def test_transport(self, category_clf):
        result = category_clf._keyword_classify("otobüs saatleri neler")
        assert result is not None
        assert result.label == "ULASIM_TRAFIK"

    def test_social_services(self, category_clf):
        result = category_clf._keyword_classify("sosyal yardım başvurusu")
        assert result is not None
        assert result.label == "SOSYAL_HIZMETLER"

    def test_marriage(self, category_clf):
        result = category_clf._keyword_classify("nikah randevusu almak istiyorum")
        assert result is not None
        assert result.label == "NIKAH_EVLENDIRME"

    def test_taxes(self, category_clf):
        result = category_clf._keyword_classify("emlak vergisi nasıl ödenir")
        assert result is not None
        assert result.label == "MALI_ISLER"

    def test_culture(self, category_clf):
        result = category_clf._keyword_classify("bu ay hangi etkinlikler var")
        assert result is not None
        assert result.label == "KULTUR_SANAT"

    def test_classify_always_returns(self, category_clf):
        result = category_clf.classify("belediye hizmetleri", "BILGI_TALEBI")
        assert isinstance(result, CategoryResult)
        assert result.label


# ── Confidence Scorer ─────────────────────────────────────────────────────────

class TestConfidenceScorer:
    def test_high_confidence(self, scorer):
        result = scorer.score_pre_retrieval(0.90, 0.85)
        assert result.tier == "HIGH"
        assert result.overall >= 0.65

    def test_medium_confidence(self, scorer):
        result = scorer.score_pre_retrieval(0.60, 0.55)
        assert result.tier in ("MEDIUM", "HIGH")

    def test_low_confidence(self, scorer):
        result = scorer.score_pre_retrieval(0.30, 0.25)
        assert result.tier in ("LOW", "VERY_LOW")

    def test_post_retrieval_boosts_score(self, scorer):
        pre = scorer.score_pre_retrieval(0.55, 0.50)
        post = scorer.score_post_retrieval(0.55, 0.50, [0.80, 0.75])
        assert post.overall >= pre.overall

    def test_no_retrieval_docs_zero(self, scorer):
        result = scorer.score_post_retrieval(0.70, 0.65, [])
        assert result.retrieval_confidence == 0.0

    def test_score_bounds(self, scorer):
        result = scorer.score_pre_retrieval(1.0, 1.0)
        assert 0.0 <= result.overall <= 1.0


# ── Router ────────────────────────────────────────────────────────────────────

class TestHybridRouter:
    def _make_intent(self, label: str, conf: float = 0.80) -> IntentResult:
        return IntentResult(label=label, confidence=conf, method="keyword")

    def _make_category(self, label: str = "GENEL", conf: float = 0.60) -> CategoryResult:
        return CategoryResult(label=label, confidence=conf, method="keyword")

    def _make_confidence(self, tier: str, overall: float = 0.50):
        from app.pipeline.confidence import OverallConfidence
        return OverallConfidence(
            overall=overall,
            intent_confidence=0.60,
            category_confidence=0.60,
            retrieval_confidence=0.0,
            tier=tier,
            explanation="test",
        )

    def test_emergency_route(self, router):
        intent = self._make_intent("ACIL_DURUM", 0.95)
        route, canned = router.decide(intent, self._make_category(), self._make_confidence("HIGH"))
        assert route == RouteDecision.EMERGENCY
        assert "110" in canned or "112" in canned

    def test_greeting_direct_answer(self, router):
        intent = self._make_intent("SELAMLASMA", 0.95)
        route, canned = router.decide(intent, self._make_category(), self._make_confidence("HIGH"))
        assert route == RouteDecision.DIRECT_ANSWER
        assert canned

    def test_oos_route(self, router):
        intent = self._make_intent("KAPSAM_DISI", 0.85)
        route, canned = router.decide(intent, self._make_category(), self._make_confidence("HIGH"))
        assert route == RouteDecision.OUT_OF_SCOPE

    def test_fallback_very_low(self, router):
        intent = self._make_intent("BILGI_TALEBI", 0.20)
        route, canned = router.decide(intent, self._make_category(), self._make_confidence("VERY_LOW", 0.15))
        assert route == RouteDecision.FALLBACK

    def test_clarification_low(self, router):
        intent = self._make_intent("BILGI_TALEBI", 0.35)
        route, canned = router.decide(intent, self._make_category(), self._make_confidence("LOW", 0.32))
        assert route == RouteDecision.CLARIFICATION
        assert canned

    def test_high_confidence_focused(self, router):
        intent = self._make_intent("BILGI_TALEBI", 0.85)
        route, canned = router.decide(intent, self._make_category("IMAR_INSAAT", 0.78), self._make_confidence("HIGH", 0.72))
        assert route == RouteDecision.RETRIEVAL_FOCUSED
        assert canned == ""

    def test_medium_confidence_broad(self, router):
        intent = self._make_intent("BILGI_TALEBI", 0.55)
        route, canned = router.decide(intent, self._make_category(), self._make_confidence("MEDIUM", 0.52))
        assert route == RouteDecision.RETRIEVAL_BROAD

    def test_retrieval_config_focused_has_category_filter(self, router):
        cat = self._make_category("IMAR_INSAAT", 0.80)
        cfg = router.get_retrieval_config(RouteDecision.RETRIEVAL_FOCUSED, cat)
        assert cfg["category_filter"] == "IMAR_INSAAT"
        assert cfg["score_threshold"] >= 0.30

    def test_retrieval_config_broad_no_filter(self, router):
        cat = self._make_category("GENEL", 0.30)
        cfg = router.get_retrieval_config(RouteDecision.RETRIEVAL_BROAD, cat)
        assert cfg["category_filter"] is None
