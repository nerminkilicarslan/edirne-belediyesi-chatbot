"""
Pipeline entegrasyon testleri.
Bu testler gerçek embedding modeli gerektirmez — pipeline akışını mock'larla test eder.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.orchestrator import DecisionPipeline
from app.api.models import ChatResponse, RouteDecision


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline():
    return DecisionPipeline()


# ── Pipeline Akış Testleri ─────────────────────────────────────────────────────

class TestDecisionPipelineFlow:
    @pytest.mark.asyncio
    async def test_greeting_skips_retrieval(self, pipeline):
        """Selamlama niyeti retrieval ve LLM'i atlamalı."""
        # Mock: intent classifier selamlama döndürür
        with patch("app.pipeline.orchestrator.intent_classifier") as mock_intent, \
             patch("app.pipeline.orchestrator.category_classifier") as mock_cat, \
             patch("app.pipeline.orchestrator.retrieval_service") as mock_ret, \
             patch("app.pipeline.orchestrator.llm_service") as mock_llm:

            from app.pipeline.intent_classifier import IntentResult
            from app.pipeline.category_classifier import CategoryResult

            mock_intent.classify.return_value = IntentResult(
                label="SELAMLASMA", confidence=0.95, method="keyword"
            )
            mock_cat.classify.return_value = CategoryResult(
                label="GENEL", confidence=0.50, method="keyword"
            )

            response = await pipeline.process("merhaba")

            assert isinstance(response, ChatResponse)
            assert response.route == RouteDecision.DIRECT_ANSWER.value
            assert "Edirne Belediyesi" in response.answer
            # Retrieval ve LLM çağrılmamalı
            mock_ret.retrieve.assert_not_called()
            mock_llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_emergency_returns_numbers(self, pipeline):
        """Acil durum her zaman acil numaraları döndürmeli."""
        with patch("app.pipeline.orchestrator.intent_classifier") as mock_intent, \
             patch("app.pipeline.orchestrator.category_classifier") as mock_cat:

            from app.pipeline.intent_classifier import IntentResult
            from app.pipeline.category_classifier import CategoryResult

            mock_intent.classify.return_value = IntentResult(
                label="ACIL_DURUM", confidence=0.95, method="keyword"
            )
            mock_cat.classify.return_value = CategoryResult(
                label="GENEL", confidence=0.50, method="keyword"
            )

            response = await pipeline.process("yangın var yardım")
            assert response.route == RouteDecision.EMERGENCY.value
            assert "110" in response.answer or "112" in response.answer

    @pytest.mark.asyncio
    async def test_oos_no_retrieval(self, pipeline):
        """Kapsam dışı sorgular retrieval ve LLM'i atlamalı."""
        with patch("app.pipeline.orchestrator.intent_classifier") as mock_intent, \
             patch("app.pipeline.orchestrator.category_classifier") as mock_cat, \
             patch("app.pipeline.orchestrator.retrieval_service") as mock_ret, \
             patch("app.pipeline.orchestrator.llm_service") as mock_llm:

            from app.pipeline.intent_classifier import IntentResult
            from app.pipeline.category_classifier import CategoryResult

            mock_intent.classify.return_value = IntentResult(
                label="KAPSAM_DISI", confidence=0.85, method="keyword"
            )
            mock_cat.classify.return_value = CategoryResult(
                label="GENEL", confidence=0.30, method="keyword"
            )

            response = await pipeline.process("galatasaray kaç aldı")
            assert response.route == RouteDecision.OUT_OF_SCOPE.value
            mock_ret.retrieve.assert_not_called()
            mock_llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_docs_returns_fallback(self, pipeline):
        """Retrieval döküman bulamazsa fallback döndürmeli."""
        with patch("app.pipeline.orchestrator.intent_classifier") as mock_intent, \
             patch("app.pipeline.orchestrator.category_classifier") as mock_cat, \
             patch("app.pipeline.orchestrator.retrieval_service") as mock_ret, \
             patch("app.pipeline.orchestrator.llm_service") as mock_llm:

            from app.pipeline.intent_classifier import IntentResult
            from app.pipeline.category_classifier import CategoryResult

            mock_intent.classify.return_value = IntentResult(
                label="BILGI_TALEBI", confidence=0.80, method="keyword"
            )
            mock_cat.classify.return_value = CategoryResult(
                label="IMAR_INSAAT", confidence=0.70, method="keyword"
            )
            mock_ret.retrieve.return_value = []  # Boş sonuç

            response = await pipeline.process("imar durumu nedir")
            assert response.is_fallback is True
            assert response.route == RouteDecision.FALLBACK.value
            mock_llm.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_rag_flow(self, pipeline):
        """Başarılı RAG akışı: retrieval + LLM çağrılmalı."""
        with patch("app.pipeline.orchestrator.intent_classifier") as mock_intent, \
             patch("app.pipeline.orchestrator.category_classifier") as mock_cat, \
             patch("app.pipeline.orchestrator.retrieval_service") as mock_ret, \
             patch("app.pipeline.orchestrator.llm_service") as mock_llm:

            from app.pipeline.intent_classifier import IntentResult
            from app.pipeline.category_classifier import CategoryResult
            from app.api.models import DocumentSource

            mock_intent.classify.return_value = IntentResult(
                label="BILGI_TALEBI", confidence=0.82, method="keyword"
            )
            mock_cat.classify.return_value = CategoryResult(
                label="SU_ALTYAPI", confidence=0.75, method="embedding"
            )
            mock_ret.retrieve.return_value = [
                DocumentSource(
                    content="Su faturası online portal üzerinden ödenebilir.",
                    metadata={"source": "test"},
                    score=0.85,
                )
            ]
            mock_llm.chat = AsyncMock(return_value="Su faturanızı online ödeyebilirsiniz.")

            response = await pipeline.process("su faturası nasıl ödenir")

            assert response.is_fallback is False
            assert response.answer == "Su faturanızı online ödeyebilirsiniz."
            assert response.confidence > 0
            mock_ret.retrieve.assert_called_once()
            mock_llm.chat.assert_called_once()

    def test_normalize_query_truncates(self, pipeline):
        """Uzun sorgu MAX_QUERY_LENGTH'e kırpılmalı."""
        long_query = "a" * 1000
        from app.pipeline.orchestrator import _normalize_query
        result = _normalize_query(long_query)
        from app.config import settings
        assert len(result) <= settings.MAX_QUERY_LENGTH

    def test_normalize_query_removes_extra_spaces(self):
        from app.pipeline.orchestrator import _normalize_query
        result = _normalize_query("  merhaba    dünya  ")
        assert result == "merhaba dünya"
