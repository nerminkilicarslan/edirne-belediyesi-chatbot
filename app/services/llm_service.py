"""
LLM Service — Ollama async client ile Qwen2.5 / Mistral entegrasyonu.

Mevcut koddan temel fark:
  1. /api/chat kullanılıyor (/api/generate değil) → model'in kendi chat
     template'i otomatik uygulanır (Qwen2.5 için çok önemli)
  2. Async: FastAPI event loop'unu bloklamaz
  3. Timeout yönetimi: Ollama yanıt vermezse fallback mesajı döner
  4. System + User mesaj ayrımı: LLM bağlamı daha iyi anlar

Apple Silicon (M4) notu:
  Ollama Metal backend kullandığı için MPS desteği otomatik.
  OLLAMA_NUM_GPU=1 ayarlı olduğundan ek yapılandırma gerekmez.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import ollama
from loguru import logger

from app.config import settings


class LLMService:
    def __init__(self) -> None:
        self._client = ollama.AsyncClient(
            host=settings.OLLAMA_BASE_URL,
            timeout=settings.OLLAMA_TIMEOUT,
        )
        logger.info(
            "LLM servisi hazır | model: {} | url: {}",
            settings.OLLAMA_MODEL,
            settings.OLLAMA_BASE_URL,
        )

    async def chat(
        self,
        user_message: str,
        system_prompt: str = "",
        temperature: float | None = None,
        context_window: int | None = None,
    ) -> str:
        """
        Ollama chat API üzerinden cevap üret.
        Chat API, model'in kendi instruction template'ini kullanır;
        bu sayede Qwen2.5 daha tutarlı Türkçe cevaplar üretir.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        options = {
            "temperature": temperature or settings.OLLAMA_TEMPERATURE,
            "top_p": settings.OLLAMA_TOP_P,
            "repeat_penalty": settings.OLLAMA_REPEAT_PENALTY,
            "num_ctx": context_window or settings.OLLAMA_CONTEXT_WINDOW,
        }

        try:
            response = await self._client.chat(
                model=settings.OLLAMA_MODEL,
                messages=messages,
                options=options,
            )
            content = response.message.content or ""
            return content.strip()
        except asyncio.TimeoutError:
            logger.error("Ollama zaman aşımı ({}s)", settings.OLLAMA_TIMEOUT)
            return settings.FALLBACK_MESSAGE
        except Exception as exc:
            logger.error("Ollama hatası: {}", exc)
            return settings.FALLBACK_MESSAGE

    async def is_available(self) -> bool:
        """Ollama sunucusunun erişilebilir olup olmadığını kontrol eder."""
        try:
            await self._client.list()
            return True
        except Exception:
            return False

    # Geriye dönük uyumluluk: eski synchronous kodu kırmamak için
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
    ) -> str:
        """
        Senkron wrapper — mevcut kodla uyumluluk için.
        Yeni kodda `await llm_service.chat(...)` kullanın.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # FastAPI event loop içindeyiz: coroutine döndür
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.chat(prompt, system_prompt=system_prompt, temperature=temperature),
                    )
                    return future.result(timeout=settings.OLLAMA_TIMEOUT)
            else:
                return asyncio.run(
                    self.chat(prompt, system_prompt=system_prompt, temperature=temperature)
                )
        except Exception as exc:
            logger.error("LLM generate hatası: {}", exc)
            return settings.FALLBACK_MESSAGE


llm_service = LLMService()
