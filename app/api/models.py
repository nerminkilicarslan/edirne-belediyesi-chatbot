"""
Pydantic request/response models and shared enumerations.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class IntentType(str, Enum):
    SELAMLASMA = "SELAMLASMA"           # greeting
    VEDALASMA = "VEDALASMA"             # farewell
    TESEKKUR = "TESEKKUR"               # thanks
    BILGI_TALEBI = "BILGI_TALEBI"       # information request
    BASVURU_ISLEM = "BASVURU_ISLEM"     # application / process
    SIKAYET_BILDIRIM = "SIKAYET_BILDIRIM"  # complaint / report
    DUYURU_HABER = "DUYURU_HABER"       # announcement / news
    ILETISIM_ULASIM = "ILETISIM_ULASIM" # contact info
    YONERGE_KONUM = "YONERGE_KONUM"     # navigation / location
    ACIL_DURUM = "ACIL_DURUM"           # emergency
    KAPSAM_DISI = "KAPSAM_DISI"         # out of scope


class CategoryType(str, Enum):
    SU_ALTYAPI = "SU_ALTYAPI"
    ULASIM_TRAFIK = "ULASIM_TRAFIK"
    IMAR_INSAAT = "IMAR_INSAAT"
    CEVRE_TEMIZLIK = "CEVRE_TEMIZLIK"
    SOSYAL_HIZMETLER = "SOSYAL_HIZMETLER"
    KULTUR_SANAT = "KULTUR_SANAT"
    SAGLIK = "SAGLIK"
    MALI_ISLER = "MALI_ISLER"
    PARK_BAHCE = "PARK_BAHCE"
    NIKAH_EVLENDIRME = "NIKAH_EVLENDIRME"
    GENEL = "GENEL"


class RouteDecision(str, Enum):
    DIRECT_ANSWER = "DIRECT_ANSWER"       # canned response, no LLM needed
    RETRIEVAL_FOCUSED = "RETRIEVAL_FOCUSED"  # high-confidence: narrow retrieval
    RETRIEVAL_BROAD = "RETRIEVAL_BROAD"   # medium-confidence: wider retrieval
    CLARIFICATION = "CLARIFICATION"       # confidence too low, ask user
    FALLBACK = "FALLBACK"                 # no docs found or very low confidence
    EMERGENCY = "EMERGENCY"               # acil durum: immediate number
    OUT_OF_SCOPE = "OUT_OF_SCOPE"         # not related to municipality


# ── Sub-models ────────────────────────────────────────────────────────────────

class DocumentSource(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
    retrieval_method: str = "semantic"  # "semantic" | "bm25" | "hybrid"


class ClassificationResult(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    method: str  # "keyword" | "embedding" | "hybrid"
    alternatives: list[dict[str, Any]] = Field(default_factory=list)


class ConfidenceScore(BaseModel):
    overall: float = Field(..., ge=0.0, le=1.0)
    intent_confidence: float
    category_confidence: float
    retrieval_confidence: float = 0.0


# ── API Request / Response ─────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., description="Kullanıcının sorusu veya mesajı")
    session_id: Optional[str] = Field(None, description="Oturum takip kimliği")

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Sorgu boş olamaz")
        return v


class ChatResponse(BaseModel):
    answer: str
    intent: str
    category: str
    confidence: float
    route: str = ""
    sources: Optional[list[DocumentSource]] = None
    processing_time_ms: float
    is_fallback: bool = False
    request_id: str = ""


class HealthDetail(BaseModel):
    status: str
    version: str
    services: dict[str, str]


class IngestRequest(BaseModel):
    source_type: str = Field(..., description="'file' | 'text' | 'url'")
    source_path: Optional[str] = None
    content: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_created: int
    message: str = ""
