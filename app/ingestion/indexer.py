"""
Document Indexer — metni chunk'lara böl ve ChromaDB'ye indeksle.

Chunk stratejisi:
  - RecursiveCharacterTextSplitter (LangChain'den): paragraf → cümle → kelime
    sırasıyla böler, anlam bütünlüğünü korur
  - Chunk boyutu 600 karakter: BGE-M3'ün 512 token sınırına uyan, yeterli bağlam
  - Overlap 80 karakter: kesim noktasında bağlam kaybını azaltır

Metadata zenginleştirme:
  Her chunk'a parent döküman metadata'sı + chunk_id eklenir.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.ingestion.document_loader import load_file, load_from_json_sss, load_from_text
from app.services.vector_db import vector_db

# Chunk parametreleri
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
MIN_CHUNK_LENGTH = 40


def _make_chunk_id(text: str, source: str) -> str:
    """Deterministik chunk ID: aynı içerik + kaynak → aynı ID (deduplication)."""
    h = hashlib.md5(f"{source}::{text}".encode()).hexdigest()[:12]
    return h


class DocumentIndexer:
    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
        )

    def index_text(self, text: str, metadata: dict[str, Any] | None = None) -> dict:
        """Ham metin → chunk → ChromaDB."""
        docs = load_from_text(text, metadata)
        return self._index_docs(docs)

    def index_file(self, file_path: str, metadata: dict[str, Any] | None = None) -> dict:
        """Dosya → yükle → chunk → ChromaDB."""
        docs = load_file(file_path, metadata)
        return self._index_docs(docs)

    def index_sss_json(self, file_path: str) -> dict:
        """sss.json dosyasını JS projesinden içe aktar."""
        docs = load_from_json_sss(file_path)
        return self._index_docs(docs)

    def index_directory(self, dir_path: str, extensions: list[str] | None = None) -> dict:
        """Bir dizindeki tüm uygun dosyaları indeksle."""
        exts = extensions or [".txt", ".json", ".docx"]
        all_stats = {"documents_indexed": 0, "chunks_created": 0}
        for path in Path(dir_path).rglob("*"):
            if path.suffix.lower() in exts and path.is_file():
                stats = self.index_file(str(path))
                all_stats["documents_indexed"] += stats.get("documents_indexed", 0)
                all_stats["chunks_created"] += stats.get("chunks_created", 0)
        return all_stats

    # ── İç yardımcılar ───────────────────────────────────────────────────────

    def _index_docs(self, docs: list[dict]) -> dict:
        if not docs:
            return {"documents_indexed": 0, "chunks_created": 0}

        all_texts: list[str] = []
        all_metas: list[dict] = []
        all_ids: list[str] = []

        for doc in docs:
            raw_text = doc["text"]
            base_meta = doc.get("metadata", {})

            chunks = self._splitter.split_text(raw_text)
            valid_chunks = [c for c in chunks if len(c.strip()) >= MIN_CHUNK_LENGTH]

            for i, chunk in enumerate(valid_chunks):
                chunk_id = _make_chunk_id(chunk, base_meta.get("source", str(uuid.uuid4())))
                meta = {
                    **base_meta,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "total_chunks": len(valid_chunks),
                }
                all_texts.append(chunk)
                all_metas.append(meta)
                all_ids.append(chunk_id)

        if not all_texts:
            return {"documents_indexed": len(docs), "chunks_created": 0}

        added = vector_db.upsert(all_texts, all_metas, all_ids)
        logger.info(
            "{} döküman → {} chunk indekslendi",
            len(docs),
            added,
        )
        return {"documents_indexed": len(docs), "chunks_created": added}
