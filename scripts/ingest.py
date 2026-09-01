#!/usr/bin/env python3
"""
Veri Yükleme Scripti (CLI)

Kullanım:
    # sss.json (JS projesi çıktısı) yükle
    python scripts/ingest.py --source data/sss.json --type sss

    # Metin dosyası veya klasör yükle
    python scripts/ingest.py --source data/documents/ --type dir

    # Tek dosya
    python scripts/ingest.py --source data/faqs/faq.txt --type file

    # Mevcut koleksiyonu görmezden gel ve sıfırdan başla
    python scripts/ingest.py --source data/sss.json --type sss --reset

    # İndeks istatistikleri
    python scripts/ingest.py --stats
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proje kökünü Python path'ine ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from app.core.logger import setup_logging

setup_logging()


def _reset_collection() -> None:
    from app.services.vector_db import vector_db
    from app.config import settings
    logger.warning("Koleksiyon sıfırlanıyor: {}", settings.CHROMA_COLLECTION_NAME)
    vector_db.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
    vector_db.collection = vector_db.client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Koleksiyon sıfırlandı.")


def _show_stats() -> None:
    from app.services.vector_db import vector_db
    from app.config import settings
    count = vector_db.doc_count
    print(f"\nKoleksiyon: {settings.CHROMA_COLLECTION_NAME}")
    print(f"Toplam döküman/chunk: {count}")
    if count > 0:
        sample = vector_db.collection.peek(limit=3)
        print(f"\nÖrnek kayıtlar (ilk 3):")
        for i, doc in enumerate(sample.get("documents", [])[:3]):
            meta = (sample.get("metadatas") or [{}])[i]
            print(f"  [{i+1}] kaynak={meta.get('source', '?')} | {doc[:80]}…")


def main() -> None:
    parser = argparse.ArgumentParser(description="Edirne Belediye Chatbot — Veri Yükleme")
    parser.add_argument("--source", help="Kaynak dosya veya klasör yolu")
    parser.add_argument(
        "--type",
        choices=["sss", "file", "dir", "web"],
        default="file",
        help="Kaynak türü: sss=sss.json, web=web_pages.json, file=tek dosya, dir=klasör",
    )
    parser.add_argument("--reset", action="store_true", help="Önce koleksiyonu sıfırla")
    parser.add_argument("--stats", action="store_true", help="İndeks istatistiklerini göster")
    args = parser.parse_args()

    if args.stats:
        _show_stats()
        return

    if not args.source:
        parser.print_help()
        sys.exit(1)

    if args.reset:
        _reset_collection()

    from app.ingestion.indexer import DocumentIndexer
    indexer = DocumentIndexer()

    logger.info("İndeksleme başlıyor: {} (tür: {})", args.source, args.type)

    if args.type == "sss":
        stats = indexer.index_sss_json(args.source)
    elif args.type == "web":
        from app.ingestion.document_loader import load_from_web_pages_json
        docs = load_from_web_pages_json(args.source)
        stats = indexer._index_docs(docs)
    elif args.type == "dir":
        stats = indexer.index_directory(args.source)
    else:
        stats = indexer.index_file(args.source)

    # BM25 indeksini güncelle
    from app.services.vector_db import vector_db
    from app.services.bm25_service import bm25_service
    ids, texts, metas = vector_db.get_all_for_bm25()
    bm25_service.build(ids, texts, metas)

    print("\n" + "=" * 50)
    print("İndeksleme Tamamlandı!")
    print(f"  Döküman sayısı : {stats.get('documents_indexed', 0)}")
    print(f"  Chunk sayısı   : {stats.get('chunks_created', 0)}")
    print(f"  Toplam DB'de   : {vector_db.doc_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
