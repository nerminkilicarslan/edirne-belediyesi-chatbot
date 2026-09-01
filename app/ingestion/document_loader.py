"""
Document Loader — farklı kaynaklardan doküman yükleme.

Desteklenen kaynaklar:
  - JSON (sss.json formatı — mevcut JS projesinden taşıma)
  - Plain text / .txt
  - .docx (python-docx ile)
  - URL (BeautifulSoup ile web crawl)
  - Inline metin

Her döküman standart dict formatına normalize edilir:
  {"text": str, "metadata": {"source": str, "category": str, ...}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from loguru import logger


def _clean_text(text: str) -> str:
    """Gereksiz boşluk ve satır sonlarını temizle."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return " ".join(lines)


# ── Yükleyiciler ──────────────────────────────────────────────────────────────

def load_from_text(text: str, metadata: dict[str, Any] | None = None) -> list[dict]:
    if not text.strip():
        return []
    return [{"text": _clean_text(text), "metadata": metadata or {"source": "inline"}}]


def load_from_txt_file(file_path: str, metadata: dict[str, Any] | None = None) -> list[dict]:
    path = Path(file_path)
    if not path.exists():
        logger.error("Dosya bulunamadı: {}", file_path)
        return []
    text = path.read_text(encoding="utf-8")
    meta = {"source": str(path), "filename": path.name, **(metadata or {})}
    return [{"text": _clean_text(text), "metadata": meta}]


def load_from_docx(file_path: str, metadata: dict[str, Any] | None = None) -> list[dict]:
    try:
        import docx  # type: ignore
    except ImportError:
        logger.error("python-docx yüklü değil: pip install python-docx")
        return []

    path = Path(file_path)
    doc = docx.Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    meta = {"source": str(path), "filename": path.name, **(metadata or {})}
    return [{"text": text, "metadata": meta}]


def load_from_json_sss(file_path: str) -> list[dict]:
    """
    JS projesindeki sss.json formatını yükler.
    Format: [{"id": ..., "category": ..., "question": ..., "answer": ..., "keywords": [...], "links": [...]}]
    """
    path = Path(file_path)
    if not path.exists():
        logger.error("sss.json bulunamadı: {}", file_path)
        return []

    with open(path, encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        logger.error("sss.json beklenen format: JSON array")
        return []

    docs = []
    for item in items:
        question = item.get("question", "")
        answer = item.get("answer", "")
        category = item.get("category", "genel")
        keywords = ", ".join(item.get("keywords", []))
        links_text = ""
        if item.get("links"):
            links_text = "İlgili bağlantılar:\n" + "\n".join(
                f"- {l.get('label', '')}: {l.get('url', '')}"
                for l in item["links"]
            )

        text = f"Soru: {question}\nCevap: {answer}"
        if keywords:
            text += f"\nAnahtar kelimeler: {keywords}"
        if links_text:
            text += f"\n{links_text}"

        meta = {
            "source": f"sss_json:{item.get('id', '')}",
            "source_type": "sss_json",
            "category": _map_category(category),
            "question": question,
            "original_category": category,
        }
        docs.append({"text": text, "metadata": meta})

    logger.info("sss.json'dan {} soru-cevap yüklendi", len(docs))
    return docs


def load_from_web_pages_json(file_path: str) -> list[dict]:
    """
    JS projesinin web_pages.json formatını yükler.
    Format: [{"url": str, "content": str}, ...]

    Temizlik adımları:
      1. "Ana içeriğe atla ☰ ... Anasayfa" navigasyon başlığını kaldır
      2. Tekrarlayan menü linklerini temizle
      3. Çok kısa içerikleri (< MIN_CONTENT_LEN) atla
    """
    import re

    MIN_CONTENT_LEN = 200  # Navigasyon artıklarını eliyor

    path = Path(file_path)
    if not path.exists():
        logger.error("web_pages.json bulunamadı: {}", file_path)
        return []

    with open(path, encoding="utf-8") as f:
        pages = json.load(f)

    if not isinstance(pages, list):
        logger.error("web_pages.json beklenen format: JSON array")
        return []

    docs = []
    skipped = 0
    for page in pages:
        url = page.get("url", "")
        raw = page.get("content", "").strip()
        if not raw:
            skipped += 1
            continue

        # Navigasyon başlığını kaldır: "Ana içeriğe atla ☰ ... Anasayfa"
        raw = re.sub(r"^Ana içeriğe atla\s*☰.*?Anasayfa\s*", "", raw, flags=re.DOTALL)

        # Tekrar eden breadcrumb ve menü satırlarını kaldır
        raw = re.sub(r"(Anasayfa\s*›?\s*){2,}", "", raw)
        raw = re.sub(r"\s{3,}", " ", raw)

        cleaned = _clean_text(raw)

        if len(cleaned) < MIN_CONTENT_LEN:
            skipped += 1
            continue

        # URL'den kategori tahmini
        category = _url_to_category(url)

        meta = {
            "source": url,
            "source_type": "web_crawl",
            "category": category,
            "original_url": url,
        }
        docs.append({"text": cleaned, "metadata": meta})

    logger.info(
        "web_pages.json: {} sayfa yüklendi, {} atlandı (çok kısa/boş)",
        len(docs), skipped,
    )
    return docs


def _url_to_category(url: str) -> str:
    """URL yolundan kategori tahmini."""
    path = url.lower()
    if "iletisim" in path:
        return "GENEL"
    if "ulasim" in path or "otopark" in path or "kentkart" in path:
        return "ULASIM_TRAFIK"
    if "imar" in path or "yapi" in path or "ruhsat" in path:
        return "IMAR_INSAAT"
    if "nikah" in path or "evlend" in path:
        return "NIKAH_EVLENDIRME"
    if "sosyal" in path or "yardim" in path:
        return "SOSYAL_HIZMETLER"
    if "kultur" in path or "turizm" in path or "kirkpinar" in path:
        return "KULTUR_SANAT"
    if "saglik" in path or "eczane" in path:
        return "SAGLIK"
    if "cevre" in path or "temizlik" in path:
        return "CEVRE_TEMIZLIK"
    if "vergi" in path or "mali" in path:
        return "MALI_ISLER"
    return "GENEL"


def load_from_url(url: str, metadata: dict[str, Any] | None = None) -> list[dict]:
    """Basit web sayfası içeriği çekme."""
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "EdirneBot/2.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Script ve style etiketlerini kaldır
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = _clean_text(text)
        meta = {"source": url, "source_type": "web", **(metadata or {})}
        return [{"text": text, "metadata": meta}]
    except Exception as exc:
        logger.error("URL yükleme hatası ({}): {}", url, exc)
        return []


def load_file(file_path: str, metadata: dict[str, Any] | None = None) -> list[dict]:
    """Uzantıya göre uygun yükleyiciyi seç."""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".json":
        return load_from_json_sss(file_path)
    if ext == ".txt":
        return load_from_txt_file(file_path, metadata)
    if ext == ".docx":
        return load_from_docx(file_path, metadata)
    logger.warning("Desteklenmeyen dosya uzantısı: {}", ext)
    return []


def _map_category(raw: str) -> str:
    """sss.json kategori etiketlerini sistem kategorilerine dönüştür."""
    mapping = {
        # su
        "su_faturasi": "SU_ALTYAPI",
        "su_hizmetleri": "SU_ALTYAPI",
        "su": "SU_ALTYAPI",
        # ulaşım
        "ulasim": "ULASIM_TRAFIK",
        "otopark": "ULASIM_TRAFIK",
        # imar
        "imar": "IMAR_INSAAT",
        # sosyal
        "sosyal_yardim": "SOSYAL_HIZMETLER",
        # kültür
        "kultur_sanat": "KULTUR_SANAT",
        "kultur_turizm": "KULTUR_SANAT",
        "kultur": "KULTUR_SANAT",
        # çevre
        "sikayet_talep": "CEVRE_TEMIZLIK",
        "cevre": "CEVRE_TEMIZLIK",
        # park
        "park": "PARK_BAHCE",
        # nikah
        "nikah": "NIKAH_EVLENDIRME",
        # vergi / mali
        "vergi": "MALI_ISLER",
        "mali": "MALI_ISLER",
        # sağlık
        "saglik": "SAGLIK",
        "saglik_yonlendirme": "SAGLIK",
        # genel
        "genel": "GENEL",
        "iletisim": "GENEL",
        "duyurular": "GENEL",
        "bagis": "GENEL",
        "cenaze": "GENEL",
        "erisilebilirlik": "GENEL",
        "fallback": "GENEL",
    }
    return mapping.get(raw.lower(), "GENEL")
