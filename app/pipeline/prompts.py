"""
Prompt şablonları — belediye chatbot senaryosuna özel.

Tasarım ilkeleri:
  - System prompt: LLM'e rolünü, sınırlarını ve dil stilini açıkla
  - User prompt: bağlamı (retrieved docs) + soruyu içerir
  - Hallucination önleme: bağlamda olmayan bilgi üretme yasağı
  - Türkçe: resmi ama anlaşılır dil, teknik jargondan kaçın
"""
from __future__ import annotations

from app.api.models import DocumentSource


BASE_SYSTEM_PROMPT = """\
Rol: Edirne Belediyesi resmi AI asistanı.

ZORUNLU KURALLAR — HİÇBİRİNİ ATLAMA:
1. Cevabını YALNIZCA aşağıdaki Bağlam bölümündeki bilgilerden üret.
2. Bağlamda olmayan adım, belge, URL, birim adı veya detay EKLEME — uydurma.
3. Bağlamda link/URL varsa yaz; yoksa hiç URL yazma.
4. Bağlam kısa veya belirsizse: sadece bağlamdaki cümleyi aktar, "Detaylar için 153 Beyaz Masa'yı arayabilirsiniz" ekle, başka hiçbir şey ekleme.
5. Kısa ve öz cevap ver.\
"""

FOCUSED_SYSTEM_PROMPT = """\
Rol: Edirne Belediyesi resmi AI asistanı.

ZORUNLU KURALLAR — HİÇBİRİNİ ATLAMA:
1. Cevabını YALNIZCA verilen Bağlam bölümündeki bilgilerden üret.
2. Bağlamda olmayan adım, belge, URL veya detay EKLEME — uydurma.
3. Bağlamda link/URL varsa yaz; yoksa hiç URL yazma.
4. Bağlam yalnızca kısa bir yönlendirme içeriyorsa: o yönlendirmeyi aktar + "Detaylar için 153 Beyaz Masa'yı (0284 225 1153) arayabilirsiniz" de.
5. Kendi bilginden hiçbir şey ekleme.\
"""

COMPLAINT_SYSTEM_PROMPT = """\
Rol: Edirne Belediyesi şikayet/bildirim asistanı.

ZORUNLU KURALLAR:
1. Cevabını YALNIZCA verilen Bağlam bölümündeki bilgilerden üret.
2. Bağlamda olmayan birim, adres veya telefon EKLEME.
3. Şikayeti Beyaz Masa (153) veya bağlamdaki kanala yönlendir.
4. Kısa ve empatik ol.\
"""

APPLICATION_SYSTEM_PROMPT = """\
Rol: Edirne Belediyesi başvuru süreçleri asistanı.

ZORUNLU KURALLAR — HİÇBİRİNİ ATLAMA:
1. Cevabını YALNIZCA verilen Bağlam bölümündeki bilgilerden üret.
2. Bağlamda belge listesi varsa yaz; yoksa ekleme.
3. Bağlamda başvuru birimi/link varsa ver; yoksa URL veya adres YAZMA.
4. Bağlam kısa/belirsizse: bağlamdaki bilgiyi aktar + "Detaylar için 153 numaralı Beyaz Masa'yı arayabilirsiniz" de.\
"""


def select_system_prompt(intent_label: str) -> str:
    """Intent'e göre uygun system prompt seç."""
    mapping = {
        "SIKAYET_BILDIRIM": COMPLAINT_SYSTEM_PROMPT,
        "BASVURU_ISLEM": APPLICATION_SYSTEM_PROMPT,
    }
    return mapping.get(intent_label, FOCUSED_SYSTEM_PROMPT)


def build_rag_user_message(
    query: str,
    docs: list[DocumentSource],
    intent_label: str = "",
    category_label: str = "",
) -> str:
    """Retrieval dökümanlarını bağlam olarak içeren kullanıcı mesajı."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Bilinmeyen Kaynak")
        context_parts.append(f"[Kaynak {i}: {source}]\n{doc.content}")

    context_str = "\n\n".join(context_parts) if context_parts else "İlgili bilgi bulunamadı."

    meta_line = ""
    if intent_label or category_label:
        meta_line = f"[Niyet: {intent_label} | Kategori: {category_label}]\n\n"

    return (
        f"{meta_line}"
        f"Bağlam:\n{context_str}\n\n"
        f"Kullanıcı Sorusu: {query}"
    )
