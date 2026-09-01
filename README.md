# Edirne Belediyesi AI Chatbot

Edirne Belediyesi vatandaş hizmetleri için geliştirilmiş, tamamen yerel çalışan RAG (Retrieval-Augmented Generation) tabanlı yapay zeka chatbotudur. Belediye hizmetleri, başvuru süreçleri, iletişim bilgileri ve sık sorulan sorulara Türkçe olarak yanıt verir.

## Özellikler

- Hibrit niyet ve kategori sınıflandırması (anahtar kelime + embedding)
- Akıllı yönlendirme: acil durumlar, kapsam dışı sorular ve doğrudan cevaplar otomatik işlenir
- Semantik vektör araması (ChromaDB) + anahtar kelime araması (BM25) hibrit retrieval
- Hallüsinasyon önleyici sistem promptları
- React tabanlı belediye temalı kullanıcı arayüzü
- Tüm veriler ve modeller yerel çalışır, dışarıya veri gönderilmez

## Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| Backend | Python 3.11 + FastAPI |
| LLM | Ollama + Qwen2.5:7b |
| Embedding | BAAI/bge-m3 |
| Vektör DB | ChromaDB |
| Arama | BM25 (rank-bm25) |
| Frontend | React 19 + Vite |
| Konteyner | Docker Compose |

## Kurulum

### Gereksinimler

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) kurulu ve çalışıyor olmalı

### 1. Ollama model kurulumu

```bash
ollama pull qwen2.5:7b
```

### 2. Python bağımlılıkları

```bash
cd app
python -m venv ../.venv
source ../.venv/bin/activate   # Windows: ..\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Veri yükleme

```bash
# Sıfırdan yükle
python scripts/ingest.py --source data/sss.json --type sss --reset
python scripts/ingest.py --source data/web_pages.json --type web
```

### 4. Sunucuyu başlat

```bash
# Terminal 1 — Backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Frontend
cd frontend && npm install && npm run dev
```

Arayüz: **http://localhost:5173**
API: **http://localhost:8000/api/v1**

## Docker ile Çalıştırma

```bash
docker compose up -d

# İlk çalıştırmada modeli indir
docker exec edirne_ollama ollama pull qwen2.5:7b
```

## Ortam Değişkenleri

`app/.env` dosyasında tanımlanır:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
EMBEDDING_MODEL_NAME=BAAI/bge-m3
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
```

## API Kullanımı

```bash
# Soru sor
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Nikâh için gerekli belgeler nelerdir?"}'

# Sağlık kontrolü
curl http://localhost:8000/api/v1/health
```

## Proje Yapısı

```
edirne-chatbot/
├── app/
│   ├── api/            # HTTP endpoint'leri ve veri modelleri
│   ├── pipeline/       # Niyet/kategori sınıflandırma, yönlendirme, promptlar
│   ├── services/       # Embedding, ChromaDB, BM25, LLM servisleri
│   ├── ingestion/      # Belge yükleme ve indeksleme
│   └── main.py         # Uygulama giriş noktası
├── frontend/           # React kullanıcı arayüzü
├── scripts/            # Veri yükleme araçları
├── data/sss.json       # Sık sorulan sorular ve cevaplar
├── data/web_pages.json # Belediye web sitesi içerikleri
├── data/               # ChromaDB veritabanı
└── docker-compose.yml  # Konteyner yapılandırması
```

## Veri Güncelleme

```bash
python scripts/ingest.py --source data/sss.json --type sss --reset
python scripts/ingest.py --source data/web_pages.json --type web
```
