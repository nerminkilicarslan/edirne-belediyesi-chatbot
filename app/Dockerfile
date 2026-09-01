# ── Build Stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch (Docker'da MPS/CUDA yoktur)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt


# ── Runtime Stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Site-packages'ları builder'dan kopyala
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Model ağırlıklarını build sırasında indir → container startup hızlanır
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', device='cpu')"

# Uygulama kodu
COPY . .

# Veri ve log dizinleri
RUN mkdir -p data/chroma_db data/documents data/faqs logs

ENV EMBEDDING_DEVICE=cpu \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_JSON=true \
    ENVIRONMENT=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:8000/api/v1/health | python3 -c \
    "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status') in ('ok','degraded') else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
