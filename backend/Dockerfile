FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 / sentence-transformers builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch wheel first: the default PyPI build bundles CUDA libraries
# (multi-GB) this project never uses, since embeddings and reranking both run
# on CPU. Installing it from PyTorch's CPU index before requirements.txt
# means sentence-transformers' own torch dependency is already satisfied and
# never pulls the GPU build in.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
