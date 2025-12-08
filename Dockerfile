# Базовый образ Python (совместим с pinned зависимостями)
FROM python:3.11-slim

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Кэш HuggingFace внутри контейнера
ENV HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

# Копирование файла с зависимостями
COPY requirements.txt .

# Установка всех необходимых пакетов
RUN pip install --no-cache-dir -r requirements.txt

# Предзагрузка моделей эмбеддинга и реранкера (нужен интернет на этапе сборки)
RUN python - <<'PY'
import torch
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

embedding_model = "Qwen/Qwen3-Embedding-0.6B"
reranker_model = "BAAI/bge-reranker-v2-m3"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for model predownload: {device}")

print(f"Downloading embedding model: {embedding_model}")
_ = SentenceTransformer(embedding_model, device=device).encode(["hello"], normalize_embeddings=True)

print(f"Downloading reranker model: {reranker_model}")
_ = FlagReranker(reranker_model, use_fp16=False, device=device).compute_score([["hello", "world"]])

print("Model artifacts cached in /root/.cache/huggingface")
PY

# Копирование всех файлов проекта в контейнер
COPY . .

# Команда запуска вашего решения
CMD ["python", "main.py"]
