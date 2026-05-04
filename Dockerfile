# Use the RunPod PyTorch base image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.chatbot.txt .
COPY requirements.video.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.chatbot.txt && \
    pip install --no-cache-dir -r requirements.video.txt

# Copy scripts first for preloading
COPY scripts/preload_video_models.py scripts/preload_video_models.py

# Preload Chatbot Models (Embedding)
ARG HF_TOKEN
RUN test -n "$HF_TOKEN"
RUN HF_TOKEN="$HF_TOKEN" python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device='cpu', token='${HF_TOKEN}', tokenizer_kwargs={'padding_side': 'left'})"

# Preload Video Models (CLIP)
RUN HF_TOKEN="$HF_TOKEN" python scripts/preload_video_models.py

# Copy the rest of the source code
COPY src/chatbot_api src/chatbot_api
COPY src/video_generation_api src/video_generation_api
COPY src/models src/models
COPY src/db src/db
COPY src/resources src/resources
COPY scripts/runpod_start.sh scripts/runpod_start.sh

# Ensure start script is executable
RUN chmod +x scripts/runpod_start.sh

# Expose both ports
EXPOSE 8000
EXPOSE 8001

# Use the startup script to run both APIs
CMD ["./scripts/runpod_start.sh"]
