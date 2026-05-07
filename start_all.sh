#!/bin/bash

# Force the system to use the bundled CUDA/CuDNN libraries from your venv
export LD_LIBRARY_PATH=/workspace/venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/workspace/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/workspace/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# 1. Main Recognition & Orchestrator (Port 8010)
/workspace/venv/bin/uvicorn src.app.main:app --host 0.0.0.0 --port 8010 &

# 2. Chatbot API (Port 8000)
/workspace/venv/bin/uvicorn src.chatbot_api.main:app --host 0.0.0.0 --port 8000 &

# 3. Video Generation API (Port 8005)
/workspace/venv/bin/uvicorn src.video_generation_api.main:app --host 0.0.0.0 --port 8005 &

echo "E.C.H.O Services are starting up on ports 8010, 8000, and 8005!"
