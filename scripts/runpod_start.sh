#!/bin/bash

# Exit on any error
set -e

echo "[runpod_start] Starting Chatbot API on port 8000..."
uvicorn src.chatbot_api.main:app --host 0.0.0.0 --port 8000 &

echo "[runpod_start] Starting Video API on port 8001..."
uvicorn src.video_generation_api.main:app --host 0.0.0.0 --port 8001 &

echo "[runpod_start] Both services are starting. Keeping container alive..."
# Wait for all background processes to finish
wait
