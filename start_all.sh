#!/bin/bash



# 1. Main Recognition & Orchestrator (Port 8010)
/workspace/venv/bin/uvicorn src.app.main:app --host 0.0.0.0 --port 8010 &

# 2. Chatbot API (Port 8000)
/workspace/venv/bin/uvicorn src.chatbot_api.main:app --host 0.0.0.0 --port 8000 &

# 3. Video Generation API (Port 8005)
/workspace/venv/bin/uvicorn src.video_generation_api.main:app --host 0.0.0.0 --port 8005 &

# 4. Hieroglyph Detection API (Port 8003)
/workspace/venv/bin/uvicorn src.hieroglyph_api.main:app --host 0.0.0.0 --port 8003 &

# 5. Recognition API (Port 8002)
/workspace/venv/bin/uvicorn src.recognition_api.main:app --host 0.0.0.0 --port 8002 &

echo "E.C.H.O Services are starting up on ports 8010, 8000, 8005, 8003, and 8002!"
