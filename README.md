# ECHO – AI-Powered Ancient Egypt Explorer

ECHO is a multimodal AI system designed to explore Ancient Egypt through computer vision, natural language processing, and generative AI.

The system allows users to upload images of Egyptian landmarks, statues, or hieroglyphs and receive intelligent recognition, historical explanations, interactive conversations, and generated visual storytelling.

This project was developed as a Graduation Project in Artificial Intelligence.

## Project Overview

Ancient Egyptian history is rich but often difficult to explore interactively. ECHO bridges this gap by combining:

* Image recognition
* Large Language Models
* Embedding-based retrieval
* Video generation
* Hieroglyph translation

The system transforms static historical content into an intelligent interactive experience.

## System Architecture

ECHO uses a modern **Microservices Architecture** to separate lightweight routing/CRUD operations from heavy AI model inferences.

```mermaid
flowchart LR
    %% Aesthetics
    classDef actor fill:none,stroke:none
    classDef ui fill:#bfdbfe,stroke:#3b82f6,stroke-width:2px,color:#000
    classDef gw fill:#1e293b,stroke:#cbd5e1,stroke-width:2px,color:#fff
    classDef pipe fill:#bfdbfe,stroke:#3b82f6,stroke-width:2px,color:#000
    classDef db fill:#bfdbfe,stroke:#3b82f6,stroke-width:2px,color:#000
    classDef ext fill:none,stroke:none

    Actor(["👤<br/>Actor"]):::actor

    subgraph Vercel [▲ Vercel]
        FE(("N")):::ui
    end
    style Vercel fill:#e0f2fe,stroke:#bae6fd,stroke-width:2px,color:#000,rx:10,ry:10

    BE["⚡"]:::gw

    subgraph Runpod [📦 runpod]
        direction TB
        Chat["Chatbot Pipeline"]:::pipe
        Hiero["Hieroglyphics<br/>Translation<br/>Pipeline"]:::pipe
        Video["Video Generation<br/>Pipeline"]:::pipe
        Rec["Entity Recognition<br/>Pipeline"]:::pipe
    end
    style Runpod fill:#e0f2fe,stroke:#bae6fd,stroke-width:2px,color:#000,rx:10,ry:10

    subgraph External [ ]
        direction TB
        Groq["⚡ groq"]:::ext
        R2("☁️ Cloudflare R2 Storage"):::ext
        DB[("🐘 PostgreSQL")]:::db
    end
    style External fill:none,stroke:none,color:#000

    %% Connections
    Actor <--> FE
    
    FE <-->|"API Request<br/>Data Result"| BE

    BE <-->|"User Prompt<br/>Audio/Text Response"| Chat
    BE <-->|"Inscription Image<br/>English Text"| Hiero
    BE <-->|"Video Request<br/>MP4 Video"| Video
    BE <-->|"Entity Image<br/>Entity Description"| Rec

    Chat <-->|"Enhanced Prompt + Context<br/>LLM Response"| Groq
    
    Chat <-->|"Query Entity Context<br/>Entity Context"| DB
    Video <-->|"Fetch Images<br/>Image Files"| R2
    Video <-->|"Query Images via Text Embeddings<br/>Images URL"| DB
    Rec <-->|"Query Metadata<br/>Entity Metadata"| DB
```

### Main Components:
* **API Gateway (`src/app`)**: Built with FastAPI. Handles frontend authentication, database CRUD operations, and forwards AI-heavy requests to the dedicated microservices.
* **Recognition API (`src/recognition_api`)**: Dedicated microservice for Landmark and Statue embedding extraction and recognition.
* **Chatbot API (`src/chatbot_api`)**: Dedicated microservice for RAG, Groq LLM streaming, and Text-To-Speech generation.
* **Video Generation API (`src/video_generation_api`)**: Dedicated pipeline for automated historical video compilation.
* **Hieroglyph Detection API (`src/hieroglyph_api`)**: Dedicated microservice to detect, classify, and translate Ancient Egyptian hieroglyphs.
* **Databases**: PostgreSQL (Relational) + ChromaDB (Vector).

## Project Structure

```text
ECHO/
|-- frontend/                  # Next.js Application
|-- src/                       # Microservices Workspace
|   |-- app/                   # 🚀 API Gateway & Orchestrator
|   |-- recognition_api/       # 🔍 Landmark & Statue Recognition Microservice
|   |-- chatbot_api/           # 🤖 Chatbot & Voice Microservice
|   |-- video_generation_api/  # 🎥 Video Compiler Microservice
|   |-- hieroglyph_api/        # 𓊹 Hieroglyph Translation Microservice
|   |-- db/                    # PostgreSQL Models & Sessions
|-- infra/                     # Code for Dockerizing services
|-- alembic/                   # Database Migrations
|-- docker-compose.yml         # Docker Compose configuration for AI modules
|-- start_all.sh               # Shell script to start all services natively
|-- README.md
```

## Installation & Running

Because every feature in ECHO is containerized, you can effortlessly run all backend AI modules using Docker.

### 1. Clone the repository
```bash
git clone https://github.com/karimtawfikk/ECHO.git
cd ECHO
```

### 2. Configure Environment
Ensure you have `.env` properly configured in the root directory based on `.env.example`, including your PostgreSQL connection string, Hugging Face Token, and all required API keys.

### 3. Running the AI Microservices (Docker)
All AI microservices (Recognition, Chatbot, Video, Hieroglyph) can be spun up at once:
```bash
docker compose up --build -d
```
*Note: Ensure your Docker host has NVIDIA Container Toolkit installed to utilize GPUs.*

### 4. Running the API Gateway (Native)
To run the main server orchestrator natively:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements/main.txt -r requirements/chatbot.txt -r requirements/video.txt -r requirements/hieroglyph.txt -r requirements/recognition.txt
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8010
```
*(Alternatively, you can start all services natively by running `./start_all.sh` on Linux/Mac)*

### 5. Running the Frontend
Start the Next.js application:
```bash
cd frontend
npm install
npm run dev
```

---

## Core Modules

### 1. Landmark & Statue Recognition
Users upload an image of a historical landmark or statue. The system exacts visual embeddings and retrieves structured metadata from PostgreSQL.

### 2. Historical Video Generation
After recognition, the system generates a structured historical narration and scene descriptions derived strictly from verified data, converting historical text into short educational videos.

### 3. Conversational Historical Chatbot
Users can interact with the recognized entity through a conversational interface using RAG. The system grounds responses in stored metadata and maintains historical accuracy.

### 4. Hieroglyph Translation
Users upload an image containing hieroglyphs. The system detects symbols, classifies them, and generates structured translations via LLM reasoning.

## Database Design

The system contains structured entities such as Landmarks, Pharaohs, Builders, Dynasties, and Historical Events. Relationships are modeled using SQLAlchemy ORM and version-controlled using Alembic migrations.

## AI Techniques Used
* Multimodal Embeddings
* Similarity Search (ChromaDB)
* Retrieval-Augmented Generation (RAG)
* Diffusion Models
* Grounded Response Control
* Object Detection & Classification

## Academic Context
This project was developed as part of an Artificial Intelligence graduation project to design a scalable AI system combining computer vision, NLP, and database systems.
