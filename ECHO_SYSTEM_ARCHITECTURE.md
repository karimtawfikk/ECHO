# E.C.H.O. System Architecture

To closely match the visual density of your reference image, this diagram collapses the dozens of internal micro-steps into **four core processing entities**. The technical details are now shifted onto the **connection arrows**, explaining exactly what data is being passed between the backend, the AI microservices, and the databases.

## Architecture Diagram

```mermaid
graph LR
    %% Aesthetics
    classDef actor fill:none,stroke:none,color:#000
    classDef ui fill:#0f172a,stroke:#3b82f6,stroke-width:2px,color:#fff
    classDef gw fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#fff
    classDef pipe fill:#1e293b,stroke:#8b5cf6,stroke-width:2px,color:#fff
    classDef db fill:#0f172a,stroke:#0ea5e9,stroke-width:2px,color:#fff
    classDef ext fill:#000,stroke:#f59e0b,stroke-width:2px,color:#fff
    classDef cloud fill:none,stroke:none,color:#333

    Actor(["👤 Actor"]):::actor

    %% Compact Core Nodes
    FE["⚛️ Next.js<br>(Frontend)"]:::ui
    BE["🚀 FastAPI<br>(Gateway)"]:::gw

    %% The 4 parallel AI pipelines
    Rec["Recognition Pipeline<br>(Binary & Identity Models)"]:::pipe
    Hiero["Hieroglyph Pipeline<br>(YOLOv8 & M2M-100)"]:::pipe
    Chat["Chatbot Pipeline<br>(RAG & TTS Embeddings)"]:::pipe
    Media["Media Adaptor<br>(Video Compilation)"]:::pipe

    %% Destinations
    DB[("🐘 PostgreSQL<br>+ pgvector")]:::db
    Groq["⚡ Groq API<br>(LLM Inference)"]:::ext
    R2[("☁️ Cloudflare R2<br>(Object Storage)")]:::db

    %% Deployment Anchors
    RunPod["🚀 RunPod (GPU Cloud)"]:::cloud

    %% --- Connections ---
    Actor -->|"Interacts"| FE
    
    %% Frontend / Backend Communication
    FE -- "API Request" --> BE
    BE -- "Data Result" --> FE

    %% Backend routing to Microservices
    BE -- "Image Data" --> Rec
    BE -- "Artifact Image" --> Hiero
    BE -- "User Query" --> Chat
    BE -- "Video Request" --> Media

    %% Microservices returning requested data to Backend
    Rec -- "Detected Classes" --> BE
    Hiero -- "English Text" --> BE
    Chat -- "Audio Stream" --> BE
    Media -- "MP4 URL" --> BE

    %% AI Models to Databases & External APIs
    Rec -- "Metadata" --> DB
    Chat -- "Conversations" --> DB
    Media -- "Image URLs" --> DB
    
    %% Groq Communication
    Chat -- "Enhanced Prompt" --> Groq
    Groq -- "LLM Response" --> Chat
    
    %% Storage
    Media -- "Fetch Source Images" --> R2
    R2 -- "Image Files" --> Media
    Media -- "Video Output" --> R2

    %% Backend DB Queries
    BE -- "Database Query" --> DB
    DB -- "Query Data" --> BE

    %% Alignment rules
    FE -.-> RunPod
    BE -.-> RunPod

    %% Force straight, non-curvy lines unconditionally
    linkStyle default interpolate step
```
