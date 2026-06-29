# E.C.H.O. System Architecture

To closely match the visual density of your reference image, this diagram collapses the dozens of internal micro-steps into **four core processing entities**. The technical details are now shifted onto the **connection arrows**, explaining exactly what data is being passed between the backend, the AI microservices, and the databases.

## Architecture Diagram

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
