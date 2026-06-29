# E.C.H.O. System Architecture

To closely match the visual density of your reference image, this diagram collapses the dozens of internal micro-steps into **four core processing entities**. The technical details are now shifted onto the **connection arrows**, explaining exactly what data is being passed between the backend, the AI microservices, and the databases.

## Architecture Diagram

```mermaid
graph LR
    %% Aesthetics
    classDef actor fill:none,stroke:none,color:#000
    classDef ui fill:#bfdbfe,stroke:#3b82f6,stroke-width:2px,color:#000
    classDef gw fill:#000,stroke:#000,stroke-width:2px,color:#fff
    classDef pipe fill:#bfdbfe,stroke:#3b82f6,stroke-width:2px,color:#000
    classDef db fill:#bfdbfe,stroke:#3b82f6,stroke-width:2px,color:#000
    classDef ext fill:none,stroke:none,color:#000

    Actor(["👤<br/>Actor"]):::actor

    subgraph Vercel [▲ Vercel]
        FE(("N")):::ui
    end
    style Vercel fill:#e0f2fe,stroke:#bae6fd,stroke-width:2px,color:#000,rx:10,ry:10

    BE["⚡"]:::gw

    subgraph Runpod [📦 runpod]
        Chat["Chatbot Pipeline"]:::pipe
        Hiero["Hieroglyphics<br/>Translation<br/>Pipeline"]:::pipe
        Video["Video Generation<br/>Pipeline"]:::pipe
        Rec["Entity Recognition<br/>Pipeline"]:::pipe
    end
    style Runpod fill:#e0f2fe,stroke:#bae6fd,stroke-width:2px,color:#000,rx:10,ry:10

    Groq["⚡ groq"]:::ext
    R2("☁️ Cloudflare R2 Storage"):::ext
    DB[("🐘")]:::db

    %% Connections
    Actor <--> FE
    
    FE -->|"API<br/>Request"| BE
    BE -->|"Data<br/>Result"| FE

    BE -->|"User Prompt"| Chat
    Chat -->|"Audio/Text<br/>Response"| BE

    BE -->|"Inscription Image"| Hiero
    Hiero -->|"English Text"| BE

    BE -->|"Video Request"| Video
    Video -->|"MP4 Video"| BE

    BE -->|"Entity<br/>Image"| Rec
    Rec -->|"Entity<br/>Description"| BE

    Chat -->|"Enhanced Prompt + Context"| Groq
    Groq -->|"LLM Response"| Chat

    Chat -->|"Query Entity<br/>Context"| DB
    DB -->|"Entity<br/>Context"| Chat

    Video -->|"Fetch Images"| R2
    R2 -->|"Image Files"| Video

    Video -->|"Query Images via Text Embeddings"| DB
    DB -->|"Images URL"| Video

    Rec -->|"Query<br/>Metadata"| DB
    DB -->|"Entity<br/>Metadata"| Rec
```
