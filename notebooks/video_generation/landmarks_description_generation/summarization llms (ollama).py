import requests
from pathlib import Path

INPUT_DIR = Path("docs")
OUTPUT_DIR = Path("outputs")

'''MODELS = {
    "mistral_7b": {
        "ollama": "mistral:latest",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "type": "causal",
    },
    "qwen2_7b": {
        "ollama": "qwen2:7b",
        "hf_id": "Qwen/Qwen2-7B-Instruct",
        "type": "causal",
    },
    "llama3_8b": {
        "ollama": "llama3:8b",
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "type": "causal",
    },
}'''


MODELS = {
     "llama3_2_1b": {
        "ollama": "llama3.2:1b",  # Already downloaded (1.3 GB)
        "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
        "type": "causal",
    }
}

"""SYSTEM_PROMPT = (
    "You are a precise summarizer for Egyptian landmarks. "
    "Extract ONLY key facts: names, dates, locations, dynasties, dimensions, materials, creators, historical events. "
    "Output requirements: "
    "(1) Start immediately with the first fact - NO preamble like 'Summary:' or 'Here is...'; "
    "(2) Use ONLY plain prose sentences separated by periods; "
    "(3) Don't use any bullet points, dashes, asterisks, numbering, or markdown; "
    "(4) NO extra text, explanations, or commentary; "
    "(5) Be concise but retain all key facts. "
)"""

SYSTEM_PROMPT = (
    "SUMMARIZE IN PLAIN SENTENCES ONLY. NO BULLETS. NO DASHES. NO MARKDOWN. "
    "START IMMEDIATELY WITH THE FIRST FACT. "
    "KEY FACTS TO INCLUDE: name, date, location, dynasty, dimensions, materials, creators, historical events. "
    "OUTPUT RULES: " \
    "1. COMPLETE YOUR FINAL SENTENCE BEFORE STOPPING - NEVER END MID-SENTENCE." \
    "2. NO extra text, explanations, or commentary - ONLY the facts. "
    "3. ONLY full sentences ending with periods. "
    "4. NO symbols (- * # •). "
    "5. NO labels like 'Name:' or 'Location:'. "
    "6. NO line breaks between facts. "
    "7. ONE continuous paragraph."
    ""
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_TOKENS = 350
TEMPERATURE = 0.2


def wc(s: str) -> int:
    return len(s.split())

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def safe_model_dir(model: str) -> str:
    # "qwen2:7b" -> "qwen2_7b"
    return model.replace(":", "_")


def ollama_summarize(model: str, text: str) -> str:
    prompt = f"""{SYSTEM_PROMPT}

    Summarize the following document:

    {text}
    
    SUMMARY:
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=600)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {INPUT_DIR.resolve()}")
        return

    print(f"Found {len(files)} .txt files")

  
    for model_name, cfg in MODELS.items():
        model_dir = OUTPUT_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)


        print("\n"+ "=" * 50)
        print(f"Model      : {model_name}")
        print(f"Ollama Tag : {cfg['ollama']}")
        print(f"HF id      : {cfg['hf_id']}")
        print(f"Type       : {cfg['type']}")
        print("=" * 50)



        for path in files:
            text = read_txt(path)
            if not text:
                print(f"Skipping empty: {path.name}")
                continue

            try:
                before = wc(text)
                print(f"[Text Size] {path.name} -> {before} words")

                summary = ollama_summarize(cfg["ollama"], text)

                after = wc(summary)
                print(f"[Summary Size] {path.name} -> {after} words")

                out_path = model_dir / path.name
                out_path.write_text(summary + "\n", encoding="utf-8")

                print(f"[Saved] {out_path}")

            except Exception as e:
                print(f"[ERROR] {model_name} on {path.name}: {e}")


if __name__ == "__main__":
    main()
