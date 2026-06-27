from pathlib import Path
import numpy as np

# -----------------------------
# BASE PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent        # where the .py file lives
SCRIPTS_ROOT = BASE_DIR / "scripts//Scripts"      # input root
FINAL_ROOT   = BASE_DIR / "Final Scripts"         # output root

FINAL_ROOT.mkdir(parents=True, exist_ok=True)

def wc(text: str) -> int:
    return len(text.split())

word_counts = []
file_map = {}

# -----------------------------
# WALK THROUGH ALL SUBFOLDERS
# -----------------------------
for subdir in SCRIPTS_ROOT.iterdir():
    if not subdir.is_dir():
        continue

    # mirror folder structure
    out_subdir = FINAL_ROOT / subdir.name
    out_subdir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(subdir.glob("*.txt"))

    for f in txt_files:
        lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()

        script_started = False
        script_lines = []

        for line in lines:
            # start AFTER Type:
            if not script_started:
                if line.strip().startswith("Type:"):
                    script_started = True
                continue

            # stop at first separator
            if line.strip().startswith("="):
                break

            script_lines.append(line)

        script_text = "\n".join(script_lines).strip()
        words = wc(script_text)

        # save extracted script
        out_path = out_subdir / f.name
        out_path.write_text(script_text + "\n", encoding="utf-8")

        word_counts.append(words)
        file_map[f"{subdir.name}/{f.name}"] = words

# -----------------------------
# STATS
# -----------------------------
word_counts = np.array(word_counts)

print(f"Processed {len(word_counts)} files.")
if len(word_counts) > 0:
    print(
        f"Word count stats → "
        f"min={word_counts.min()}, "
        f"max={word_counts.max()}, "
        f"mean={word_counts.mean():.1f}"
    )

print(f"Final scripts saved to: {FINAL_ROOT}")