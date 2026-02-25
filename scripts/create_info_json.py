import os
import json
import pandas as pd
from pathlib import Path

# Paths
csv_path = Path("data/video_generation/raw/pharahos_info.csv")
descriptions_folder = Path("data/video_generation/outputs/pharahos_descriptions")
output_json_path = Path("data/video_generation/outputs/pharahos.json")

df = pd.read_csv(csv_path, encoding="cp1252")

landmarks_list = []

for idx, row in df.iterrows():
    name = row["name"]
    period = row["Period"]
    dynasty = row["Dynasty"]

    filename = name + ".txt"
    file_path = descriptions_folder / filename

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            description = f.read().strip()
    else:
        print("Missing description for:", name)
        description = ""

    landmark_entry = {
        "id": idx,
        "name": name,
        "period": period,
        "dynasty": dynasty,
        "description": description,
    }

    landmarks_list.append(landmark_entry)


with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(landmarks_list, f, indent=4, ensure_ascii=False)