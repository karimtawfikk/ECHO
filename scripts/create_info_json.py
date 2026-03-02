import os
import json
import pandas as pd
from pathlib import Path

csv_path = Path("data/video_generation/raw/pharahos_info.csv")
descriptions_folder = Path("data/video_generation/outputs/pharahos_descriptions")
output_json_path = Path("data/video_generation/outputs/pharahos.json")

df = pd.read_csv(csv_path, encoding="cp1252")

landmarks_list = []

def clean_val(val):
    """Returns None if the value is NaN (empty in CSV) or empty string."""
    if pd.isna(val) or str(val).strip() == "":
        return None
    return val

for idx, row in df.iterrows():
    name = row["Name"]
    period = clean_val(row["Period"])
    dynasty = clean_val(row["Dynasty"])
    
    filename = f"{name}.txt"
    file_path = descriptions_folder / filename

    description = None
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content:
            description = content
    
    composite_entity = clean_val(row.get("Composite_Entity", None))
    landmark_entry = {
        "id": idx,
        "name": name,
        "period": period,
        "dynasty": dynasty,
        "description": description,
        "composite_entity": composite_entity
    }

    landmarks_list.append(landmark_entry)

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(landmarks_list, f, indent=4, ensure_ascii=False)

