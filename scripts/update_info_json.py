import json
import pandas as pd
from pathlib import Path

json_path = Path("data/data/video_generation/outputs/pharaohs.json")
excel_path = Path("data/data/video_generation/raw/pharaohs_with_type.xlsx")
output_json_path = Path("data/data/video_generation/outputs/pharaohs.json")   # overwrite existing JSON

def clean_val(val):
    if pd.isna(val) or str(val).strip() == "":
        return None
    return str(val).strip()

# existing JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Excel with column Type
df = pd.read_excel(excel_path)

# Build mapping: Name -> Type
type_map = {
    clean_val(row["Name"]): clean_val(row["Type"])
    for _, row in df.iterrows()
    if clean_val(row["Name"]) is not None
}

# Add type to JSON entries
unmatched = []

for item in data:
    name = clean_val(item.get("name"))
    item["type"] = type_map.get(name)

    if item["type"] is None:
        unmatched.append(name)

# Save updated JSON
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Updated JSON saved to: {output_json_path}")
print(f"Total records: {len(data)}")
print(f"Unmatched names: {len(unmatched)}")

if unmatched:
    print("\nNames not found in Excel:")
    for name in unmatched:
        print("-", name)