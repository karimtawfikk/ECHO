import pandas as pd
from pathlib import Path

excel_path = Path(r"data\video_generation\raw\pharaos_info.xlsx")
txt_folder = Path(r"data\video_generation\outputs\pharaohs_descriptions")

df = pd.read_excel(excel_path)
excel_names = set(df["Name"].astype(str).str.strip())  

txt_names = {f.stem.strip() for f in txt_folder.glob("*.txt")}

missing_txts = sorted(excel_names - txt_names)

print("Names in Excel without matching text files:")
for name in missing_txts:
    print(name)