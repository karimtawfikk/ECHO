import boto3
from pathlib import Path
from tqdm import tqdm
import os
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
SECRET_KEY = os.getenv("R2_SECRET_KEY")
BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

session = boto3.session.Session()
client = session.client(
    "s3",
    region_name="auto",
    endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)
data_folder = Path("data")

files = [f for f in data_folder.rglob("*") if f.is_file()]

print(f"Found {len(files)} files to upload")

for file_path in tqdm(files, desc="Uploading files"):
    relative = file_path.relative_to(data_folder.parent)
    key = str(relative).replace('\\', '/')
    # ────────────────────────────────────────────────────────────────

    client.upload_file(str(file_path), BUCKET_NAME, key)

print("Upload finished.")