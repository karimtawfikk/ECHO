import boto3
from pathlib import Path
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Config ────────────────────────────────────────
ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
SECRET_KEY = os.getenv("R2_SECRET_KEY")
BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
LOCAL_ROOT = Path("data")          
REMOTE_PREFIX = "data"             
# ───────────────────────────────────────────────────

# Init R2 client
session = boto3.session.Session()
client = session.client(
    "s3",
    region_name="auto",
    endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

# Gather files (skip hidden files/dirs)
files = [
    f for f in LOCAL_ROOT.rglob("*")
    if f.is_file() and not any(part.startswith(".") for part in f.parts)
]

print(f"✓ Found {len(files)} files to upload")

# Optional: create folder marker for UI visibility
if REMOTE_PREFIX:
    client.put_object(Bucket=BUCKET_NAME, Key=f"{REMOTE_PREFIX}/")

# Upload loop
for file_path in tqdm(files, desc="Uploading"):
    # Build relative path *inside* LOCAL_ROOT
    relative = file_path.relative_to(LOCAL_ROOT)
    
    # Construct S3 key: e.g., "data/subfolder/file.jpg"
    key_parts = [REMOTE_PREFIX] if REMOTE_PREFIX else []
    key_parts += [str(part) for part in relative.parts]
    key = "/".join(key_parts).replace("\\", "/")
    
    try:
        client.upload_file(str(file_path), BUCKET_NAME, key)
    except Exception as e:
        print(f"✗ Failed: {file_path.name} → {e}")

print("✓ Upload complete!")