import boto3
from pathlib import Path
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID  = os.getenv("R2_ACCOUNT_ID")
ACCESS_KEY  = os.getenv("R2_ACCESS_KEY")
SECRET_KEY  = os.getenv("R2_SECRET_KEY")
BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

if not all([ACCOUNT_ID, ACCESS_KEY, SECRET_KEY, BUCKET_NAME]):
    print("Missing one or more R2 credentials in .env")
    exit(1)

session = boto3.session.Session()
client = session.client(
    "s3",
    region_name="auto",
    endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

data_folder = Path("data").resolve()          # .resolve() → avoids relative path surprises

files = [p for p in data_folder.rglob("*") if p.is_file()]

if not files:
    print("No files found in", data_folder)
    exit(0)

print(f"Found {len(files):,} files. Starting upload...\n")

for file_path in tqdm(files, desc="Uploading", unit="file"):
    key = str(file_path.relative_to(data_folder.parent))
    try:
        client.upload_file(str(file_path), BUCKET_NAME, key)
    except Exception as e:
        print(f"Failed → {key}\n   {type(e).__name__}: {e}")

print("\nUpload finished.")