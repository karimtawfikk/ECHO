import boto3
import os
from pathlib import Path
from dotenv import load_dotenv

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

REMOTE_PREFIX = "data/"
# 🔹 Where to save locally
LOCAL_DIR = Path(r"C:\Uni\4th Year\GP\ECHO\data")

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

paginator = client.get_paginator("list_objects_v2")

for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=REMOTE_PREFIX):
    if "Contents" not in page:
        print("No objects found in R2 with this prefix.")
        continue

    for obj in page["Contents"]:
        key = obj["Key"]
        if key.endswith("/"):
            continue

        local_path = LOCAL_DIR / key

        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"⬇️ Downloading {key} → {local_path}")
        client.download_file(BUCKET_NAME, key, str(local_path))