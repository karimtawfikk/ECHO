import boto3
import os
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

FOLDER_PREFIX = "data/video_generation/outputs/pharaohs_descriptions"

# List all objects with this prefix
objects = client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FOLDER_PREFIX)

if "Contents" in objects:
    keys_to_delete = [{"Key": obj["Key"]} for obj in objects["Contents"]]
    
    # Delete all objects in one request (max 1000 at a time)
    response = client.delete_objects(
        Bucket=BUCKET_NAME,
        Delete={"Objects": keys_to_delete}
    )
    print(f"Deleted {len(keys_to_delete)} objects from {FOLDER_PREFIX}")
else:
    print("No objects found to delete.")