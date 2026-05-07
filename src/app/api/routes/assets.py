import os
import boto3
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from botocore.exceptions import ClientError
from src.app.core.config import settings

router = APIRouter()

# Initialize R2 client using existing env vars
def get_r2_client():
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY")
    secret_key = os.getenv("R2_SECRET_KEY")
    
    if not all([account_id, access_key, secret_key]):
        return None
        
    return boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

@router.get("/r2/{key:path}")
async def proxy_r2_asset(key: str):
    """
    Proxies an asset from Cloudflare R2.
    Usage: GET /api/v1/assets/r2/data/video_generation/...
    """
    client = get_r2_client()
    if not client:
        raise HTTPException(status_code=500, detail="R2 credentials not configured")
        
    bucket_name = os.getenv("R2_BUCKET_NAME", "echo-data")
    
    try:
        # Get the object from R2
        response = client.get_object(Bucket=bucket_name, Key=key)
        
        # Stream the content back to the frontend
        return StreamingResponse(
            response['Body'],
            media_type=response.get('ContentType', 'image/jpeg'),
            headers={
                "Cache-Control": "public, max-age=31536000",
                "Content-Disposition": f"inline; filename={os.path.basename(key)}"
            }
        )
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "NoSuchKey":
            raise HTTPException(status_code=404, detail=f"Asset not found in R2: {key}")
        raise HTTPException(status_code=500, detail=str(e))
