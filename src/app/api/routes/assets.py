import os
import boto3
import time
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from botocore.exceptions import ClientError
from src.app.core.config import settings
from sqlalchemy import text
from src.db.session import SessionLocal

router = APIRouter()

# Cached global client instance
_r2_client = None

# Initialize R2 client using existing env vars
def get_r2_client():
    global _r2_client
    if _r2_client is not None:
        return _r2_client
        
    account_id = os.getenv("R2_ACCOUNT_ID")
    access_key = os.getenv("R2_ACCESS_KEY")
    secret_key = os.getenv("R2_SECRET_KEY")
    
    if not all([account_id, access_key, secret_key]):
        return None
        
    _r2_client = boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return _r2_client


@router.get("/r2/{key:path}")
def proxy_r2_asset(key: str):
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


@router.get("/r2-history/{key:path}")
def proxy_history_asset(key: str):
    """
    Proxies an asset from the user-history-data Cloudflare R2 bucket.
    Usage: GET /api/v1/assets/r2-history/recognition/...
    """
    client = get_r2_client()
    if not client:
        raise HTTPException(status_code=500, detail="R2 credentials not configured")
        
    bucket_name = "user-history-data"
    
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

@router.post("/upload/history")
async def upload_history_image(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    task_type: str = Form(...) # "recognition" or "hieroglyphics"
):
    client = get_r2_client()
    if not client:
        raise HTTPException(status_code=500, detail="R2 credentials not configured")
        
    bucket_name = "user-history-data"
    
    # Validate task_type
    if task_type not in ["recognition", "hieroglyphics"]:
        raise HTTPException(status_code=400, detail="Invalid task_type. Must be recognition or hieroglyphics")
        
    # Get file extension
    filename = file.filename or "image.jpg"
    file_ext = filename.split('.')[-1] if '.' in filename else "jpg"
    
    # Save key as task_type/user_id_timestamp.ext
    timestamp = int(time.time() * 1000)
    key = f"{task_type}/{user_id}_{timestamp}.{file_ext}"
    
    try:
        # Read the file content
        content = await file.read()
        
        # Upload to Cloudflare R2
        client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content,
            ContentType=file.content_type or "image/jpeg"
        )
        
        # Return the key so the frontend can query it via proxy
        return {"key": key}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to R2: {str(e)}")

@router.delete("/delete-account/{user_id}")
async def delete_account_all(user_id: str):
    """
    Deletes all images from Cloudflare R2 user-history-data for a specific user,
    and deletes all database records including auth.users.
    """
    client = get_r2_client()
    if client:
        bucket_name = "user-history-data"
        try:
            prefixes = [f"recognition/{user_id}", f"hieroglyphics/{user_id}"]
            for prefix in prefixes:
                response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                if 'Contents' in response:
                    objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                    if objects_to_delete:
                        client.delete_objects(
                            Bucket=bucket_name,
                            Delete={'Objects': objects_to_delete, 'Quiet': True}
                        )
        except Exception as e:
            print(f"Error deleting R2 files: {e}")
            
    # Now delete from DB using admin privilege connection
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM chat_messages_rewriter WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id = :uid)"), {"uid": user_id})
        db.execute(text("DELETE FROM chat_messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id = :uid)"), {"uid": user_id})
        db.execute(text("DELETE FROM conversations WHERE user_id = :uid"), {"uid": user_id})
        db.execute(text("DELETE FROM translation_history WHERE user_id = :uid"), {"uid": user_id})
        db.execute(text("DELETE FROM recognition_history WHERE user_id = :uid"), {"uid": user_id})
        db.execute(text("DELETE FROM profiles WHERE id = :uid"), {"uid": user_id})
        db.execute(text("DELETE FROM auth.users WHERE id = :uid"), {"uid": user_id})
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database deletion failed: {e}")
    finally:
        db.close()
        
    return {"status": "success", "message": f"Deleted all data and account for user {user_id}"}


# Pre-warm the R2 client connection pool on backend startup
try:
    get_r2_client()
except Exception as e:
    pass

