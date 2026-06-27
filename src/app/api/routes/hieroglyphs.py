from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
import httpx
import base64

router = APIRouter()

# Hieroglyph API microservice URL 
HIEROGLYPH_API_URL = "http://127.0.0.1:8003"

@router.post("/translate")
async def translate_hieroglyphs(image: UploadFile = File(...)):
    """
    Proxies the hieroglyph translation request to the dedicated microservice.
    """
    # Validate file type
    is_image = image.content_type and image.content_type.startswith("image/")
    is_heic = image.filename and image.filename.lower().endswith(('.heic', '.heif'))
    if not (is_image or is_heic):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image_data = await image.read()
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            files = {"image": (image.filename, image_data, image.content_type)}
            response = await client.post(f"{HIEROGLYPH_API_URL}/translate/upload", files=files)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"Hieroglyph API error: {response.text}")
                
            return response.json()
            
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Hieroglyph microservice unavailable: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal proxy error: {str(e)}")

@router.post("/translate/stream")
async def translate_hieroglyphs_stream(image: UploadFile = File(...)):
    """
    Proxies the streaming hieroglyph translation request.
    """
    is_image = image.content_type and image.content_type.startswith("image/")
    is_heic = image.filename and image.filename.lower().endswith(('.heic', '.heif'))
    if not (is_image or is_heic):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    image_data = await image.read()
    image_b64 = base64.b64encode(image_data).decode("utf-8")
    
    async def stream_proxy():
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST", 
                f"{HIEROGLYPH_API_URL}/translate/stream", 
                json={"image_base64": image_b64}
            ) as response:
                async for chunk in response.aiter_raw():
                    yield chunk

    return StreamingResponse(stream_proxy(), media_type="text/event-stream")
