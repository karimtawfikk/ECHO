from fastapi import APIRouter, UploadFile, File, HTTPException
import httpx
import os

router = APIRouter()

# Hieroglyph API microservice URL (defaults to localhost:8003)
HIEROGLYPH_API_URL = os.getenv("HIEROGLYPH_API_URL", "http://127.0.0.1:8003")

@router.post("/translate")
async def translate_hieroglyphs(file: UploadFile = File(...)):
    """
    Proxies the hieroglyph translation request to the dedicated microservice.
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image_data = await file.read()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            files = {"file": (file.filename, image_data, file.content_type)}
            response = await client.post(f"{HIEROGLYPH_API_URL}/api/v1/hieroglyphs/translate", files=files)
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"Hieroglyph API error: {response.text}")
                
            return response.json()
            
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Hieroglyph microservice unavailable: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal proxy error: {str(e)}")
