from fastapi import Request
from fastapi.responses import JSONResponse
from app.config import API_KEY

async def check_api_key(request: Request, call_next):
    if request.url.path == "/process_resume":
        api_key = request.headers.get("x-api-key")
        if not api_key or api_key != API_KEY:
            return JSONResponse(status_code=401, content={"message": "Invalid or missing API key."})
    return await call_next(request)
