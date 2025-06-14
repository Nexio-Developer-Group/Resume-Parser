import uuid
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.models import ResumeRequest
from app.services.resume_processor import process_resume_logic
from app.config import TEMP_DIR

router = APIRouter()

@router.post("/process_resume")
def process_resume(req: ResumeRequest):
    return process_resume_logic(req)
