import os
import uuid
import requests
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from app.extractor.resume_pipeline import ResumePipeline
from app.config import TEMP_DIR

def process_resume_logic(req):
    pdf_filename = f"{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(TEMP_DIR, pdf_filename)

    try:
        # Download PDF
        try:
            r = requests.get(req.live_link, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Could not download PDF: {str(e)}")

        with open(pdf_path, "wb") as f:
            f.write(r.content)

        # Extract text and check identity
        try:
            pipeline = ResumePipeline(pdf_path, method="auto", debug=False)
            pipeline.extract_text()
            if not pipeline.check_identity_in_text(req.username, req.email, n_lines=20):
                return JSONResponse(status_code=422, content={
                    "message": "The PDF does not appear to belong to the provided username or email.",
                    "code": 422
                })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text extraction or identity check failed: {str(e)}")

        # Segment and structure
        try:
            pipeline.segment_text()
            structured_resume = pipeline.structure_resume()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Resume structuring failed: {str(e)}")

        return {
            "message": "Resume processed successfully.",
            "code": 200,
            "data": structured_resume
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "message": f"An error occurred: {str(e)}",
            "code": 500
        })
    finally:
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
        except Exception:
            pass
