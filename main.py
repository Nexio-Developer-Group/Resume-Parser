from pdf_text_extractor import PDFTextExtractor
from segregator import EnhancedResumeParser
from resume_structurer import ResumeStructurer
from resume_pipeline import ResumePipeline
import time

if __name__ == "__main__":
    pdf_path = "resume1.pdf"  # Change as needed
    method = "auto"
    debug = False

    pipeline = ResumePipeline(pdf_path, method, debug)
    pipeline.extract_text()
    pipeline.segment_text()
    structured_resume = pipeline.structure_resume()
    print(structured_resume)

    # Optional: Export to JSON
    pipeline.export_to_json("structured_resume.json")
