import time
import json
from app.extractor.pdf_text_extractor import PDFTextExtractor
from app.extractor.segregator import EnhancedResumeParser
from app.extractor.resume_structurer import ResumeStructurer

class ResumePipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text_extractor = PDFTextExtractor()
        self.segregator = EnhancedResumeParser()
        self.structurer = ResumeStructurer()

    def extract_text(self):
        # Extract text from PDF using the text extractor
        raw_text = self.text_extractor.extract(self.file_path)
        return raw_text

    def parse_enhanced(self, raw_text):
        # Parse the raw text to structured data using the enhanced parser
        structured_data = self.segregator.parse(raw_text)
        return structured_data

    def structure_resume(self, structured_data):
        # Structure the resume data into a desired format
        resume_structure = self.structurer.structure(structured_data)
        return resume_structure

    def run_pipeline(self):
        # Run the complete pipeline: extract -> parse -> structure
        raw_text = self.extract_text()
        structured_data = self.parse_enhanced(raw_text)
        resume_structure = self.structure_resume(structured_data)
        return resume_structure