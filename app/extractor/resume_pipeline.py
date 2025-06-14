import time
import json
import re
from app.extractor.pdf_text_extractor import PDFTextExtractor
from app.extractor.segregator import EnhancedResumeParser
from app.extractor.resume_structurer import ResumeStructurer

class ResumePipeline:
    def __init__(self, pdf_path, method='auto', debug=False):
        self.pdf_path = pdf_path
        self.method = method
        self.debug = debug
        self.text = None
        self.segmented = None
        self.structured = None

    def extract_text(self):
        extractor = PDFTextExtractor(debug=self.debug)
        self.text = extractor.extract_text(self.pdf_path, method=self.method)
        print(self.text)
        return self.text

    def segment_text(self):
        if self.text is None:
            raise ValueError("Text not extracted yet.")
        parser = EnhancedResumeParser(threshold=0.6, max_words_in_header=4)
        parsed_data = parser.parse_resume(self.text)
        parser.print_parsed_sections(parsed_data)
        self.segmented = {k: v for k, v in parsed_data.items() if k != "debug_info"}
        return self.segmented

    def structure_resume(self):
        if self.segmented is None:
            raise ValueError("Text not segmented yet.")
        structurer = ResumeStructurer(self.segmented)
        self.structured = structurer.get_structured_resume()
        return self.structured

    def export_to_json(self, output_path):
        if self.structured is None:
            raise ValueError("Resume not structured yet.")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.structured, f, ensure_ascii=False, indent=2)
        return output_path

    def _normalize(self, s, is_email=False):
        s = s.lower()
        if is_email:
            # Remove all spaces and keep only allowed email chars
            s = re.sub(r'\s+', '', s)
            s = re.sub(r'[^a-z0-9@._+-]', '', s)
        else:
            # Remove all non-alphanumeric, collapse spaces
            s = re.sub(r'[^a-z0-9]', '', s)
        return s

    def check_identity_in_text(self, username, email, n_lines=20):
        # Always extract text directly from the PDF for identity check
        extractor = PDFTextExtractor(debug=self.debug)
        text = extractor.extract_text(self.pdf_path, method = "single_column")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        first_n = lines[:n_lines]
        norm_username = self._normalize(username)
        norm_email = self._normalize(email, is_email=True)
        name_found = False
        email_found = False
        for line in first_n:
            norm_line = self._normalize(line)
            norm_line_email = self._normalize(line, is_email=True)
            if norm_username in norm_line:
                name_found = True
            if norm_email in norm_line_email:
                email_found = True
            if name_found and email_found:
                break
        return name_found and email_found