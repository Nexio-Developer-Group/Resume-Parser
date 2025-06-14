import time
import json
from pdf_text_extractor import PDFTextExtractor
from segregator import EnhancedResumeParser
from resume_structurer import ResumeStructurer

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