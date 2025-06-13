from pdf_text_extractor import PDFTextExtractor
from segregator import EnhancedResumeParser
import time
import json


def main(pdf_path, method='auto', debug=False):
    start_time = time.time()
    extractor = PDFTextExtractor(debug=debug)
    text = extractor.extract_text(pdf_path, method=method)
    print(text)

    # segregate the text into sections
    parser = EnhancedResumeParser(threshold=0.6, max_words_in_header=4)
    parsed_data = parser.parse_resume(text)
    parser.print_parsed_sections(parsed_data)
    # print(json.dumps({k: v for k, v in parsed_data.items() if k != "debug_info"}, indent=2))
    with open('resume_sections.json', 'w') as f:
        json.dump({k: v for k, v in parsed_data.items() if k != "debug_info"}, f, indent=2)

    end_time = time.time()
    print(f"\n[INFO] main() completed in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    pdf_path = "resume1.pdf"  # Change as needed
    method = "auto"            # 'auto', 'single_column', 'dual_column', or 'opencv'
    debug = False              # Set True for debug output
    main(pdf_path, method, debug)

