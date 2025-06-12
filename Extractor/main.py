from pdf_text_extractor import PDFTextExtractor


def main(pdf_path, method='auto', debug=False):
    extractor = PDFTextExtractor(debug=debug)
    text = extractor.extract_text(pdf_path, method=method)
    print(text)

if __name__ == "__main__":
    pdf_path = "resume1.pdf"  # Change as needed
    method = "auto"            # 'auto', 'single_column', 'dual_column', or 'opencv'
    debug = False              # Set True for debug output
    main(pdf_path, method, debug)

