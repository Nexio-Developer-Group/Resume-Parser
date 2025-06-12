import pdfplumber
import re
import logging
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

class PDFTextExtractor:
    """
    A robust PDF text extractor that handles single/dual-column PDFs (pdfplumber) and OpenCV+OCR extraction.
    """
    def __init__(self, debug: bool = False):
        self.debug = debug
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)

    def extract_text(self, pdf_path: str, method: str = 'auto') -> str:
        if method == 'opencv':
            return self.extract_text_opencv(pdf_path)
        try:
            if method == 'auto':
                return self._auto_extract(pdf_path)
            elif method == 'single_column':
                return self._extract_single_column(pdf_path)
            elif method == 'dual_column':
                return self._extract_dual_column(pdf_path)
            else:
                raise ValueError(f"Unknown extraction method: {method}")
        except Exception as e:
            logging.error(f"Error extracting text: {e}") if self.debug else None
            raise

    def extract_text_opencv(self, pdf_path: str) -> str:
        pages = convert_from_path(pdf_path, dpi=300)
        all_text = []
        for i, pil_img in enumerate(pages):
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            vertical_sum = np.sum(edges, axis=0)
            img_width = img.shape[1]
            third = img_width // 3
            middle_section = vertical_sum[third: 2 * third]
            split_offset = np.argmin(middle_section)
            split_col = split_offset + third  # shift back to real image coordinate
            # Safety check
            if split_col <= 10 or split_col >= img_width - 10:
                print(f"⚠️ [Page {i+1}] Invalid split detected at column {split_col}, falling back to midpoint.")
                split_col = img_width // 2
            print(f"✅ [Page {i+1}] Final split at column: {split_col}")
            # === Step 4: Crop the left and right pages ===
            left_image = img[:, :split_col]
            right_image = img[:, split_col:]
            # Convert to PIL for Tesseract OCR
            left_pil = Image.fromarray(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
            right_pil = Image.fromarray(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
            # === Step 5: OCR the cropped halves ===
            left_text = pytesseract.image_to_string(left_pil, config='--psm 3')
            right_text = pytesseract.image_to_string(right_pil, config='--psm 3')
            # === Step 6: Output the results ===
            all_text.append(f"\n=== Page {i+1} - Left Half ===\n{left_text}\n")
            all_text.append(f"=== Page {i+1} - Right Half ===\n{right_text}\n")
        return '\n'.join(all_text)

    def _auto_extract(self, pdf_path: str) -> str:
        gap_position = self._find_column_gap(pdf_path)
        return self._extract_dual_column_with_gap(pdf_path, gap_position) if gap_position else self._extract_single_column(pdf_path)

    def _find_column_gap(self, pdf_path: str) -> Optional[float]:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                first_page = pdf.pages[0]
                chars = first_page.chars
                if not chars or len(chars) < 50:
                    return None

                page_width = first_page.width
                total_chars = len(chars)
                char_positions = [(char['x0'], char['y0']) for char in chars]

                window_width = page_width * 0.05
                scan_range = (page_width * 0.25, page_width * 0.75)
                step = page_width * 0.02

                best_gap, min_chars = None, float('inf')

                x = scan_range[0]
                while x <= scan_range[1]:
                    count = sum(1 for x0, _ in char_positions if x <= x0 <= x + window_width)
                    if count < min_chars:
                        min_chars = count
                        best_gap = x + window_width / 2
                    x += step

                if best_gap and (min_chars / total_chars) * 100 < 1.0:
                    left = sum(1 for x0, _ in char_positions if x0 < best_gap)
                    right = sum(1 for x0, _ in char_positions if x0 > best_gap)
                    if (left / total_chars) > 0.2 and (right / total_chars) > 0.2:
                        return best_gap
        except Exception as e:
            logging.error(f"Error detecting column gap: {e}") if self.debug else None
        return None

    def _extract_single_column(self, pdf_path: str) -> str:
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n\n"
            return self._clean_text(text)
        except Exception as e:
            logging.error(f"Single-column extraction error: {e}") if self.debug else None
            raise

    def _extract_dual_column(self, pdf_path: str) -> str:
        return self._extract_dual_column_with_gap(pdf_path, self._find_column_gap(pdf_path))

    def _extract_dual_column_with_gap(self, pdf_path: str, gap_position: Optional[float]) -> str:
        """
        Extract text from a dual-column PDF using the detected or default gap position.

        Args:
            pdf_path: Path to the PDF file.
            gap_position: X-coordinate to split the page into two columns.
                        If None, splits at 50% of the page width.

        Returns:
            Cleaned, extracted text from both columns.
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    if self.debug:
                        logging.debug(f"Processing dual-column page {page_num + 1}")

                    page_width, page_height = page.width, page.height
                    split_x = gap_position if gap_position else page_width * 0.5

                    if self.debug:
                        logging.debug(f"Splitting page at x={split_x:.1f} (page width: {page_width:.1f})")

                    # Define crop boxes
                    left_column = (0, 0, split_x, page_height)
                    right_column = (split_x, 0, page_width, page_height)

                    # Safely crop and extract each column
                    try:
                        left_crop = self.safe_crop(page, left_column)
                        left_text = left_crop.extract_text() or ""

                        right_crop = self.safe_crop(page, right_column)
                        right_text = right_crop.extract_text() or ""

                        if self.debug:
                            logging.debug(f"Left column length: {len(left_text)}")
                            logging.debug(f"Right column length: {len(right_text)}")

                        combined_text = ""
                        if left_text.strip():
                            combined_text += left_text.strip() + "\n"
                        if right_text.strip():
                            combined_text += right_text.strip() + "\n"

                        text += combined_text + "\n"

                    except Exception as crop_error:
                        if self.debug:
                            logging.warning(f"Column cropping failed: {crop_error}")
                        # Fallback to extracting the full page text
                        fallback_text = page.extract_text() or ""
                        text += fallback_text + "\n\n"

        except Exception as e:
            if self.debug:
                logging.error(f"Dual-column extraction failed: {str(e)}")
            raise

        return self._clean_text(text)


    def safe_crop(self, page, crop_box):
        """
        Safely crop a PDF page within its bounding box.
        If crop_box exceeds the page bounds, adjust it.
        """
        try:
            x0, y0, x1, y1 = crop_box
            page_x0, page_y0, page_x1, page_y1 = page.bbox

            # Clamp crop box to page bounds
            x0 = max(x0, page_x0)
            y0 = max(y0, page_y0)
            x1 = min(x1, page_x1)
            y1 = min(y1, page_y1)

            return page.crop((x0, y0, x1, y1))
        except Exception as e:
            if self.debug:
                logging.warning(f"Safe crop failed: {e}")
            return page  # fallback to full page


    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n ', '\n', text)
        text = re.sub(r'[^\w\s@.,;:(){}\[\]/"&%$#+=<>|\\\'\-`~!?*]', '', text)
        text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€\x9d', '"').replace('â€"', '-')
        return text.strip()

    def get_pdf_info(self, pdf_path: str) -> Dict:
        info = {'pages': 0, 'layout_type': 'unknown', 'gap_position': None, 'extraction_method': 'auto'}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                info['pages'] = len(pdf.pages)
                gap_position = self._find_column_gap(pdf_path)
                info['layout_type'] = 'dual_column' if gap_position else 'single_column'
                info['gap_position'] = gap_position
        except Exception as e:
            logging.error(f"Error reading PDF info: {e}") if self.debug else None
        return info

    def extract_with_fallback(self, pdf_path: str) -> Tuple[str, str]:
        try:
            gap = self._find_column_gap(pdf_path)
            text = self._extract_dual_column_with_gap(pdf_path, gap) if gap else self._extract_single_column(pdf_path)
            if text and len(text.strip()) > 50:
                return text, 'dual_column' if gap else 'single_column'
        except Exception as e:
            logging.warning(f"Auto extraction failed: {e}") if self.debug else None

        try:
            text = self._extract_single_column(pdf_path)
            if text and len(text.strip()) > 50:
                return text, 'single_column_fallback'
        except Exception as e:
            logging.warning(f"Fallback failed: {e}") if self.debug else None

        return "", "failed"

