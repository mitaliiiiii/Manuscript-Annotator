import os
from pathlib import Path
import fitz # PyMuPDF
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import urljoin
import base64
import re
import cv2
import numpy as np

class DocumentParser:
    def __init__(self, output_dir="outputs/images"):
        """
        Initializes the DocumentParser.
        :param output_dir: Directory to save the extracted images.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, input_path, max_pages=None):
        """
        Main entry point for parsing a document (PDF or HTML).
        :param input_path: Path to the input file.
        :param max_pages: Optional limit on the number of pages/images to extract.
        :return: List of paths to the extracted images.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if input_path.suffix.lower() == '.pdf':
            return self._parse_pdf(input_path, max_pages=max_pages)
        elif input_path.suffix.lower() in ['.html', '.htm']:
            return self._parse_html(input_path, max_pages=max_pages)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

    def _parse_pdf(self, pdf_path, max_pages=None):
        """
        Extracts images from a PDF file using PyMuPDF (fitz).
        """
        print(f"Parsing PDF: {pdf_path}")
        extracted_images = []
        try:
            doc = fitz.open(str(pdf_path))
            num_pages = len(doc)
            if max_pages:
                num_pages = min(num_pages, max_pages)
                
            for i in range(num_pages):
                page = doc.load_page(i)
                # Render to high-res image (e.g., 300 DPI) for better OCR accuracy
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                output_path = self.output_dir / f"{pdf_path.stem}_page_{i+1}.jpg"
                pix.save(str(output_path))
                extracted_images.append(str(output_path))
                print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Error converting PDF {pdf_path}: {e}")
        return extracted_images

    def _parse_html(self, html_path):
        """
        Extracts images (embedded base64 and linked) from an HTML file.
        """
        print(f"Parsing HTML: {html_path}")
        extracted_images = []
        
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                
            img_tags = soup.find_all('img')
            for i, img in enumerate(img_tags):
                src = img.get('src')
                if not src:
                    continue
                    
                output_path = self.output_dir / f"{html_path.stem}_img_{i+1}.jpg"
                
                # Handle base64 embedded images
                if src.startswith('data:image'):
                    try:
                        base64_data = re.sub('^data:image/.+;base64,', '', src)
                        img_data = base64.b64decode(base64_data)
                        with open(output_path, 'wb') as f:
                            f.write(img_data)
                        extracted_images.append(str(output_path))
                        print(f"Saved base64 image: {output_path}")
                    except Exception as e:
                        print(f"Failed to decode base64 image: {e}")
                # Handle linked images
                elif not src.startswith(('http://', 'https://')):
                    # Local path relative to HTML file
                    try:
                        local_img_path = html_path.parent / src
                        if local_img_path.exists():
                            # Standardize format to jpg
                            img_frame = cv2.imread(str(local_img_path))
                            if img_frame is not None:
                                cv2.imwrite(str(output_path), img_frame)
                                extracted_images.append(str(output_path))
                                print(f"Saved local image: {output_path}")
                    except Exception as e:
                        print(f"Failed to process local image {src}: {e}")
                else:
                    # Online Image link - Optional since we are focusing on local dataset
                    print(f"Skipping external URL image: {src}")

        except Exception as e:
            print(f"Error parsing HTML {html_path}: {e}")
            
        return extracted_images

if __name__ == '__main__':
    # Basic testing entry
    parser = DocumentParser()
    print("DocumentParser initialized successfully.")
