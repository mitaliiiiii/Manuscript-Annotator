import easyocr
import cv2
import numpy as np

class OCRAdapter:
    def __init__(self, languages=['hi', 'en']):
        """
        Initializes EasyOCR reader with specified languages.
        Defaults to Hindi ('hi') and English ('en').
        """
        print(f"Initializing EasyOCR with languages: {languages}...")
        self.reader = easyocr.Reader(languages, gpu=False) # Fallback to CPU by default for broader compatibility, change gpu=True if CUDA is configured
        print("EasyOCR loaded successfully.")

    def run_ocr(self, original_image, bounding_boxes):
        """
        Runs OCR on specific ROI cropped from the original image.
        
        :param original_image: A 3D numpy array (Color image)
        :param bounding_boxes: List of bounding boxes [x, y, w, h]
        :return: List of dictionaries with bbox and text mappings
        """
        results = []
        
        # Verify image format
        if isinstance(original_image, str):
            original_image = cv2.imread(original_image)
            
        if original_image is None:
            raise ValueError("Invalid original image provided to OCR adapter.")
            
        img_h, img_w = original_image.shape[:2]

        for bbox in bounding_boxes:
            x, y, w, h = [int(v) for v in bbox]
            
            # Boundary checks to prevent crop indexing errors
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            
            # Crop ROI
            roi = original_image[y:y+h, x:x+w]
            
            # Skip if ROI is invalid
            if roi.size == 0 or h <= 1 or w <= 1:
                continue

            # Run OCR
            # readtext returns a list of tuples: ([(x,y)...], text, confidence)
            # detail=0 returns just the string parts
            text_lines = self.reader.readtext(roi, detail=0)
            
            # Join text if multiple lines are somehow detected in the box
            extracted_text = " ".join(text_lines).strip()
            
            # Skip empty OCR results
            if not extracted_text:
                continue
                
            results.append({
                "bbox": [x, y, w, h],
                "text": extracted_text
            })

        return results

if __name__ == '__main__':
    adapter = OCRAdapter()
    print("OCR Adapter initialized properly.")
