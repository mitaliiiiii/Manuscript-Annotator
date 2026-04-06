import cv2
import numpy as np

class TextDetector:
    def __init__(self):
        pass

    def detect(self, preprocessed_image):
        """
        Detects layout regions in a preprocessed image.
        Returns a list of dictionaries with 'bbox', 'polygon', 'category_id', and 'area'.
        """
        if preprocessed_image is None or len(preprocessed_image.shape) > 2:
            raise ValueError("Detector requires a 2D binary image.")

        num_white = cv2.countNonZero(preprocessed_image)
        num_black = preprocessed_image.size - num_white
        
        if num_white > num_black:
            binary_inv = cv2.bitwise_not(preprocessed_image)
        else:
            binary_inv = preprocessed_image

        # Dilation to connect disjoint parts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(binary_inv, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100: # Filter small noise
                continue

            x, y, w, h = cv2.boundingRect(contour)
            
            # Simplify contour to a polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Flatten polygon points for COCO [x1, y1, x2, y2, ...]
            polygon = approx.flatten().tolist()
            
            # --- Classification Heuristics ---
            img_h, img_w = preprocessed_image.shape[:2]
            aspect_ratio = w / float(h)
            fill_ratio = area / float(w * h)
            
            category_id = 1 # Default: text_region
            
            # Heuristic for Marginalia (near edges)
            margin_threshold = 0.1
            is_marginal = (x < img_w * margin_threshold or (x + w) > img_w * (1 - margin_threshold))
            
            if is_marginal:
                category_id = 2 # marginalia/notes
            
            # Heuristic for Illustration/Diagram
            is_large_blob = (area > (img_w * img_h) * 0.05) # More than 5% of page
            is_square_ish = (0.5 < aspect_ratio < 2.0)
            
            if is_large_blob or (is_square_ish and area > 1000):
                category_id = 3 # illustration/diagram

            # Special case: Page Frame (extremely large)
            if area > (img_w * img_h) * 0.8:
                category_id = 4 # page_frame

            # Heuristic for Damage/Hole (Category 5)
            # Irregular, non-horizontal, often "hollow" or strange fill_ratio
            if not is_marginal and area > 500 and aspect_ratio < 3.0 and fill_ratio < 0.4:
                 category_id = 5 # damage/hole

            # --- Baseline Extraction (Horizontal Polyline through center) ---
            # We'll create a 2-point horizontal line through the Y-center for Track 2
            baseline = [[int(x), int(y + h/2)], [int(x + w), int(y + h/2)]]

            regions.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "polygon": [int(p) for p in polygon],
                "baseline": baseline,
                "category_id": category_id,
                "area": float(area)
            })

        return regions
    
if __name__ == '__main__':
    detector = TextDetector()
    print("TextDetector initialized successfully.")
