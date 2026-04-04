import cv2
import numpy as np

class TextDetector:
    def __init__(self):
        pass

    def detect(self, preprocessed_image):
        """
        Detects text bounding boxes in a preprocessed (mostly binary) image.
        Uses contour-based detection and a simplified projection profiling hybrid.
        
        :param preprocessed_image: A 2D numpy array (binary/grayscale image)
        :return: A list of bounding boxes: [x, y, w, h]
        """
        # Ensure we have a binary image where text is white and background is black
        # Depending on adaptive threshold, we might need to invert it based on pixel mass.
        # Assuming typical output from enhancer: background might be mostly 255.
        # Let's count white/black pixels to determine if inversion is necessary.
        if preprocessed_image is None or len(preprocessed_image.shape) > 2:
            raise ValueError("Detector requires a 2D binary image.")

        num_white = cv2.countNonZero(preprocessed_image)
        num_black = preprocessed_image.size - num_white
        
        if num_white > num_black:
            # Assuming background is white, text is black -> invert to make text white
            binary_inv = cv2.bitwise_not(preprocessed_image)
        else:
            binary_inv = preprocessed_image

        # Dilation to connect disjoint parts of the same character/word/line
        # Rectangular kernel tailored for Hindi text lines (horizontal bias)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary_inv, kernel, iterations=2)

        # Find contours on dilated image
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Simple heuristic filtering to avoid extreme noise
            if w > 10 and h > 10 and w < preprocessed_image.shape[1] * 0.95:
                bounding_boxes.append([int(x), int(y), int(w), int(h)])

        # Sort bounding boxes top-to-bottom, then left-to-right
        bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))
        
        return bounding_boxes
    
if __name__ == '__main__':
    detector = TextDetector()
    print("TextDetector initialized successfully.")
