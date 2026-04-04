import cv2
import numpy as np
from pathlib import Path

class ImageEnhancer:
    def __init__(self):
        # Setup CLAHE
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def enhance(self, image_input):
        """
        Enhances the input image for OCR processing based on the required pipeline.
        :param image_input: numpy array of the image (BGR from cv2.imread) or path to the image
        :return: enhanced numpy array image (grayscale/binary)
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not read image from {image_input}")
        else:
            img = image_input

        # 1. Grayscale Conversion
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        enhanced_contrast = self.clahe.apply(gray)

        # 3. Gaussian Blur (noise reduction)
        blurred = cv2.GaussianBlur(enhanced_contrast, (5, 5), 0)

        # 4. Adaptive Thresholding
        # Using ADAPTIVE_THRESH_GAUSSIAN_C to handle uneven lighting in manuscripts
        binary = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )

        return binary

if __name__ == '__main__':
    # Test initialization
    enhancer = ImageEnhancer()
    print("ImageEnhancer initialized locally.")
