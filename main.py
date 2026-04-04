import os
import json
import argparse
from pathlib import Path
import cv2

# Import pipeline components
from preprocessing.document_parser import DocumentParser
from preprocessing.enhancer import ImageEnhancer
from layout.detector import TextDetector
from ocr.model_adapter import OCRAdapter

def run_pipeline(input_file, output_dir="outputs"):
    print(f"Starting OCR Pipeline for: {input_file}")
    
    # Ensure outputs directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize all modules
    parser = DocumentParser(output_dir=os.path.join(output_dir, "images"))
    enhancer = ImageEnhancer()
    detector = TextDetector()
    ocr = OCRAdapter()
    
    # 1 & 2. Parse document to get images
    extracted_image_paths = parser.parse(input_file)
    if not extracted_image_paths:
        print("No images extracted. Exiting.")
        return

    all_results = {}

    # Run remaining pipeline on each extracted image
    for img_path in extracted_image_paths:
        print(f"\nProcessing image: {img_path}")
        try:
            # Load Original Image (BGR)
            original_image = cv2.imread(img_path)
            if original_image is None:
                print(f"Skipping {img_path}, could not read image.")
                continue

            # 3. Preprocess image for detection
            print("Enhancing image...")
            enhanced_binary = enhancer.enhance(original_image)
            
            # 4. Detect Text Regions
            print("Detecting bounding boxes...")
            bboxes = detector.detect(enhanced_binary)
            print(f"Detected {len(bboxes)} bounding boxes.")
            
            # 5. Run OCR (Hindi Support) on text regions
            print("Running OCR on bounding boxes...")
            ocr_results = ocr.run_ocr(original_image, bboxes)
            
            # 6. Collect Result Format
            base_name = Path(img_path).stem
            res_json_path = os.path.join(output_dir, f"{base_name}_results.json")
            
            with open(res_json_path, 'w', encoding='utf-8') as f:
                json.dump(ocr_results, f, ensure_ascii=False, indent=4)
                
            all_results[img_path] = res_json_path
            print(f"Results saved to: {res_json_path}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    print("\nPipeline execution complete.")
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OCR Pipeline on a Document")
    parser.add_argument('input', type=str, help="Path to input PDF or HTML")
    args = parser.parse_args()
    
    run_pipeline(args.input)
