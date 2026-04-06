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

from layout.model_segmenter import LayoutSegmenter
from postprocess.reading_order import ReadingOrderSorter

def run_pipeline(input_file, output_dir="outputs", model_path=None, max_pages=None):
    print(f"Starting Auto-Annotation Pipeline for: {input_file}")
    
    # Ensure outputs directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize all modules
    parser = DocumentParser(output_dir=os.path.join(output_dir, "images"))
    enhancer = ImageEnhancer()
    detector = TextDetector()
    segmenter = LayoutSegmenter(model_path=model_path)
    sorter = ReadingOrderSorter()
    
    # Trace statistics
    stats = {}

    # 1 & 2. Parse document to get images
    extracted_image_paths = parser.parse(input_file, max_pages=max_pages)
    if not extracted_image_paths:
        print("No images extracted. Exiting.")
        return

    all_page_data = []

    # Run remaining pipeline on each extracted image
    for img_path in extracted_image_paths:
        print(f"\nProcessing image: {img_path}")
        try:
            # Load Original Image (BGR)
            original_image = cv2.imread(img_path)
            if original_image is None:
                print(f"Skipping {img_path}, could not read image.")
                continue

            h, w = original_image.shape[:2]

            # 3. Preprocess image for detection
            print("Enhancing image...")
            enhanced_binary = enhancer.enhance(original_image)
            
            # 4. Detect Layout Regions
            if segmenter.model is not None:
                print("Running deep learning layout segmentation...")
                regions = segmenter.segment(img_path)
            else:
                print("Detecting layout regions using heuristic-based detector...")
                regions = detector.detect(enhanced_binary)
            
            # 5. Apply Smart Reading Order Sorting
            regions = sorter.sort(regions)
            print(f"Detected and sorted {len(regions)} regions.")
            
            # Record stats
            for r in regions:
                cat_id = r["category_id"]
                stats[cat_id] = stats.get(cat_id, 0) + 1

            # Store data for COCO formatting
            all_page_data.append({
                "file_path": img_path,
                "width": w,
                "height": h,
                "regions": regions
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    # 6. Format and save as COCO JSON
    print("\nFormatting results to COCO JSON...")
    from postprocess.coco_formatter import COCOFormatter
    formatter = COCOFormatter()
    coco_output = formatter.format(all_page_data)
    
    coco_path = os.path.join(output_dir, "annotations.json")
    with open(coco_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print("-" * 30)
    print("RUN SUMMARY")
    print(f"Total Pages Processed: {len(all_page_data)}")
    cat_names = {1: "Text Regions", 2: "Marginalia", 3: "Illustrations", 4: "Page Frames", 5: "Damage/Holes"}
    for cat_id, count in stats.items():
        print(f" - {cat_names.get(cat_id, 'Unknown')}: {count}")
    print("-" * 30)
    print(f"COCO annotations saved to: {coco_path}")
    print("\nPipeline execution complete.")
    return coco_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Auto-Annotation Pipeline on a Document")
    parser.add_argument('input', type=str, help="Path to input PDF or HTML")
    parser.add_argument('--out', type=str, default="outputs", help="Output directory (default: outputs)")
    parser.add_argument('--limit', type=int, default=None, help="Max number of pages to process")
    parser.add_argument('--model', type=str, default=None, help="Path to YOLOv8 .pt model weights")
    args = parser.parse_args()
    
    run_pipeline(args.input, output_dir=args.out, model_path=args.model, max_pages=args.limit)
