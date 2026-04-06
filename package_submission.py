import os
import json
import shutil
from pathlib import Path
import datetime

def package_submission(output_dir="outputs", submission_dir="submission_package"):
    """
    Organizes the COCO annotations and processed images into a clean 
    submission folder for the hackathon.
    """
    print(f"Creating submission package in: {submission_dir}...")
    
    # Create structure
    sub_path = Path(submission_dir)
    images_sub = sub_path / "images"
    sub_path.mkdir(parents=True, exist_ok=True)
    images_sub.mkdir(parents=True, exist_ok=True)

    # 1. Copy the COCO JSON
    coco_src = Path(output_dir) / "annotations.json"
    if not coco_src.exists():
        print("Error: annotations.json not found in output directory.")
        return
        
    shutil.copy(coco_src, sub_path / "predictions.json")
    print(" - Copied predictions.json")

    # 2. Copy the processed images referenced in the JSON
    with open(coco_src, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    print(f" - Copying {len(coco_data['images'])} images...")
    for img_entry in coco_data["images"]:
        image_path = Path(img_entry["file_name"])
        if image_path.exists():
            shutil.copy(image_path, images_sub / image_path.name)
        else:
            print(f"   Warning: Could not find image {image_path}")

    # 3. Generate Method Description (Required for Winner Verification)
    method_desc = f"""# Auto-Annotation Submission Method Description
**Generated on**: {datetime.datetime.now().strftime("%Y-%m-%d")}

## Overview
Our approach focuses on high-precision layout segmentation and baseline extraction for Indic manuscripts (Standard Indic & Ramcharitmanas).

## Technical Implementation
1. **Preprocessing**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) and Adaptive Gaussian Thresholding to handle bleed-through and fading common in palm-leaf and aged paper manuscripts.
2. **Layout Parsing**: Utilizes an enhanced contour-based segmentation with horizontal dilation bias to tightly wrap text lines. Includes heuristics for:
   - `marginalia/notes`: Detected via edge-proximity thresholds.
   - `illustration/diagram`: Detected via aspect-ratio and area density checks.
   - `damage/hole`: Detected via irregular blob analysis.
3. **Reading Order**: Implements a "Smart Reading Order" graph that priorities main text flow over marginal components, significantly reducing the "Human Effort Score" (E).
4. **Baselines**: Generates horizontal polylines (baselines) through the geometric center of text regions for Track 2 compliance.

## Reproducibility
The pipeline is modular and can be run using `python main.py <input>`. It includes support for a pre-trained YOLOv8-segmentation weights (`--model`) once fine-tuned on the provided seed set.
"""

    with open(sub_path / "METHODOLOGY.md", "w", encoding="utf-8") as f:
        f.write(method_desc)
    print(" - Generated METHODOLOGY.md")

    print(f"\nPackage created successfully at: {submission_dir}")
    print("You can now zip this folder and upload it as your submission!")

if __name__ == "__main__":
    package_submission()
