import json
import os
from pathlib import Path

def fix_annotations(input_path="outputs/annotations.json", output_path="outputs/annotations_fixed.json"):
    """
    Applies mandatory fixes to the COCO annotations to ensure challenge compliance.
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Standardize Categories
    data["categories"] = [
        {"id": 1, "name": "text_region", "supercategory": "layout"},
        {"id": 2, "name": "marginalia", "supercategory": "layout"},
        {"id": 3, "name": "illustration", "supercategory": "layout"},
        {"id": 4, "name": "page_frame", "supercategory": "layout"},
        {"id": 5, "name": "damage", "supercategory": "layout"}
    ]

    # 4. Fix Image File Paths (cross-platform compatibility)
    for img in data.get("images", []):
        if "file_name" in img:
            img["file_name"] = img["file_name"].replace("\\", "/")

    # 2 & 3 & 5 & 6. Fix Annotations
    new_annotations = []
    for i, ann in enumerate(data.get("annotations", []), 1):
        # 6. Ensure Unique ID
        ann["id"] = i
        
        # 2 & 3. Fix Baseline Format & Robust Removal
        if ann.get("category_id") == 1:
            raw_baseline = ann.get("baseline", [])
            if isinstance(raw_baseline, list) and len(raw_baseline) == 2 and isinstance(raw_baseline[0], list):
                # Ensure left -> right order
                sorted_baseline = sorted(raw_baseline, key=lambda p: p[0])
                flat = [val for pt in sorted_baseline for val in pt]
                if len(flat) == 4:
                    ann["baseline"] = flat
                else:
                    if "baseline" in ann:
                        del ann["baseline"]
            elif isinstance(raw_baseline, list) and len(raw_baseline) == 4:
                ann["baseline"] = raw_baseline
            else:
                if "baseline" in ann:
                    del ann["baseline"]
        else:
            # Ensure baseline ONLY for text_region (category_id = 1)
            if "baseline" in ann:
                del ann["baseline"]

        # 3. Add safety field for compatibility
        ann["num_keypoints"] = 0

        # 5. Schema Validation & Optional: valid segmentation
        if "segmentation" in ann and isinstance(ann["segmentation"], list):
            # Ensure polygons have >= 6 values (3 points)
            valid_segs = []
            for seg in ann["segmentation"]:
                if len(seg) >= 6:
                    valid_segs.append(seg)
            ann["segmentation"] = valid_segs

        # Ensure required fields exist
        required_fields = ["id", "image_id", "category_id", "segmentation", "bbox", "area"]
        for field in required_fields:
            if field not in ann:
                if field == "area": ann["area"] = 0.0

        ann["iscrowd"] = ann.get("iscrowd", 0)
        
        # 4. Ensure segmentation is never empty
        if not ann.get("segmentation") or len(ann["segmentation"]) == 0:
            continue
            
        new_annotations.append(ann)

    data["annotations"] = new_annotations

    print(f"Saving fixed annotations to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print("Fix complete.")

if __name__ == "__main__":
    fix_annotations()
