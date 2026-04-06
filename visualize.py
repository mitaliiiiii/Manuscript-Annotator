import cv2
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def visualize_coco(coco_path, image_id=None, limit=None, output_dir="outputs/visualized"):
    """
    Visualizes COCO annotations. 
    If image_id is None, processes all images (up to limit).
    """
    import json
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(coco_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # Filter images to process
    if image_id is not None:
        images_to_process = [img for img in coco_data["images"] if img["id"] == image_id]
    else:
        images_to_process = coco_data["images"]

    if limit is not None:
        images_to_process = images_to_process[:limit]

    print(f"Starting batch visualization for {len(images_to_process)} images...")

    # Load categories for mapping
    cat_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Attempt to load font
    try:
        font = ImageFont.truetype("Nirmala.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for img_info in images_to_process:
        image_path = img_info["file_name"]
        curr_id = img_info["id"]
        
        if not Path(image_path).exists():
            print(f"Skipping {image_path}, file not found.")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue
            
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Draw annotations for this image
        page_seq = 1
        for ann in coco_data["annotations"]:
            if ann["image_id"] != curr_id:
                continue
                
            # Draw Polygon (Red)
            for seg in ann["segmentation"]:
                if len(seg) >= 4:
                    points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    draw.polygon(points, outline=(255, 0, 0), width=3)
                    
                    # Draw Label + Sequence Order (Reading Order)
                    label = f"#{page_seq} {cat_map.get(ann['category_id'], 'Unknown')}"
                    x, y = points[0]
                    draw.text((x, max(0, y-25)), label, font=font, fill=(0, 0, 255))
                    page_seq += 1
                    
            # Draw Baseline (Green)
            baseline = ann.get("baseline", [])
            if baseline and len(baseline) >= 4:
                # Handle flat list [x1, y1, x2, y2, ...]
                if isinstance(baseline[0], (int, float)):
                    line_points = [(baseline[i], baseline[i+1]) for i in range(0, len(baseline), 2)]
                else:
                    # Handle list of lists [[x1, y1], [x2, y2]] (back-compat)
                    line_points = [(p[0], p[1]) for p in baseline]
                
                if len(line_points) >= 2:
                    draw.line(line_points, fill=(0, 255, 0), width=4)

        final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        output_path = Path(output_dir) / f"vis_{Path(image_path).name}"
        cv2.imwrite(str(output_path), final_img)
        print(f" - Saved: {output_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize COCO Results")
    parser.add_argument('json', type=str, help="Path to COCO annotations.json")
    parser.add_argument('--id', type=int, default=None, help="ID of a specific image to visualize")
    parser.add_argument('--limit', type=int, default=None, help="Max number of images to visualize")
    parser.add_argument('--out', type=str, default="outputs/visualized", help="Output directory")
    args = parser.parse_args()
    
    visualize_coco(args.json, image_id=args.id, limit=args.limit, output_dir=args.out)
