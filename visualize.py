import cv2
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def visualize_results(image_path, json_path, output_path=None):
    """
    Draws bounding boxes and superimposed Hindi text on an image.
    Uses PIL for robust unicode text rendering (OpenCV lacks native Hindi text support).
    """
    print(f"Visualizing results for {image_path}...")
    
    # Verify inputs
    if not Path(image_path).exists() or not Path(json_path).exists():
        print("Image or JSON file not found.")
        return

    # Load image and JSON
    img = cv2.imread(str(image_path))
    if img is None:
        print("Failed to read image.")
        return
        
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Convert to PIL Image for drawing Unicode text
    # OpenCV's putText doesn't handle complex scripts perfectly by default
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Attempt to load a default font that supports Devanagari
    # In Windows, Mangal or Nirmala UI are generally available
    try:
        # Fallback list of fonts typical on Windows systems
        font = ImageFont.truetype("Nirmala.ttf", 24)
    except IOError:
        try:
            font = ImageFont.truetype("mangal.ttf", 24)
        except IOError:
            print("Warning: Hindi compatible font not found. Text rendering may display blocks.")
            font = ImageFont.load_default()

    for item in results:
        bbox = item.get("bbox", [])
        text = item.get("text", "")
        
        if len(bbox) == 4:
            x, y, w, h = bbox
            
            # Draw Bounding Box (Red)
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=3)
            
            # Superimpose recognized Hindi text slightly above the bounding box (Blue)
            # Add a small semi-transparent background to text for readability
            text_x = x
            text_y = max(0, y - 30)
            
            draw.text((text_x, text_y), text, font=font, fill=(0, 0, 255))

    # Convert back to OpenCV format
    final_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Determine Output Path
    if output_path is None:
        base = Path(image_path)
        output_path = base.parent / f"{base.stem}_visualized.jpg"
        
    cv2.imwrite(str(output_path), final_img)
    print(f"Visualized image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize OCR Pipeline Results")
    parser.add_argument('image', type=str, help="Path to original image")
    parser.add_argument('json', type=str, help="Path to JSON results file")
    parser.add_argument('--out', type=str, help="Optional output path", default=None)
    args = parser.parse_args()
    
    visualize_results(args.image, args.json, args.out)
