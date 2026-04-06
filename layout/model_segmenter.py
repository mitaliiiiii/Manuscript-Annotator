import os
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class LayoutSegmenter:
    def __init__(self, model_path=None):
        """
        Initializes the YOLOv8-segmentation model for layout analysis.
        :param model_path: Path to the .pt model weights. 
                           If None, it can be initialized later or use a default.
        """
        self.model = None
        if YOLO and model_path and os.path.exists(model_path):
            print(f"Loading YOLO model from {model_path}...")
            self.model = YOLO(model_path)
        elif not YOLO:
            print("Warning: 'ultralytics' library not found. Run 'pip install ultralytics' to use YOLO models.")

    def segment(self, image_path):
        """
        Runs inference on an image and returns regions in the standard format.
        """
        if not self.model:
            return None

        results = self.model(image_path)[0]
        regions = []
        
        if results.masks is not None:
            # results.masks.xy is a list of segments [N, [x1, y1, ...]]
            for i, mask in enumerate(results.masks.xy):
                cls = int(results.boxes.cls[i])
                conf = float(results.boxes.conf[i])
                bbox = results.boxes.xywh[i].tolist() # [xc, yc, w, h]
                
                # Convert xc, yc to top-left x, y for COCO
                x = int(bbox[0] - bbox[2]/2)
                y = int(bbox[1] - bbox[3]/2)
                w = int(bbox[2])
                h = int(bbox[3])

                regions.append({
                    "bbox": [x, y, w, h],
                    "polygon": [int(p) for p in mask.flatten().tolist()],
                    "category_id": cls + 1, # Shift by 1 if labels start at 0
                    "confidence": conf,
                    "area": float(w * h) # Approximation or calculate from mask
                })
        
        return regions
