import datetime

class COCOFormatter:
    def __init__(self):
        self.categories = [
            {"id": 1, "name": "text_region", "supercategory": "layout"},
            {"id": 2, "name": "marginalia/notes", "supercategory": "layout"},
            {"id": 3, "name": "illustration/diagram", "supercategory": "layout"},
            {"id": 4, "name": "page_frame", "supercategory": "layout"},
            {"id": 5, "name": "damage/hole", "supercategory": "layout"}
        ]

    def format(self, all_page_data):
        """
        Converts internal page data into standard COCO JSON format.
        """
        coco_output = {
            "info": {
                "description": "Auto-Annotation Indic Manuscript Dataset",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "Auto-Annotator Pipeline",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d")
            },
            "images": [],
            "annotations": [],
            "categories": self.categories
        }

        ann_id = 1
        for img_id, page_data in enumerate(all_page_data, 1):
            # Image info
            coco_output["images"].append({
                "id": img_id,
                "width": page_data["width"],
                "height": page_data["height"],
                "file_name": page_data["file_path"]
            })

            # Annotations
            for region in page_data["regions"]:
                coco_output["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": region["category_id"],
                    "segmentation": [region["polygon"]],
                    "baseline": region.get("baseline", []),
                    "area": region["area"],
                    "bbox": region["bbox"],
                    "iscrowd": 0
                })
                ann_id += 1

        return coco_output
