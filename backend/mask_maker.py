import json
import cv2
import numpy as np
import os

class MaskMaker:
    def draw_polygon_mask(self, image, points, color=(255, 255, 255)):
        if points:
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(image, [pts], color)

    def draw_box_mask(self, image, x1, y1, x2, y2, color=(255, 255, 255)):
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)

    def process_json_and_mask(self, image_path, json_path, output_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return

        # Create a blank mask image (same size, black)
        image = image * 0

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Process "text" and "onomatope" sections
        for section in ["text", "onomatope"]:
            for item in data.get(section, []):
                mask = item.get("mask")
                if mask:
                    self.draw_polygon_mask(image, mask)
                else:
                    x1, y1 = int(item.get("x1", 0)), int(item.get("y1", 0))
                    x2, y2 = int(item.get("x2", 0)), int(item.get("y2", 0))
                    self.draw_box_mask(image, x1, y1, x2, y2)

        cv2.imwrite(output_path, image)
        print(f"✅ Masked image saved to {output_path}")

    def process_all_json_in_folder(self, json_folder, image_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(json_folder):
            if filename.endswith(".json"):
                json_path = os.path.join(json_folder, filename)
                base_name = os.path.splitext(filename)[0]

                # Look for image with any common extension
                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    image_path = os.path.join(image_folder, base_name + ext)
                    if os.path.exists(image_path):
                        break
                else:
                    print(f"⚠️ Image not found for {filename}, skipping.")
                    continue

                output_path = os.path.join(output_folder, base_name + "_masked.jpg")
                self.process_json_and_mask(image_path, json_path, output_path)
