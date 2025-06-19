## FINAL SCRIPT TO MAKE BOUNDING BOX USING YOLO (NO CHANGE NEEDED)

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np
import os
import json
from IPython.display import clear_output
import shutil
class yolo_detect:

    def __init__(self, input_folders):
        self.input_folders = input_folders

        # self.model_bubble = YOLO("./backend/YOLO/best_YOLOV11.pt")
        self.model_bubble = YOLO("./backend/YOLO/yolov11_best_5_class.pt")
        self.model_onomatope = YOLO("./backend/YOLO/best_yoloV11Seeg.pt")
        self.model_text = YOLO("./backend/YOLO/comic-text-segmenter.pt")

        self.output_folder_image = f"{self.input_folders}/image/"
        self.output_folder_mask = f"{self.input_folders}/mask/"
        self.output_folder_json = f"{self.input_folders}/bbox/"

        os.makedirs(self.output_folder_image, exist_ok=True)
        os.makedirs(self.output_folder_mask, exist_ok=True)
        os.makedirs(self.output_folder_json, exist_ok=True)

    def scale_polygon(self, polygon, scale=1.1):
        # Compute centroid
        M = np.mean(polygon, axis=0)
        # Scale around centroid
        scaled = (polygon - M) * scale + M
        return scaled.astype(np.int32)


    def expand_box(self, x1, y1, x2, y2, scale=1.1, img_shape=None):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = (x2 - x1) * scale, (y2 - y1) * scale
        new_x1 = int(max(cx - w / 2, 0))
        new_y1 = int(max(cy - h / 2, 0))
        new_x2 = int(min(cx + w / 2, img_shape[1] - 1))
        new_y2 = int(min(cy + h / 2, img_shape[0] - 1))
        return new_x1, new_y1, new_x2, new_y2

    def yolo_det(self):
        # Folder containing the images
        input_folder = self.input_folders

        # Load models once
        model_bubble = self.model_bubble
        model_onomatope = self.model_onomatope
        model_text = self.model_text

        # Process each image in the folder
        for filename in os.listdir(input_folder):
            bbox_data = {
            "text": [],
            "bubble": [],
            "onomatope": [],
            }
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_folder, filename)
                img_input = cv2.imread(input_path)
                # print(input_path)
                # print(img_input is None)

                if img_input is None:
                    print(f"Failed to load {filename}")
                    continue
                
                img_shape = img_input.shape

                img_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                masking = img_rgb * 0

                result_bubble = model_bubble(img_input, verbose=False)
                result_onomatope = model_onomatope(img_input, verbose=False)
                result_text = model_text(img_input, verbose=False)

                for result in result_bubble:
                    for box in result.boxes:
                        cls = int(box.cls)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox_data["bubble"].append({
                            "x1": x1,
                            "y1": y1, 
                            "x2": x2,
                            "y2": y2,
                            "cls": cls
                        })
                        # cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        # Optional: cv2.rectangle(masking, (x1, y1), (x2, y2), (0, 255, 0), -1)

                #Bounding box only
                for result in result_text:
                    if result.boxes is not None:
                        for i, box in enumerate(result.boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            x1,y1,x2,y2 = self.expand_box(x1, y1, x2, y2, scale=1.05, img_shape=img_shape)
                            
                            text_entry = {
                                "x1": x1,
                                "y1": y1, 
                                "x2": x2,
                                "y2": y2,
                            }
                            
                            # Add mask if available
                            if result.masks is not None and i < len(result.masks.xy):
                                mask_points = result.masks.xy[i].tolist()
                                text_entry["mask"] = mask_points
                            
                            bbox_data["text"].append(text_entry)

                for result in result_text:
                    if result.masks is not None:
                        for mas in result.masks.xy:
                            polygon = np.array(mas, np.int32)
                            cv2.polylines(img_rgb, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)
                            cv2.fillPoly(masking, [polygon], color=(255, 255, 255))

                for result in result_onomatope:
                    if result.masks is not None and result.boxes is not None:
                        for i, (mask, box) in enumerate(zip(result.masks.xy, result.boxes)):
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cls = int(box.cls) if hasattr(box, 'cls') else 0
                            
                            # Get mask polygon
                            mask_points = mask.tolist()
                            
                            # Save both bbox and mask data
                            bbox_data["onomatope"].append({
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "cls": cls,
                                "mask": mask_points
                            })
                            
                            polygon = np.array(mask, np.int32)
                            cv2.polylines(img_rgb, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)
                            cv2.fillPoly(masking, [polygon], color=(255, 255, 255))


                # clear_output()
                # filename = os.path.splitext(filename)[0]
                # if filename == "25-656077ad1fe4c":
                        
                #     # Display the results
                #     plt.imshow(img_rgb)
                #     plt.title(f"Detections: {filename}")
                #     plt.show()

                #     plt.imshow(masking)
                #     plt.title(f"Mask: {filename}")
                #     plt.show()
                #     break



                # DEBUG PURPOSES
                # break

                # Optional: save outputs
                filename = os.path.splitext(filename)[0]
                # cv2.imwrite(f"{self.output_folder_mask}{filename}.png", masking)
                cv2.imwrite(f"{self.output_folder_image}{filename}.png", img_input)

                # output_image_path = os.path.join(self.output_folder_image, filename)
                # shutil.move(input_path, output_image_path)

                json_path = os.path.join(self.output_folder_json, f"{filename}.json")
                with open(json_path, 'w') as json_file:
                    json.dump(bbox_data, json_file, indent=2, ensure_ascii=False)
