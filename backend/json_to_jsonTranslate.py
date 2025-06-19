import os
import json
import cv2
import numpy as np
from PIL import Image
from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator
from tqdm import tqdm
from IPython.display import clear_output
import re

class JsonToJsonTranslate:
    def __init__(self, image_folder, json_folder, output_folder, language='en'):
        self.ocr = MangaOcr()
        self.translator_en = GoogleTranslator(source='ja', target=language)
        self.image_folder = image_folder 
        self.json_folder = json_folder
        self.output_folder = output_folder
        pass

    def shorten_repetitive_words(self,text, max_repeats=3):
        # This pattern finds any word repeated more than once, like "hello hello hello hello"
        pattern = r'\b(\w+)(?:\s+\1){' + str(max_repeats) + r',}'

        def replacer(match):
            word = match.group(1)
            return ' '.join([word] * max_repeats)

        shortened_text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)
        return shortened_text

    # Check if text box is inside a bubble
    def is_inside_bubble(self,x1, y1, x2, y2, bubble_boxes):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for b in bubble_boxes:
            if b["x1"] <= cx <= b["x2"] and b["y1"] <= cy <= b["y2"]:
                return True, b["cls"]
        return False, 5

    # Sort boxes top-to-bottom, right-to-left
    def sort_boxes(self,boxes, y_thresh=20):
        boxes.sort(key=lambda b: b[0][1])
        sorted_boxes = []
        current_line = []
        for (coords, crop) in boxes:
            if not current_line:
                current_line.append((coords, crop))
                continue
            prev_y = current_line[-1][0][1]
            if abs(coords[1] - prev_y) < y_thresh:
                current_line.append((coords, crop))
            else:
                current_line.sort(key=lambda b: b[0][2], reverse=True)
                sorted_boxes.extend(current_line)
                current_line = [(coords, crop)]
        if current_line:
            current_line.sort(key=lambda b: b[0][2], reverse=True)
            sorted_boxes.extend(current_line)
        return sorted_boxes

    def translate_and_save_json(self):
        for filename in tqdm(os.listdir(self.image_folder)):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            base_name = os.path.splitext(filename)[0]
            img_path = os.path.join(self.image_folder, filename)
            json_path = os.path.join(self.json_folder, base_name + '.json')
            
            if not os.path.exists(json_path):
                print(f"Missing JSON for {filename}")
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            img_input = cv2.imread(img_path)
            if img_input is None:
                print(f"Failed to load image: {img_path}")
                continue

            text_boxes = data.get("text", [])
            bubble_boxes = data.get("bubble", [])
            boxes_with_coords = []

            # Prepare boxes with coordinates and cropped images
            for box in text_boxes:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                cropped = img_input[y1:y2, x1:x2]
                boxes_with_coords.append(((x1, y1, x2, y2), cropped))

            # Sort boxes
            sorted_boxes = self.sort_boxes(boxes_with_coords)

            # Create a new data structure for translations
            translations = {
                "original_data": data,
                "translations": []
            }

            # Process each text box
            for (x1, y1, x2, y2), cropped in sorted_boxes:
                try:
                    pil_img = Image.fromarray(cropped)
                    jp_text = self.ocr(pil_img)
                    en_text = self.translator_en.translate(jp_text)
                    en_text = self.shorten_repetitive_words(en_text, 3)
                    
                    inside_bubble, cls = self.is_inside_bubble(x1, y1, x2, y2, bubble_boxes)
                    
                    # Add translation data
                    translations["translations"].append({
                        "coords": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "japanese_text": jp_text,
                        "english_text": en_text,
                        "inside_bubble": inside_bubble,
                        "bubble_class": cls
                    })
                    
                    # print(f"Translated: {jp_text} → {en_text}")
                    
                except Exception as e:
                    print(f"Error processing box at {x1},{y1},{x2},{y2}: {e}")
                    translations["translations"].append({
                        "coords": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "japanese_text": "[Error]",
                        "english_text": "[Error]",
                        "inside_bubble": self.is_inside_bubble(x1, y1, x2, y2, bubble_boxes),
                        "bubble_class": cls
                    })
                    return

            # Save translations to a new JSON file
            translation_json_path = os.path.join(self.output_folder, base_name + '_translated.json')
            with open(translation_json_path, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
                
            print(f"Saved translations for {filename} to {translation_json_path}")
        return