#Version 3 Draw text to image

import os
import json
import cv2
import numpy as np
import freetype
from tqdm import tqdm
from collections import defaultdict

# Paths
# inpainted_folder = './output_inpaint_zeitaku_7'
# translation_json_folder = './translated_json_zeitaku_7'
# output_folder = './outpu_translated_zeitaku_7_3_V3'


# # Create output directory if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

class TranslationDrawer:
    def __init__(self, inpainted_folder, translation_json_folder, output_folder):
        self.inpainted_folder = inpainted_folder
        self.translation_json_folder = translation_json_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def detect_overlaps(self,translations):
        """
        Detect and resolve overlapping bounding boxes by adjusting their rendering order.
        Returns a list of translations sorted by rendering priority.
        """
        # Sort boxes by area (smallest to largest)
        sorted_translations = sorted(translations, key=lambda t: 
                                (t["coords"]["x2"] - t["coords"]["x1"]) * 
                                (t["coords"]["y2"] - t["coords"]["y1"]))
        
        # Check for overlaps
        for i in range(len(sorted_translations)):
            box1 = sorted_translations[i]["coords"]
            for j in range(i + 1, len(sorted_translations)):
                box2 = sorted_translations[j]["coords"]
                
                # Check for overlap
                if (box1["x1"] < box2["x2"] and box1["x2"] > box2["x1"] and
                    box1["y1"] < box2["y2"] and box1["y2"] > box2["y1"]):
                    # Mark as overlapping
                    sorted_translations[i]["overlaps"] = True
                    sorted_translations[j]["overlaps"] = True
                    
                    # Calculate overlap area
                    overlap_width = min(box1["x2"], box2["x2"]) - max(box1["x1"], box2["x1"])
                    overlap_height = min(box1["y2"], box2["y2"]) - max(box1["y1"], box2["y1"])
                    overlap_area = overlap_width * overlap_height
                    
                    # Store overlap information for later use
                    if "overlap_info" not in sorted_translations[i]:
                        sorted_translations[i]["overlap_info"] = []
                    if "overlap_info" not in sorted_translations[j]:
                        sorted_translations[j]["overlap_info"] = []
                        
                    sorted_translations[i]["overlap_info"].append({
                        "with_idx": j,
                        "area": overlap_area
                    })
                    sorted_translations[j]["overlap_info"].append({
                        "with_idx": i,
                        "area": overlap_area
                    })
        
        # Sort by overlap status (non-overlapping first, then by size)
        return sorted(sorted_translations, key=lambda t: (t.get("overlaps", False), 
                                                        -(t["coords"]["x2"] - t["coords"]["x1"]) * 
                                                        (t["coords"]["y2"] - t["coords"]["y1"])))

    def is_point_in_bubble(self, point, mask, tolerance=10):
        """Check if a point is inside a speech bubble using the mask"""
        x, y = point
        # Create a small region around the point to check
        x_min, x_max = max(0, x - tolerance), min(mask.shape[1] - 1, x + tolerance)
        y_min, y_max = max(0, y - tolerance), min(mask.shape[0] - 1, y + tolerance)
        
        # If any pixel in this region is non-zero (part of a bubble), consider it inside
        return np.any(mask[y_min:y_max+1, x_min:x_max+1] > 0)

    def create_bubble_mask(self, img):
        """Create a mask for potential speech bubbles"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get potential bubble areas (usually white or very light)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        
        # Close small gaps
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours (bubble boundaries)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask
        mask = np.zeros_like(gray)
        for contour in contours:
            # Only use contours of reasonable size
            area = cv2.contourArea(contour)
            if area > 1000:  # Adjust threshold as needed
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        return mask

    def draw_text_on_image_freetype(self, img, text, x1, y1, x2, y2, inside_bubble, bubble_mask=None, box_expansion=0, font_path="./fonts/CC Wild Words Roman.ttf", min_text_size = 12, max_text_size = 40, auto_font_size=False):
        # Apply box expansion to the coordinates
        x1 = max(0, x1 - box_expansion)
        y1 = max(0, y1 - box_expansion)
        x2 = min(img.shape[1], x2 + box_expansion)
        y2 = min(img.shape[0], y2 + box_expansion)
        print("apply box expansion to the coordinates done")


        # Respect bubble boundaries if needed
        if inside_bubble and bubble_mask is not None:
            # Shrink box if it extends beyond bubble
            box_points = [(x1+i, y1+j) for i in range(0, x2-x1, 5) for j in [0, y2-y1]] + \
                        [(x1+i, y2-j) for i in range(0, x2-x1, 5) for j in [0, y2-y1]] + \
                        [(x1+i, y1+j) for i in [0, x2-x1] for j in range(0, y2-y1, 5)] + \
                        [(x2-i, y1+j) for i in [0, x2-x1] for j in range(0, y2-y1, 5)]
            
            outside_points = [p for p in box_points if not self.is_point_in_bubble(p, bubble_mask)]
            
            if outside_points:
                # Shrink the box to stay within the bubble
                shrink_factor = 0.9  # Adjust as needed
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                new_width = int((x2 - x1) * shrink_factor)
                new_height = int((y2 - y1) * shrink_factor)
                x1 = max(0, cx - new_width // 2)
                y1 = max(0, cy - new_height // 2)
                x2 = min(img.shape[1], cx + new_width // 2)
                y2 = min(img.shape[0], cy + new_height // 2)
        
        text_padding = 4
        box_padding = 6
        line_spacing = 6
        stroke_width = 2  # Width of the outline/stroke

        x1_p, y1_p = x1 + text_padding, y1 + text_padding
        x2_p, y2_p = x2 - text_padding, y2 - text_padding
        if inside_bubble:
            x1_p += 6
            y1_p += 6
            x2_p -= 6
            y2_p -= 6

        if x2_p <= x1_p or y2_p <= y1_p:
            return

        cropped_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        brightness = np.mean(cropped_gray)
        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        # Opposite color for the stroke/outline
        stroke_color = (255, 255, 255) if text_color == (0, 0, 0) else (0, 0, 0)

        max_width = x2_p - x1_p
        max_height = y2_p - y1_p

        # Create the FreeType Face with proper error handling
        try:
            face = freetype.Face(font_path)
            
            def get_line_width(text_line, font_size):
                face.set_char_size(font_size * 48)
                width = 0
                for char in text_line:
                    face.load_char(char)
                    width += face.glyph.advance.x >> 6
                return width

            def wrap_text_freetype(text, font_size):
                face.set_char_size(font_size * 48)
                words = text.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = f"{current_line} {word}".strip()
                    width = get_line_width(test_line, font_size)
                    if width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                return lines

            
            if auto_font_size:
                best_font_size = min_text_size
                best_lines = wrap_text_freetype(text, best_font_size)
                face.set_char_size(best_font_size * 48)
                line_height = face.size.height >> 6

                for size in range(min_text_size, max_text_size + 1):
                    face.set_char_size(size * 48)
                    lines = wrap_text_freetype(text, size)
                    line_height = face.size.height >> 6
                    total_height = len(lines) * (line_height + line_spacing) - line_spacing
                    max_line_width = max([get_line_width(line, size) for line in lines] + [0])

                    if total_height <= (max_height - stroke_width * 2) and max_line_width <= (max_width - stroke_width * 2):
                        best_font_size = size
                        best_lines = lines
                face.set_char_size(best_font_size * 48)
            else:
                best_font_size = min_text_size
                best_lines = wrap_text_freetype(text, best_font_size)
                face.set_char_size(best_font_size * 48)

            reduced_max_width = max_width - stroke_width * 2
            reduced_max_height = max_height - stroke_width * 2


            for size in range(min_text_size, max_text_size + 1, 1):
                lines = wrap_text_freetype(text, size)
                face.set_char_size(size * 48)
                line_height = face.size.height >> 6
                total_height = len(lines) * (line_height + line_spacing) - line_spacing
                max_line_width = max([get_line_width(line, size) for line in lines] + [0])
                if total_height <= reduced_max_height and max_line_width <= reduced_max_width:
                    best_font_size = size
                    best_lines = lines
            
            print("best_font_size:", best_font_size)

            face.set_char_size(best_font_size * 48)
            line_height = face.size.height >> 6

            text_block_height = len(best_lines) * (line_height + line_spacing) - line_spacing
            
            # Center the text block both horizontally and vertically within the box
            cx, cy = (x1_p + x2_p) // 2, (y1_p + y2_p) // 2
            top_y = cy - text_block_height // 2

            # Create a buffer to hold character bitmap data
            def draw_characters(lines, color, offset_x=0, offset_y=0):
                for i, line in enumerate(lines):
                    line_width = get_line_width(line, best_font_size)
                    start_x = cx - line_width // 2  # Center each line horizontally
                    pen_x = start_x
                    pen_y = top_y + i * (line_height + line_spacing) + line_height

                    for ch in line:
                        face.load_char(ch)
                        bitmap = face.glyph.bitmap
                        top = face.glyph.bitmap_top
                        left = face.glyph.bitmap_left
                        w, h = bitmap.width, bitmap.rows

                        x = pen_x + left + offset_x
                        y = pen_y - top + offset_y

                        for row in range(h):
                            for col in range(w):
                                val = bitmap.buffer[row * w + col]
                                if val > 0:
                                    px, py = x + col, y + row
                                    if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
                                        # Only draw if we're inside the bubble or bubble check isn't needed
                                        if not inside_bubble or bubble_mask is None or self.is_point_in_bubble((px, py), bubble_mask):
                                            alpha = val / 255.0
                                            for c in range(3):  # BGR channels
                                                img[py, px, c] = int((1 - alpha) * img[py, px, c] + alpha * color[c])

                        pen_x += face.glyph.advance.x >> 6

            # Draw stroke/outline first (in 8 directions to create a complete outline)
            offsets = [
                (-stroke_width, -stroke_width), (0, -stroke_width), (stroke_width, -stroke_width),
                (-stroke_width, 0),                               (stroke_width, 0),
                (-stroke_width, stroke_width),  (0, stroke_width),  (stroke_width, stroke_width)
            ]
            
            for offset_x, offset_y in offsets:
                draw_characters(best_lines, stroke_color, offset_x, offset_y)
            
            # Then draw the main text on top
            draw_characters(best_lines, text_color, 0, 0)
        except Exception as e:
            print(f"Error in draw_text_on_image_freetype: {e}")
        finally:
            # Clean up the FreeType face after usage
            if 'face' in locals():
                del face

    def draw_translations(self,font_config_path, box_expansion=0, auto_expand=False, auto_font_size=False, min_text_size=16, base_font_location="fonts/", max_text_size = 40):
        """
        Draw translations on images with expanded text boxes
        
        Parameters:
        - box_expansion: Number of pixels to expand each bounding box in all directions (manual mode)
        - auto_expand: If True, automatically determine expansion based on text content
        - min_text_size: Minimum font size to aim for in auto expansion mode
        - font_path: Path to the font file
        """
        # Create a single FreeType face for auto-expansion calculations to avoid resource leaks
        with open(font_config_path, 'r') as f:
            font_config = json.load(f)
        fonts = font_config.get("font_choices", [])

        # temp_face = None
        if auto_expand:
            try:
                # temp_face = freetype.Face(font_path)
                # temp_face.set_char_size(min_text_size * 48)
                for i in range(0,5):
                    font_path = os.path.join(base_font_location, fonts[str(i)])
                    temp_face = freetype.Face(font_path)
                    temp_face.set_char_size(min_text_size * 48)
            except Exception as e:
                print(f"Error loading font for auto-expansion: {e}")
                auto_expand = False  # Disable auto expansion if font loading fails
        
        try:
            for filename in tqdm(os.listdir(self.translation_json_folder)):
                if not filename.lower().endswith('_translated.json'):
                    continue

                base_name = filename.replace('_translated.json', '')
                image_filename = base_name + '.jpg'  # Assuming jpg, adjust if needed
                
                # Check for png if jpg doesn't exist
                if not os.path.exists(os.path.join(self.inpainted_folder, image_filename)):
                    image_filename = base_name + '.png'
                    if not os.path.exists(os.path.join(self.inpainted_folder, image_filename)):
                        print(f"Could not find image file for {base_name}")
                        continue
                
                inpainted_path = os.path.join(self.inpainted_folder, image_filename)
                translation_json_path = os.path.join(self.translation_json_folder, filename)

                # Load translation data
                with open(translation_json_path, 'r', encoding='utf-8') as f:
                    translation_data = json.load(f)

                # Load inpainted image
                inpainted_img = cv2.imread(inpainted_path)
                if inpainted_img is None:
                    print(f"Failed to load inpainted image: {inpainted_path}")
                    continue
                    
                # Create bubble mask for this image
                bubble_mask = self.create_bubble_mask(inpainted_img)
                
                # Detect and resolve overlapping boxes
                sorted_translations = self.detect_overlaps(translation_data["translations"])
                
                # Draw text on image - draw from bottom to top for overlaps
                for translation in sorted_translations:
                    try:
                        coords = translation["coords"]
                        x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
                        english_text = translation["english_text"]
                        inside_bubble = translation["inside_bubble"]
                        cls = translation.get("bubble_class", 6)  # Default to 6 if not specified
                        temp_face = None
                        font_path = None

                        try:
                            # Load the font for this translation
                            font_path = os.path.join(base_font_location, fonts[str(cls)])
                            temp_face = freetype.Face(font_path)
                            temp_face.set_char_size(min_text_size * 48)
                        except Exception as e:
                            print("Error loading font for translation:", e, font_path)
                            print(font_path)
                        
                        if english_text != "[Error]":
                            # Calculate auto expansion if enabled
                            current_expansion = box_expansion
                            if auto_expand and temp_face:
                                print("masuk auto expand")
                                # Determine how much space we need for this text
                                original_width = x2 - x1
                                original_height = y2 - y1

                                print("original_width dan height done", original_width, original_height)
                                
                                # Estimate how much space the text would need at min_text_size
                                text_length = len(english_text)
                                print("text length done", text_length)
                                print(temp_face.size.max_advance)
                                print(temp_face.size.max_advance >> 6)
                                chars_per_line = max(1, original_width // (temp_face.size.max_advance >> 6))
                                print("chars per line done", chars_per_line)
                                print(text_length)
                                print(chars_per_line)
                                estimated_lines = max(1, text_length // chars_per_line)
                                print("estimated lines done", estimated_lines)
                                line_height = temp_face.size.height >> 6
                                print(line_height)
                                estimated_height = estimated_lines * line_height * 1.2  # 1.2 for line spacing
                                print("estimate space needed for text done", estimated_height)
                                
                                # Calculate expansion needed
                                width_ratio = estimated_height / original_height
                                expansion_needed = int(original_width * (width_ratio - 1) / 2)
                                print("calculate expansion needed done", expansion_needed)
                                
                                # Limit expansion to reasonable values
                                current_expansion = min(100, max(0, expansion_needed))
                                print(f"Auto expansion for text: {english_text[:20]}... - {current_expansion}px")
                            
                            # If overlapping, reduce expansion to minimize overlap
                            if translation.get("overlaps", False):
                                current_expansion = max(0, current_expansion - 5)
                                print("Overlapping detected, reducing expansion to", current_expansion)
                            
                            self.draw_text_on_image_freetype(
                                inpainted_img, 
                                english_text, 
                                x1, y1, x2, y2, 
                                inside_bubble,
                                bubble_mask=bubble_mask if inside_bubble else None,
                                box_expansion=current_expansion,
                                font_path=font_path,
                                min_text_size=min_text_size,
                                max_text_size=max_text_size
                            , auto_font_size=auto_font_size)
                            print(f"Drew text: {english_text}")
                    except Exception as e:
                        print(f"Error drawing text: {e}")

                # Save the output image
                output_path = os.path.join(self.output_folder, image_filename)
                cv2.imwrite(output_path, inpainted_img)
                print(f"Saved image with translations to {output_path}")
        finally:
            # Clean up the FreeType face when done
            if temp_face is not None:
                del temp_face

# if __name__ == "__main__":
#     # Choose one of these options:
#     try:
#         # Option 1: Manual fixed expansion
#         # draw_translations(box_expansion=15)  # Expand boxes by 15 pixels in all directions
        
#         # Option 2: Auto expansion based on text content
#         draw_translations(auto_expand=True, min_text_size=22, max_text_size=28)
        
#         # Option 3: Both manual and auto (will use the larger of the two)
#         # draw_translations(box_expansion=10, auto_expand=True, min_text_size=16)
#     except Exception as e:
#         print(f"An error occurred during translation: {e}")
#     finally:
#         # Force Python's garbage collector to clean up
#         import gc
#         gc.collect()