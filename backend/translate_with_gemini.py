import google.generativeai as genai
import os
import json
import time

class MangaTranslator:
    """
    A class to handle the batch translation of manga text from JSON files
    using the Google Gemini API.
    """
    def __init__(self, api_key: str, json_folder: str, model_name: str = "gemini-1.5-flash"):
        """
        Initializes the MangaTranslator.

        Args:
            api_key (str): Your Google Gemini API key.
            json_folder (str): The path to the folder containing the JSON files.
            model_name (str, optional): The Gemini model to use for translation. 
                                        Defaults to "gemini-1.5-flash".
        """
        self.api_key = api_key
        self.json_folder = json_folder
        self.model_name = model_name
        self.model = None
        self._file_text_map = [] # To map translations back to files
        self._all_japanese_text = [] # To hold all text for one API call

    def _configure_gemini(self) -> bool:
        """Configures the Gemini API and model."""
        if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
            print("ERROR: API key is not set. Please provide a valid Gemini API key.")
            return False
            
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print("Gemini API configured successfully.")
            return True
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            return False

    def _read_source_files(self) -> bool:
        """Reads all JSON files and extracts Japanese text."""
        if not os.path.isdir(self.json_folder):
            print(f"Error: The folder '{self.json_folder}' does not exist.")
            return False
            
        json_files = sorted([f for f in os.listdir(self.json_folder) if f.endswith('.json')])

        if not json_files:
            print(f"No JSON files found in '{self.json_folder}'.")
            return False

        print(f"Found {len(json_files)} JSON files to process.")
        
        for filename in json_files:
            filepath = os.path.join(self.json_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if "translations" in data and isinstance(data["translations"], list):
                    for i, item in enumerate(data["translations"]):
                        if "japanese_text" in item:
                            self._all_japanese_text.append(item["japanese_text"])
                            self._file_text_map.append({'file': filepath, 'index': i})
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Could not process {filename}: {e}")
        
        return True

    def _translate_batch(self) -> list | None:
        """Sends the collected text to the Gemini API for translation."""
        if not self._all_japanese_text:
            print("No text was extracted for translation.")
            return []

        prompt_lines = [f"{i+1}. {text}" for i, text in enumerate(self._all_japanese_text)]
        full_prompt = (
            "You are an expert Japanese manga translator. I will provide a numbered list of Japanese lines from a manga chapter. "
            "Please provide a natural-sounding English translation for each line. "
            "Maintain the original numbering in your response. Do not add any extra commentary, just the translated lines.\n\n"
            "Here is the text:\n"
            + "\n".join(prompt_lines)
        )

        print(f"Sending {len(self._all_japanese_text)} lines to Gemini for translation...")
        try:
            response = self.model.generate_content(full_prompt)
            translated_lines = response.text.strip().split('\n')
            
            clean_translations = []
            for line in translated_lines:
                if ". " in line:
                    clean_translations.append(line.split('. ', 1)[1])
                else:
                    clean_translations.append(line)

            if len(clean_translations) != len(self._all_japanese_text):
                print("Warning: Mismatch between original and translated line counts.")
                return None
            
            print("Translation successful.")
            return clean_translations
        except Exception as e:
            print(f"An error occurred during translation: {e}")
            return None

    def _update_files(self, translated_texts: list):
        """Updates the JSON files with the new translations."""
        if not translated_texts:
            print("No translations to apply.")
            return

        print("Updating JSON files...")
        for i, translation_info in enumerate(self._file_text_map):
            filepath = translation_info['file']
            item_index = translation_info['index']
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                data["translations"][item_index]["english_text"] = translated_texts[i]
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Failed to update {filepath}: {e}")

    def run(self):
        """Executes the entire translation process."""
        print("--- Starting Manga Batch Translation ---")
        if not self._configure_gemini():
            return
        
        if not self._read_source_files():
            return
            
        translated_texts = self._translate_batch()
        
        if translated_texts is not None:
            self._update_files(translated_texts)
            print("--- Translation process completed successfully! ---")
        else:
            print("--- Translation process failed. ---")


# if __name__ == "__main__":
#     # --- How to Use ---
#     # 1. Fill in your API Key.
#     # 2. Set the folder path to where your JSON files are.
#     # 3. Run the script from your terminal: python your_script_name.py

#     YOUR_API_KEY = "YOUR_API_KEY_HERE" 
#     FOLDER_WITH_JSONS = "chapter_1"    

#     # Create an instance of the translator
#     translator = MangaTranslator(api_key=YOUR_API_KEY, json_folder=FOLDER_WITH_JSONS)
    
#     # Run the translation process
#     translator.run()
