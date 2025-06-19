import google.generativeai as genai
import os
import json
import time

#PROVIDE CONTEXT FOR THE


class MangaTranslator:
    """
    A class to handle the batch translation of manga text from JSON files
    using the Google Gemini API.
    """
    def __init__(self, 
                 api_key: str, 
                 json_folder: str, 
                 target_language: str = "English",
                 manga_context_url: str = None,
                 model_name: str = "gemini-1.5-flash"):
        """
        Initializes the MangaTranslator.

        Args:
            api_key (str): Your Google Gemini API key.
            json_folder (str): The path to the folder containing the JSON files.
            target_language (str, optional): The language to translate the text into. Defaults to "English".
            manga_context_url (str, optional): A URL providing context for the manga series. Defaults to None.
            model_name (str, optional): The Gemini model to use for translation. Defaults to "gemini-1.5-flash".
        """
        self.api_key = api_key
        self.json_folder = json_folder
        self.target_language = target_language
        self.manga_context_url = manga_context_url
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

        # Dynamically build the prompt
        prompt_header_lines = [
            f"You are an expert Japanese manga translator. I will provide a numbered list of Japanese lines from a manga chapter.",
            f"Please provide a natural-sounding {self.target_language} translation for each line.",
            "Maintain the original numbering in your response. Do not add any extra commentary, just the translated lines."
        ]
        if self.manga_context_url:
            prompt_header_lines.append(f"For context on the series, please refer to this link: {self.manga_context_url}")
        
        prompt_header = "\n".join(prompt_header_lines)
        
        prompt_body_lines = [f"{i+1}. {text}" for i, text in enumerate(self._all_japanese_text)]
        full_prompt = (
            prompt_header + "\n\n"
            "Here is the text:\n"
            + "\n".join(prompt_body_lines)
        )

        print(f"Sending {len(self._all_japanese_text)} lines to Gemini for translation into {self.target_language}...")
        try:
            response = self.model.generate_content(full_prompt)
            translated_lines = response.text.strip().split('\n')
            
            clean_translations = []
            for line in translated_lines:
                # This handles lines that might not have a number, e.g., if the model adds extra newlines
                if ". " in line:
                    clean_translations.append(line.split('. ', 1)[1])
                else:
                    clean_translations.append(line)

            if len(clean_translations) != len(self._all_japanese_text):
                print(f"Warning: Mismatch between original ({len(self._all_japanese_text)}) and translated ({len(clean_translations)}) line counts.")
                # Even with a mismatch, we can try to proceed if needed, but it's risky.
                # For now, we'll return None to indicate an issue.
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

        print(f"Updating JSON files with {self.target_language} translations...")
        for i, translation_info in enumerate(self._file_text_map):
            filepath = translation_info['file']
            item_index = translation_info['index']
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # We'll use a dynamic key for the translation, e.g., 'english_text', 'spanish_text'
                translation_key = f'{self.target_language.lower()}_text'
                data["translations"][item_index][translation_key] = translated_texts[i]
                
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
#     # 3. Optionally, set the target language and context URL.
#     # 4. Run the script from your terminal: python your_script_name.py

#     YOUR_API_KEY = "YOUR_API_KEY_HERE" 
#     FOLDER_WITH_JSONS = "chapter_1"    

#     # --- Example 1: Basic English Translation (Default) ---
#     # translator = MangaTranslator(api_key=YOUR_API_KEY, json_folder=FOLDER_WITH_JSONS)
    
#     # --- Example 2: Translate to Spanish with Context ---
#     translator = MangaTranslator(
#         api_key=YOUR_API_KEY, 
#         json_folder=FOLDER_WITH_JSONS,
#         target_language="Spanish",
#         manga_context_url="https://mangadex.org/title/5d2c516a-b041-421b-a9a2-ba231e2081c8/"
#     )
    
#     # Run the translation process
#     translator.run()
