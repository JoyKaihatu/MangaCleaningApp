# 📚 MangaCleaningApp

**MangaCleaningApp** is a Flask-based web application that automatically translates Japanese manga or comic pages into English (or other languages). It combines speech bubble detection, text extraction, inpainting, translation, and text re-rendering into a complete end-to-end pipeline.

![Screenshot Placeholder](https://github.com/JoyKaihatu/MangaCleaningApp/assets/your_screenshot.png)

---

## ✨ Features

* 🚀 Upload single or multiple manga pages (ZIP)
* 🔍 YOLO-based detection of:

  * Speech bubbles
  * Text regions
  * Onomatopoeia
* ✏️ Visual editor to tweak bounding boxes before processing
* 🔠 Choose translation method:

  * Google Gemini (via API key)
  * Google Translate
  * JSON-only mode
* 🌍 Supports multiple output languages (e.g. English, Indonesian, etc.)
* ⚖️ Adjustable font type per bubble type
* ⬆️ Auto-adjusting font size with stroke rendering
* 📄 Clean ZIP download of final images + JSON metadata

---

## ♻️ How It Works

1. **Upload Pages**

   * Upload an image or a `.zip` of multiple manga pages
2. **Detect Annotations**

   * YOLO models detect speech bubbles, text, and onomatopoeia
3. **Edit Annotations** *(Optional)*

   * Move, resize, or delete detected regions via an intuitive editor
4. **Choose Options**

   * Pick fonts per bubble type
   * Select translation method
   * Choose target language (e.g. `en`, `id`, `es`)
   * Set min/max font size
5. **Preview Output**

   * See rendered translations or clean pages before download
6. **Download Results**

   * Download a zip of translated images and translation JSON

---

## 📁 Folder Structure (per session)

```
uploads/
  ├── <session_id>/
  │   ├── image/               # Original input images
  │   ├── bbox/                # Initial YOLO annotation JSONs
  │   ├── edited/              # User-edited annotations
  │   ├── mask/                # Mask images per region
  │   ├── translated_json/     # Translated + OCR JSON
  │   ├── final_output/
  │   │   ├── translated_images/
  │   │   └── inpainted/
  │   ├── translation_config.json
  │   └── status.json
```

---

## ⚡ Requirements

* Python 3.11+
* Flask
* OpenCV
* Pillow
* PyTorch (for running YOLO models)
* FreeType
* `googletrans` (for Google Translate fallback)
* Access to Google AI Lab API Key (for Google Gemini translation mode)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
flask run
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## ✉ TODO / Improvements

* [ ] Add user login/session history
* [ ] Add language auto-detect
* [ ] Deploy to Hugging Face Spaces / Render
* [ ] Add loading indicator during processing
* [ ] Add downloadable logs or translation text preview

---

## 📑 License

MIT License

---

## 🙏 Credits

Made by [Joy Kaihatu](https://github.com/JoyKaihatu)

YOLO model training, OCR integration, FreeType rendering, and full-stack architecture designed and built from scratch as part of a university thesis.

---
