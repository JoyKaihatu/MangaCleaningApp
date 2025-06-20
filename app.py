from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, send_file, current_app, session
import os
import zipfile
import json
from werkzeug.utils import secure_filename
import shutil
from threading import Thread
import shutil
import zipfile

from backend.mask_maker import MaskMaker as make_mask
from backend.yolo_detection import yolo_detect as y  # Import your YOLO detection module
from backend.inpainting_script import InpaintingScript as inpaint
from backend.json_to_jsonTranslate import JsonToJsonTranslate as json_translate
from backend.draw_translation import TranslationDrawer as draw_translate
from backend.draw_translation_refined_v2 import TranslationDrawer as draw_translate_v2
from backend.translate_with_gemini import MangaTranslator as GeminiTranslator
from backend.translate_with_gemini_v2 import MangaTranslator as GeminiTranslatorV2


with open('./PROJECT_KEY(DONT_PUSH).json', 'r') as f:
    PROJECT_STUFF = json.load(f)

GEMINI_API_KEY = PROJECT_STUFF['GEMINI_KEY']


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'zip'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
FONTS_DIR = os.path.join(BASE_DIR, 'fonts')
FINAL_OUTPUT_FOLDER = os.path.join(BASE_DIR, 'final_output')




# app.config['FINAL_OUTPUT_FOLDER'] = FINAL_OUTPUT_FOLDER
# app.config['JSON_FOLDER'] = 'json_output'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
# app.config['TRANSLATE_FONTS_FOLDER'] = FONTS_DIR

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def convert_language_for_gemini(language:str):
    iso_map = {
    "en": "English",
    "id": "Indonesian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ko": "Korean",
    "zh": "Chinese",
    "th": "Thai",
}

    return iso_map.get(language.lower(), language)
    


def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_file(filename):
    """Check if the file is an image."""
    image_extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in image_extensions

def extract_zip(zip_path, extract_to):
    """Extract ZIP file and return list of extracted image files."""
    extracted_images = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                # Skip directories and hidden files
                if file_info.is_dir() or file_info.filename.startswith('.'):
                    continue
                
                # Check if it's an image file
                if is_image_file(file_info.filename):
                    # Extract with secure filename
                    filename = secure_filename(os.path.basename(file_info.filename))
                    if filename:  # Make sure filename is not empty after securing
                        extract_path = os.path.join(extract_to, filename)
                        
                        # Ensure we don't overwrite existing files
                        counter = 1
                        base_name, ext = os.path.splitext(filename)
                        while os.path.exists(extract_path):
                            filename = f"{base_name}_{counter}{ext}"
                            extract_path = os.path.join(extract_to, filename)
                            counter += 1
                        
                        # Extract the file
                        with zip_ref.open(file_info) as source, open(extract_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        
                        extracted_images.append(filename)
        
        return extracted_images
    
    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP file")
    except Exception as e:
        raise ValueError(f"Error extracting ZIP file: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preview_results/<folder>')
def preview_results(folder):
    # session_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    session_folder = os.path.join(UPLOAD_FOLDER, folder)
    method_path = os.path.join(session_folder, 'translation_config.json')

    # Load translation method and check output image folder
    try:
        with open(method_path, 'r') as f:
            config = json.load(f)
            translation_method = config.get('translation_method', 'chatgpt')
    except Exception as e:
        flash(f'Failed to read translation config: {e}', 'error')
        return redirect(url_for('upload_files'))

    # Determine the correct image folder
    if translation_method == 'json_only':
        image_dir = os.path.join(session_folder, 'final_output','inpainted')
    else:
        image_dir = os.path.join(session_folder, 'final_output', 'translated_images')

    if not os.path.exists(image_dir):
        flash('Processed images not found.', 'error')
        return redirect(url_for('upload_files'))

    images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    return render_template(
        'preview_results.html',
        images=images,
        translation_method=translation_method,
        folder=folder
    )


@app.route('/uploads/<folder>/final_output/<subfolder>/<filename>')
def serve_final_output(folder, subfolder, filename):
    # subfolder = inpainted OR translated_images
    # base = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'final_output', subfolder)
    base = os.path.join(UPLOAD_FOLDER, folder, 'final_output', subfolder)
    file_path = os.path.join(base, filename)

    if not os.path.exists(file_path):
        return "Image not found", 404

    return send_from_directory(base, filename)

@app.route('/upload_files', methods=['GET', 'POST'])
def upload_files():
    """Handle single file upload and processing."""
    
    if request.method == 'GET':
        return render_template('upload.html')
    
    # Handle POST request
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash('File format not supported. Please upload JPG, JPEG, PNG, or ZIP files.', 'error')
        return redirect(request.url)
    
    # Create a unique folder for this upload session
    import uuid
    session_id = str(uuid.uuid4())[:8]
    # session_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    session_folder = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    try:
        filename = secure_filename(file.filename)
        if not filename:
            flash('Invalid filename', 'error')
            return redirect(request.url)
        
        # Save the uploaded file
        file_path = os.path.join(session_folder, filename)
        file.save(file_path)
        
        uploaded_files = []
        
        # If it's a ZIP file, extract it
        if filename.lower().endswith('.zip'):
            try:
                extracted_images = extract_zip(file_path, session_folder)
                if extracted_images:
                    uploaded_files.extend(extracted_images)
                    # Remove the ZIP file after extraction
                    os.remove(file_path)
                    flash(f'Successfully extracted {len(extracted_images)} image(s) from ZIP file', 'success')
                else:
                    flash('No valid images found in ZIP file', 'error')
                    # Clean up
                    shutil.rmtree(session_folder)
                    return redirect(request.url)
            except ValueError as e:
                flash(f'Error processing ZIP file: {str(e)}', 'error')
                # Clean up
                shutil.rmtree(session_folder)
                return redirect(request.url)
        
        # If it's an image file, add it to the list
        elif is_image_file(filename):
            uploaded_files.append(filename)
            flash('Successfully uploaded image file', 'success')
        
        else:
            flash('Invalid file type', 'error')
            # Clean up
            shutil.rmtree(session_folder)
            return redirect(request.url)
        
        # Run YOLO detection on the uploaded files
        try:
            flash('Running YOLO detection...', 'info')
            # y.yolo_det(session_folder)

            run_yolo = y(session_folder)

            print("input folders: ",run_yolo.input_folders)
            print("output_mask folders: ", run_yolo.output_folder_mask)
            print("output_image_folders: ", run_yolo.output_folder_image)
            print("output_json_folders: ", run_yolo.output_folder_json)
            run_yolo.yolo_det()

            # print()
            flash('YOLO detection completed successfully!', 'success')
        except Exception as e:
            flash(f'YOLO detection failed: {str(e)}', 'warning')
            print(f'YOLO detection failed: {str(e)}', 'warning')
            # Continue anyway - the files are still uploaded
        
        # Redirect to edit page with session folder
        print("edit_page kepanggil")
        return redirect(url_for('edit_page', folder=session_id))
        
    except Exception as e:
        # Clean up on error
        try:
            shutil.rmtree(session_folder)
        except:
            pass
        
        flash(f'Upload failed: {str(e)}', 'error')
        return redirect(request.url)

# @app.route('/edit/<folder>')
def edit_page2(folder):
    """
    Edit page showing uploaded files after YOLO detection.
    """
    # session_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'image')
    
    if not os.path.exists(session_folder):
        flash('Upload session not found', 'error')
        return redirect(url_for('upload_files'))
    
    # Get list of uploaded files
    try:
        files = [f for f in os.listdir(session_folder) 
                if os.path.isfile(os.path.join(session_folder, f)) and is_image_file(f)]
        files.sort()  # Sort files alphabetically
    except Exception as e:
        flash(f'Error reading uploaded files: {str(e)}', 'error')
        return redirect(url_for('upload_files'))
    
    # Basic edit page with file listing
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Comic Translation - Edit</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/4.6.2/css/bootstrap.min.css">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            body {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
            .edit-container {{ 
                background: rgba(255, 255, 255, 0.95); 
                border-radius: 20px; 
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1); 
                margin-top: 2rem; 
                padding: 2rem; 
            }}
            .file-item {{ 
                background: #f8f9fa; 
                border: 1px solid #dee2e6; 
                border-radius: 8px; 
                padding: 1rem; 
                margin-bottom: 0.5rem; 
            }}
            .status-badge {{ 
                background: #28a745; 
                color: white; 
                padding: 0.25rem 0.75rem; 
                border-radius: 15px; 
                font-size: 0.85rem; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="row justify-content-center">
                <div class="col-lg-10">
                    <div class="edit-container">
                        <div class="d-flex justify-content-between align-items-center mb-4">
                            <h1 class="mb-0">
                                <i class="fas fa-edit text-primary mr-3"></i>Edit Comic Translation
                            </h1>
                            <span class="status-badge">
                                <i class="fas fa-check-circle mr-1"></i>YOLO Detection Complete
                            </span>
                        </div>
                        
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle mr-2"></i>
                            <strong>Session ID:</strong> {folder} | 
                            <strong>Images Processed:</strong> {len(files)}
                        </div>
                        
                        <h5 class="mb-3">
                            <i class="fas fa-images text-success mr-2"></i>Processed Images:
                        </h5>
                        
                        <div class="row">
                            {''.join([f'''
                            <div class="col-md-6 col-lg-4 mb-3">
                                <div class="file-item">
                                    <div class="d-flex align-items-center">
                                        <i class="fas fa-file-image text-success mr-2"></i>
                                        <span class="font-weight-medium">{file}</span>
                                    </div>
                                </div>
                            </div>
                            ''' for file in files])}
                        </div>
                        
                        <div class="mt-4 text-center">
                            <a href="{url_for('upload_files')}" class="btn btn-secondary mr-3">
                                <i class="fas fa-arrow-left mr-2"></i>Upload New File
                            </a>
                            <button class="btn btn-primary" disabled>
                                <i class="fas fa-magic mr-2"></i>Continue Translation (Coming Soon)
                            </button>
                        </div>
                        
                        <div class="mt-4 text-muted text-center">
                            <small>
                                <i class="fas fa-folder mr-1"></i>
                                Files are stored in: uploads/{folder}/
                            </small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


@app.route('/edit/<folder>')
def edit_page(folder):
    """
    Interactive edit page for comic translation with canvas editing.
    """
    # session_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'image')
    # json_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'bbox')

    session_folder = os.path.join(UPLOAD_FOLDER, folder, 'image')
    json_folder = os.path.join(UPLOAD_FOLDER, folder, 'bbox')

    print(json_folder)
    
    if not os.path.exists(session_folder):
        print("Upload session not found")
        flash('Upload session not found', 'error')
        return redirect(url_for('upload_files'))
    
    # Get list of uploaded files
    try:
        files = [f for f in os.listdir(session_folder) 
                if os.path.isfile(os.path.join(session_folder, f)) and is_image_file(f)]
        files.sort()  # Sort files alphabetically
    except Exception as e:
        print(f"error reading uploaded file: {str(e)}")
        flash(f'Error reading uploaded files: {str(e)}', 'error')
        return redirect(url_for('upload_files'))
    
    if not files:
        flash('No images found in upload session', 'error')
        print("No Image Found")
        return redirect(url_for('upload_files'))
    
    print("edit.html sukses kepanggil")
    return render_template('edit.html', folder=folder, files=files)

@app.route('/uploads/<folder>/<subfolder>/<filename>')
def uploaded_file(folder, subfolder, filename):
    """Serve uploaded files (images)."""
    try:
        # directory = os.path.join(app.config['UPLOAD_FOLDER'], folder, subfolder)
        directory = os.path.join(UPLOAD_FOLDER, folder, subfolder)
        return send_from_directory(directory, filename)
    except Exception as e:
        flash(f'File not found: {str(e)}', 'error')
        print(f"Masuk di uploaded_file: {str(e)}")
        return redirect(url_for('upload_files'))

# @app.route('/get_annotations/<folder>/<filename>')
def get_annotations_2(folder, filename):
    """Get JSON annotations for a specific image."""
    try:
        # Remove file extension and add .json
        base_name = os.path.splitext(filename)[0]
        json_filename = f"{base_name}.json"
        json_path = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'bbox', json_filename)
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert YOLO format to editor format if needed
            annotations = convert_yolo_to_editor_format(data)
            
            return jsonify({
                'success': True,
                'annotations': annotations
            })
        else:
            # Return empty annotations if no JSON file exists
            return jsonify({
                'success': True,
                'annotations': []
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'annotations': []
        })

@app.route('/get_annotations/<folder>/<image>')
def get_annotations(folder, image):
    base = os.path.splitext(image)[0] + ".json"
    # edited_path = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'edited', base)
    # original_path = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'bbox', base)

    edited_path = os.path.join(UPLOAD_FOLDER, folder, 'edited', base)
    original_path = os.path.join(UPLOAD_FOLDER, folder, 'bbox', base)

    if os.path.exists(edited_path):
        with open(edited_path, 'r') as f:
            data = json.load(f)
    elif os.path.exists(original_path):
        with open(original_path, 'r') as f:
            data = json.load(f)
    else:
        return jsonify({'annotations': []})  # empty

    editor_data = convert_yolo_to_editor_format(data)
    return jsonify({'annotations': editor_data})



@app.route('/save_annotations/<folder>', methods=['POST'])
def save_annotations(folder):
    """Save edited annotations to JSON file."""
    try:
        # print("Masuk save annotation")
        data = request.get_json()
        # print("done request data")
        image_name = data.get('image')
        print("image_name: ", image_name)
        # print("done taking image from data")
        annotations = data.get('annotations', [])
        # print("done annotation from data")
        
        if not image_name:
            print("no image name provided")
            return jsonify({'success': False, 'error': 'No image name provided'})
        
        # Create JSON folder if it doesn't exist
        # json_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'edited')
        json_folder = os.path.join(UPLOAD_FOLDER, folder, 'edited')
        os.makedirs(json_folder, exist_ok=True)
        
        # Generate JSON filename
        base_name = os.path.splitext(image_name)[0]
        print("base_name: ", base_name)
        json_filename = f"{base_name}.json"
        print("json_filename: ", json_filename)
        json_path = os.path.join(json_folder, json_filename)
        print("json_path: ", json_path)
        
        # Convert editor format to YOLO format if needed
        yolo_data = convert_editor_to_yolo_format(annotations, image_name)

        print("Done everything on save annotation about to write")
        
        # Save JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(yolo_data, f, indent=2, ensure_ascii=False)

        # base_image_name = image_name
        # image_location = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'image', image_name)
        image_location = os.path.join(UPLOAD_FOLDER, folder, 'image', image_name)
        print("image_location: ", image_location)
        # mask_output = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'mask', image_name)
        mask_output = os.path.join(UPLOAD_FOLDER, folder, 'mask', image_name)
        os.makedirs(os.path.dirname(mask_output), exist_ok=True)

        mask_maker = make_mask()

        mask_maker.process_json_and_mask(image_location, json_path, mask_output)
        
        print("Write done. about to return success")

        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Somethings wrong in save annotation. Error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/reset_annotations/<folder>/<image>')
def reset_annotations(folder, image):
    base = os.path.splitext(image)[0] + ".json"
    # original_path = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'bbox', base)
    original_path = os.path.join(UPLOAD_FOLDER, folder, 'bbox', base)

    if os.path.exists(original_path):
        with open(original_path, 'r') as f:
            data = json.load(f)
        editor_data = convert_yolo_to_editor_format(data)
        return jsonify({'annotations': editor_data})
    else:
        return jsonify({'annotations': []})

@app.route('/process_translation/<folder>', methods=['POST'])
def process_translation(folder):
    # session_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    session_folder = os.path.join(UPLOAD_FOLDER, folder)
    edited_folder = os.path.join(session_folder, 'edited')
    bbox_folder = os.path.join(session_folder, 'bbox')
    config_path = os.path.join(session_folder, 'translation_config.json')
    status_path = os.path.join(session_folder, 'status.json')
    image_folder = os.path.join(session_folder, 'image')
    mask_folder = os.path.join(session_folder, 'mask')
    output_folder = os.path.join(session_folder, 'final_output')
    json_for_work_path = os.path.join(session_folder, 'translated_json')
    json_for_final_path = os.path.join(output_folder, 'translated_json')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(edited_folder, exist_ok=True)
    progress_path = os.path.join(session_folder, 'preview_progress.json')
    with open(progress_path, 'w') as f:
        json.dump({
            "inpainting": False,
            "translating": False,
            "rendering": False
        }, f)

    # ✅ 1. Write status: processing
    # with open(status_path, 'w') as f:
    #     json.dump({'status': 'processing'}, f)

    try:
        # ✅ 2. Ensure all annotations copied
        for filename in os.listdir(bbox_folder):
            if filename.endswith('.json'):
                src = os.path.join(bbox_folder, filename)
                dst = os.path.join(edited_folder, filename)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

        # ✅ 3. Load translation config
        if not os.path.exists(config_path):
            raise Exception("Missing translation_config.json")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        method = config.get('translation_method')
        font_choices = config.get('font_choices', {})
        openai_key = config.get('openai_api_key')
        gemini_context = config.get('gemini_context')
        min_font_size = int(config.get('min_font_size'))
        max_font_size = int(config.get('max_font_size'))
        target_language = config.get('target_language')


        inpaint_output = os.path.join(output_folder, 'inpainted')
        os.makedirs(inpaint_output, exist_ok=True)

        # Run Inpainting Here
        print("INPAINTING STARTING")
        inpaint_model = inpaint(image_folder, mask_folder, inpaint_output)
        print("INPAINTING MODEL CONFIG SET")
        inpaint_model.run_inpainting()
        print("INPAINTING DONE")
        with open(progress_path, 'r') as f:
            progress = json.load(f)

        progress["inpainting"] = True

        with open(progress_path, 'w') as f:
            json.dump(progress, f)

        # ✅ 4. Handle each method
        if method == 'json_only':
            print("MASUK JSON ONLY")
            # generate only json (no rendering)
            # ... do your JSON generation

            print("MAKING DIR FOR JSON")
            os.makedirs(json_for_work_path, exist_ok=True)
            os.makedirs(json_for_final_path, exist_ok=True)

            make_json_for_work = json_translate(image_folder, edited_folder, json_for_work_path)
            # make_json_final = json_translate(image_folder, edited_folder, json_for_final_path)
            make_json_for_work.translate_and_save_json()
            with open(progress_path, 'r') as f:
                progress = json.load(f)

            progress["translating"] = True
            progress["rendering"] = True

            with open(progress_path, 'w') as f:
                json.dump(progress, f)
            # make_json_final.translate_and_save_json()

            for filename in os.listdir(json_for_work_path):
                if filename.endswith('.json'):
                    src = os.path.join(json_for_work_path, filename)
                    dst = os.path.join(json_for_final_path, filename)
                    shutil.copy2(src, dst)

            shutil.copytree(mask_folder, os.path.join(output_folder, 'mask'), copy_function=shutil.copy2, dirs_exist_ok= True)

        elif method == 'google':
            # ... call Google Translate and render
            os.makedirs(json_for_work_path, exist_ok=True)
            os.makedirs(json_for_final_path, exist_ok=True)

            make_json_for_work = json_translate(image_folder, edited_folder, json_for_work_path, language=target_language)
            # make_json_final = json_translate(image_folder, edited_folder, json_for_final_path)
            make_json_for_work.translate_and_save_json()
            with open(progress_path, 'r') as f:
                progress = json.load(f)

            progress["translating"] = True

            with open(progress_path, 'w') as f:
                json.dump(progress, f)
            # make_json_final.translate_and_save_json()

            for filename in os.listdir(json_for_work_path):
                if filename.endswith('.json'):
                    src = os.path.join(json_for_work_path, filename)
                    dst = os.path.join(json_for_final_path, filename)
                    shutil.copy2(src, dst)
            
            translated_path = os.path.join(output_folder, 'translated_images')

            g_method = draw_translate(inpaint_output, json_for_work_path, translated_path)
            #OLD
            g_method.draw_translations(config_path, base_font_location="fonts/" ,auto_expand=False , min_text_size= min_font_size, max_text_size=max_font_size,target_language=target_language)
            with open(progress_path, 'r') as f:
                progress = json.load(f)

            progress["rendering"] = True

            with open(progress_path, 'w') as f:
                json.dump(progress, f)            
            # NEW    
            # g_method.draw_translations(config_path, base_font_location="fonts/", auto_expand=True, auto_font_size=True, min_text_size=22, max_text_size=28)

            #NEW V2
            # g_method = draw_translate_v2(inpaint_output, json_for_work_path, translated_path)
            # g_method.draw_translations(config_path, base_font_location="fonts/", auto_expand=False, max_text_size=52, auto_font_size=True)


        elif method == 'chatgpt':
            # if not openai_key:
            #     raise Exception("Missing OpenAI API key")
            # # ... call OpenAI API
            os.makedirs(json_for_work_path, exist_ok=True)
            os.makedirs(json_for_final_path, exist_ok=True)

            make_json_for_work = json_translate(image_folder, edited_folder, json_for_work_path)
            make_json_for_work.translate_and_save_json()

            # gemini_for_translate = GeminiTranslator(GEMINI_API_KEY, json_for_work_path)

            # gemini_for_translate.run()

            gemini_for_translate_v2 = GeminiTranslatorV2(GEMINI_API_KEY, json_for_work_path, manga_context_url=gemini_context, target_language=convert_language_for_gemini(target_language))
            
            gemini_for_translate_v2.run()

            for filename in os.listdir(json_for_work_path):
                if filename.endswith('.json'):
                    src = os.path.join(json_for_work_path, filename)
                    dst = os.path.join(json_for_final_path, filename)
                    shutil.copy2(src,dst)
            with open(progress_path, 'r') as f:
                progress = json.load(f)

            progress["translating"] = True

            with open(progress_path, 'w') as f:
                json.dump(progress, f)
            
            translated_path = os.path.join(output_folder, 'translated_images')

            cgpt_method = draw_translate(inpaint_output, json_for_work_path, translated_path)

            cgpt_method.draw_translations(config_path, base_font_location='fonts/', auto_expand=False, min_text_size=min_font_size, max_text_size=max_font_size, target_language=target_language)

            with open(progress_path, 'r') as f:
                progress = json.load(f)

            progress["rendering"] = True

            with open(progress_path, 'w') as f:
                json.dump(progress, f)

            pass

        else:
            raise Exception("Unknown translation method")


        # final_output_folder = os.path.join(app.config['FINAL_OUTPUT_FOLDER'])
        final_output_folder = os.path.join(FINAL_OUTPUT_FOLDER)
            
        os.makedirs(final_output_folder, exist_ok=True)

        # session_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder)
        # final_output_folder = os.path.join(app.config['FINAL_OUTPUT_FOLDER'])

        session_folder = os.path.join(UPLOAD_FOLDER, folder)
        final_output_folder = os.path.join(FINAL_OUTPUT_FOLDER)
        

        zip_filename = f"{folder}_output.zip"
        zip_path = os.path.join(final_output_folder, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(os.path.join(UPLOAD_FOLDER, folder, 'final_output')):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, os.path.join(UPLOAD_FOLDER, folder, 'final_output'))
                    zipf.write(full_path, arcname=rel_path)
        # try:
        #     shutil.rmtree(session_folder)
        # except Exception as e:
        #     print(f"[WARN] Could not remove session folder: {e}")


        # ✅ 5. Set status to done
        with open(os.path.join(final_output_folder, f"{folder}_status.json"), 'w') as f:
            json.dump({'status': 'done', 'zip': zip_filename}, f)

        status_data = {}
        if os.path.exists(status_path):
            try:
                with open(status_path, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
            except:
                pass
        
        # Update status
        status_data['status'] = 'done'

        # Save back to the same file
        with open(status_path, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2)

    except Exception as e:
        # ❌ 6. If error, log it and keep status as 'processing' or write 'failed'
        print(f"[ERROR] Translation failed: {e}")
        with open(status_path, 'w') as f:
            json.dump({'status': 'failed', 'error': str(e)}, f)

    return jsonify({'success': True})



def process_translation_2(folder):
    """Process all images for translation and inpainting."""
    try:
        # print("Masuk process translation")
        # session_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder)
        session_folder = os.path.join(UPLOAD_FOLDER, folder)
        # print("session folder checking done")

        status_path = os.path.join(session_folder, 'status.json')

        # Set status to processing
        with open(status_path, 'w') as f:
            json.dump({'status': 'processing'}, f)


        
        if not os.path.exists(session_folder):
            return jsonify({'success': False, 'error': 'Session folder not found'})

        bbox_folder = os.path.join(session_folder, 'bbox')
        edited_folder = os.path.join(session_folder, 'edited')
        config_path = os.path.join(session_folder, 'translation_config.json')
        os.makedirs(edited_folder, exist_ok=True)

        for filename in os.listdir(bbox_folder):
            if filename.endswith('.json'):
                src_path = os.path.join(bbox_folder, filename)
                dst_path = os.path.join(edited_folder, filename)
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    print(f"[Auto-Copy] {filename} → /edited")

        # Load config
        if not os.path.exists(config_path):
            return jsonify({'success': False, 'error': 'Missing translation config'})

        with open(config_path, 'r') as f:
            config = json.load(f)

        font_choices = config.get('font_choices', {})
        method = config.get('translation_method')
        openai_key = config.get('openai_api_key', '')
        
        print(f"[CONFIG] Fonts: {font_choices}")
        print(f"[CONFIG] Method: {method}")

        if method == 'json_only':
            # ✅ Just prepare the output JSON, skip rendering
            return jsonify({
                'success': True,
                'mode': 'json_only',
                'message': 'Exporting OCR + translation as JSON only',
                'redirect_url': url_for('results_page', folder=folder)
            })

        # Here you would call your translation/inpainting backend
        # For now, we'll just simulate the process
        
        # Example: Call your translation pipeline
        # translation_result = your_translation_module.process_folder(session_folder)
        
        flash('All images processed successfully!', 'success')

        print("Everything done, about to return jsonify")

        # Done
        with open(status_path, 'w') as f:
            json.dump({'status': 'done'}, f)
        
        return jsonify({
            'success': True,
            'message': 'Translation processing completed',
            'redirect_url': url_for('results_page', folder=folder)  # Optional results page
        })
        
    except Exception as e:
        print(f"Failed. Error str{e}")
        return jsonify({'success': False, 'error': str(e)})

def convert_yolo_to_editor_format_2(yolo_data):
    """
    Convert YOLO detection format to editor format.
    Adjust this function based on your YOLO output format.
    """
    annotations = []
    
    # Example YOLO format conversion
    # Adjust based on your actual YOLO output structure
    if isinstance(yolo_data, dict) and 'detections' in yolo_data:
        for detection in yolo_data['detections']:
            annotation = {
                'type': 'rectangle',  # or 'polygon' based on detection
                'label': detection.get('class', 'text'),  # Map YOLO classes to your labels
                'confidence': detection.get('confidence', 1.0)
            }
            
            # Handle bounding box format
            if 'bbox' in detection:
                bbox = detection['bbox']
                if len(bbox) == 4:  # [x, y, width, height]
                    annotation.update({
                        'x': bbox[0],
                        'y': bbox[1],
                        'width': bbox[2],
                        'height': bbox[3]
                    })
                    
            # Handle polygon format
            elif 'polygon' in detection:
                annotation.update({
                    'type': 'polygon',
                    'points': detection['polygon']
                })
            
            annotations.append(annotation)
    
    # Handle simple list format
    elif isinstance(yolo_data, list):
        for item in yolo_data:
            if isinstance(item, dict):
                annotation = {
                    'type': item.get('type', 'rectangle'),
                    'label': item.get('label', 'text'),
                    'confidence': item.get('confidence', 1.0)
                }
                
                if 'x' in item and 'y' in item:
                    annotation.update({
                        'x': item['x'],
                        'y': item['y'],
                        'width': item.get('width', 50),
                        'height': item.get('height', 50)
                    })
                elif 'points' in item:
                    annotation.update({
                        'type': 'polygon',
                        'points': item['points']
                    })
                
                annotations.append(annotation)
    
    return annotations

def convert_editor_to_yolo_format_2(annotations, image_name):
    """
    Convert editor format back to YOLO format for saving.
    Adjust this function based on your required output format.
    """
    yolo_data = {
        'image': image_name,
        'detections': []
    }
    
    for annotation in annotations:
        detection = {
            'class': annotation.get('label', 'text'),
            'confidence': annotation.get('confidence', 1.0)
        }
        
        if annotation.get('type') == 'rectangle':
            detection['bbox'] = [
                annotation.get('x', 0),
                annotation.get('y', 0),
                annotation.get('width', 0),
                annotation.get('height', 0)
            ]
        elif annotation.get('type') == 'polygon':
            detection['polygon'] = annotation.get('points', [])
        
        yolo_data['detections'].append(detection)
    
    return yolo_data


def convert_yolo_to_editor_format(yolo_data):
    annotations = []
    bubble_cls_map = {
        0: 'Ellipse',
        1: 'Cloud',
        2: 'Other',
        3: 'Rectangle',
        4: 'Sea Urchin',
        5: 'Thorn'
    }

    for label_type in ['onomatope', 'bubble', 'text']:
        items = yolo_data.get(label_type, [])
        for item in items:
            # cls_name = bubble_cls_map.get(item.get('cls'), f"Class {item.get('cls')}") if label_type == 'bubble' else None

            cls_id = item.get('cls')
            cls_name = bubble_cls_map.get(cls_id, f"Class {cls_id}") if label_type == 'bubble' else None


            if 'mask' in item:
                annotations.append({
                    'type': 'polygon',
                    'label': label_type,
                    'points': [coord for point in item['mask'] for coord in point],
                    'className': cls_name
                })
            else:
                x1, y1 = item['x1'], item['y1']
                x2, y2 = item['x2'], item['y2']
                annotations.append({
                    'type': 'rectangle',
                    'label': label_type,
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'className': cls_name,
                    'cls': cls_id
                })
    return annotations


def convert_editor_to_yolo_format(annotations, image_name):
    """
    Convert editor format to your custom YOLO format (with mask or bbox).
    """
    yolo_data = {
        "onomatope": [],
        "bubble": [],
        "text": []
    }

    for ann in annotations:
        label = ann.get("label", "text")
        if label not in yolo_data:
            yolo_data[label] = []

        if ann.get("type") == "rectangle":
            obj = {
                "x1": int(ann["x"]),
                "y1": int(ann["y"]),
                "x2": int(ann["x"] + ann["width"]),
                "y2": int(ann["y"] + ann["height"]),
            }
        elif ann.get("type") == "polygon":
            points = ann.get("points", [])
            obj = {
                "x1": int(min(points[::2])),
                "y1": int(min(points[1::2])),
                "x2": int(max(points[::2])),
                "y2": int(max(points[1::2])),
                "mask": [[points[i], points[i+1]] for i in range(0, len(points), 2)]
            }
        else:
            continue

        if label == "bubble":
            obj["cls"] = int(ann.get("cls", 3))
        
        yolo_data[label].append(obj)

    return yolo_data

@app.route('/translation_config/<folder>', methods=['GET'])
def translation_config(folder):
    # fonts_path = app.config['TRANSLATE_FONTS_FOLDER']
    fonts_path = FONTS_DIR
    font_extensions = ('.ttf', '.otf', '.woff', '.woff2')
    print(fonts_path)
    # fonts = [f for f in os.listdir(fonts_path) if os.path.isdir(os.path.join(fonts_path, f))]
    # fonts = [f for f in os.listdir(fonts_path) if f.lower().endswith('.ttf')]
    fonts = [f for f in os.listdir(fonts_path)if f.lower().endswith(font_extensions)]
    print(fonts)
    bubble_classes = {
        0: 'Ellipse',
        1: 'Cloud',
        2: 'Rectangle',
        3: 'Other',
        4: 'Thorn',
        5: 'Outside'
    }
    return render_template('translation_options.html', folder=folder, available_fonts=fonts, bubble_classes=bubble_classes)


@app.route('/configure_translation/<folder>', methods=['POST'])
def configure_translation(folder):
    try:
        # Save config
        font_choices = {}
        for key in request.form:
            if key.startswith("font_"):
                cls_id = key.replace("font_", "")
                font_choices[cls_id] = request.form[key]

        translation_method = request.form.get("translation_method")
        # openai_api_key = request.form.get("chatgpt_api_key", "") # DISABLED OPENAI_API_KEY FOR NOW
        openai_api_key = ""
        gemini_context = request.form.get("chatgpt_api_key", "")
        min_font_size = int(request.form.get("min_font_size", 12))
        max_font_size = int(request.form.get("max_font_size", 48))
        target_language = request.form.get("target_language", "en")


        config_data = {
            "font_choices": font_choices,
            "translation_method": translation_method,
            "openai_api_key": openai_api_key,
            "gemini_context": gemini_context,
            "min_font_size" : min_font_size,
            "max_font_size" : max_font_size,
            "target_language" : target_language
            
        }

        # session_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder)
        session_folder = os.path.join(UPLOAD_FOLDER, folder)
        config_path = os.path.join(session_folder, 'translation_config.json')

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)

        # ✅ Run processing in background thread
        def run_translation():
            with app.app_context():
                status_path = os.path.join(session_folder, 'status.json')
                with open(status_path, 'w') as f:
                    json.dump({'status': 'processing'}, f)
                # Simulate a POST request to /process_translation/<folder>
                client = current_app.test_client()
                client.post(f'/process_translation/{folder}')

        Thread(target=run_translation).start()

        # ✅ Redirect immediately to results page (will show status/progress)
        # return redirect(url_for('results_page', folder=folder))
        return redirect(url_for('preview_status', folder=folder))


    except Exception as e:
        return f"Error saving configuration: {str(e)}", 500



@app.route('/preview_status/<folder>')
def preview_status(folder):
    # status_path = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'status.json')
    status_path = os.path.join(UPLOAD_FOLDER, folder, 'status.json')

    progress_path = os.path.join(UPLOAD_FOLDER, folder, 'preview_progress.json')

    # Default values
    status = "processing"
    stage_data = {
        "inpainting": False,
        "translating": False,
        "rendering": False
    }

    if os.path.exists(status_path):
        try:
            with open(status_path, 'r') as f:
                status = json.load(f).get("status", "processing")
        except:
            status = "processing"
    else:
        status = "processing"

        # Read progress stages if available
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r') as f:
                stage_data = json.load(f)
        except:
            pass

    if status == "done":
        return redirect(url_for('preview_results', folder=folder))
    else:
        return render_template('preview_status.html', folder=folder, stage_data=stage_data)



def configure_translation_2(folder):
    font_choices = {}
    bubble_classes = {
        0: 'Ellipse',
        1: 'Cloud',
        2: 'Other',
        3: 'Rectangle',
        4: 'Sea Urchin',
        5: 'Thorn',
        6: 'Outside'
    }
    for cls_id, cls_name in bubble_classes.items():
        font_choices[cls_id] = request.form.get(f'font_{cls_id}')

    translation_method = request.form.get('translation_method')
    openai_api_key = request.form.get('openai_api_key', '')

    config = {
        'font_choices': font_choices,
        'translation_method': translation_method,
        'openai_api_key': openai_api_key
    }

    # config_path = os.path.join(app.config['UPLOAD_FOLDER'], folder, 'translation_config.json')
    config_path = os.path.join(UPLOAD_FOLDER, folder, 'translation_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return redirect(url_for('process_translation', folder=folder))


# Optional: Results page to show processed images
@app.route('/results/<folder>')
def results_page(folder):
    
    # status_path = os.path.join(app.config['FINAL_OUTPUT_FOLDER'], f"{folder}_status.json")
    status_path = os.path.join(FINAL_OUTPUT_FOLDER, f"{folder}_status.json")
    if not os.path.exists(status_path):
        return render_template('results.html', status='processing', folder=folder)

    with open(status_path, 'r') as f:
        status_data = json.load(f)

    # session_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    session_folder = os.path.join(UPLOAD_FOLDER, folder)
    
    try:
        shutil.rmtree(session_folder)
    except Exception as e:
        print(f"[WARN] Could not remove session folder: {e}")

    return render_template('results.html', status=status_data.get('status'), zip_filename=status_data.get('zip'), folder=folder)



def results_page_2(folder):
    """Show translation results."""
    # session_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    session_folder = os.path.join(UPLOAD_FOLDER, folder)
    
    if not os.path.exists(session_folder):
        flash('Session not found', 'error')
        return redirect(url_for('upload_files'))
    
    # You can implement a results page here
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Translation Results</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/4.6.2/css/bootstrap.min.css">
    </head>
    <body>
        <div class="container mt-4">
            <h2>Translation Results</h2>
            <p>Session: {folder}</p>
            <div class="alert alert-success">
                Translation processing completed successfully!
            </div>
            <a href="{url_for('upload_files')}" class="btn btn-primary">
                <i class="fas fa-upload mr-2"></i>Upload New Files
            </a>
        </div>
    </body>
    </html>
    """


@app.route('/download/<filename>')
def download_result_zip(filename):
    # final_output_folder = app.config['FINAL_OUTPUT_FOLDER']
    final_output_folder = FINAL_OUTPUT_FOLDER
    file_path = os.path.join(final_output_folder, filename)

    print(file_path)

    if not os.path.exists(file_path):
        return "Zip file not found.", 404

    return send_file(file_path, mimetype='application/zip', as_attachment=True, download_name=filename)



# Error handlers
@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 100MB.', 'error')
    return redirect(url_for('upload_files'))

@app.errorhandler(Exception)
def handle_exception(e):
    flash(f'An error occurred: {str(e)}', 'error')
    return redirect(url_for('upload_files'))

if __name__ == '__main__':
    app.run(debug=False)