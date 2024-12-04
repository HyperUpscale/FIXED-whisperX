import os
import sys
import logging
import torch
import warnings
import time
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import datetime
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import gc
from youtube_transcribe import YouTubeTranscriber
import whisperx
import traceback

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")

# Detailed logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Log to console
        logging.FileHandler('whisperx_transcription.log')  # Log to file
    ]
)

# Global variable for the WhisperX model
global_whisper_model = None

def initialize_whisper_model():
    global global_whisper_model
    try:
        import torch
        import whisperx
        from whisperx.asr import load_model
        
        # Detailed system and environment logging
        print("DEBUG: Checking system environment for WhisperX model initialization")
        logger.info("Checking system environment for WhisperX model initialization")
        
        # Log Python and Torch versions
        print(f"DEBUG: Python Version: {sys.version}")
        print(f"DEBUG: Torch Version: {torch.__version__}")
        logger.info(f"Python Version: {sys.version}")
        logger.info(f"Torch Version: {torch.__version__}")
        
        # Detect device and compute type
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"DEBUG: Detected Device: {device}")
        print(f"DEBUG: Compute Type: {compute_type}")
        logger.info(f"Detected Device: {device}")
        logger.info(f"Compute Type: {compute_type}")
        
        # Check CUDA availability if GPU is desired
        if device == "cuda":
            print("DEBUG: CUDA Devices:")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            print(f"Primary CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Devices: {torch.cuda.device_count()}")
            logger.info(f"Primary CUDA Device: {torch.cuda.get_device_name(0)}")
            
            # Log CUDA memory info
            print(f"DEBUG: CUDA Memory Allocated: {torch.cuda.memory_allocated(0)}")
            print(f"DEBUG: CUDA Memory Reserved: {torch.cuda.memory_reserved(0)}")
        
        print("DEBUG: Attempting to load WhisperX model...")
        logger.info("Attempting to load WhisperX model...")
        
        # Attempt model loading with comprehensive error handling
        try:
            # Set default ASR options
            asr_options = {

                "multilingual": True,  # Required argument
                "hotwords": None  # Required argument
            }
            
            global_whisper_model = load_model(
                "medium",
                device,
                compute_type=compute_type,
                download_root="models",
                asr_options=asr_options
            )
            
            print("DEBUG: WhisperX model loaded successfully")
            logger.info("WhisperX model loaded successfully")
            
            if device == "cuda":
                print(f"DEBUG: CUDA Memory After Model Load: {torch.cuda.memory_allocated(0)}")
            
            return global_whisper_model
        
        except Exception as load_error:
            print(f"DEBUG: Model loading failed: {load_error}")
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            logger.error(f"Model loading failed: {load_error}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    except ImportError as import_error:
        print(f"DEBUG: Import error - {import_error}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        logger.error(f"Import error: {import_error}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    
    except Exception as e:
        print(f"DEBUG: Unexpected error during model initialization: {e}")
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        logger.error(f"Unexpected error during model initialization: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Attempt model initialization during import
try:
    print("DEBUG: Attempting to initialize WhisperX model during import")
    logging.info("Attempting to initialize WhisperX model during import")
    global_whisper_model = initialize_whisper_model()
except Exception as initialization_error:
    print(f"DEBUG: Global model initialization failed: {initialization_error}")
    print(f"DEBUG: Traceback: {traceback.format_exc()}")
    logging.error(f"Global model initialization failed: {initialization_error}")
    logging.error(f"Traceback: {traceback.format_exc()}")
    global_whisper_model = None

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable TF32 to avoid reproducibility issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Attempt to set CUDA device with error handling
def setup_cuda_device():
    try:
        if torch.cuda.is_available():
            # Attempt to set the default CUDA device
            torch.cuda.set_device(0)
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return "cuda"
    except Exception as e:
        logger.warning(f"CUDA setup failed: {e}")
    return "cpu"

# Global configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 1024  # 1000MB max file size
app.config['TRANSCRIPTION_TIMEOUT'] = 300  # 5 minutes timeout

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}
REQUIRED_SAMPLE_RATE = 16000  # WhisperX expects 16kHz audio

# Global variables to store models
whisper_model = None
model_name = "medium" ##"large-v2"
device = setup_cuda_device()
compute_type = "float16" if device == "cuda" else "int8"
use_whisperx = True

def cleanup_gpu():
    """Clean up GPU memory"""
    global whisper_model
    try:
        if device == "cuda":
            # Force synchronize before cleanup
            torch.cuda.synchronize()
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # Force garbage collection
            gc.collect()
            # Final synchronize
            torch.cuda.synchronize()
            if whisper_model is not None:
                del whisper_model
                whisper_model = None
    except Exception as e:
        logger.error(f"GPU cleanup error: {e}")

def initialize_model(force_basic_whisper=False, retry_count=0):
    """
    Initialize Whisper model with robust error handling and retry mechanism
    
    Args:
        force_basic_whisper (bool): Force using basic Whisper instead of WhisperX
        retry_count (int): Number of retry attempts
    """
    global whisper_model, use_whisperx
    
    # Clean up before initialization
    cleanup_gpu()
    
    # Prevent infinite recursion
    if retry_count > 3:
        raise RuntimeError("Failed to initialize Whisper model after multiple attempts")
    
    # Reset global model
    whisper_model = None
    
    # Determine which model to use
    use_whisperx = not force_basic_whisper
    
    # Ensure WhisperX is used as the primary transcription model
    use_whisperx = True
    
    try:
        logger.info("Loading WhisperX model...")
        asr_options = {
         
            "multilingual": True,  # Add required multilingual parameter
            "hotwords": []  # Add required hotwords parameter with empty list
        }
        
        whisper_model = whisperx.load_model(
            "large-v2",
            device=device,
            compute_type=compute_type,
            download_root="models",
            asr_options=asr_options
        )
        logger.info("WhisperX model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load WhisperX model: {e}")
        raise RuntimeError("WhisperX model initialization failed")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_audio(input_path):
    """
    Convert audio to 16kHz mono WAV using ffmpeg
    """
    try:
        output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
        logger.info(f"Converting audio file to 16kHz WAV: {output_path}")
        
        # ffmpeg command to convert to 16kHz mono WAV
        command = [
            'ffmpeg',
            '-i', input_path,  # Input file
            '-ar', str(REQUIRED_SAMPLE_RATE),  # Output sample rate
            '-ac', '1',  # Mono audio
            '-y',  # Overwrite output file if exists
            output_path
        ]
        
        # Run ffmpeg command
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"FFmpeg error: {stderr.decode()}")
            raise Exception("Failed to convert audio file")
            
        # Replace original file with converted file
        os.replace(output_path, input_path)
        logger.info("Audio conversion completed")
        
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        raise

def transcribe_audio(audio_path):
    try:
        logger.info("Starting transcription...")
        
        # Ensure model is initialized
        global whisper_model, use_whisperx
        if whisper_model is None:
            initialize_model()
        
        # Transcribe using the appropriate model
        if use_whisperx:
            # WhisperX transcription with more options
            result = whisper_model.transcribe(
                audio_path,
                batch_size=16 if device == "cuda" else 1,
                language="en"  # You can make this configurable
            )
        else:
            # Basic Whisper transcription
            result = whisper_model.transcribe(
                audio_path,
                language="en"  # You can make this configurable
            )
        
        logger.info("Transcription completed successfully!")
        return result
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        # If WhisperX fails, try basic Whisper
        if use_whisperx:
            try:
                logger.warning("Falling back to basic Whisper...")
                whisper_model = None  # Reset model
                use_whisperx = False  # Switch to basic Whisper
                return transcribe_audio(audio_path)
            except Exception as fallback_error:
                logger.error(f"Fallback transcription failed: {fallback_error}")
                raise
    finally:
        cleanup_gpu()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_model')
def check_model():
    global whisper_model
    if whisper_model is None:
        success = initialize_model()
        return jsonify({'initialized': success})
    return jsonify({'initialized': True})

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Get output format preference
        output_format = request.form.get('format', 'json')  # Default to json if not specified
        
        # Save and process the file
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(filename)[0]
        saved_filename = f"{base_filename}_{timestamp}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        
        file.save(filepath)
        logger.info(f"Saving uploaded file to {filepath}")
        
        # Convert audio if needed
        try:
            converted_path = filepath
            result = transcribe_audio(converted_path)
            
            # Format the response based on preference
            if output_format == 'text':
                # Combine all segments into a single text
                text = ' '.join([segment['text'].strip() for segment in result['segments']])
                return text, 200, {'Content-Type': 'text/plain'}
            else:
                # Return full JSON response
                return jsonify(result)
                
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # Cleanup temporary files
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(app.config['UPLOAD_FOLDER'], filename),
        as_attachment=True
    )

@app.route('/transcribe_youtube', methods=['GET'])
def transcribe_youtube():
    global global_whisper_model
    
    try:
        # Validate global model
        if global_whisper_model is None:
            print("DEBUG: Whisper model is NOT initialized")
            logger.error("Whisper model is NOT initialized")
            
            # Attempt re-initialization
            try:
                global_whisper_model = initialize_whisper_model()
            except Exception as reinit_error:
                print(f"DEBUG: Model re-initialization failed: {reinit_error}")
                logger.error(f"Model re-initialization failed: {reinit_error}")
                return jsonify({
                    'error': 'Transcription model failed to initialize',
                    'details': str(reinit_error)
                }), 500

        # Extract YouTube URL and parameters
        url = request.args.get('youtube_url')
        if not url:
            return jsonify({'error': 'No YouTube URL provided'}), 400

        chunk_size = int(request.args.get('chunk_size', 6))
        format_type = request.args.get('format', 'text')  # Default to text format
        
        print(f"DEBUG: Starting YouTube transcription for URL: {url} (format: {format_type})")
        logger.info(f"Starting YouTube transcription for URL: {url}")
        
        def generate_transcription():
            try:
                # Create YouTubeTranscriber instance with the global model
                youtube_transcriber = YouTubeTranscriber(
                    output_dir='downloads',
                    whisper_model=global_whisper_model
                )
                
                # Process the URL and stream results
                for result in youtube_transcriber.process_url_streaming(url, chunk_size):
                    if format_type == 'text':
                        # For text format, extract only the text
                        if isinstance(result, dict):
                            if 'segments' in result:
                                text = ' '.join(segment['text'].strip() for segment in result['segments'])
                            else:
                                text = result.get('text', str(result)).strip()
                        else:
                            text = str(result).strip()
                            
                        chunk_data = {
                            'type': 'chunk',
                            'text': text
                        }
                    else:
                        # For JSON format, preserve the full structure
                        chunk_data = {
                            'type': 'chunk',
                            'data': result
                        }
                    
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                # Send completion message
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                    
            except ValueError as e:
                print(f"DEBUG: ValueError in transcribe_youtube: {e}")
                logger.error(f"ValueError in transcribe_youtube: {e}")
                error_data = {
                    'type': 'error',
                    'error': str(e),
                    'error_type': 'youtube_error'
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            except Exception as e:
                print(f"DEBUG: Unexpected error in transcribe_youtube: {e}")
                logger.error(f"Unexpected error in transcribe_youtube: {e}")
                error_data = {
                    'type': 'error',
                    'error': str(e),
                    'error_type': 'transcription_error'
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return Response(
            generate_transcription(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        print(f"DEBUG: YouTube processing error: {str(e)}")
        logger.error(f"YouTube processing error: {str(e)}")
        return jsonify({
            'error': 'Unexpected error during YouTube transcription',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)  # Create directory for model storage
    os.makedirs('downloads', exist_ok=True)  # Create directory for YouTube downloads
    
    # Register cleanup handler
    import atexit
    atexit.register(cleanup_gpu)
    
    app.run(debug=True, use_reloader=False)  # Disable auto-reloader to prevent memory issues
