import yt_dlp
import subprocess
import os
import logging
import time
from pathlib import Path
import torch
import whisperx

# Global model variables
whisper_model = None
use_whisperx = True

logger = logging.getLogger(__name__)

def initialize_model():
    global whisper_model, use_whisperx
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    try:
        logger.info(f"Initializing model - Device: {device}, Compute Type: {compute_type}")
        
        # Load the model
        whisper_model = whisperx.load_model(
            "large-v2",  # You can make this configurable
            device, 
            asr_options = {
                    "hotwords": None,
                    "multilingual": True
                },
            compute_type=compute_type
        )
        
        logger.info("Model initialized successfully")
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise

# Ensure model is initialized when the module is imported
try:
    initialize_model()
except Exception as e:
    logger.error(f"Failed to initialize model on import: {e}")

class YouTubeTranscriber:
    def __init__(self, output_dir='downloads', whisper_model=None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.whisper_model = whisper_model

    def process_url_streaming(self, url, chunk_duration_minutes=5):
        try:
            # Download and convert YouTube video to WAV
            wav_path = self.download_and_convert(url)
            
            if not wav_path or not os.path.exists(wav_path):
                raise ValueError("Failed to download or convert YouTube video")
            
            if os.path.getsize(wav_path) == 0:
                raise ValueError("Downloaded WAV file is empty")
            
            try:
                # Use the provided whisper model for transcription
                if self.whisper_model is None:
                    raise ValueError("Whisper model not initialized")
                
                result = self.whisper_model.transcribe(
                    wav_path,
                    batch_size=16 if torch.cuda.is_available() else 1,
                    language="en"
                )
                
                # Return the complete result - let app-uv.py handle the format
                yield result
                    
            finally:
                # Cleanup the downloaded file
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                    
        except Exception as e:
            raise

    def download_and_convert(self, url):
        try:
            # Extract video ID from URL
            if "youtube.com" in url or "youtu.be" in url:
                with yt_dlp.YoutubeDL() as ydl:
                    info = ydl.extract_info(url, download=False)
                    video_id = info['id']
            else:
                raise ValueError("Not a valid YouTube URL")

            # Configure yt-dlp options with video ID as filename
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'outtmpl': os.path.join(self.output_dir, '%(id)s.%(ext)s'),
                'quiet': True
            }
            
            # Download the audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            # Construct the output path using video ID
            wav_path = os.path.join(self.output_dir, f"{video_id}.wav")
            
            # Verify file exists and has content
            if not os.path.exists(wav_path):
                raise ValueError(f"WAV file not found")
            
            if os.path.getsize(wav_path) == 0:
                raise ValueError("Downloaded WAV file is empty")
                
            return wav_path
                
        except Exception as e:
            raise

    def process_url(self, url, chunk_size=6):
        try:
            wav_path, video_id = self.download_and_convert(url)
            transcription = self.transcribe(wav_path, chunk_size)
            return transcription
        finally:
            # Cleanup downloaded files
            try:
                if 'wav_path' in locals() and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception as e:
                logger.error(f"Error cleaning up files: {e}")

    def validate_youtube_url(self, url):
        if not url:
            raise ValueError("No URL provided")
        
        # Basic YouTube URL validation
        valid_hosts = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if not parsed.scheme:
                url = 'https://' + url
                parsed = urlparse(url)
            
            if parsed.netloc not in valid_hosts:
                raise ValueError("Invalid YouTube URL. Please provide a valid YouTube URL.")
                
            return url
        except Exception as e:
            raise ValueError(f"Invalid URL format: {str(e)}")

    def transcribe(self, wav_path, chunk_size=6, max_retries=3, initial_wait=60):
        import whisperx
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        for attempt in range(max_retries + 1):
            try:
                # Simplified ASR options
                asr_options = {
                    "hotwords": None,
                    "multilingual": True
                }
                
                logger.info("Loading WhisperX model...")
                model = whisperx.load_model(
                    "medium",
                    device=device,
                    compute_type=compute_type,
                    asr_options=asr_options,
                    download_root="models"
                )
                
                logger.info("Starting transcription...")
                # Process audio in chunks and stream results
                result = model.transcribe(
                    wav_path,
                    batch_size=16 if device == "cuda" else 1
                )
                
                # Clean up GPU memory
                if device == "cuda":
                    del model
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                return result["text"] if isinstance(result, dict) else result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if "rate limit exceeded" in str(e).lower():
                    if attempt < max_retries:
                        wait_time = initial_wait * (2 ** attempt)
                        logger.info(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception("Rate limit exceeded. Please try again later.")
                elif attempt < max_retries:
                    continue
                raise Exception(f"Transcription failed: {str(e)}")

    def process_audio_in_chunks(self, audio_path, chunk_duration=300):  
        """Process audio file in chunks using pydub"""
        from pydub import AudioSegment
        import tempfile
        
        # Load the audio file
        audio = AudioSegment.from_wav(audio_path)
        chunk_length_ms = chunk_duration * 1000  # Convert to milliseconds
        
        chunks = []
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            # Create a temporary file for the chunk
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                chunk.export(temp_file.name, format='wav')
                chunks.append(temp_file.name)
        
        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks
