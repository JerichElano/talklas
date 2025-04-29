import os
os.environ["HOME"] = "/root"
os.environ["HF_HOME"] = "/tmp/hf_cache"

import logging
import threading
import tempfile
import uuid
import torch
import numpy as np
import soundfile as sf
import torchaudio
import wave
import time
import re
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta

# Import our model manager
from model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("talklas-api")

app = FastAPI(title="Talklas API")

# Mount a directory to serve audio files
AUDIO_DIR = "/tmp/audio_output"  # Use /tmp for temporary files
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio_output", StaticFiles(directory=AUDIO_DIR), name="audio_output")

# Create model manager instance
model_manager = ModelManager(max_idle_time=300, memory_check_interval=60)  # 5 minutes idle timeout

# Define the valid languages and mappings
LANGUAGE_MAPPING = {
    "English": "eng",
    "Tagalog": "tgl",
    "Cebuano": "ceb",
    "Ilocano": "ilo",
    "Waray": "war",
    "Pangasinan": "pag"
}

# Define which languages use Whisper vs MMS for STT
WHISPER_LANGUAGES = {"eng", "tgl"}  # English and Tagalog use Whisper
MMS_LANGUAGES = {"ceb", "ilo", "war", "pag"}  # Other Philippine languages use MMS

NLLB_LANGUAGE_CODES = {
    "eng": "eng_Latn",
    "tgl": "tgl_Latn",
    "ceb": "ceb_Latn",
    "ilo": "ilo_Latn",
    "war": "war_Latn",
    "pag": "pag_Latn"
}

# List of inappropriate words/phrases for content filtering
INAPPROPRIATE_WORDS = [
    # English inappropriate words
    "fuck", "shit", "bitch", "ass", "damn", "hell", "bastard", "cunt", "son of a bitch", "dick", "pussy", "motherfucker", 
    # Philippine languages
    "agka baboy", "puta", "putang ina", "gago", "tanga", "hayop", "ulol", "lintik", "animal ka",
    "paki", "pakyu", "yawa", "bungol", "gingan", "yawa ka", "peste", "irig",
    "pakit", "ayat", "pua", "kayat mo ti agsardeng", "hinampak", "iring ka"
]

# Function to check for inappropriate content
def check_inappropriate_content(text: str) -> bool:
    """
    Check if the text contains inappropriate content.
    Returns True if inappropriate content is detected, False otherwise.
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check for inappropriate words
    for word in INAPPROPRIATE_WORDS:
        # Use word boundary matching to avoid false positives
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text_lower):
            logger.warning(f"Inappropriate content detected: {word}")
            return True
    
    return False

# Function to save PCM data as a WAV file
def save_pcm_to_wav(pcm_data: list, sample_rate: int, output_path: str):
    # Convert pcm_data to a NumPy array of 16-bit integers
    pcm_array = np.array(pcm_data, dtype=np.int16)
    
    with wave.open(output_path, 'wb') as wav_file:
        # Set WAV parameters: 1 channel (mono), 2 bytes per sample (16-bit), sample rate
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        # Write the 16-bit PCM data as bytes (little-endian)
        wav_file.writeframes(pcm_array.tobytes())

# Function to detect speech using an energy-based approach
def detect_speech(waveform: torch.Tensor, sample_rate: int, threshold: float = 0.01, min_speech_duration: float = 0.5) -> bool:
    """
    Detects if the audio contains speech using an energy-based approach.
    Returns True if speech is detected, False otherwise.
    """
    # Convert waveform to numpy array
    waveform_np = waveform.numpy()
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=0)  # Convert stereo to mono

    # Compute RMS energy
    rms = np.sqrt(np.mean(waveform_np**2))
    logger.info(f"RMS energy: {rms}")

    # Check if RMS energy exceeds the threshold
    if rms < threshold:
        logger.info("No speech detected: RMS energy below threshold")
        return False

    # Optionally, check for minimum speech duration (requires more sophisticated VAD)
    # For now, we assume if RMS is above threshold, there is speech
    return True

# Function to clean up old audio files
def cleanup_old_audio_files():
    logger.info("Starting cleanup of old audio files...")
    expiration_time = datetime.now() - timedelta(minutes=10)  # Files older than 10 minutes
    for filename in os.listdir(AUDIO_DIR):
        file_path = os.path.join(AUDIO_DIR, filename)
        if os.path.isfile(file_path):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_mtime < expiration_time:
                try:
                    os.unlink(file_path)
                    logger.info(f"Deleted old audio file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {str(e)}")

# Background task to periodically clean up audio files
def schedule_cleanup():
    while True:
        cleanup_old_audio_files()
        time.sleep(300)  # Run every 5 minutes (300 seconds)

# Function to synthesize speech from text
def synthesize_speech(text: str, target_code: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert text to speech for the specified language.
    Returns a tuple of (output_path, error_message).
    """
    tts_model_name = f"tts_{target_code}"
    
    # Get the TTS model and tokenizer
    tts_model = model_manager.get_model(tts_model_name)
    tts_tokenizer = model_manager.get_tokenizer(tts_model_name)
    
    if tts_model is None or tts_tokenizer is None:
        return None, f"Failed to load TTS model for {target_code}"
    
    request_id = str(uuid.uuid4())
    output_path = os.path.join(AUDIO_DIR, f"{request_id}.wav")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Mark the model as being used
        model_manager.use_model(tts_model_name)
        
        inputs = tts_tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = tts_model(**inputs)
        speech = output.waveform.cpu().numpy().squeeze()
        speech = (speech * 32767).astype(np.int16)
        sample_rate = tts_model.config.sampling_rate

        # Save the audio as a WAV file
        save_pcm_to_wav(speech.tolist(), sample_rate, output_path)
        logger.info(f"Saved synthesized audio to {output_path}")
        
        return output_path, None
    except Exception as e:
        error_msg = f"Error during TTS conversion: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

# Start the background processes when the app starts
@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    
    # Start cleanup task
    cleanup_thread = threading.Thread(target=schedule_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Pre-load English Whisper model as it's commonly used
    model_manager.load_model("stt_whisper")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    model_manager.shutdown()

@app.get("/")
async def root():
    """Root endpoint for default health check"""
    logger.info("Root endpoint requested")
    return {"status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint that returns model status"""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "model_status": model_manager.get_model_status()
    }

@app.post("/translate-text")
async def translate_text(text: str = Form(...), source_lang: str = Form(...), target_lang: str = Form(...)):
    """Endpoint to translate text and convert to speech"""
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    if source_lang not in LANGUAGE_MAPPING or target_lang not in LANGUAGE_MAPPING:
        raise HTTPException(status_code=400, detail="Invalid language selected")
    
    logger.info(f"Translate-text requested: {text} from {source_lang} to {target_lang}")
    request_id = str(uuid.uuid4())
    
    # Translate the text
    source_code = LANGUAGE_MAPPING[source_lang]
    target_code = LANGUAGE_MAPPING[target_lang]
    translated_text = "Translation not available"
    
    # Get the MT model and tokenizer
    mt_model = model_manager.get_model("mt")
    mt_tokenizer = model_manager.get_tokenizer("mt")
    
    if mt_model is not None and mt_tokenizer is not None:
        try:
            # Mark the model as being used
            model_manager.use_model("mt")
            
            source_nllb_code = NLLB_LANGUAGE_CODES[source_code]
            target_nllb_code = NLLB_LANGUAGE_CODES[target_code]
            mt_tokenizer.src_lang = source_nllb_code
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = mt_tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                generated_tokens = mt_model.generate(
                    **inputs,
                    forced_bos_token_id=mt_tokenizer.convert_tokens_to_ids(target_nllb_code),
                    max_length=448
                )
            translated_text = mt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            logger.info(f"Translation completed: {translated_text}")
        except Exception as e:
            logger.error(f"Error during translation: {str(e)}")
            translated_text = f"Translation failed: {str(e)}"
    else:
        logger.warning("MT model not loaded, skipping translation")

    # Check for inappropriate content in the source text and translated text
    is_inappropriate = check_inappropriate_content(text) or check_inappropriate_content(translated_text)
    if is_inappropriate:
        logger.warning("Inappropriate content detected in translation request")

    # Convert translated text to speech
    output_audio_url = None
    output_path, error = synthesize_speech(translated_text, target_code)
    if output_path:
        output_filename = os.path.basename(output_path)
        output_audio_url = f"https://legendary-halibut-4p76969wqgr27xjw-8000.app.github.dev/audio_output/{output_filename}"
        logger.info("TTS conversion completed")
    else:
        logger.error(f"TTS conversion failed: {error}")
    
    return {
        "request_id": request_id,
        "status": "completed",
        "message": "Translation and TTS completed (or partially completed).",
        "source_text": text,
        "translated_text": translated_text,
        "output_audio": output_audio_url,
        "is_inappropriate": is_inappropriate
    }

@app.post("/translate-audio")
async def translate_audio(audio: UploadFile = File(...), source_lang: str = Form(...), target_lang: str = Form(...)):
    """Endpoint to transcribe, translate, and convert audio to speech"""
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file provided")
    if source_lang not in LANGUAGE_MAPPING or target_lang not in LANGUAGE_MAPPING:
        raise HTTPException(status_code=400, detail="Invalid language selected")
    
    source_code = LANGUAGE_MAPPING[source_lang]
    target_code = LANGUAGE_MAPPING[target_lang]
    
    logger.info(f"Translate-audio requested: {audio.filename} from {source_lang} ({source_code}) to {target_lang} ({target_code})")
    request_id = str(uuid.uuid4())
    
    # Determine which STT model to use based on source language
    use_whisper = source_code in WHISPER_LANGUAGES
    use_mms = source_code in MMS_LANGUAGES
    
    # Get the appropriate STT model
    stt_model_name = "stt_whisper" if use_whisper else "stt_mms"
        
    # Get the STT model and processor
    stt_model = model_manager.get_model(stt_model_name)
    stt_processor = model_manager.get_processor(stt_model_name)
    
    if stt_model is None or stt_processor is None:
        return {
            "request_id": request_id,
            "status": "failed",
            "message": f"Failed to load {stt_model_name} model",
            "source_text": "Transcription not available",
            "translated_text": "Translation not available",
            "output_audio": None,
            "is_inappropriate": False
        }

    # Save the uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await audio.read())
        temp_path = temp_file.name

    transcription = "Transcription not available"
    translated_text = "Translation not available"
    output_audio_url = None
    is_inappropriate = False

    try:
        # Process the audio file
        waveform, sample_rate = torchaudio.load(temp_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Check for speech
        if not detect_speech(waveform, sample_rate):
            return {
                "request_id": request_id,
                "status": "failed",
                "message": "No speech detected in the audio.",
                "source_text": "No speech detected",
                "translated_text": "No translation available",
                "output_audio": None,
                "is_inappropriate": False
            }

        # Mark the STT model as being used
        model_manager.use_model(stt_model_name)

        # Transcribe the audio
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if use_whisper:
            inputs = stt_processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                forced_language = "en" if source_code == "eng" else "tl"
                generated_ids = stt_model.generate(**inputs, language=forced_language, task="transcribe")
                transcription = stt_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            inputs = stt_processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = stt_model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = stt_processor.batch_decode(predicted_ids)[0]

        logger.info(f"Transcription completed: {transcription}")

        # Get the MT model and process translation
        mt_model = model_manager.get_model("mt")
        mt_tokenizer = model_manager.get_tokenizer("mt")

        if mt_model is not None and mt_tokenizer is not None:
            try:
                model_manager.use_model("mt")
                source_nllb_code = NLLB_LANGUAGE_CODES[source_code]
                target_nllb_code = NLLB_LANGUAGE_CODES[target_code]
                mt_tokenizer.src_lang = source_nllb_code
                inputs = mt_tokenizer(transcription, return_tensors="pt").to(device)
                with torch.no_grad():
                    generated_tokens = mt_model.generate(
                        **inputs,
                        forced_bos_token_id=mt_tokenizer.convert_tokens_to_ids(target_nllb_code),
                        max_length=448
                    )
                translated_text = mt_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                logger.info(f"Translation completed: {translated_text}")
            except Exception as e:
                logger.error(f"Error during translation: {str(e)}")
                translated_text = f"Translation failed: {str(e)}"

        # Check for inappropriate content
        is_inappropriate = check_inappropriate_content(transcription) or check_inappropriate_content(translated_text)
        if is_inappropriate:
            logger.warning("Inappropriate content detected")

        # Convert to speech if appropriate
        if not is_inappropriate:
            output_path, error = synthesize_speech(translated_text, target_code)
            if output_path:
                output_filename = os.path.basename(output_path)
                output_audio_url = f"https://legendary-halibut-4p76969wqgr27xjw-8000.app.github.dev/audio_output/{output_filename}"
                logger.info("TTS conversion completed")

        return {
            "request_id": request_id,
            "status": "completed",
            "message": "Processing completed successfully",
            "source_text": transcription,
            "translated_text": translated_text,
            "output_audio": output_audio_url,
            "is_inappropriate": is_inappropriate
        }

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return {
            "request_id": request_id,
            "status": "failed",
            "message": f"Processing failed: {str(e)}",
            "source_text": transcription,
            "translated_text": translated_text,
            "output_audio": output_audio_url,
            "is_inappropriate": is_inappropriate
        }
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")