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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("talklas-api")

app = FastAPI(title="Talklas API")

# Mount a directory to serve audio files
AUDIO_DIR = "/tmp/audio_output"  # Use /tmp for temporary files
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio_output", StaticFiles(directory=AUDIO_DIR), name="audio_output")

# Global variables to track application state
models_loaded = False
loading_in_progress = False
loading_thread = None
model_status = {
    "stt_whisper": "not_loaded",
    "stt_mms": "not_loaded",
    "mt": "not_loaded",
    "tts": "not_loaded"
}
error_message = None
current_tts_language = "tgl"  # Track the current TTS language

# Model instances
whisper_processor = None
whisper_model = None
mms_processor = None
mms_model = None
mt_model = None
mt_tokenizer = None
tts_model = None
tts_tokenizer = None

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

# Function to load models in background
def load_models_task():
    global models_loaded, loading_in_progress, model_status, error_message
    global whisper_processor, whisper_model, mms_processor, mms_model
    global mt_model, mt_tokenizer, tts_model, tts_tokenizer
    
    try:
        loading_in_progress = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Whisper STT model for English and Tagalog
        logger.info("Starting to load Whisper STT model...")
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        try:
            logger.info("Loading Whisper STT model...")
            model_status["stt_whisper"] = "loading"
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            whisper_model.to(device)
            logger.info("Whisper STT model loaded successfully")
            model_status["stt_whisper"] = "loaded"
        except Exception as whisper_error:
            logger.error(f"Failed to load Whisper STT model: {str(whisper_error)}")
            model_status["stt_whisper"] = "failed"
            error_message = f"Whisper STT model loading failed: {str(whisper_error)}"
            return
        
        # Load MMS STT model for other Philippine languages
        logger.info("Starting to load MMS STT model...")
        from transformers import AutoProcessor, AutoModelForCTC
        
        try:
            logger.info("Loading MMS STT model...")
            model_status["stt_mms"] = "loading"
            mms_processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
            mms_model = AutoModelForCTC.from_pretrained("facebook/mms-1b-all")
            mms_model.to(device)
            logger.info("MMS STT model loaded successfully")
            model_status["stt_mms"] = "loaded"
        except Exception as mms_error:
            logger.error(f"Failed to load MMS STT model: {str(mms_error)}")
            model_status["stt_mms"] = "failed"
            error_message = f"MMS STT model loading failed: {str(mms_error)}"
            return

        # Load MT model
        logger.info("Starting to load MT model...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        try:
            logger.info("Loading NLLB-200-distilled-600M model...")
            model_status["mt"] = "loading"
            mt_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
            mt_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            mt_model.to(device)
            logger.info("MT model loaded successfully")
            model_status["mt"] = "loaded"
        except Exception as e:
            logger.error(f"Failed to load MT model: {str(e)}")
            model_status["mt"] = "failed"
            error_message = f"MT model loading failed: {str(e)}"
            return

        # Load TTS model (default to Tagalog, will be updated dynamically)
        logger.info("Starting to load TTS model...")
        from transformers import VitsModel, AutoTokenizer
        
        try:
            logger.info("Loading MMS-TTS model for Tagalog...")
            model_status["tts"] = "loading"
            tts_model = VitsModel.from_pretrained("facebook/mms-tts-tgl")
            tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tgl")
            tts_model.to(device)
            logger.info("TTS model loaded successfully")
            model_status["tts"] = "loaded"
        except Exception as e:
            logger.error(f"Failed to load TTS model for Tagalog: {str(e)}")
            # Fallback to English TTS if the target language fails
            try:
                logger.info("Falling back to MMS-TTS English model...")
                tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
                tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
                tts_model.to(device)
                logger.info("Fallback TTS model loaded successfully")
                model_status["tts"] = "loaded (fallback)"
                current_tts_language = "eng"
            except Exception as e2:
                logger.error(f"Failed to load fallback TTS model: {str(e2)}")
                model_status["tts"] = "failed"
                error_message = f"TTS model loading failed: {str(e)} (fallback also failed: {str(e2)})"
                return
        
        models_loaded = True
        logger.info("Model loading completed successfully")
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in model loading task: {str(e)}")
    finally:
        loading_in_progress = False

# Start loading models in background
def start_model_loading():
    global loading_thread, loading_in_progress
    if not loading_in_progress and not models_loaded:
        loading_in_progress = True
        loading_thread = threading.Thread(target=load_models_task)
        loading_thread.daemon = True
        loading_thread.start()

# Start the background cleanup task
def start_cleanup_task():
    cleanup_thread = threading.Thread(target=schedule_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()

# Function to load or update TTS model for a specific language
def load_tts_model_for_language(target_code: str) -> bool:
    """
    Load or update the TTS model for the specified language.
    Returns True if successful, False otherwise.
    """
    global tts_model, tts_tokenizer, current_tts_language, model_status
    
    if target_code not in LANGUAGE_MAPPING.values():
        logger.error(f"Invalid language code: {target_code}")
        return False
    
    # Skip if the model is already loaded for the target language
    if current_tts_language == target_code and model_status["tts"].startswith("loaded"):
        logger.info(f"TTS model for {target_code} is already loaded.")
        return True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        logger.info(f"Loading MMS-TTS model for {target_code}...")
        from transformers import VitsModel, AutoTokenizer
        tts_model = VitsModel.from_pretrained(f"facebook/mms-tts-{target_code}")
        tts_tokenizer = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{target_code}")
        tts_model.to(device)
        current_tts_language = target_code
        logger.info(f"TTS model updated to {target_code}")
        model_status["tts"] = "loaded"
        return True
    except Exception as e:
        logger.error(f"Failed to load TTS model for {target_code}: {str(e)}")
        try:
            logger.info("Falling back to MMS-TTS English model...")
            tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
            tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            tts_model.to(device)
            current_tts_language = "eng"
            logger.info("Fallback TTS model loaded successfully")
            model_status["tts"] = "loaded (fallback)"
            return True
        except Exception as e2:
            logger.error(f"Failed to load fallback TTS model: {str(e2)}")
            model_status["tts"] = "failed"
            return False

# Function to synthesize speech from text
def synthesize_speech(text: str, target_code: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert text to speech for the specified language.
    Returns a tuple of (output_path, error_message).
    """
    global tts_model, tts_tokenizer
    
    request_id = str(uuid.uuid4())
    output_path = os.path.join(AUDIO_DIR, f"{request_id}.wav")
    
    # Make sure the TTS model is loaded for the target language
    if not load_tts_model_for_language(target_code):
        return None, "Failed to load TTS model for the target language"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
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
    start_model_loading()
    start_cleanup_task()

@app.get("/")
async def root():
    """Root endpoint for default health check"""
    logger.info("Root endpoint requested")
    return {"status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint that always returns successfully"""
    global models_loaded, loading_in_progress, model_status, error_message
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "loading_in_progress": loading_in_progress,
        "model_status": model_status,
        "error": error_message
    }

@app.post("/translate-text")
async def translate_text(text: str = Form(...), source_lang: str = Form(...), target_lang: str = Form(...)):
    """Endpoint to translate text and convert to speech"""
    global mt_model, mt_tokenizer, tts_model, tts_tokenizer, current_tts_language
    
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
    
    if model_status["mt"] == "loaded" and mt_model is not None and mt_tokenizer is not None:
        try:
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
    if model_status["tts"].startswith("loaded"):
        # Load or update TTS model for the target language
        if load_tts_model_for_language(target_code):
            try:
                output_path, error = synthesize_speech(translated_text, target_code)
                if output_path:
                    output_filename = os.path.basename(output_path)
                    output_audio_url = f"https://legendary-halibut-4p76969wqgr27xjw-8000.app.github.dev/audio_output/{output_filename}"
                    logger.info("TTS conversion completed")
            except Exception as e:
                logger.error(f"Error during TTS conversion: {str(e)}")
    
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
    global whisper_processor, whisper_model, mms_processor, mms_model
    global mt_model, mt_tokenizer, tts_model, tts_tokenizer, current_tts_language
    
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
    
    # Check if the appropriate STT model is loaded
    if use_whisper and (model_status["stt_whisper"] != "loaded" or whisper_processor is None or whisper_model is None):
        logger.warning("Whisper STT model not loaded for English/Tagalog, returning placeholder response")
        return {
            "request_id": request_id,
            "status": "processing",
            "message": "Whisper STT model not loaded yet. Please try again later.",
            "source_text": "Transcription not available",
            "translated_text": "Translation not available",
            "output_audio": None,
            "is_inappropriate": False
        }
    
    if use_mms and (model_status["stt_mms"] != "loaded" or mms_processor is None or mms_model is None):
        logger.warning("MMS STT model not loaded for Philippine languages, returning placeholder response")
        return {
            "request_id": request_id,
            "status": "processing",
            "message": "MMS STT model not loaded yet. Please try again later.",
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
        # Step 1: Load and resample the audio using torchaudio
        logger.info(f"Reading audio file: {temp_path}")
        waveform, sample_rate = torchaudio.load(temp_path)
        logger.info(f"Audio loaded: sample_rate={sample_rate}, waveform_shape={waveform.shape}")

        # Resample to 16 kHz if needed (required by Whisper and MMS models)
        if sample_rate != 16000:
            logger.info(f"Resampling audio from {sample_rate} Hz to 16000 Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Step 2: Detect speech
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

        # Step 3: Transcribe the audio (STT)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device} for STT")
        
        if use_whisper:
            # Use Whisper model for English and Tagalog
            logger.info(f"Using Whisper model for language: {source_code}")
            
            # Prepare audio for Whisper
            inputs = whisper_processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").to(device)
            logger.info("Audio processed for Whisper, generating transcription...")
            
            with torch.no_grad():
                # For English, we can specify the language; for Tagalog we use 'tl'
                forced_language = "en" if source_code == "eng" else "tl"
                generated_ids = whisper_model.generate(
                    **inputs, 
                    language=forced_language,
                    task="transcribe"
                )
                transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        else:
            # Use MMS model for other Philippine languages
            logger.info(f"Using MMS model for language: {source_code}")
            
            # Prepare audio for MMS
            inputs = mms_processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").to(device)
            logger.info("Audio processed for MMS, generating transcription...")
            
            with torch.no_grad():
                # Process with MMS
                logits = mms_model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = mms_processor.batch_decode(predicted_ids)[0]
        
        logger.info(f"Transcription completed: {transcription}")

        # Step 4: Translate the transcribed text (MT)
        if model_status["mt"] == "loaded" and mt_model is not None and mt_tokenizer is not None:
            try:
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
        else:
            logger.warning("MT model not loaded, skipping translation")

        # Step 5: Check for inappropriate content
        is_inappropriate = check_inappropriate_content(transcription) or check_inappropriate_content(translated_text)
        if is_inappropriate:
            logger.warning("Inappropriate content detected in audio transcription or translation")

        # Step 6: Convert translated text to speech (TTS)
        if load_tts_model_for_language(target_code):
            try:
                output_path, error = synthesize_speech(translated_text, target_code)
                if output_path:
                    output_filename = os.path.basename(output_path)
                    output_audio_url = f"https://legendary-halibut-4p76969wqgr27xjw-8000.app.github.dev/audio_output/{output_filename}"
                    logger.info("TTS conversion completed")
            except Exception as e:
                logger.error(f"Error during TTS conversion: {str(e)}")
        
        return {
            "request_id": request_id,
            "status": "completed",
            "message": "Transcription, translation, and TTS completed (or partially completed).",
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
        logger.info(f"Cleaning up temporary file: {temp_path}")
        os.unlink(temp_path)

@app.post("/text-to-speech")
async def text_to_speech(text: str = Form(...), target_lang: str = Form(...)):
    """Endpoint to convert text to speech in the specified language"""
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    if target_lang not in LANGUAGE_MAPPING:
        raise HTTPException(status_code=400, detail="Invalid language selected")
    
    logger.info(f"Text-to-speech requested for text in {target_lang}")
    request_id = str(uuid.uuid4())
    
    target_code = LANGUAGE_MAPPING[target_lang]
    
    # Check for inappropriate content
    is_inappropriate = check_inappropriate_content(text)
    if is_inappropriate:
        logger.warning("Inappropriate content detected in text-to-speech request")
    
    # Synthesize speech
    output_audio_url = None
    if model_status["tts"].startswith("loaded") or load_tts_model_for_language(target_code):
        try:
            output_path, error = synthesize_speech(text, target_code)
            if output_path:
                output_filename = os.path.basename(output_path)
                output_audio_url = f"https://legendary-halibut-4p76969wqgr27xjw-8000.app.github.dev/audio_output/{output_filename}"
                logger.info("TTS conversion completed")
            else:
                logger.error(f"TTS conversion failed: {error}")
        except Exception as e:
            logger.error(f"Error during TTS conversion: {str(e)}")
    else:
        logger.warning("TTS model not loaded and could not be loaded")
    
    return {
        "request_id": request_id,
        "status": "completed" if output_audio_url else "failed",
        "message": "TTS completed" if output_audio_url else "TTS failed",
        "text": text,
        "output_audio": output_audio_url,
        "is_inappropriate": is_inappropriate
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)