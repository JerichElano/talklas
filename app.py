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
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, Tuple
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("talklas-simplified-api")

app = FastAPI(title="Talklas Simplified API")

# Mount a directory to serve audio files
AUDIO_DIR = "/tmp/audio_output"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio_output", StaticFiles(directory=AUDIO_DIR), name="audio_output")

# Global variables to track application state
models_loaded = False
loading_in_progress = False
model_status = {
    "stt_whisper": "not_loaded",
    "tts": "not_loaded"
}
error_message = None
current_tts_language = "eng"

# Model instances
whisper_processor = None
whisper_model = None
tts_model = None
tts_tokenizer = None

# Define valid languages
LANGUAGE_MAPPING = {
    "English": "eng",
    "Tagalog": "tgl",
    "Cebuano": "ceb",
    "Ilocano": "ilo",
    "Waray": "war",
    "Pangasinan": "pag"
}

# Common phrases and their translations
COMMON_PHRASES = {
    "hello": {"eng": "Hello", "tgl": "Kamusta", "ceb": "Kumusta", "ilo": "Hello", "war": "Kamusta", "pag": "Kumusta"},
    "thank you": {"eng": "Thank you", "tgl": "Salamat", "ceb": "Salamat", "ilo": "Agyamanak", "war": "Salamat", "pag": "Salamat"},
    "how are you": {"eng": "How are you", "tgl": "Kumusta ka", "ceb": "Kumusta ka", "ilo": "Kumusta ka", "war": "Kumusta ka", "pag": "Kumusta ka"},
    "good morning": {"eng": "Good morning", "tgl": "Magandang umaga", "ceb": "Maayong buntag", "ilo": "Naimbag a bigat", "war": "Maupay nga aga", "pag": "Masanto ya kabwasan"},
    "goodbye": {"eng": "Goodbye", "tgl": "Paalam", "ceb": "Paalam", "ilo": "Agpakada", "war": "Paaram", "pag": "Agya"}
}

# Function to save PCM data as a WAV file
def save_pcm_to_wav(pcm_data: list, sample_rate: int, output_path: str):
    pcm_array = np.array(pcm_data, dtype=np.int16)
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_array.tobytes())

# Function to detect speech using an energy-based approach
def detect_speech(waveform: torch.Tensor, sample_rate: int, threshold: float = 0.01) -> bool:
    waveform_np = waveform.numpy()
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=0)
    rms = np.sqrt(np.mean(waveform_np**2))
    logger.info(f"RMS energy: {rms}")
    if rms < threshold:
        logger.info("No speech detected: RMS energy below threshold")
        return False
    return True

# Function to clean up old audio files
def cleanup_old_audio_files():
    logger.info("Starting cleanup of old audio files...")
    expiration_time = datetime.now() - timedelta(minutes=10)
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
        time.sleep(300)

# Function to load models in background
def load_models_task():
    global models_loaded, loading_in_progress, model_status, error_message
    global whisper_processor, whisper_model, tts_model, tts_tokenizer
    
    try:
        loading_in_progress = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Whisper STT model
        logger.info("Loading Whisper STT model...")
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        try:
            model_status["stt_whisper"] = "loading"
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            whisper_model.to(device)
            logger.info("Whisper STT model loaded successfully")
            model_status["stt_whisper"] = "loaded"
        except Exception as e:
            logger.error(f"Failed to load Whisper STT model: {str(e)}")
            model_status["stt_whisper"] = "failed"
            error_message = f"Whisper STT model loading failed: {str(e)}"
            return
        
        # Load TTS model (default to English)
        logger.info("Loading MMS-TTS model for English...")
        from transformers import VitsModel, AutoTokenizer
        try:
            model_status["tts"] = "loading"
            tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
            tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            tts_model.to(device)
            logger.info("TTS model loaded successfully")
            model_status["tts"] = "loaded"
        except Exception as e:
            logger.error(f"Failed to load TTS model: {str(e)}")
            model_status["tts"] = "failed"
            error_message = f"TTS model loading failed: {str(e)}"
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
    global loading_in_progress, models_loaded
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
    global tts_model, tts_tokenizer, current_tts_language, model_status
    
    if target_code not in LANGUAGE_MAPPING.values():
        logger.error(f"Invalid language code: {target_code}")
        return False
    
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
        model_status["tts"] = "failed"
        return False

# Function to find a matching phrase
def find_matching_phrase(transcription: str, source_code: str) -> Tuple[Optional[str], Optional[str]]:
    transcription = transcription.lower().strip()
    best_match = None
    best_score = 0
    threshold = 80  # Fuzzy matching threshold
    
    for phrase_key, translations in COMMON_PHRASES.items():
        source_phrase = translations.get(source_code, "").lower()
        if source_phrase:
            score = fuzz.ratio(transcription, source_phrase)
            if score > best_score and score >= threshold:
                best_score = score
                best_match = phrase_key
    
    if best_match:
        logger.info(f"Matched phrase: {best_match} (score: {best_score})")
        return best_match, COMMON_PHRASES[best_match]
    else:
        logger.info("No matching phrase found")
        return None, None

# Function to synthesize speech from text
def synthesize_speech(text: str, target_code: str) -> Tuple[Optional[str], Optional[str]]:
    global tts_model, tts_tokenizer
    
    request_id = str(uuid.uuid4())
    output_path = os.path.join(AUDIO_DIR, f"{request_id}.wav")
    
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

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "loading_in_progress": loading_in_progress,
        "model_status": model_status,
        "error": error_message
    }

@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...), source_lang: str = Form(...), target_lang: str = Form(...)):
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file provided")
    if source_lang not in LANGUAGE_MAPPING or target_lang not in LANGUAGE_MAPPING:
        raise HTTPException(status_code=400, detail="Invalid language selected")
    
    source_code = LANGUAGE_MAPPING[source_lang]
    target_code = LANGUAGE_MAPPING[target_lang]
    
    logger.info(f"Process-audio requested: {audio.filename} from {source_lang} ({source_code}) to {target_lang} ({target_code})")
    request_id = str(uuid.uuid4())
    
    if model_status["stt_whisper"] != "loaded" or whisper_processor is None or whisper_model is None:
        logger.warning("Whisper STT model not loaded")
        return {
            "request_id": request_id,
            "status": "processing",
            "message": "Whisper STT model not loaded yet. Please try again later.",
            "source_text": "Transcription not available",
            "translated_text": "Translation not available",
            "output_audio": None
        }
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await audio.read())
        temp_path = temp_file.name
    
    transcription = "Transcription not available"
    translated_text = "Translation not available"
    output_audio_url = None
    
    try:
        # Load and resample audio
        logger.info(f"Reading audio file: {temp_path}")
        waveform, sample_rate = torchaudio.load(temp_path)
        if sample_rate != 16000:
            logger.info(f"Resampling audio from {sample_rate} Hz to 16000 Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Detect speech
        if not detect_speech(waveform, sample_rate):
            return {
                "request_id": request_id,
                "status": "failed",
                "message": "No speech detected in the audio.",
                "source_text": "No speech detected",
                "translated_text": "No translation available",
                "output_audio": None
            }

        # Transcribe audio using Whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using Whisper model for language: {source_code}")
        inputs = whisper_processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            forced_language = "en" if source_code == "eng" else "tl"
            generated_ids = whisper_model.generate(
                **inputs, 
                language=forced_language,
                task="transcribe"
            )
            transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info(f"Transcription completed: {transcription}")

        # Find matching phrase
        matched_phrase_key, translations = find_matching_phrase(transcription, source_code)
        if matched_phrase_key and translations:
            translated_text = translations.get(target_code, translations["eng"])
            logger.info(f"Using matched translation: {translated_text}")
        else:
            translated_text = "No matching phrase found"
            logger.info("No matching phrase, using default message")

        # Convert to speech
        if load_tts_model_for_language(target_code):
            try:
                output_path, error = synthesize_speech(translated_text, target_code)
                if output_path:
                    output_filename = os.path.basename(output_path)
                    output_audio_url = f"/audio_output/{output_filename}"  # Relative URL for simplicity
                    logger.info("TTS conversion completed")
            except Exception as e:
                logger.error(f"Error during TTS conversion: {str(e)}")
        
        return {
            "request_id": request_id,
            "status": "completed",
            "message": "Transcription, phrase matching, and TTS completed.",
            "source_text": transcription,
            "translated_text": translated_text,
            "output_audio": output_audio_url
        }
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        return {
            "request_id": request_id,
            "status": "failed",
            "message": f"Processing failed: {str(e)}",
            "source_text": transcription,
            "translated_text": translated_text,
            "output_audio": output_audio_url
        }
    finally:
        logger.info(f"Cleaning up temporary file: {temp_path}")
        os.unlink(temp_path)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)