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
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, Optional, Tuple
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
    "stt": "not_loaded",
    "mt": "not_loaded",
    "tts": "not_loaded"
}
error_message = None
current_tts_language = "tgl"  # Track the current TTS language

# Model instances
stt_processor = None
stt_model = None
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

NLLB_LANGUAGE_CODES = {
    "eng": "eng_Latn",
    "tgl": "tgl_Latn",
    "ceb": "ceb_Latn",
    "ilo": "ilo_Latn",
    "war": "war_Latn",
    "pag": "pag_Latn"
}

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
    global stt_processor, stt_model, mt_model, mt_tokenizer, tts_model, tts_tokenizer
    
    try:
        loading_in_progress = True
        
        # Load STT model (MMS with fallback to Whisper)
        logger.info("Starting to load STT model...")
        from transformers import AutoProcessor, AutoModelForCTC, WhisperProcessor, WhisperForConditionalGeneration
        
        try:
            logger.info("Loading MMS STT model...")
            model_status["stt"] = "loading"
            stt_processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
            stt_model = AutoModelForCTC.from_pretrained("facebook/mms-1b-all")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            stt_model.to(device)
            logger.info("MMS STT model loaded successfully")
            model_status["stt"] = "loaded_mms"
        except Exception as mms_error:
            logger.error(f"Failed to load MMS STT model: {str(mms_error)}")
            logger.info("Falling back to Whisper STT model...")
            try:
                stt_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
                stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
                stt_model.to(device)
                logger.info("Whisper STT model loaded successfully as fallback")
                model_status["stt"] = "loaded_whisper"
            except Exception as whisper_error:
                logger.error(f"Failed to load Whisper STT model: {str(whisper_error)}")
                model_status["stt"] = "failed"
                error_message = f"STT model loading failed: MMS error: {str(mms_error)}, Whisper error: {str(whisper_error)}"
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

@app.post("/update-languages")
async def update_languages(source_lang: str = Form(...), target_lang: str = Form(...)):
    global stt_processor, stt_model, tts_model, tts_tokenizer, current_tts_language
    
    if source_lang not in LANGUAGE_MAPPING or target_lang not in LANGUAGE_MAPPING:
        raise HTTPException(status_code=400, detail="Invalid language selected")
    
    source_code = LANGUAGE_MAPPING[source_lang]
    target_code = LANGUAGE_MAPPING[target_lang]
    
    # Update the STT model based on the source language (MMS or Whisper)
    try:
        logger.info("Updating STT model for source language...")
        from transformers import AutoProcessor, AutoModelForCTC, WhisperProcessor, WhisperForConditionalGeneration
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info(f"Loading MMS STT model for {source_code}...")
            stt_processor = AutoProcessor.from_pretrained("facebook/mms-1b-all")
            stt_model = AutoModelForCTC.from_pretrained("facebook/mms-1b-all")
            stt_model.to(device)
            # Set the target language for MMS
            if source_code in stt_processor.tokenizer.vocab.keys():
                stt_processor.tokenizer.set_target_lang(source_code)
                stt_model.load_adapter(source_code)
                logger.info(f"MMS STT model updated to {source_code}")
                model_status["stt"] = "loaded_mms"
            else:
                logger.warning(f"Language {source_code} not supported by MMS, using default")
                model_status["stt"] = "loaded_mms_default"
        except Exception as mms_error:
            logger.error(f"Failed to load MMS STT model for {source_code}: {str(mms_error)}")
            logger.info("Falling back to Whisper STT model...")
            try:
                stt_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
                stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
                stt_model.to(device)
                logger.info("Whisper STT model loaded successfully as fallback")
                model_status["stt"] = "loaded_whisper"
            except Exception as whisper_error:
                logger.error(f"Failed to load Whisper STT model: {str(whisper_error)}")
                model_status["stt"] = "failed"
                error_message = f"STT model update failed: MMS error: {str(mms_error)}, Whisper error: {str(whisper_error)}"
                return {"status": "failed", "error": error_message}
    except Exception as e:
        logger.error(f"Error updating STT model: {str(e)}")
        model_status["stt"] = "failed"
        error_message = f"STT model update failed: {str(e)}"
        return {"status": "failed", "error": error_message}
    
    # Update the TTS model based on the target language
    try:
        logger.info(f"Loading MMS-TTS model for {target_code}...")
        from transformers import VitsModel, AutoTokenizer
        tts_model = VitsModel.from_pretrained(f"facebook/mms-tts-{target_code}")
        tts_tokenizer = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{target_code}")
        tts_model.to(device)
        current_tts_language = target_code
        logger.info(f"TTS model updated to {target_code}")
        model_status["tts"] = "loaded"
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
        except Exception as e2:
            logger.error(f"Failed to load fallback TTS model: {str(e2)}")
            model_status["tts"] = "failed"
            error_message = f"TTS model loading failed: {str(e)} (fallback also failed: {str(e2)})"
            return {"status": "failed", "error": error_message}
    
    logger.info(f"Updating languages: {source_lang} → {target_lang}")
    return {"status": f"Languages updated to {source_lang} → {target_lang}"}

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

    # Update TTS model if the target language doesn't match the current TTS language
    if current_tts_language != target_code:
        try:
            logger.info(f"Updating TTS model for {target_code}...")
            from transformers import VitsModel, AutoTokenizer
            tts_model = VitsModel.from_pretrained(f"facebook/mms-tts-{target_code}")
            tts_tokenizer = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{target_code}")
            tts_model.to(device)
            current_tts_language = target_code
            logger.info(f"TTS model updated to {target_code}")
            model_status["tts"] = "loaded"
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
            except Exception as e2:
                logger.error(f"Failed to load fallback TTS model: {str(e2)}")
                model_status["tts"] = "failed"

    # Convert translated text to speech
    output_audio_url = None
    if model_status["tts"].startswith("loaded") and tts_model is not None and tts_tokenizer is not None:
        try:
            inputs = tts_tokenizer(translated_text, return_tensors="pt").to(device)
            with torch.no_grad():
                output = tts_model(**inputs)
            speech = output.waveform.cpu().numpy().squeeze()
            speech = (speech * 32767).astype(np.int16)
            sample_rate = tts_model.config.sampling_rate

            # Save the audio as a WAV file
            output_filename = f"{request_id}.wav"
            output_path = os.path.join(AUDIO_DIR, output_filename)
            save_pcm_to_wav(speech.tolist(), sample_rate, output_path)
            logger.info(f"Saved synthesized audio to {output_path}")

            # Generate a URL to the WAV file
            output_audio_url = f"https://jerich-talklasapp.hf.space/audio_output/{output_filename}"
            logger.info("TTS conversion completed")
        except Exception as e:
            logger.error(f"Error during TTS conversion: {str(e)}")
            output_audio_url = None
    
    return {
        "request_id": request_id,
        "status": "completed",
        "message": "Translation and TTS completed (or partially completed).",
        "source_text": text,
        "translated_text": translated_text,
        "output_audio": output_audio_url
    }

@app.post("/translate-audio")
async def translate_audio(audio: UploadFile = File(...), source_lang: str = Form(...), target_lang: str = Form(...)):
    """Endpoint to transcribe, translate, and convert audio to speech"""
    global stt_processor, stt_model, mt_model, mt_tokenizer, tts_model, tts_tokenizer, current_tts_language
    
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file provided")
    if source_lang not in LANGUAGE_MAPPING or target_lang not in LANGUAGE_MAPPING:
        raise HTTPException(status_code=400, detail="Invalid language selected")
    
    logger.info(f"Translate-audio requested: {audio.filename} from {source_lang} to {target_lang}")
    request_id = str(uuid.uuid4())
    
    # Check if STT model is loaded
    if model_status["stt"] not in ["loaded_mms", "loaded_mms_default", "loaded_whisper"] or stt_processor is None or stt_model is None:
        logger.warning("STT model not loaded, returning placeholder response")
        return {
            "request_id": request_id,
            "status": "processing",
            "message": "STT model not loaded yet. Please try again later.",
            "source_text": "Transcription not available",
            "translated_text": "Translation not available",
            "output_audio": None
        }
    
    # Save the uploaded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await audio.read())
        temp_path = temp_file.name
    
    transcription = "Transcription not available"
    translated_text = "Translation not available"
    output_audio_url = None
    
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
                "output_audio": None
            }

        # Step 3: Transcribe the audio (STT)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        inputs = stt_processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt").to(device)
        logger.info("Audio processed, generating transcription...")
        
        with torch.no_grad():
            if model_status["stt"] == "loaded_whisper":
                # Whisper model
                generated_ids = stt_model.generate(**inputs, language="en")
                transcription = stt_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                # MMS model
                logits = stt_model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = stt_processor.batch_decode(predicted_ids)[0]
        logger.info(f"Transcription completed: {transcription}")

        # Step 4: Translate the transcribed text (MT)
        source_code = LANGUAGE_MAPPING[source_lang]
        target_code = LANGUAGE_MAPPING[target_lang]
        
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

        # Step 5: Update TTS model if the target language doesn't match the current TTS language
        if current_tts_language != target_code:
            try:
                logger.info(f"Updating TTS model for {target_code}...")
                from transformers import VitsModel, AutoTokenizer
                tts_model = VitsModel.from_pretrained(f"facebook/mms-tts-{target_code}")
                tts_tokenizer = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{target_code}")
                tts_model.to(device)
                current_tts_language = target_code
                logger.info(f"TTS model updated to {target_code}")
                model_status["tts"] = "loaded"
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
                except Exception as e2:
                    logger.error(f"Failed to load fallback TTS model: {str(e2)}")
                    model_status["tts"] = "failed"

        # Step 6: Convert translated text to speech (TTS)
        if model_status["tts"].startswith("loaded") and tts_model is not None and tts_tokenizer is not None:
            try:
                inputs = tts_tokenizer(translated_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = tts_model(**inputs)
                speech = output.waveform.cpu().numpy().squeeze()
                speech = (speech * 32767).astype(np.int16)
                sample_rate = tts_model.config.sampling_rate

                # Save the audio as a WAV file
                output_filename = f"{request_id}.wav"
                output_path = os.path.join(AUDIO_DIR, output_filename)
                save_pcm_to_wav(speech.tolist(), sample_rate, output_path)
                logger.info(f"Saved synthesized audio to {output_path}")

                # Generate a URL to the WAV file
                output_audio_url = f"https://jerich-talklasapp.hf.space/audio_output/{output_filename}"
                logger.info("TTS conversion completed")
            except Exception as e:
                logger.error(f"Error during TTS conversion: {str(e)}")
                output_audio_url = None
        
        return {
            "request_id": request_id,
            "status": "completed",
            "message": "Transcription, translation, and TTS completed (or partially completed).",
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
