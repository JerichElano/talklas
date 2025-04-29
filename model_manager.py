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
from tenacity import retry, stop_after_attempt, wait_exponential
import requests.exceptions
from model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("talklas-api")

app = FastAPI(title="Talklas API")

AUDIO_DIR = "/tmp/audio_output"
os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio_output", StaticFiles(directory=AUDIO_DIR), name="audio_output")

model_manager = ModelManager(max_idle_time=300, memory_check_interval=60)

LANGUAGE_MAPPING = {
    "English": "eng",
    "Tagalog": "tgl",
    "Cebuano": "ceb",
    "Ilocano": "ilo",
    "Waray": "war",
    "Pangasinan": "pag"
}

WHISPER_LANGUAGES = {"eng", "tgl"}
MMS_LANGUAGES = {"ceb", "ilo", "war", "pag"}

NLLB_LANGUAGE_CODES = {
    "eng": "eng_Latn",
    "tgl": "tgl_Latn",
    "ceb": "ceb_Latn",
    "ilo": "ilo_Latn",
    "war": "war_Latn",
    "pag": "pag_Latn"
}

INAPPROPRIATE_WORDS = [
    "fuck", "shit", "bitch", "ass", "damn", "hell", "bastard", "cunt", "son of a bitch", "dick", "pussy", "motherfucker",
    "agka baboy", "puta", "putang ina", "gago", "tanga", "hayop", "ulol", "lintik", "animal ka",
    "paki", "pakyu", "yawa", "bungol", "gingan", "yawa ka", "peste", "irig",
    "pakit", "ayat", "pua", "kayat mo ti agsardeng", "hinampak", "iring ka"
]

def check_inappropriate_content(text: str) -> bool:
    text_lower = text.lower()
    for word in INAPPROPRIATE_WORDS:
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text_lower):
            logger.warning(f"Inappropriate content detected: {word}")
            return True
    return False

def save_pcm_to_wav(pcm_data: list, sample_rate: int, output_path: str):
    pcm_array = np.array(pcm_data, dtype=np.int16)
    with wave.open(output_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_array.tobytes())

def detect_speech(waveform: torch.Tensor, sample_rate: int, threshold: float = 0.01, min_speech_duration: float = 0.5) -> bool:
    waveform_np = waveform.numpy()
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.mean(axis=0)
    rms = np.sqrt(np.mean(waveform_np**2))
    logger.info(f"RMS energy: {rms}")
    if rms < threshold:
        logger.info("No speech detected: RMS energy below threshold")
        return False
    return True

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

def schedule_cleanup():
    while True:
        cleanup_old_audio_files()
        time.sleep(300)

def synthesize_speech(text: str, target_code: str) -> Tuple[Optional[str], Optional[str]]:
    tts_model_name = f"tts_{target_code}"
    tts_model = model_manager.get_model(tts_model_name)
    tts_tokenizer = model_manager.get_tokenizer(tts_model_name)
    
    if tts_model is None or tts_tokenizer is None:
        return None, f"Failed to load TTS model for {target_code}"
    
    request_id = str(uuid.uuid4())
    output_path = os.path.join(AUDIO_DIR, f"{request_id}.wav")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model_manager.use_model(tts_model_name)
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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: False
)
async def load_model_with_retry(model_name: str, model_manager: ModelManager) -> bool:
    try:
        success = model_manager.load_model(model_name)
        if success:
            logger.info(f"Successfully loaded {model_name}")
            return True
        raise Exception(f"Failed to load {model_name}")
    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
        logger.error(f"Network error loading {model_name}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading {model_name}: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")
    
    cleanup_thread = threading.Thread(target=schedule_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Pre-load STT models
    default_models = [
        "stt_whisper",
        "stt_mms"
    ]
    
    for model_name in default_models:
        try:
            logger.info(f"Pre-loading STT model: {model_name}")
            success = await load_model_with_retry(model_name, model_manager)
            if not success:
                logger.error(f"Failed to pre-load {model_name} after retries")
            else:
                model_manager.mark_default(model_name)
        except Exception as e:
            logger.error(f"Error pre-loading {model_name}: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    model_manager.shutdown()

@app.get("/")
async def root():
    logger.info("Root endpoint requested")
    return {"status": "healthy"}

@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "model_status": model_manager.get_model_status()
    }

@app.post("/translate-text")
async def translate_text(text: str = Form(...), source_lang: str = Form(...), target_lang: str = Form(...)):
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    if source_lang not in LANGUAGE_MAPPING or target_lang not in LANGUAGE_MAPPING:
        raise HTTPException(status_code=400, detail="Invalid language selected")
    
    logger.info(f"Translate-text requested: {text} from {source_lang} to {target_lang}")
    request_id = str(uuid.uuid4())
    
    source_code = LANGUAGE_MAPPING[source_lang]
    target_code = LANGUAGE_MAPPING[target_lang]
    translated_text = "Translation not available"
    
    mt_model = model_manager.get_model("mt")
    mt_tokenizer = model_manager.get_tokenizer("mt")
    
    if mt_model is not None and mt_tokenizer is not None:
        try:
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

    is_inappropriate = check_inappropriate_content(text) or check_inappropriate_content(translated_text)
    if is_inappropriate:
        logger.warning("Inappropriate content detected in translation request")

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
    if not audio:
        raise HTTPException(status_code=400, detail="No audio file provided")
    if source_lang not in LANGUAGE_MAPPING or target_lang not in LANGUAGE_MAPPING:
        raise HTTPException(status_code=400, detail="Invalid language selected")
    
    source_code = LANGUAGE_MAPPING[source_lang]
    target_code = LANGUAGE_MAPPING[target_lang]
    
    logger.info(f"Translate-audio requested: {audio.filename} from {source_lang} ({source_code}) to {target_lang} ({target_code})")
    request_id = str(uuid.uuid4())
    
    # Determine which STT model to use
    use_whisper = source_code in WHISPER_LANGUAGES
    stt_model_name = "stt_whisper" if use_whisper else "stt_mms"
    
    # Ensure only the needed STT model is loaded
    model_manager.mark_default(stt_model_name)
    model_manager.restore_default_models()
    
    # Get STT model
    stt_model = model_manager.get_model(stt_model_name)
    stt_processor = model_manager.get_tokenizer(stt_model_name)
    
    if stt_model is None or stt_processor is None:
        model_manager.restore_default_models()
        return {
            "request_id": request_id,
            "status": "failed",
            "message": f"Failed to load {stt_model_name} model",
            "source_text": "Transcription not available",
            "translated_text": "Translation not available",
            "output_audio": None,
            "is_inappropriate": False
        }

    # Save uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(await audio.read())
        temp_path = temp_file.name

    transcription = "Transcription not available"
    translated_text = "Translation not available"
    output_audio_url = None
    is_inappropriate = False

    try:
        # Process audio
        waveform, sample_rate = torchaudio.load(temp_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        if not detect_speech(waveform, sample_rate):
            model_manager.restore_default_models()
            return {
                "request_id": request_id,
                "status": "failed",
                "message": "No speech detected in the audio.",
                "source_text": "No speech detected",
                "translated_text": "No translation available",
                "output_audio": None,
                "is_inappropriate": False
            }

        model_manager.use_model(stt_model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Transcribe
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

        # Unload STT and load MT
        model_manager.unload_model(stt_model_name)
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

        # Unload MT and load TTS
        model_manager.unload_model("mt")
        is_inappropriate = check_inappropriate_content(transcription) or check_inappropriate_content(translated_text)
        if is_inappropriate:
            logger.warning("Inappropriate content detected")

        if not is_inappropriate:
            output_path, error = synthesize_speech(translated_text, target_code)
            if output_path:
                output_filename = os.path.basename(output_path)
                output_audio_url = f"https://legendary-halibut-4p76969wqgr27xjw-8000.app.github.dev/audio_output/{output_filename}"
                logger.info("TTS conversion completed")

        # Unload TTS and restore default STT
        model_manager.unload_model(f"tts_{target_code}")
        model_manager.restore_default_models()

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
        model_manager.restore_default_models()
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
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.error(f"Error removing temporary file: {str(e)}")