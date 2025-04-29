---
title: Talklas API
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
health_check_path: /health
---

# Talklas API

This is a FastAPI app deployed on Hugging Face Spaces for audio transcription, translation, and text-to-speech (TTS). It includes the following endpoints:

- `/`: Returns a simple health check response.
- `/health`: Health check endpoint for Hugging Face Spaces.
- `/update-languages`: Updates the source and target languages for STT and TTS models.
- `/translate-text`: Translates text and converts it to speech.
- `/translate-audio`: Transcribes audio, translates the text, and converts the translated text to speech. Includes speech detection to handle silent audio gracefully.

## Features

- **Speech Detection**: The `/translate-audio` endpoint detects if the audio is silent (no speech) and returns a user-friendly response.
- **Transcription (STT)**: Uses MMS or Whisper models to transcribe audio.
- **Translation (MT)**: Uses the NLLB-200 model to translate text between supported languages.
- **Text-to-Speech (TTS)**: Uses MMS-TTS models to convert translated text to speech.

## Supported Languages

- English
- Tagalog
- Cebuano
- Ilocano
- Waray
- Pangasinan

## Deployment

This app uses a `Dockerfile` to deploy a FastAPI app with Uvicorn. The health check path is set to `/health` to ensure Hugging Face Spaces can verify the app is running.
