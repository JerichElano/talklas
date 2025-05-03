# Talklas Simplified API

This FastAPI application processes audio input for speech transcription, matches the transcribed text against common phrases, and generates speech output using text-to-speech (TTS). It uses Whisper Small for transcription and MMS-TTS for speech synthesis, designed to run in GitHub Codespaces or locally.


## Features

- **Speech Detection**: Detects if the audio contains speech using an energy-based approach.
- **Transcription (STT)**: Uses Whisper Small to transcribe audio in English or Tagalog.
- **Phrase Matching**: Matches transcribed text to a predefined list of common phrases using fuzzy matching.
- **Text-to-Speech (TTS)**: Converts matched phrases to speech in the target language using MMS-TTS.
- **Supported Languages**: English, Tagalog, Cebuano, Ilocano, Waray, Pangasinan.


## Endpoints

- `/health`: Returns the status of model loading and any errors.
- `/process-audio`: Transcribes audio, matches the transcription to common phrases, and generates speech in the target language.


## Setup in GitHub Codespaces

1. Create a Codespace:

- Fork or clone this repository to your GitHub account.
- Open the repository in GitHub Codespaces by clicking "Code" > "Open with Codespaces" > "New codespace".


2. Install Dependencies:

- The .devcontainer/devcontainer.json file automatically sets up the environment.
- If needed, run pip install -r requirements.txt in the Codespace terminal.


3. Run the Application:

- Start the FastAPI server with:uvicorn app: `app --host 0.0.0.0 --port 8000`
- Codespaces will prompt to open port 8000. Make it public to access the app externally.
- Access the app at the provided Codespaces URL (e.g., `https://<codespace-name>-8000.app.github.dev`).


4. Test the Endpoint:

- Use a tool like  `curl` to send a POST request to `/process-audio` with an audio file, `source_lang`, and `target_lang`.
- Example: `curl -X POST -F "audio=@sample.wav" -F "source_lang=English" -F "target_lang=Tagalog" http://localhost:8000/process-audio`


## Local Setup

1. Install Dependencies:
`pip install -r requirements.txt`

2. Run the Application:
`uvicorn app:app --host 0.0.0.0 --port 8000`

3. Access the App:

Open `http://localhost:8000/health` in a browser to check the status.
Test the `/process-audio` endpoint as described above.



### Notes

- The app stores temporary audio files in /tmp/audio_output, which are cleaned up every 5 minutes.
- Ensure sufficient memory (at least 4GB) in Codespaces for model loading.
- Models are downloaded from Hugging Face on first run, which may take time depending on network speed.

