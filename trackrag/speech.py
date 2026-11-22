
"""Speech-to-text adapter for drop-off instructions.

Transcribes audio from a file or microphone, shows the transcript for
correction, and returns the final text through the same interface used
by typed instructions.
"""

from __future__ import annotations

import os
from pathlib import Path

from trackrag.local_speech import LocalSpeechRecognizer, SpeechInputError, SpeechTranscript


def transcribe_file(audio_path: str | Path) -> str:
    """Transcribe an audio file and return its initial machine text."""
    transcript, _ = transcribe_file_with_backend(audio_path)
    return transcript


def transcribe_file_with_backend(audio_path: str | Path) -> tuple[str, str]:
    """Return the unedited text and the backend that produced it."""
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        transcript = _transcribe_with_gemini(audio_path, api_key)
        return transcript, "gemini:" + os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    try:
        return _transcribe_with_whisper(audio_path), "openai-whisper:base"
    except ImportError:
        return _transcribe_with_speech_recognition(audio_path), "google-web-speech"


def _transcribe_with_gemini(audio_path: Path, api_key: str) -> str:
    import base64
    import mimetypes

    from dotenv import load_dotenv
    from google import genai
    from google.genai import types as genai_types

    load_dotenv()

    mime_type, _ = mimetypes.guess_type(str(audio_path))
    if not mime_type or not mime_type.startswith("audio/"):
        mime_type = "audio/wav"

    encoded = base64.b64encode(audio_path.read_bytes()).decode("utf-8")

    client = genai.Client(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    response = client.models.generate_content(
        model=model_name,
        contents=[
            "Transcribe the following audio exactly as spoken. "
            "Return only the transcript text, nothing else.",
            genai_types.Part.from_bytes(
                data=base64.b64decode(encoded),
                mime_type=mime_type,
            ),
        ],
    )

    return (response.text or "").strip()


def _transcribe_with_whisper(audio_path: Path) -> str:
    try:
        import whisper
        import numpy as np
        import wave
    except ImportError as exc:
        raise ImportError(
            "Install whisper: pip install openai-whisper"
        ) from exc

    with wave.open(str(audio_path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()

    audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

    if rate != 16000:
        import math
        ratio = 16000 / rate
        new_len = int(len(audio_np) * ratio)
        indices = np.linspace(0, len(audio_np) - 1, new_len)
        audio_np = np.interp(indices, np.arange(len(audio_np)), audio_np)

    model = whisper.load_model("base")
    result = model.transcribe(audio_np)
    return result["text"].strip()


def _transcribe_with_speech_recognition(audio_path: Path) -> str:
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise ImportError(
            "Install speech_recognition: pip install SpeechRecognition"
        ) from exc

    recognizer = sr.Recognizer()

    with sr.AudioFile(str(audio_path)) as source:
        audio_data = recognizer.record(source)

    return recognizer.recognize_google(audio_data)


def record_from_microphone(duration_s: int = 5) -> bytes:
    """Record audio from the default microphone and return raw WAV bytes."""
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise ImportError(
            "Install speech_recognition: pip install SpeechRecognition"
        ) from exc

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print(f"Recording for {duration_s} seconds…")
        audio_data = recognizer.record(source, duration=duration_s)
        print("Recording complete.")

    return audio_data.get_wav_data()


def prompt_correction(transcript: str) -> str:
    """Show the transcript to the user and allow correction before returning."""
    print(f"\nTranscript: {transcript!r}")
    corrected = input("Press Enter to accept, or type a correction: ").strip()
    return corrected if corrected else transcript


def transcribe_and_correct(
    audio_path: str | Path,
    *,
    interactive: bool = True,
) -> str:
    """Transcribe audio and optionally prompt the user to correct it."""
    transcript = transcribe_file(audio_path)

    if interactive:
        return prompt_correction(transcript)

    return transcript
