
"""Speech-to-text adapter for drop-off instructions.

Transcribes audio from a file or microphone, shows the transcript for
correction, and returns the final text through the same interface used
by typed instructions.
"""

from __future__ import annotations

import os
from pathlib import Path


def transcribe_file(audio_path: str | Path) -> str:
    """Transcribe an audio file using Gemini's speech-to-text capability.

    Falls back to SpeechRecognition + Google Web Speech API when
    GEMINI_API_KEY is unavailable.
    """
    audio_path = Path(audio_path)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return _transcribe_with_gemini(audio_path, api_key)

    return _transcribe_with_speech_recognition(audio_path)


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
