
"""Collect spoken footage-QA recordings from multiple speakers.

Usage:
  python tools/collect_recordings.py --speaker SP01 --count 10

Each recording is saved to data/recordings/<speaker_id>/<timestamp>.wav and
registered in data/recordings/manifest.jsonl.

For new takes, raw_asr_transcript is the machine output before review and
transcript is the final editable text. A null raw transcript means recognition
did not return text. An unverified_prompt_fallback copies the printed question.
Older entries provide only their original fields. duration_s is the requested
microphone interval and audio_duration_s is measured from the stored frames.
"""

import argparse
import hashlib
import io
import json
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
MANIFEST_PATH = RECORDINGS_DIR / "manifest.jsonl"

if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))


def save_wav(audio_bytes: bytes, path: Path) -> dict:
    """Validate the microphone's WAV container and store its original bytes."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
        frame_count = wav.getnframes()
        rate = wav.getframerate()
        frame_size = wav.getnchannels() * wav.getsampwidth()
        if rate < 1:
            raise ValueError("The microphone WAV has no valid sample rate")
        if frame_count < 1 or len(wav.readframes(frame_count)) != frame_size * frame_count:
            raise ValueError("The microphone returned empty or truncated WAV data")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(audio_bytes)
    return {
        "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
        "audio_sample_rate_hz": rate,
        "audio_frame_count": frame_count,
        "audio_duration_s": frame_count / rate,
    }


def append_manifest(record: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def review_transcript(audio_path: Path, prompt_text: str) -> dict:
    from trackrag.speech import prompt_correction, transcribe_file_with_backend

    raw = None
    backend = None
    failure = None
    try:
        raw, backend = transcribe_file_with_backend(audio_path)
    except Exception as exc:
        failure = type(exc).__name__
        print(f"Auto-transcription failed ({failure}).")
    if raw and raw.strip():
        final = prompt_correction(raw)
        source = "reviewed_asr_unchanged" if final == raw else "reviewed_asr_edited"
    else:
        if failure is None:
            failure = "empty_transcript"
        print("Please type the words spoken. A blank entry keeps the script as an unverified placeholder.")
        manual = input("Transcript: ").strip()
        final = manual or prompt_text
        source = "manual_after_asr_failure" if manual else "unverified_prompt_fallback"
    return {
        "transcript": final,
        "raw_asr_transcript": raw,
        "asr_backend": backend,
        "asr_error": failure,
        "transcript_source": source,
    }


def collect_one(
    speaker_id: str,
    prompt_text: str,
    duration_s: int,
    index: int,
) -> None:
    from trackrag.speech import record_from_microphone

    print(f"\n[Recording {index}]")
    print(f"Speaker: {speaker_id}")
    print(f"Please say: {prompt_text!r}")
    input("Press Enter when ready…")

    audio_bytes = record_from_microphone(duration_s=duration_s)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{index:03d}.wav"
    audio_path = RECORDINGS_DIR / speaker_id / filename

    audio_metadata = save_wav(audio_bytes, audio_path)
    print(f"Saved to {audio_path.relative_to(PROJECT_ROOT)}")

    transcription = review_transcript(audio_path, prompt_text)
    record = {
        "recording_id": f"{speaker_id}_{timestamp}_{index:03d}",
        "speaker_id": speaker_id,
        "audio_path": str(audio_path.relative_to(PROJECT_ROOT)),
        "prompt": prompt_text,
        **transcription,
        "duration_s": duration_s,
        "collected_at": timestamp,
        **audio_metadata,
    }

    append_manifest(record)
    print(f"Logged to manifest.")


PROMPTS = [
    "How many pedestrians are visible near the crossing?",
    "Is there a vehicle stopped at the junction in this frame?",
    "Does the pedestrian track appear in all four frames?",
    "Which tracks are present at both the start and end of the interval?",
    "Is a small motorised vehicle visible at the right edge of the frame?",
    "How many road users are annotated in this frame?",
    "Does the same emergency vehicle appear in both frames?",
    "Are there any pedestrians crossing the road in frame two hundred and five?",
    "Which pedestrian track IDs are visible near the traffic light?",
    "Is the pedestrian track continuous across the selected frames?",
]


def main():
    parser = argparse.ArgumentParser(description="Collect spoken footage-QA recordings.")
    parser.add_argument("--speaker", required=True, help="Speaker ID, e.g. SP01")
    parser.add_argument("--count", type=int, default=10, help="Number of recordings to collect")
    parser.add_argument("--duration", type=int, default=5, help="Seconds to record per instruction")
    parser.add_argument("--start-index", type=int, default=1, help="Index of the first recording")
    args = parser.parse_args()

    prompts = PROMPTS[:args.count]

    print(f"\nCollecting {len(prompts)} recording(s) for speaker {args.speaker}")
    print("Audio will be saved to:", RECORDINGS_DIR / args.speaker)

    for i, prompt in enumerate(prompts, start=args.start_index):
        collect_one(
            speaker_id=args.speaker,
            prompt_text=prompt,
            duration_s=args.duration,
            index=i,
        )

    print(f"\nDone. Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
