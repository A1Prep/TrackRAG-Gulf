
"""Collect spoken drop-off instruction recordings from multiple speakers.

Usage:
  python tools/collect_recordings.py --speaker SP01 --count 10

Each recording is saved to data/recordings/<speaker_id>/<timestamp>.wav and
registered in data/recordings/manifest.jsonl.
"""

import argparse
import json
import wave
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
MANIFEST_PATH = RECORDINGS_DIR / "manifest.jsonl"


def save_wav(audio_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)


def append_manifest(record: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def collect_one(
    speaker_id: str,
    prompt_text: str,
    duration_s: int,
    index: int,
) -> None:
    from trackrag.speech import record_from_microphone, transcribe_and_correct

    print(f"\n[Recording {index}]")
    print(f"Speaker: {speaker_id}")
    print(f"Please say: {prompt_text!r}")
    input("Press Enter when ready…")

    audio_bytes = record_from_microphone(duration_s=duration_s)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{timestamp}_{index:03d}.wav"
    audio_path = RECORDINGS_DIR / speaker_id / filename

    save_wav(audio_bytes, audio_path)
    print(f"Saved to {audio_path.relative_to(PROJECT_ROOT)}")

    transcript = transcribe_and_correct(audio_path, interactive=True)

    record = {
        "recording_id": f"{speaker_id}_{timestamp}_{index:03d}",
        "speaker_id": speaker_id,
        "audio_path": str(audio_path.relative_to(PROJECT_ROOT)),
        "prompt": prompt_text,
        "transcript": transcript,
        "duration_s": duration_s,
        "collected_at": timestamp,
    }

    append_manifest(record)
    print(f"Logged to manifest.")


PROMPTS = [
    "Leave the package at the front door.",
    "Please drop it at the reception desk.",
    "Hand it to the security guard at the gate.",
    "Leave it with the neighbour in flat twelve.",
    "Place it in the parcel locker on the left.",
    "Ring the bell and wait for someone to answer.",
    "Leave it by the side entrance, not the main door.",
    "Put the parcel inside the green bin area.",
    "Give it to anyone at the counter.",
    "Drop it at the building concierge.",
]


def main():
    parser = argparse.ArgumentParser(description="Collect spoken drop-off instruction recordings.")
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
