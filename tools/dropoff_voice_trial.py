"""Generate or replay synthetic-English passenger audio with the real offline recognizer."""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from trackrag.speech import LocalSpeechRecognizer, SpeechInputError


# Declared before the first ASR run. Critical terms use whole words in spoken order.
# Each case is spoken by two Windows desktop synthetic voices. No passenger recordings.
PROMPTS = (
    ("bus-left", "Please drop me beside the yellow bus on the left.", ("beside", "yellow", "bus", "left")),
    ("deictic", "Drop me here.", ("here",)),
    ("black-cars", "Drop me between the two parked black cars on the right.",
     ("between", "two", "parked", "black", "cars", "right")),
    ("truck-ahead", "I mean the area near the large white truck ahead.", ("near", "large", "white", "truck", "ahead")),
    ("negation", "Near the white truck, not the black car.", ("white", "truck", "not", "black", "car")),
    ("ordinal", "Please stop in front of the second blue car.", ("front", "second", "blue", "car")),
    ("correction", "Sorry, I meant the one on the right.", ("right",)),
    ("cancel", "Cancel that request.", ("cancel",)),
)
VOICES = (("david", "Microsoft David Desktop"), ("zira", "Microsoft Zira Desktop"))


def synthesize(audio_dir, prompts=PROMPTS):
    if sys.platform != "win32":
        raise ValueError("Generating the reference voices requires Windows. Replay saved WAVs on other platforms.")
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"path": str((audio_dir / f"{case}-{voice}.wav").resolve()), "text": phrase, "voice": name}
        for case, phrase, _ in prompts for voice, name in VOICES
    ]
    if any(Path(row["path"]).exists() for row in rows):
        raise FileExistsError("Use a new audio folder or replay the existing files.")
    script = """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$rows = [Console]::In.ReadToEnd() | ConvertFrom-Json
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($row in $rows) {
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
        $synth.SelectVoice($row.voice)
        $synth.SetOutputToWaveFile($row.path, $format)
        $synth.Speak($row.text)
    } finally { $synth.Dispose() }
}
"""
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                   input=json.dumps(rows), text=True, check=True, timeout=120)


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for position, expected in enumerate(reference, start=1):
        row = [position]
        for j, actual in enumerate(hypothesis, start=1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (expected != actual)))
        previous = row
    return previous[-1]


def retains_terms(hypothesis, terms):
    remaining = iter(hypothesis)
    return all(any(word == term for word in remaining) for term in terms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--generate", action="store_true", help="Generate the fixed Windows TTS WAV files first.")
    parser.add_argument("--out", type=Path, required=True, help="New JSON results file")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out exists; choose a new results file")
    if args.generate:
        synthesize(args.audio_dir)
    started = perf_counter()
    asr = LocalSpeechRecognizer(args.model_dir)
    load_s = perf_counter() - started
    rows = []
    for case, phrase, critical in PROMPTS:
        reference = words(phrase)
        for voice, name in VOICES:
            filename = f"{case}-{voice}.wav"
            audio = (args.audio_dir / filename).read_bytes()
            row = {"case": case, "voice": name, "filename": filename,
                   "reference": phrase, "critical_terms": critical}
            output = []
            try:
                receipt = asr.transcribe_wav(audio)
                row["receipt"] = asdict(receipt)
                output = words(receipt.transcript)
            except SpeechInputError as error:
                row["error"] = {"code": error.code, "message": str(error)}
            row.update(exact=reference == output, word_edits=edit_distance(reference, output),
                       reference_words=len(reference), critical_terms_preserved=retains_terms(output, critical))
            rows.append(row)
    results = {
        "source": "Windows System.Speech scripted text-to-speech, two synthetic voices, no added noise",
        "model_load_s": load_s, "utterances": len(rows), "exact": sum(row["exact"] for row in rows),
        "critical_terms_preserved": sum(row["critical_terms_preserved"] for row in rows),
        "recognition_errors": sum("error" in row for row in rows),
        "word_edits": sum(row["word_edits"] for row in rows),
        "reference_words": sum(row["reference_words"] for row in rows), "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")
    print(json.dumps({k: v for k, v in results.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
