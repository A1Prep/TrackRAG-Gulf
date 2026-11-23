# Thirty passenger voice clips

Task 3 asks for about 30 consented recordings from at least three people. Assign the anonymous codes **S01, S02 and S03** to three different volunteers. Each person reads the same ten English instructions. Nine are exact daytime scene instructions. The tenth rephrases the pickup/SUV case to include the spoken count “two”. This is a read-speech study. It measures the recognizer’s hearing of an instruction independently of the model’s choice of a road location. [Team task](https://github.com/A1Prep/TrackRAG-Gulf/issues/3)

The Task 3 team provided a separate manifest for spoken footage questions. It records 31 takes under three speaker codes. Each code covers all ten questions, with one extra take. All 31 stored transcripts match the printed question. At collection time, the tool kept the final text after editing and could copy the prompt if recognition failed and the manual entry was blank. The reviewed handoffs contain no matching WAV, first machine transcript or individual consent record. This supports reporting coverage of the footage-QA scripts. Word error rate remains unmeasured for those recordings. None of the ten questions matches an instruction in the passenger study. [Team manifest](https://github.com/A1Prep/TrackRAG-Gulf/blob/05426cdfc296d8ebeff6d606438f86dd937d3b9b/annotations/recording_manifest.jsonl) · [Original collection code](https://github.com/A1Prep/TrackRAG-Gulf/blob/05426cdfc296d8ebeff6d606438f86dd937d3b9b/tools/collect_recordings.py#L43-L92) · [Aggregate audit](../experiments/dropoff/passenger-task3-reported-speech-manifest-20261122.json)

For new footage-QA recordings the repaired Task 3 tool preserves the microphone’s original WAV bytes, sample rate, first recognizer output and final edited text. It labels manual entries and copies of the prompt. A combined software test exercised the real collection command with generated 48 kHz audio. [Recording repair merged into Task 3](https://github.com/A1Prep/TrackRAG-Gulf/pull/40)

## Record on your own laptop

1. Download the blank **TrackRAG-passenger-voice-study.html** from the private project release and double-click it in Chrome or Edge. Choose the code given to you (S01, S02 or S03). If you agree to participate, give consent and record each of the ten instructions, or choose an original WAV of your own voice. Play each take through to the end. Confirm the displayed words or type exactly what you said, then save the take. [Download the blank voice form](https://github.com/A1Prep/TrackRAG-Gulf/releases/download/passenger-feature-20261122/TrackRAG-passenger-voice-study.html)
2. Download your **private return** and give the JSON file privately to the study coordinator. You can stop before ten, keep the partial return, and load it in the same blank form under the same code to continue. If you rerecord a take, send the return containing the one you chose. Only the blank form belongs on GitHub.

The coordinator can rebuild the blank and import one or more private returns. Prepare a new study once, or point the import at the existing prepared directory. Identical takes in overlapping returns count once. A conflict leaves the existing take intact. If a volunteer replaces an earlier take already imported, retain their choice and rebuild the study in a fresh directory from the selected returns. The browser records microphone or original-file provenance and shows no recognition output. Score both models later against the words the speaker verified. [Offline-form checks](../experiments/dropoff/passenger-portable-speech-handoff-20261122.json) · [Whisper replay](passenger-whisper.md)

```powershell
python -m tools.passenger_voice_form build --out TrackRAG-passenger-voice-study.html
python -m tools.passenger_speech_study prepare --out runs/passenger-team-voice
python -m tools.passenger_voice_form import --study runs/passenger-team-voice --return "D:\private\speaker-one.json"
```

For a withdrawal, remove the volunteer’s folders under `recordings/<code>` and `private-portable-returns/<code>`, any received copies and backups holding their voice, and regenerate the scores. The original WAV and privately retained return contain the participant’s voice.

## On the project laptop

The blank study lives in `runs/passenger-human-speech-20261122`. A volunteer enters consent before each collection session and starts each microphone take explicitly. The person listens to the original take and confirms or supplies the *actually spoken* wording before seeing the recognizer. Then they can accept or edit the recognized text. No account, GPU or hosted speech service is involved.

```powershell
$study = "runs/passenger-human-speech-20261122"
$asr = Join-Path $env:LOCALAPPDATA "TrackRAG-method-cache/asr-assets/vosk-model-small-en-us-0.15"
$env:PYTHONPATH = Join-Path $env:LOCALAPPDATA "TrackRAG-method-cache/speech-capture-py311"
.venv\Scripts\python -m tools.passenger_speech_study collect --study $study --speaker S01 --noise quiet --model-dir $asr
```

Use **S02** and **S03** for the other volunteers. Choose `real_background` or `added_noise` when appropriate and describe it briefly. The default microphone window is eight seconds. Use `--seconds 12` for a longer window, or `--prompt b13` for a single take. Repeating the command continues with unfinished cases. A volunteer can leave before saving a take and can ask the team to remove their coded folder and regenerate the scores.

To use a participant’s own WAVs, run the same collection with `--wav-dir <folder>`. Put the ten originals in that folder as `a01.wav`, `a06.wav`, `a10.wav`, `b01.wav`, `b02.wav`, `b06.wav`, `b11.wav`, `b13.wav`, `b16.wav` and `b06-count.wav`. The passenger browser stores the original under its session’s `audio/<hash>.wav`. Copy the recording into the matching prompt filename. Use uncompressed 16-bit PCM, 16–48 kHz, mono or stereo. Keep the participant present for consent and listen-back during import. A laptop without the recognition model can omit `--model-dir` during collection. Its original WAV and verified transcript can be scored later; an on-screen ASR correction is available when the model is supplied.

## Combine and score

For volunteers on different machines, start from the same branch and blank protocol. Privately transfer each completed `recordings/Sxx` folder into the coordinating study. Keep audio, private transcripts, background descriptions and individual reports out of Git. The coordinator can share aggregate results.

```powershell
.venv\Scripts\python -m tools.passenger_speech_study score --study $study --model-dir $asr --out runs/passenger-human-speech-score-01.json
```

The scorer verifies the WAV, script and scene and refuses a duplicate recording. It measures the original transcript the volunteer saw and a fresh Vosk replay separately. Each has its own WER, critical-word metrics and speaker breakdown. The report also keeps the participant’s corrected wording, decode timing and recording condition. A failed recognition scores as an empty delivered transcript. Missing recordings remain visible.

WER is total word insertions, deletions and substitutions divided by reference words. It lowercases English, retains apostrophes, and drops punctuation. The critical-word check compares ordered sequences of the declared colours, directions, counts and negation. It equates digits 1–10 with the corresponding count words. Other spellings and contractions are distinct. A newly inserted “not” counts as an error even when the reference contained no negation. Landmark names and all other words remain in WER and the transcripts. [Study verification and current return count](../experiments/dropoff/passenger-speech-study-readiness-20261122.json)

## From a fresh team checkout

Clone the private project’s `main` branch and run these commands from its root using Python 3.11 in a virtual environment. On Linux, the system’s PortAudio development package may be needed to build PyAudio. WAV-only teams can install `requirements-speech.txt` and import their audio.

```powershell
python -m pip install -r requirements-speech-capture.txt
python -m tools.passenger_release fetch-support
python -m tools.passenger_speech_study prepare --out runs/passenger-human-speech
python -m tools.passenger_speech_study collect --study runs/passenger-human-speech --speaker S01 --noise quiet --model-dir models/passenger-support/vosk-model-small-en-us-0.15
```

For the **separate visual annotation task**, each teammate can rebuild the two blank offline forms from their own EMT images. Install `requirements-integration.txt`. Point `$emt` to the directory containing `images/train/video_054604` and `images/train/video_054907`. The two commands were checked in an isolated source checkout without private run folders or answer files. Identical originals produce byte-identical forms. [Human review instructions](passenger-human-intake.md)

```powershell
python -m tools.dropoff_passenger_references_pack --image-root "$emt" --out blank-train.html
python -m tools.dropoff_passenger_references_pack --partition development --image-root "$emt" --out blank-development.html
```

## External human speech check

The release also supports English Whisper Small on CPU. It improved the paired public transport study and has its own replay command for the consented team recordings. [Whisper setup, results and local scoring](passenger-whisper.md)

The unmodified passenger recognizer transcribed 16 public transport requests from 16 people. The cohort included six taxi requests, six travel queries and four destination ticket requests. It made **84 word edits in 160 reference words**, a **52.5% WER**. It kept 7 of 20 annotated place names in order and every place name in 5 of 16 clips. All 16 produced text and none matched the complete reference exactly. The source supplied the reference prompts. A speaker may have said something different. The subset was selected by archive order and eligibility before inference. Ten distant-microphone recordings and six headset recordings were included. The project-specific volunteer total remains 0 of 30 at the recorded intake. [Fixed protocol](../experiments/dropoff/passenger-public-speech-protocol-20261122.json) · [Per-clip measurements](../experiments/dropoff/passenger-public-human-speech-20261122.json)

The original Talk2Car authors collected written passenger commands. Room-Across-Room collected spoken spatial instructions but its released dataset omitted the voices. For this academic check we used the separately published **SLURP** audio by Emanuele Bastianelli, Andrea Vanzo, Pawel Swietojanski and Verena Rieser. The authors release the original audio under CC BY-NC 4.0 and the text under CC BY 4.0. The report publishes numerical results and source identifiers. The FLACs, converted WAVs and transcript comparisons remain in the local cache. [Talk2Car paper](https://aclanthology.org/D19-1215/) · [Room-Across-Room paper](https://aclanthology.org/2020.emnlp-main.356/) · [SLURP paper](https://aclanthology.org/2020.emnlp-main.588/) · [Authors’ audio release](https://zenodo.org/records/4274930) · [SLURP license and annotation format](https://github.com/pswietojanski/slurp#license)

To reproduce the check, use a private cache outside the repository. `fetch` retrieves the published real-speech metadata and the first 160 MiB of the real archive. `select` seals the unscored speaker and utterance list and converts each selected original with a local FLAC decoder. Copy the printed SHA-256 into the scoring command.

```powershell
$cache = Join-Path $env:LOCALAPPDATA "TrackRAG-method-cache/slurp-reproduction"
$flacExe = python -c "from speech_recognition import get_flac_converter; print(get_flac_converter())"
python -m tools.passenger_external_speech fetch --cache $cache
$selected = python -m tools.passenger_external_speech select --cache $cache --flac "$flacExe" | ConvertFrom-Json
python -m tools.passenger_external_speech score --cache $cache --model-dir "$asr" --selection-sha $selected.selection_sha256 --out runs/slurp-replayed.json
```
