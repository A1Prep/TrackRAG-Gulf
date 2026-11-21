# Passenger speech and browser demonstration

The local passenger page opens a paused source image, accepts a typed or spoken instruction and shows the interpreted destination or a clarification question. An explicit button confirms the current area. A passenger can also draw a correction and confirm it separately. The recording is an observation of the scene. The highlighted area represents the interpreted request and cannot establish clearance or legal stopping suitability.

## Launch

Use the prepared inference environment with the tested Qwen base and adapter, the learned geometry artifact, the SegFormer cache and one or more prepared request banks. Paths can point outside the checkout. The service listens on the laptop's loopback address.

```powershell
python -m pip install -r requirements-speech.txt
python -m trackrag.passenger_app --requests path/to/first/requests.jsonl --requests path/to/second/requests.jsonl --config path/to/trained-config.json --geometry path/to/conditioned.json --vision-cache path/to/segformer-b0 --asr-model path/to/vosk-model-small-en-us-0.15 --out runs/passenger-sessions
```

Open the address printed by the command, normally `http://127.0.0.1:8765/`. `--verify-landmarks` explicitly selects the additional landmark-verification condition. Use the setting and model hashes associated with the reported experiment. Omitting `--asr-model` starts a text-only session. The existing cited-footage reviewer remains accessible through its existing `python -m trackrag review` command.

For voice, download the English small-model archive and unpack it outside Git. The application verifies the installed model contents on startup. The 41,205,931-byte ZIP used here has SHA-256 `30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498`. The publisher lists the model under Apache 2.0. [Official Vosk models](https://alphacephei.com/vosk/models) · [Model archive](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip)

## Use

1. Select the recording and original frame from **Road scene**.
2. Type the instruction, choose **Record voice**, or upload a 16-bit PCM WAV. The microphone records one turn locally. The upload supports mono or stereo, 16–48 kHz and up to 60 seconds.
3. Read the transcript and edit any misunderstood words before sending. Colour, side, count, object names and negation can change the intended location.
4. If the assistant asks a question, send the clarification on the same frame. A yellow outline and dot show the model proposal in original-image coordinates.
5. Choose **Confirm this area** or **Cancel request**. To correct the location, choose **Mark an area myself**, drag across the image and choose **Use my marked area**. The page identifies the passenger-marked area in cyan. Confirmation applies only to the currently displayed proposal.

Each session has its own directory below `--out`. Its JSONL journal records the original frame hash, input, turn history, semantic or operational outcome, source of each proposal and explicit confirmation/cancellation. The audio directory stores original WAV bytes by SHA-256 plus transcription-attempt receipts. A failed or replaced recording blocks confirmation of an older proposal. Short generic prompts such as “here” can legitimately need clarification. A model error is visible separately from an unresolved destination.

## Speech evidence

The recognition model runs locally on CPU with a new decoder for each utterance. It receives audio samples alone. The displayed transcript and passenger correction reach the same grounding interface as a typed instruction, with the original recognition retained. The original Task 3 hosted/file and console-microphone interfaces and tests are Aditi Pateriya's contribution, brought forward from commit `474713a70ca9405e2df9245ba16f6c0e109b20b1`. The passenger browser selects the offline recognizer explicitly. [Task 3 source](https://github.com/A1Prep/TrackRAG-Gulf/blob/474713a70ca9405e2df9245ba16f6c0e109b20b1/trackrag/speech.py) · [Vosk decoder example](https://github.com/alphacep/vosk-api/blob/master/python/example/test_simple.py)

Two machine-generated Windows voices each spoke eight scripted instructions. Vosk transcribed all 16 clips, reproduced the normalized text on 15, and retained the predeclared critical words in order on 15. One voice said “parked” and the recognizer returned “park.” There was one word edit in 124 reference words. The mean decoding time was 1.42 seconds per clip on the test laptop. In six additional clips using three scene-specific instructions, “SUV” became “as you v” or “s u v,” and “metal” sometimes became “medal.” A one-second silent WAV returned `no_speech`. These data measure clean synthesized English. Passenger accents, physical microphones and moving-cabin noise require additional recordings. [Per-clip speech results](../experiments/dropoff/passenger-speech-results-20261121.json)

The browser capture path was also exercised in headless Chromium using an audio-file-fed virtual microphone. The actual Web Audio worklet encoded 5.104 seconds of WAV audio and real Vosk transcribed the truck instruction plus the beginning of a second playback. The test edited that transcript and confirmed the resulting software-test proposal. A separate all-silent virtual capture correctly produced an empty-recognition error.

To reproduce the eight-instruction speech trial on Windows with the Microsoft David Desktop and Zira Desktop voices installed, use a new output directory. To replay existing WAVs on another operating system, omit `--generate`.

```powershell
python -m tools.dropoff_voice_trial --generate --model-dir path/to/vosk-model-small-en-us-0.15 --audio-dir runs/speech-replay/clips --out runs/speech-replay/results.json
python -m pytest tests/test_local_speech.py tests/test_speech.py tests/test_passenger_app.py -q
```

For browser trials, the independent driver accepts a frozen JSON plan containing the URL, ordered scene selections, source WAV paths/hashes, permitted transcript edits and passenger actions. Start the application first. Use Node with the `playwright` package and Chromium installed in its testing environment. The optional `PASSENGER_CHROMIUM` variable selects an existing Chromium executable. The result directory must be new.

```powershell
node tools/dropoff_browser_trial.cjs path/to/protocol.json runs/browser-replay
```

The driver saves the original plan, each initial outcome, overlay dimensions and painted-pixel count, the passenger-marked result and any stale-token check, scripted confirmation or cancellation. Its screenshots contain scene pixels and stay in the local run directory. Measure initial grounding against separate references before counting any passenger assistance.
