# Passenger speech and browser demonstration

The local passenger page opens a paused source image, accepts a typed or spoken instruction and shows the interpreted destination or a clarification question. An explicit button confirms the current area. A passenger can also draw a correction and confirm it separately. The recording is an observation of the scene. The highlighted area represents the interpreted request and cannot establish clearance or legal stopping suitability.

## Launch

On the configured project laptop, launch the packaged model, prepared scenes and speech recognizer from the project checkout. The first command checks the exported model files. The second starts the local page and prints its address.

```powershell
.\.venv-grounding-local\Scripts\python.exe -m tools.passenger_release verify
.\.venv-grounding-local\Scripts\python.exe -m tools.passenger_release start
```

The **Search footage** link opens the existing recording-review task when the original recording and saved prediction folders are available. **Find evidence locally** retrieves the saved tracks and associated frames up to the selected moment. The separately selectable Gemini answer mode processes evidence through the hosted answerer.

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
4. If the assistant asks a question, send the clarification on the same frame. A yellow dot marks the proposed road location. A dashed outline shows its approximate extent, and the original pixels remain visible inside the outline.
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

## Packaged release trial on 22 November

The packaged application completed the four actions declared before the run. Chromium opened two real EMT recordings, the local Vosk recognizer processed every turn, and the local fine-tuned Qwen3-VL-8B and learned spatial model produced the proposals. Two Windows voices supplied the speech and the test script performed the editing, confirmation and cancellation. [Release trial, model digests, speech receipts and per-turn results](../experiments/dropoff/passenger-release-browser-results-20261122.json)

| What the passenger did | What appeared in the actual browser |
| --- | --- |
| Said “Drop me here” through Chromium’s file-fed microphone. | The browser captured 4.304 seconds of WAV. Vosk recognized the words and the system asked for a visible landmark. |
| On the same frame, asked for the road just behind the white SUV on the left. | After editing “white as you v” in the transcript, the model marked `(577.5, 1202.5)`. An explicit click confirmed that model proposal. |
| Asked for the space between the white SUV on the left and the large open-sided truck on the right, just behind them. | The reviewed transcript identified both vehicles. The initial model response named the truck. Two additional image queries independently located the SUV and the truck. The learned spatial model marked `(1037.5, 1282.5)` on the visible pavement between them. The script confirmed it without drawing. |
| On another recording, asked for the yellow bus and then cancelled. | A proposal appeared at `(627.5, 1117.5)`. Cancelling cleared its highlight, and trying the old confirmation token returned HTTP 409. |

Time from starting each turn to the first response was 50.5 seconds for the cold microphone turn, 31.3 for the follow-up, 33.4 for the two-landmark request and 10.4 for the bus. The center point for the two-landmark request was on road pixels. Part of its coarse rectangle reached the dashcam hood. Human acceptance, cabin noise and the appropriateness of a real-world stop require separate evaluation.

The updated display identifies the proposed road location with a solid dot and the approximate area with an unfilled dashed outline. Chromium replayed four exact saved states at three screen widths. The original image hash, coordinates, previous model outputs and confirmation history were preserved. The measured center pixel was yellow for model proposals, cyan for a passenger-marked area, and the clarification displayed the original photograph alone. [Visual replay measurements and screenshot hashes](../experiments/dropoff/passenger-visual-replay-results-20261122.json)

The same packaged entry point also opened the original footage-review page. Two real local searches returned three intervals and nine evidence thumbnails each. All 18 thumbnail responses matched their saved hashes and came from frames at or before the selected moment. Both pages rendered at 320, 390, 768 and 1280 pixels without horizontal overflow. The private image-free evidence ZIP is `runs/passenger-release-browser-evidence-20261122.zip`, SHA-256 `b024af41bfa31f2f2a32edc08b64bae0c4c146c48239aa65f2c3ae8ff2e15af3`.

## Earlier browser results on 21 November

These observations used Chromium, offline Vosk and the fine-tuned Qwen3-VL-8B model on the laptop. The confirmed locations below came from two different original recordings, both 2560 × 1440. A script operated the passenger buttons. [Per-turn outcomes, input hashes and raw receipt hashes](../experiments/dropoff/passenger-browser-results-20261121.json)

| Spoken request | First model response | Scripted passenger action |
| --- | --- | --- |
| Road behind the white SUV on the left, recording `054907`, frame 1 | Vosk heard “white as you v.” After the transcript edit to “white SUV,” the model proposed pixel `(577.5, 1222.5)`. | Confirmed the displayed model proposal. |
| Beside the yellow bus on the left, recording `054604`, frame 150 | With the unedited transcript, the model proposed `(627.5, 1117.5)`. | Confirmed the displayed model proposal. |
| Behind the white truck, recording `054907`, frame 1 | The model proposed `(1807.5, 1282.5)`. | Drew a different rectangle. The obsolete confirmation token returned HTTP 409, then the new passenger-marked rectangle was confirmed. |
| Behind “that white vehicle,” followed by “the big truck with the metal sides on the right” | The model proposed a specific area for the vague first turn. On the follow-up it requested help placing the area. | Left the first proposal unconfirmed and cancelled the follow-up. |
| At the foot of the white panels, recording `054604`, frame 83 | In both browser runs the model named an unavailable detector identifier. The photograph had zero detector candidates. | Saw an explicit error and cancelled. |

All three low-memory errors from the first model run cleared on the repeat with eight resident layers, software browser rendering and the updated road-mask cache. The repeat’s four warm voice turns reached their first response in 13.3–14.0 seconds. The initial run’s screenshot and confirmed-session evidence remains available, including the original failures. Two earlier driver attempts stopped before model inference. The separate CPU tests verify that changing frames now preserves a finished request's confirmation history.

The private evidence archive is `runs/passenger-browser-evidence-20261121.zip`, SHA-256 `f426a7492ae1e15df5b981bb1a7a690d53cfe4071a61f84c992687c5c6241308`. Its manifest covers the original synthetic WAVs, frozen protocols, recognition receipts and session journals. Screenshots remain on the local machine. These selected development trials demonstrate the software interaction. Held-out localization and genuine passenger voice require their own evidence.
