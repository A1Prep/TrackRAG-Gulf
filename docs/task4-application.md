# Task 4 local application

For the presentation, use the shared passenger and footage launcher with speech and the optional cloud parking model. Below is the development interface and its independently usable adapters. [Final start command](../README.md#run-the-demonstration)

The local application brings the existing cited footage-QA route and the new
drop-off dialogue into one browser workspace. It opens exact recorded frames,
shows destination and landmark overlays, supports typed/audio input and transcript
correction, and binds explicit confirmation to the proposal shown to the passenger.

The [application](../trackrag/dropoff/app.py) uses the existing EMT reader,
prediction adapters, lexical retriever, evidence exporter, answerer and
[dialogue controller](../trackrag/dropoff/interaction.py). Its browser assets are
under [web](../trackrag/dropoff/web/). No frontend package installation or build is
required. [Part 2](final-completion-part2.md) remains the acceptance plan.

## Start in the project environment

Use Python 3.11 or newer in the isolated project environment. Install the
[hosted QA requirements](../requirements-hosted.txt) when using the existing Gemini
answerer; these also include the integration/reviewer dependencies:

```powershell
& '.venv/Scripts/python.exe' -m pip install -r requirements-hosted.txt
& '.venv/Scripts/python.exe' -m trackrag.dropoff.app --help
```

Task 5's actual model inference dependencies and checkpoint must be available in
the environment running the app. Reviewer dependencies alone do not provide a
trained model. Reuse the [model handoff](dropoff-model-handoff.md) for its pinned
model, artifact checks, configuration and GPU requirements.

For the normal saved-prediction and trained-model route, replace the three external
paths below with the actual shared locations:

```powershell
& '.venv/Scripts/python.exe' -m trackrag.dropoff.app --data-root 'D:/EMT_IMAGES' --vision-config configs/vision/finetuned.json --predictions 'D:/TASK1_SAVED_PREDICTIONS' --grounding-config 'D:/TASK5_TRAINED_RUN/inference_config.json'
```

Open the printed local URL, normally `http://127.0.0.1:8765`. The server binds only
to loopback. Run the application in the environment that can load Task 5's model;
do not assume an owner-specific GPU/cache/checkpoint path exists on another machine.

Each launch creates a unique directory under the ignored runs folder. Supply
`--out <new-directory>` to choose it explicitly; an existing directory is rejected.
The actual receipt directory is printed at startup.
Use another `--port` if needed. Ctrl+C stops the server, waits for model workers
and saves the final session audit. Pending requests are also saved before shutdown.
The application starts a new session on each process start; it does not restore
confirmation authority from historical files.

Other execution options:

| Option | Meaning |
| --- | --- |
| --vision-mode saved | Default. Requires the Task 1 tracks/provenance folder and matching vision configuration. Strictly validated saved predictions are labelled as saved. |
| --vision-mode fresh | Runs Task 1's real predictor on the available prefix through the selected original frame. Use Task 1's configured weight location/environment. |
| --qa-answerer gemini | Default. Calls the existing Task 3 answerer. Its existing provider configuration/credentials remain outside Git. Provider errors remain visible. |
| --qa-answerer none | Explicit evidence-only condition. The interface labels answer generation as not requested. |
| --grounding-config | The explicit pretrained or trained inference configuration. adapter_dir selects the trained artifact; unavailable artifacts remain operational errors. |
| --grounding-backend ollama | Uses the separately documented [local pretrained comparison](dropoff-local-pretrained.md), with its explicit configuration. It is not the trained Task 5 condition. |
| --speech-adapter | Task 3's module exporting transcribe_audio(path), as described below. |
| --asr-model | Select the existing verified local Vosk recognizer using its model folder. Accepts 16-bit PCM WAV. |
| --whisper-model | Select the existing pinned CPU Whisper Small English recognizer using its model folder. Accepts 16-bit PCM WAV. |
| --frame-rate / --split | Explicit EMT reader overrides. Frame IDs always remain original IDs. |

Without a grounding configuration the QA application can still run; a grounding
attempt returns configuration_unavailable and produces no proposal. Without a
speech adapter, audio attempts retain transcription errors and typed input remains
available. The UI does not replace either missing component with an answer provider
or static coordinates.

## Development wiring demonstration

This checkout has two local EMT clips under the baseline data root. To inspect the
UI before teammates' real artifacts arrive:

```powershell
& '.venv/Scripts/python.exe' -m trackrag.dropoff.app --data-root data/emt-task4-baseline --adapters tools.dropoff_app_demo
```

The [development module](../tools/dropoff_app_demo.py) generates track observations,
grounding coordinates, a canned cited answer and a fixed transcript. Its banner,
model provenance and saved session explicitly identify this condition. Original
EMT images can be real while these outputs remain generated fixtures. This command
does not satisfy detector, learned-grounding, live-answer or ASR acceptance.

The default route never imports that module. A custom development module must be
selected explicitly with --adapters and supply predict_tracks_with_receipt and
create_grounding_adapter. Optional answer_question and transcribe_audio hooks
allow isolated wiring checks. Their call shapes are demonstrated in the module.
The shipped fixture cannot be selected as a normal speech adapter to bypass the
development-mode label.

## Walk through both routes

1. Choose a recording, enter its exact original frame ID, and click Open scene.
   Previous/next navigate actual available frame IDs, including sparse recordings.
   Editing the recording/frame selection immediately invalidates the old proposal.
   A missing original frame produces an explicit error rather than a nearby image.
2. In Footage question, type a question and click Find evidence and answer. Inspect
   the outcome label, citation links and original-frame evidence thumbnails. Empty
   evidence, provider error and evidence-only mode remain different outcomes.
3. In Drop-off instruction, type an instruction such as "Drop me off beside the
   white car." The configured grounding adapter receives the exact image, predicted
   observations and text through the shared request format.
4. If the model needs clarification, answer in the clarification field. The next
   request contains the original instruction, actual model question and reply.
   Relevant same-scene dialogue is available under Dialogue context.
5. Review the proposed coarse region and point. The landmark outline is separate
   from the destination region. Use Confirm this destination only for the proposal
   currently displayed. A point marker is not itself the learned coarse region.
6. Edit Passenger text to revise the latest passenger utterance, including a
   clarification reply. Update interpretation creates a new request. New instruction
   starts a fresh dialogue instead. Edits clear previous proposal authority.
7. Cancel clears the current proposal, pending clarification and confirmation.
   A late response cannot revive it or overwrite another selected scene.

The region interprets the requested destination. It does not establish physical
clearance, stopping legality or any vehicle-control action. Bare "here" has no
pointing context in this interface and cannot produce an accepted proposal.

## Audio and transcript handoff

Select exactly one of --speech-adapter, --asr-model and --whisper-model. The native
options reuse the existing recognizers and preserve their audio hash, original
transcript, model identity and timing receipt through the application's normal
audio interface. An explicitly named module must provide a callable
transcribe_audio; an incompatible module fails at startup. In particular, the
older trackrag.speech.transcribe_file returns only text and is not that interface.

Install [Vosk dependencies](../requirements-speech.txt) or the separately pinned
[Whisper dependencies](../requirements-speech-whisper.txt) in the actual application
environment. Model locations remain explicit arguments. Missing or mismatched
models do not switch to a hosted recognizer. Native speech errors keep their
original codes, and a supplied receipt's audio hash must match the saved bytes.

Upload an audio file, or click Record voice and explicitly permit the browser's
microphone request. Stop the recording when finished; a recording is capped at
60 seconds. Microphone permission/device failures are shown explicitly. Changing
the scene while recording prevents that audio from being assigned to a new scene.
The browser now captures mono 16-bit PCM WAV at its actual supported sample rate
(16-48 kHz), using the existing passenger application's AudioWorklet approach.
Scene/input changes discard in-flight capture and late permission grants. Native
recognizers require this PCM format; a WebM/MP4 upload is not converted or renamed
to appear compatible. Their existing limits are 60 seconds and 12 MiB.

Accepted files are bounded to 16 MiB and saved before transcription. The adapter
decides which audio containers/codecs it actually supports. Task 3 supplies this
callable in the module named by --speech-adapter:

```python
def transcribe_audio(audio_path):
    # Run the fixed recognizer and return its actual text and receipt metadata.
    return {"text": recognized_text, "provenance": recognizer_receipt}
```

The example variable names are supplied by Task 3's implementation. text must be
nonempty bounded text; provenance is a finite JSON object and can include model
identity and timing. Empty/failed transcription is an error, not a grounding or
confirmation event. A failed audio replacement clears the previous proposal too.

The recognized text appears in the editable Passenger text field. Original
recognized transcript retains the untouched ASR wording. Correct it before
Interpret instruction; the current text becomes the model input while the source
transcript and correction history remain in the audit. Use typed text switches
explicitly to a new typed instruction. A late ASR result cannot replace a newer
typed instruction. Recognition alone never invokes grounding or confirms a point.

## Saved records and provenance

Download session returns the complete JSON audit. The run directory also contains:

| Output | Contents |
| --- | --- |
| session.json | Continuously saved application audit: execution conditions, scene preparation, grounding requests/outcomes, available raw adapter receipts, QA rows, audio outcomes and input errors. |
| images | Exact selected-image snapshots named by content hash. They are used by the overlay and grounding route. |
| qa | Existing reviewer run folders, including public questions, predicted observations, evidence images and any hosted-provider receipts. |
| audio | Accepted source-audio bytes and hashes, retained even when transcription fails. |
| vision-config.json / grounding-config.json | Consumed configuration snapshots when supplied. No environment credentials are copied. |

Grounding receipts keep model errors, semantic unresolved results, discarded late
responses and unfinished calls distinct. QA records are added before review starts,
so an unfinished or failed request remains in the audit even if no reviewer folder
can be published. Every requested bank row must still be retained during final
acceptance; a successful subset is not a complete quality result.

The selected image is checked and snapshotted before use. QA rejects a selected
source image that changes after selection. Historical Task 1 prediction bundles
still lack per-image byte hashes: their full artifact integrity and declared model
identity are checked, but the application labels that historical image-binding gap.

The JSON is an audit, not a model-input manifest or a confirmation restore file.
It retains original request image paths; copy the images/QA/audio folders with it
for review. Use the existing [portable scene preparation](dropoff-scene-inputs.md)
workflow when transferring replayable inference requests to another host.

## Checks and remaining acceptance

```powershell
& '.venv/Scripts/python.exe' -m pytest tests/test_dropoff_app.py tests/test_dropoff_interaction.py tests/test_dropoff_interaction_contracts.py tests/test_dropoff_inputs.py -q
& '.venv/Scripts/python.exe' -m pytest tests -q
node --check trackrag/dropoff/web/app.js
```

The [application tests](../tests/test_dropoff_app.py) use generated image-only EMT
recordings, synthetic saved-prediction artifacts and controlled adapters. They
exercise both routes through the same application, exact frames, real existing
retrieval/evidence export, original coordinates, transcript correction, cancelled
and late requests, errors, source changes, strict saved-prediction validation and
HTTP media boundaries. These are software integration checks.

Browser wiring also uses original local frames from video_054604 and video_054907
with the explicit development adapters. Typed dialogue, proposal/confirmation,
scene changes, an uploaded silent audio fixture, transcript correction and QA
evidence are checked in that condition. It is not real ASR or model-quality evidence.

On 22 November 2026, the focused application/controller/contracts/builder command
passed 197 tests. The full suite passed 1,011 tests and 401 subtests, with 5 skipped.
JavaScript syntax, editor diagnostics, whitespace and all three CLI help checks
passed. A fresh-process launch was checked. Desktop and narrow layouts were
inspected; a simulated unavailable microphone produced an explicit error and
cleared the previous proposal/confirmation without recording real audio.

Before declaring all Task 4 work complete, obtain and test:

- Task 1's genuine prediction bundle/configuration, including exact-frame checks
  on both real recordings, and independent integration/scoring review.
- Task 3's actual speech adapter and successful, reviewed live QA acceptance on both
  recordings, with valid insufficient-evidence and provider failures retained.
- Task 5's accessible trained checkpoint/configuration and a real load/inference
  run through this application; a development or pretrained-only run is different.
- The frozen regression cohort, complete receipts and a fresh-start rehearsal by
  another teammate, followed by the final release/report updates.

## Check real module handoffs before a live trial

The [read-only integration preflight](../tools/task4_check_integration.py) checks
the normal saved-prediction path, exact images on two recordings, trained adapter
manifest, base/processor accessibility, dependency metadata and the selected
native speech model. It reports every requested check, including independent
failures; it does not load weights, recognize speech, download files or score
grounding quality. Supply the reviewed Task 1 prediction manifest when checking
the shared bundle, so its complete prediction/provenance hashes are compared too.

For example, after placing the actual artifacts at their documented locations:

```powershell
& '.venv-grounding/Scripts/python.exe' -m tools.task4_check_integration --data-root data/emt-task4-baseline --vision-config configs/vision/finetuned.json --predictions data/task1-shared-20261119/predictions-finetuned --grounding-config models/passenger-reviewed-clarification/inference_config.json --scene video_054604:150 --scene video_054907:150 --prediction-manifest experiments/vision/shared_prediction_runs.json --asr-model models/passenger-support/vosk-model-small-en-us-0.15
```

These paths are examples of the documented handoff layout, not artifacts included
in this checkout. Replace them with the actual shared locations. An exit code of
zero means the checked files and dependency metadata permit a runtime attempt;
actual GPU/ABI compatibility, model load, latency and quality remain live checks.

The [current native model handoff](passenger-evaluator-handoff.md) documents the
trained four-field adapter installed under models/passenger-reviewed-clarification.
The composite passenger application uses additional landmark/region/road modules
and is a different condition. Do not use its adapter package or historical
application results as proof of native Task 4 inference. The earlier public-only
V2 adapter is also a separately labelled training stage.

Follow the [separate Python 3.11 GPU environment recipe](dropoff-local-training.md)
for the pinned streamed inference stack. The Python 3.14 reviewer environment
does not establish compatibility with that stack. Keep the reviewer environment
intact and run the application from the configured inference environment. The
pinned quantized base alone requires approximately 7.24 GB of weight downloads;
the small trained adapter does not replace those base and processor files.

Once preflight permits an attempt, use the same paths in the normal application:

```powershell
& '.venv-grounding/Scripts/python.exe' -m trackrag.dropoff.app --data-root data/emt-task4-baseline --vision-config configs/vision/finetuned.json --predictions data/task1-shared-20261119/predictions-finetuned --grounding-config models/passenger-reviewed-clarification/inference_config.json --asr-model models/passenger-support/vosk-model-small-en-us-0.15
```

Retain the two-recording typed and actual speech trials, transcript corrections,
clarifications, confirmations, errors and complete QA cohort. The application
drains pending jobs and closes its owned grounding worker during shutdown,
including a server bind/start failure. This is a resource-lifecycle guarantee,
not evidence that real inference has been performed on this machine.

Additional compatibility checks use controlled recognizers and generated saved
artifacts rather than real ASR/grounding quality:

```powershell
& '.venv/Scripts/python.exe' -m pytest tests/test_dropoff_speech.py tests/test_dropoff_app_lifecycle.py tests/test_task4_check_integration.py tests/test_dropoff_app.py -q
node --test tests/test_dropoff_web_audio.cjs
```

Compatibility verification on 22 November 2026 passed 62 focused Python tests,
2,135 full-suite tests and 401 subtests, with 41 tests skipped. All nine
Node audio tests, JavaScript syntax checks, CLI help checks and editor diagnostics
passed. Independent code review found no blocking issue in the PCM capture,
speech bridge or artifact preflight. These use generated scenes, controlled
recognizers and supplied-artifact fixtures. No genuine speech recognition or
trained-checkpoint inference was performed in this verification phase; those
trials still require confirmed artifact locations and the inference environment.

The handoff machine's Windows hardware inventory reported Intel Iris Xe graphics
and installed Python 3.12/3.14 interpreters. It did not provide the NVIDIA
CUDA/BF16 hardware required by the documented streamed runtime. The receiving
teammate should use the prepared Task 5 GPU machine, run the preflight with the
actual artifact paths, then perform the two-recording application trial above.
Keep that live acceptance evidence separate from the software test results.
