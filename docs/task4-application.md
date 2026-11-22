# Task 4 local application

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

Upload an audio file, or click Record voice and explicitly permit the browser's
microphone request. Stop the recording when finished; a recording is capped at
60 seconds. Microphone permission/device failures are shown explicitly. Changing
the scene while recording prevents that audio from being assigned to a new scene.

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
