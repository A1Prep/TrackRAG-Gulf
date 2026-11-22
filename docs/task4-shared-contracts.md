# Task 4: shared drop-off records and dialogue controller

Shared records prepared for teammate review on 21 November 2026; the dialogue
controller was added on 22 November. These implement the shared-record and dialogue
milestones in [Part 2](final-completion-part2.md). Agreement by the four other owners
is still pending; the complete application and real-artifact acceptance remain unfinished.

## What was already available

- The [footage reviewer](../trackrag/review.py) resolves exact original frames,
  exports marked evidence and calls the configured answerer. Its existing command
  is `python -m trackrag review --help`; the
  [entry point](../trackrag/__main__.py) wires the task modules together. The
  [HTML report tool](../tools/review_report.py) displays completed runs with
  evidence images, citations and outcome labels.
- The [lexical retriever](../trackrag/retrieval.py) builds track windows and ranks
  bounded evidence. [Candidate preparation](../trackrag/retrieval_candidates.py)
  and the [HTML review worksheet](../trackrag/retrieval_review_html.py) are reusable
  preparation tools. They do not establish a trained grounding model.
- The [original shared records](../trackrag/contracts.py) define Frame,
  TrackObservation, Question, RankedInterval and the QA evidence/answer formats.
- The [existing drop-off contracts](../trackrag/dropoff/contracts.py),
  [scene request builder](../trackrag/dropoff/inputs.py),
  [scene preparation command](../tools/dropoff_prepare_scene.py) and
  [Task 5 model adapter](../trackrag/dropoff/model.py) already supply much of the
  new input/model boundary. Reuse the
  [scene handoff](dropoff-scene-inputs.md) and
  [model handoff](dropoff-model-handoff.md).

The shared-record milestone extends those records with app-session identity,
app-owned proposal, confirmation and transcript-correction records, and validators
for returned outcomes and interaction snapshots. The next milestone adds the
[dialogue controller](../trackrag/dropoff/interaction.py), documented below.
These records now support the [local application](task4-application.md), which
provides the unified UI and startup command. The actual speech recognizer and
trained model remain Task 3/5 handoffs; their final acceptance runs are separate.
The original Question and RankedInterval schemas and the existing model response
fields remain unchanged. Existing batch requests without session_id remain valid.

## The records each owner exchanges

All records are ordinary dictionaries and can be saved as JSON. TypedDict describes
their shape; call the runtime validators at a module boundary.

| Record | Contents and responsibility |
| --- | --- |
| SceneContext | Existing Frame, explicit predicted source, TrackObservation list and source/model provenance. Task 1 supplies genuine images and predictions. |
| InstructionRequest | Unique request_id, optional session_id, scene, current text, typed/voice input mode, relevant same-scene dialogue and optional original_transcript. Task 4 supplies session identity and builds the request using Task 1/3 inputs. |
| GroundingResult | proposed, needs_clarification or insufficient_visual_evidence; destination point/box or nulls, clarification question or null, optional visual landmark anchor boxes. Task 5 supplies the interpretation. |
| GroundingOutcome | Either ok=true with result, or ok=false with code/message error. Both retain request_id and model/configuration provenance. Every requested attempt belongs in the run receipts and evaluation denominator. |
| DestinationProposal | App-generated proposal_id, request_id, recording/frame/source reference, validated proposed result and model provenance. Task 4 creates this only from a proposed semantic result. |
| ProposalConfirmation | Exact proposal_id, request_id, scene reference and proposal_sha256. Task 4 records it only after the passenger explicitly confirms. |
| TranscriptCorrection | Request identity, untouched original ASR transcript and corrected text. Task 3 supplies recognition; Task 4 records the user's correction. |
| InteractionState | session_id, current request, status, pending clarification, current proposal, confirmation and correction history. Task 4 owns transitions and presentation. |

GroundingAdapter describes the existing `interpret(request) -> outcome` callable
boundary. It is distinct from Task 5's lower-level generation backend. No heavy
model dependencies are needed to import these contracts.

## Boundary rules

- Use original image pixels throughout the shared format. For width W and height H,
  points satisfy `0 <= x < W`, `0 <= y < H`; boxes satisfy
  `0 <= left < right <= W`, `0 <= top < bottom <= H`. The point must be contained
  in the coarse destination box. The model adapter converts its internal normalized
  coordinates before returning a result. Validation rejects invalid values without
  clipping or guessing a repair.
- Preserve original frame IDs. Bind all track identities to their recording and
  source. Track IDs from human labels and detector predictions are independent.
  Requests reject mismatched recording/source, future observations and dialogue
  from another selected frame. The scene builder selects current-frame observations.
- Do not add reference destinations, masks, answer keys, relevance labels or
  future observations to requests or their metadata. Training targets and independent
  scoring references remain separate. Provenance is receipt metadata, excluded from
  the model prompt; it is not an extra channel for supervision.
- `validate_request` checks structure. The existing scene builder also opens the
  exact image, checks dimensions and binds its SHA256. Task 1's saved-prediction
  adapter checks prediction provenance. A well-formed dictionary alone cannot prove
  that predictions are genuine or that an image depicts a white car.
- `validate_outcome(outcome, request)` checks the current request identity, strict
  envelope, semantic status/geometry and finite JSON provenance. Failure envelopes
  cannot contain a result. A successful semantic call can still be unresolved.
  A malformed envelope raises ValueError; the later controller must retain the raw
  outcome and an invalid_model_output error receipt with no proposal.
- `validate_proposal(proposal, request)` accepts only a proposed semantic result
  bound to the current request and recording/frame/source.
- `validate_interaction_state(state)` checks consistent pending/proposed/confirmed/
  cancelled/error states. Unresolved, pending, cancelled and error states have no
  current proposal or confirmation. Only confirmed states carry confirmation.
  These checks validate snapshots; they do not implement dialogue transitions or
  establish that a passenger actually clicked Confirm.

Create a fresh request_id for each submitted instruction, clarification reply or
correction. Create a fresh proposal_id for every accepted new proposal and never
reuse it for revised content. Include session_id for interactive app requests;
legacy batch inputs can omit it. A voice interaction snapshot must preserve
original_transcript, even when the user replaces failed ASR text manually.

Confirmation uses `proposal_fingerprint(proposal, request)`. The hash binds the
proposal content and the full request, including text, dialogue, image reference,
image hash when supplied, and session identity. Old confirmation fails validation
after any of these change. This is a consistency check, not authentication. It
does not reopen or monitor an image file; the scene/model adapter must verify the
image bytes. On scene changes or new input, the later controller must clear the old
proposal and confirmation immediately, including while the next call is pending.

## Worked example: "Drop me off beside the white car"

The following image, track and coordinates are illustrative. They are not outputs
from a real recording or a trained model. In the application, obtain the exact
image, predicted observations and receipt from Task 1 through the existing helper:

```python
from trackrag.dropoff.inputs import build_instruction_request

request = build_instruction_request(
    recording, validated_predicted_observations,
    frame_id=150,
    request_id="instruction-1",
    session_id="passenger-session-1",
    text="Drop me off beside the white car.",
    provenance=validated_prediction_provenance,
)
```

The recording, observations and provenance above are caller-provided inputs, not
new global variables supplied by these contracts. The resulting shape is:

```json
{
  "request_id": "instruction-1",
  "session_id": "passenger-session-1",
  "scene": {
    "frame": {
      "recording_id": "example-recording",
      "frame_id": 150,
      "timestamp_s": 6.0,
      "width": 1280,
      "height": 720,
      "image_path": "original-frame.png"
    },
    "source": "predicted",
    "tracks": [{
      "recording_id": "example-recording", "frame_id": 150, "track_id": 7,
      "class_name": "Car", "bbox_xyxy": [780, 450, 910, 610], "source": "predicted"
    }],
    "provenance": {"condition": "illustrative_fixture"}
  },
  "text": "Drop me off beside the white car.",
  "input_mode": "typed",
  "dialogue": []
}
```

The real builder also supplies image_sha256. The original image must be accessible
to the inference host. There is no colour field in a TrackObservation; the model
must inspect the image to interpret "white".

Call Task 5's configured adapter and validate its response:

```python
from trackrag.dropoff.contracts import validate_outcome

outcome = validate_outcome(grounding_model.interpret(request), request)
```

The model can return four kinds of outcome:

| Condition | Envelope and result | Application response |
| --- | --- | --- |
| Destination resolved | ok=true, status=proposed, point and coarse region supplied | Validate; create an app proposal ID; display the destination; wait for explicit confirmation. |
| Multiple plausible references | ok=true, status=needs_clarification, destination fields null, distinguishing question supplied | Store pending clarification; ask the passenger; resubmit their reply with relevant same-scene dialogue. |
| Requested visual reference unavailable | ok=true, status=insufficient_visual_evidence, destination and question fields null | Explain the missing evidence; keep no proposal or confirmation. |
| Checkpoint/load, timeout, memory or malformed-response failure | ok=false, error code/message, no result | Retain an error receipt and the requested row; show the failure and permit retry. |

For example, an illustrative proposal response is:

```json
{
  "ok": true,
  "request_id": "instruction-1",
  "result": {
    "status": "proposed",
    "destination_point_xy": [620, 650],
    "destination_box_xyxy": [560, 610, 700, 700],
    "clarification_question": null,
    "anchor_boxes_xyxy": [[780, 450, 910, 610]]
  },
  "provenance": {"backend_kind": "illustrative_fixture", "model_id": "fixture-only"}
}
```

The destination box is beside the car; the anchor box identifies the referred
visual landmark. These are different roles. The result establishes an interpreted
destination; it does not establish physical clearance or stopping legality.

An operational failure instead has this shape:

```json
{
  "ok": false,
  "request_id": "instruction-1",
  "error": {"code": "checkpoint_unavailable", "message": "Configured checkpoint is unavailable"},
  "provenance": {"requested_condition": "fine_tuned"}
}
```

Model provenance in a real outcome retains the actual model/revision, pretrained
or trained condition, adapter identity and generation settings. Keep fixture,
pretrained, trained, cached-vision and historical outcomes accurately labelled.

Task 4 creates the current proposal from the validated proposed outcome:

```python
from copy import deepcopy
from uuid import uuid4
from trackrag.dropoff.contracts import (
    proposal_fingerprint, validate_interaction_state, validate_proposal,
)

frame = request["scene"]["frame"]
proposal = validate_proposal({
    "proposal_id": str(uuid4()),
    "request_id": request["request_id"],
    "scene": {"recording_id": frame["recording_id"], "frame_id": frame["frame_id"],
              "source": request["scene"]["source"]},
    "result": outcome["result"],
    "provenance": outcome["provenance"],
}, request)
state = validate_interaction_state({
    "session_id": request["session_id"], "request": request, "status": "proposed",
    "pending_clarification": None, "proposal": proposal,
    "confirmation": None, "transcript_corrections": [],
})

# Only in the later controller's explicit passenger-confirmation handler:
state["status"] = "confirmed"
state["confirmation"] = {
    "proposal_id": proposal["proposal_id"], "request_id": request["request_id"],
    "scene": deepcopy(proposal["scene"]),
    "proposal_sha256": proposal_fingerprint(proposal, request),
}
state = validate_interaction_state(state)
```

If clarification is needed, a second request's dialogue includes the original
instruction and the assistant's distinguishing question, each with recording_id
and frame_id. Its text holds the passenger's new reply, for example "The one on the
right." The controller decides when to call again. Bare "here" without pointing
or selection context remains unresolved; the contract cannot determine visual
meaning by itself.

For voice input, Task 3 may recognize "beside the wide car". Task 4 preserves that
in original_transcript, puts the passenger's correction "beside the white car" in
text, sets input_mode to voice and records a TranscriptCorrection for the new
request. Both typed and corrected voice text use the same interpret interface.

## Dialogue controller milestone: 22 November

The [controller](../trackrag/dropoff/interaction.py) now implements the transitions
that the shared records describe. It accepts an explicitly configured Task 5 adapter
and creates proposal IDs itself. It never generates destination coordinates or
silently selects a different model. It is a Python API for the upcoming local UI,
not a finished application or a new CLI command.

| App action | Controller method | Result |
| --- | --- | --- |
| Submit typed/voice input | submit(request) | Clears old authority, records the request, calls the adapter, validates and applies the outcome. |
| Answer a clarification | reply(text, ...) | Sends a fresh request containing the original instruction, actual model question and new reply, all bound to the same scene. |
| Edit current instruction/transcript | revise_instruction(text) | Sends a fresh request; preserves original ASR and correction history for voice; clears the old proposal/confirmation. |
| Explicitly confirm the displayed proposal | confirm(binding) | Accepts only the exact proposal/request/scene and fingerprint shown to the passenger. |
| Cancel | cancel() | Removes the live proposal, clarification and confirmation. |
| Change recording/frame/image/source | reset_scene() | Immediately clears the current interaction, including confirmation. |
| Export audit | snapshot() / save(path) | Retains requests, validated outcomes, bounded raw outcomes, available adapter receipts, events and counts. save requires a new file in an existing directory. |

For example, reuse the earlier built request and Task 5's configured adapter:

```python
from trackrag.dropoff.interaction import InteractionController

controller = InteractionController(
    configured_grounding_adapter,
    session_id=request["session_id"],
)
state = controller.submit(request)
if state is not None and state["status"] == "needs_clarification":
    # A real UI obtains this reply from the passenger.
    state = controller.reply("The one on the right.")
```

The adapter and request are supplied by the application, as in the model and scene
handoffs. Use the actual model configuration/checkpoint and retain its declared
condition. A test adapter is acceptable for wiring tests and must stay labelled.

When rendering a proposed destination, retain that exact returned snapshot with
the displayed Confirm control. In its explicit passenger-click handler:

```python
from copy import deepcopy
from trackrag.dropoff.contracts import proposal_fingerprint

# shown_state is the proposed snapshot used to render this particular control.
proposal = shown_state["proposal"]
confirmed = controller.confirm({
    "proposal_id": proposal["proposal_id"],
    "request_id": proposal["request_id"],
    "scene": deepcopy(proposal["scene"]),
    "proposal_sha256": proposal_fingerprint(proposal, shown_state["request"]),
})
```

Do not reconstruct this binding from a newer state when an old control is clicked.
The controller rejects stale bindings, leaves the valid current state unchanged,
and never interprets a text reply such as "yes" as an automatic confirmation.
It also rejects a model proposal for bare "here" because these contracts have no
pointing/selection context. The faulty response is retained as invalid_model_output,
not relabelled as a genuine semantic clarification or abstention.

For a UI that needs to render pending state before inference, use the split API:

```python
pending = controller.begin(request)
request_id = pending["request"]["request_id"]
# Render pending now; run this next line in the application's model worker.
state = controller.run_pending(request_id)
```

There are matching begin_reply and begin_revision methods. If an external worker
already calls the model, supply its result with complete(request_id, outcome)
instead of run_pending. Use one completion path per request. Cancellation and scene
changes remain available during inference; late results are saved as discarded and
cannot replace a newer interaction. Adapter calls are serialized. Backend resource
and timeout enforcement remain the supplied adapter/configuration's responsibility.

Further boundary behavior:

- Initial request IDs come from the caller; clarification/revision IDs are generated
  by the controller. IDs cannot be reused in a session. Legacy inputs receive the
  session ID before inference. Foreign sessions, future tracks, wrong sources,
  invalid geometry and extra target fields are rejected.
- A rejected replacement input clears the old current state and records an
  input_rejected event containing a bounded input snapshot. It does not invent a
  model attempt. Dialogue/correction budgets are enforced without silent truncation.
- The controller catches adapter exceptions as operational errors. Successful
  insufficient-visual-evidence decisions remain distinct successful model calls.
- Outward state and receipt values are copies. Caller or adapter mutation cannot
  change internal state or previously saved evidence.

The JSON audit schema is trackrag.dropoff.interaction.v1. counts.submission_rows
includes accepted requests and rejected inputs; counts.requested_rows counts
accepted requests only. requested_rows equals completed_rows plus unfinished_rows.
Rejected inputs remain in events and rejected_inputs, so a whole submitted cohort
can be accounted for. Discarded late outcomes and cancelled-before-dispatch requests
remain in calls; an unfinished outcome is null. These counts are execution receipts,
not grounding quality scores. Task 1 still owns independent scoring.

Available adapter last_receipt snapshots preserve raw model text and image/model
provenance before a subsequent inference can overwrite them. Oversized or non-JSON
values have explicit bounded capture-failure summaries rather than invalid JSON.
Their declared request identity is labelled matches, different_request or unverified;
a stale receipt left after an adapter exception is not presented as a matching one.
Request-start events retain transcript-correction history even after a scene reset.
Saving refuses to overwrite an existing receipt. Snapshots are audits, not a restore
or authentication mechanism for confirmations.

The [unified local application](task4-application.md) now connects recording/frame
selection, the existing cited-QA route, instruction/audio input, editable transcript,
destination overlay and controls to this controller. Real speech, real trained-model
inference and the final two-recording acceptance trial still require teammates'
actual artifacts and independently checked results.

## What to share for review

Share the small branch diff, [contracts](../trackrag/dropoff/contracts.py),
[scene builder](../trackrag/dropoff/inputs.py), this walkthrough,
[contract tests](../tests/test_dropoff_interaction_contracts.py) and
[builder tests](../tests/test_dropoff_inputs.py). For the dialogue milestone, also
include the [controller](../trackrag/dropoff/interaction.py) and its
[interaction tests](../tests/test_dropoff_interaction.py). Ask for these specific checks:

| Owner | Review and return handoff |
| --- | --- |
| Task 1 | Verify original frame/source/coordinate identities, no-future policy and image/prediction provenance. Return an exact-frame genuine predicted scene and receipt. Confirm saved outcomes and operational failures can be consumed by independent scoring. |
| Task 2 | Verify image/instruction identifiers and coordinate conventions align with manifests, mask-derived interior points and coarse boxes. Keep targets/masks and reserved references outside requests; supply one validated input/target pair separately. |
| Task 3 | Verify typed/voice modes, original versus corrected transcript and same-scene clarification dialogue. Supply speech/transcript handoff and reviewed clarification examples. Confirm the public QA schemas remain compatible. |
| Task 5 | Verify interpret accepts the existing request plus optional session_id, returns the unchanged semantic/error envelope and converts internal coordinates to original pixels. Supply explicit model/configuration provenance, a proposed example, an unresolved example and a failure receipt. |

Do not claim agreement until those owners review it. Task 4 owns the later local
application, dialogue controller, overlays, correction, confirmation/cancellation,
module wiring, receipts, regressions and release instructions. Tasks 1/2/3/5 own
vision/evaluation, data, speech/QA and learned grounding respectively.

## Checks

Use the project's isolated Python environment with integration dependencies:

```text
python -m pytest tests/test_dropoff_interaction_contracts.py tests/test_dropoff_inputs.py -q
python -m pytest tests/test_dropoff_interaction.py tests/test_dropoff_interaction_contracts.py tests/test_dropoff_inputs.py -q
python -m pytest tests -q
```

The focused tests cover the illustrative white-car exchange, existing adapter
compatibility using a labelled injected backend, semantic versus operational
outcomes, invalid geometry/envelopes, transcript correction, session compatibility
and stale confirmation after input/proposal changes. They use generated images
and fixture outputs; they do not demonstrate real grounding quality, trained
checkpoint availability or final two-recording application acceptance.

The controller checks additionally exercise full clarification/confirmation flows,
voice correction, real thread cancellation, stale/duplicate completions, reset on
scene change, immutable receipts, rejected-input counts, saved errors/unfinished
rows, bare-here rejection and compatibility with the existing Task 5 adapter using
a labelled injected generation backend. The focused controller/contracts/builder
command passed 174 tests on 22 November. These are wiring and regression checks,
not real speech or trained-model accuracy evidence.
The final full suite on 22 November passed 988 tests and 401 subtests, with 5
skipped. Editor diagnostics and the whitespace check reported no errors.

Local verification on 21 November 2026 used the existing Python 3.14.7 project
environment: the focused command passed 112 tests; the full suite passed 926 tests
and 401 subtests, with 5 skipped. The three JSON examples in this walkthrough also
passed the runtime request/outcome validators. No real grounding model was run.
