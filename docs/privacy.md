# Privacy handling for hosted answers

Task 3 sends processed copies of the evidence exported by Task 5. It reads `tracks.jsonl` in that run folder, using only observations at the actual supplied image frames. It does not require a copied EMT installation, a hardcoded test partition or hidden human annotations when the chosen source is predictions.

Before resizing, the processor masks the entire box of every observed person, pedestrian, cyclist or motorbike/rider. For other observed vehicles it masks the lower 40 percent of the box. It then resizes within 1024 by 576 pixels. Originals remain unchanged. This follows the policy used by the earlier development trials; it is a coarse mask, not a trained face or registration-plate detector.

The policy depends on the supplied observations. Missing detections or annotations can leave identifying details visible; lower-box masking is not proof that every plate has been covered. No complete anonymisation guarantee is made. Inspect actual processed inputs before an authorized hosted run.

Task 3 copies the exact processed files it passes to the model into `hosted/<question-hash>/`. Its receipt records the evidence/frame inventory, source-labeled metadata, privacy policy, sent-byte hashes, configured model, provider outcome and raw/parsed response. The evaluator checks the saved sent-byte hashes and binds the inputs to review packets. These checks establish stored consistency; they do not independently prove semantic correctness or that an external server saw particular bytes.

The default model client uses Google's Interactions API with `store=false`, no previous interaction, and a JSON schema including `answer_status`, answer text and citations. This disables stored interaction state for this request; provider terms still apply. `request_arguments_sha256` hashes canonical pre-SDK arguments, not the SDK-normalized HTTP body. API keys remain in the environment or an ignored `.env` file and are never intentionally saved to receipts.

Install hosted dependencies with `python -m pip install -r requirements-hosted.txt`. Set `GEMINI_API_KEY` and optionally `GEMINI_MODEL`; the contributor's default remains `gemini-3.8-flash`. Select the model explicitly for controlled comparisons. The SDK is pinned to the version used by the offline transport checks. Dependency imports and client creation are deferred so readiness checks and no-evidence results need neither an API key nor a hosted request.

The client bounds request time, disables automatic continuation and limits transient retries. HTTP 429 is propagated without an automatic retry. A provider/validation failure is an error, separate from a valid insufficient-evidence response. Empty evidence makes no model call. Raw answer and processed-input receipts are local artifacts and should not be committed with sensitive footage.

Implementation references: [Google image inputs](https://aistudio.google.com/docs/image-understanding), [structured output](https://aistudio.google.com/docs/structured-output), and [stateless Interactions requests](https://aistudio.google.com/docs/interactions-overview#data-storage-retention).
