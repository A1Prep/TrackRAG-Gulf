# Follow-up on the five empty development searches

The original TF-IDF development baseline searches short descriptions of tracked road users. Signs, bridges, bed structures and concealed contents are not included in that vocabulary. These five questions therefore had no lexical overlap and returned empty results. Empty retrieval is different from a question whose answer cannot be established from the footage.

| Question | Missing representation | Change and practical limit |
| --- | --- | --- |
| temporal-16: street name on the exit sign | Sign text is absent from track descriptions | Supply the exact scene at frame 150. Reading the street name still depends on image legibility and the answerer. |
| second-03: structure above the white pickup's bed | The labels/descriptions do not identify the pickup's bed structure | Supply scene 100, where its metal framework is visible. The fallback assigns no vehicle identity. |
| second-07: overpass above the roadway | Overpasses are not tracked road users | Supply scene 75 so a reviewer/answerer can inspect the requested moment. |
| second-10: pickup's actual speed | Physical speed is absent from the track vocabulary and is not established by pixel motion | Supply scene 100 as context. The dashcam speed overlay describes the camera vehicle; this does not establish the pickup's speed. |
| second-11: cargo inside a closed truck | Concealed contents are absent from the descriptions and are not visible | Supply scene 150 as context. This does not establish the hidden cargo. |

The development adapter now returns one exact query-frame scene, with `track_id: null` and score `0.0`, only when lexical search returns no match. It uses the same rule for every question and has no question-ID, reference-answer or object-to-track lookup. A past observation must establish the source; future-only tracks cannot enable a fallback. Existing nonempty lexical intervals, order and scores are preserved.

`tools.emt_trial_adapters.explain_retrieval` returns pure diagnostics separately from the shared evidence records. The verification report separately counts zero-score scene contexts and their temporal comparisons. The HTML evidence report warns that relevance and answerability are unassessed. Nonsense or unrelated questions can receive the same context; that is an exposed limitation, not a semantic match.

The simple hosted trial adapter describes scene inputs correctly and warns that fallback context does not establish a fact. Tracked-only request prompts and image labels retain their previous content. Mocked provider checks verify request construction without an upload. No live answer improvement is implied.

## Regression protocol

Run both original question files together with the new coverage controls through the same two-recording entry point:

```sh
python -m evaluation verify-retrieval --data-root data/emt-official --questions experiments/temporal-development-questions.jsonl experiments/temporal-second-questions.jsonl experiments/scene-context-controls.jsonl --source annotated --retrieval-module tools.emt_trial_adapters --out runs/task5-scene-context-01
```

The 12 extra questions include paraphrases, hidden/nonvisual information, a future-time request and an unrelated string. They test fallback classification and time bounds. They have no answer key and are not an answer-accuracy benchmark. Do not add their zero-score contexts to a semantic retrieval success count.

Compare every original nonempty ranking with `runs/task5-retrieval-acceptance-02`, including IDs, time spans, rank, source and exact scores. Check that each original empty case becomes exactly its query-frame scene; keep answerability references unchanged. Repeat all future-observation, prefix and future-pixel controls. The five new contexts require live-model and human review before any answer-quality claim.

## Handoff to the final retriever

Task 4 should include exact scene candidates alongside tracked histories before learned reranking. Track labels alone do not describe sign text, bridge structure or a pickup's bed. The model needs appropriate visual/text inputs for those distinctions. Keep extra candidate features inside the index; returned evidence still follows the shared interval contract.

Report the fallback rate separately when comparing frozen and fine-tuned models. Use reviewed relevance labels to determine whether a selected context supports the question, and preserve an insufficient-evidence outcome for hidden or unmeasured facts. All questions used in this investigation are development data; reserve new recordings and questions for final evaluation.

Keep candidate sets and fallback policy fixed between frozen and fine-tuned comparisons. This software change is not a training improvement.

## Measured outcome

Code commit `18913b1b195f6347b99cda976a86257b6ab4ea77` completed the command above. The [committed summary](../experiments/scene-context-summary.json) records the input/report hashes and exact comparison results.

| Recording | Original plus control questions | Preserved lexical matches | Zero-score scene contexts | Nonempty temporal comparisons (context subset) |
| --- | ---: | ---: | ---: | ---: |
| video_054604 | 30 | 23 | 7 | 54 (18) |
| video_054907 | 18 | 8 | 10 | 42 (27) |

The original 36 questions retain 31 lexical results and gain five zero-score scene contexts. All 12 additional controls also receive context, including unrelated/nonvisual questions, demonstrating why fallback availability cannot count as semantic success. No contract, control-stability or temporal-perturbation failure occurred; 16 final-frame queries still have no future test. Both recording statuses remain `passed_with_gaps`.

All 31 previous result records and 98 cited image files are unchanged. Comparing the old and new answer-request builders on these same inputs produced 31 byte-identical provider request bodies with HTTP mocked. This verifies unchanged inputs, not unchanged model answers. No live model call was made.

The five new scene exports are pixel-identical to their original frames. Four unique processed inputs were generated locally using the actual privacy/resize function, at 1024 by 576, without uploading them during the offline check. The processed sign and pickup frames were inspected: the Hessa St lettering and metal framework remain visible. The overpass is visible in the original query-frame export. Those offline checks did not test answer generation; the subsequent live check is recorded below. The speed and concealed-cargo facts remain unestablished.

The full local suite passes 181 tests and 67 subtests. Independent review passes 12 focused tests and nine subtests, with no blocking finding; Windows/Linux CI passed on the implementation. The local `report.html` files in each recording's run folder show the evidence and context-only warnings.

## Subsequent bounded live check

The five new scene cases were then tested through the production reviewer and actual hosted answer adapter, using code `5c3a56c2f82fa9855413e294b70c8cc03f2cdecf` and explicitly selecting `TRACKRAG_TRIAL_MODEL=gemini-3.5-flash-lite`. Five requests completed, with starts spaced at least 13 seconds apart. Only the authorized, privacy-processed public EMT scene frames were sent. The [live trial summary](../experiments/scene-answer-trial-summary.json) preserves the questions, answers, provider usage, source revision, request settings and hashes for results, evidence, processed inputs and receipts.

| Case | Delivered result | Status / citations |
| --- | --- | --- |
| temporal-16 | Hessa St | Answered, E1 |
| second-03 | Metal cage-like/rack structure above the pickup bed | Answered, E1 |
| second-07 | Overpass directly above the roadway in frame 75 | Answered, E1 |
| second-10 | Actual pickup speed is not established by camera-vehicle telemetry | Insufficient evidence, no citations |
| second-11 | Cargo is concealed by the closed compartment | Insufficient evidence, no citations |

An independent AI-assisted reviewer inspected the saved answers and processed images against the development references. All five had supported factual/status/citation outcomes, and all five passed the artifact auditor. This is not human validation or held-out accuracy: the cases were selected because their earlier searches were empty. The other 31 live answers were not replayed; their evidence and request bodies were verified unchanged in the offline regression.

The legacy provider receipt captures the requested model but not a separate provider-reported revision. These runs use human annotations and the simple development retriever. The trained detector/tracker, Qwen reranker, Task 3 answerer and independent evaluation remain the next integration work. For reproduction, select each saved four-field question from the existing banks and run the standard `trackrag review` command for its recording with `--source annotated --answerer gemini --adapters tools.emt_trial_adapters`, the explicit model environment setting above, and a fresh output directory. Respect the provider's current limits.
