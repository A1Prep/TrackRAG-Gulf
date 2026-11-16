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
