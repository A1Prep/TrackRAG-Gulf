# Fresh live answer investigation

The remaining 31 questions were submitted as fresh Gemini 3.5 Flash-Lite requests after the five scene-context cases. The initial condition used the explicitly selected TF-IDF development retriever and simple answer adapter. No gold answers were sent. Every attempt and processed input remains in `runs/task5-live-remaining-01`.

All 31 were attempted. Two questions had two HTTP 503 responses and no answer to grade. Independent AI-assisted development review found 15 content/status passes, five qualifications, eight failures, two unavailable answers and the predeclared manual entry diagnostic. Some correct words were supported only by incidental scene content while the highlighted vehicle was wrong. QA and target retrieval therefore need separate judgments. This condition had no previous complete live counterpart; these are newly measured baseline defects, not proof that the recent code change caused them.

## Corrected development condition

`tools.grounded_trial_adapters` is an explicit alternative for annotated development data. It does not replace the simple comparator or the teammates' trained-model interfaces.

- Use one stable display token per observed identity, its full query-bounded history and a privacy-processed identity gallery. This removes competing overlapping-window choices. Broad annotation categories remain separate from visible vehicle body shape. Original pixel areas/dimensions prevent gallery magnification from being mistaken for apparent size. The latest selector uses opaque letter aliases consistently across its catalog, galleries and scene labels, while saved evidence preserves the actual recording-local track IDs. The frozen complete v1 trial used T-number labels; these conditions must remain distinguishable.
- Resolve explicit user track references against the supplied annotated identities. Do not silently substitute independently numbered predictions. Other questions use model-assisted visual evidence selection over the query-bounded candidate catalog.
- Keep selected identity histories, refine available multi-frame comparison endpoints, and reserve the exact current scene within the three evidence slots. A single numeric timestamp must not erase the beginning of a comparison.
- Give the answerer the actual sent-frame inventory, observed class/box/size/motion facts, source-resolution privacy-masked object crops and scene context. Computed image motion does not establish physical speed or driver intent.
- Validate exact temporal coverage and cited identities. An identity may use its own tracked frames plus cited full-scene frames; unrelated tracked identities cannot donate endpoints. Comma-separated frame references are supported. Correct scene-backed absence must remain answerable.
- Generate a fresh answer stage with cache reads/writes disabled. When the model abstains, check its explanation against the same pixels/facts in a fresh support-review stage, preserving the classification and raw draft receipt. A bounded validation repair may make another fresh request; rejected attempts remain in the final receipt. Selector caching remains explicit and separate from answer freshness.

The reviewer accepts an optional `export_retrieval(question, index, run_dir)` hook. This binds selected catalog/gallery inputs to the current run before generation. The artifact auditor checks selector scene/gallery times, available identity observations and returned intervals, alongside normal answer images and citations.

## Run and preserve a cohort

```sh
python -m tools.live_trial --data-root data/emt-official --questions experiments/temporal-development-questions.jsonl experiments/temporal-second-questions.jsonl --adapters tools.grounded_trial_adapters --model gemini-3.5-flash-lite --out runs/task5-grounded-live-01
```

The command keeps a fresh directory for each question/attempt and updates `summary.json` after every case. It records real HTTP events and model response IDs; it never substitutes answers. The default 13-second spacing applies to actual HTTP requests, including selection and validation repairs. Daily quota errors stop the remaining submissions; provider failures remain distinct from model-quality failures. A bounded second complete attempt is allowed after an error. Use `--include` or `--exclude` with public question IDs and a new output directory for focused follow-up. Existing directories are not overwritten.

The full rerun includes the prior five scene cases to check for new regressions. All of these questions and recordings are development data. Keep hidden references out of model inputs, identify AI-assisted review as such, and reserve independent recordings/questions for final evaluation. The detector/tracker, trained local Qwen reranker and normal Task 3 answerer remain separate integration requirements.

## Measured results and remaining validation

The complete grounded v1 run executed commit `80aaa0d`, before the later focused fixes. All 36 questions received fresh Gemini 3.5 Flash-Lite answers, including the two previously unavailable questions. Independent AI-assisted review checked the delivered answers and selected evidence against the development rubric. Human validation remains pending.

| Complete development condition | Clean pass | Qualified | Fail | Manual diagnostic | Provider errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple baseline: requested 31 plus the previous five scene checks | 18 | 7 | 10 | 1 | 2 |
| Frozen grounded v1: all 36 fresh answers | 28 | 4 | 3 | 1 | 0 |

Provider errors are included among failures; the columns are not additive. A joint clean pass requires supported answer content/status and suitable target evidence. The baseline's separate content-only review was more generous in two cases whose highlighted vehicles were wrong. The original 31-question trial itself has 13 joint passes, seven qualifications, ten failures including two unavailable answers, and one manual diagnostic. In the complete grounded v1 run, the requested 31 have 23 passes, four qualifications, three failures and the same manual diagnostic; the prior five all pass.

The paired comparison preserves every ordinary question: 12 clean-pass gains and two losses. The losses, `temporal-20` and `temporal-23`, retain appropriate abstention statuses but invent unsupported explanations about masked windows/interiors. They must remain visible as regressions. Multiple pipeline components changed, so this comparison does not isolate a model or training effect.

Subsequent focused live tests are separate evidence:

| Case | Problem in complete v1 | Follow-up result |
| --- | --- | --- |
| `temporal-18` | Abstains despite a supplied scene showing no visible bus at frame 30 | Pilot 03 correctly answers the visible-absence question |
| `second-04` | A single explicit timestamp erases the comparison's opening frames | Pilot 03 retains pickup 6 from frames 1–100 and correctly reports image-left motion |
| `second-05` | Highlights a different car and adds an unsupported physical-distance claim | Pilot 03 selects pickup 6 and supports its image growth with the observed dimensions |
| `temporal-20` | Correctly abstains but invents a masking explanation | Pilot 04 preserves the abstention and gives a supported explanation after a fresh review; the original draft is saved |
| `second-02` | Describes or selects the pickup instead of the very large truck | Pilots 03, 05 and 06 remain failures even when the direction wording is correct. Pilot 07 selects truck 17 at frames 1–57 plus scene 60, but its answer request hits the daily quota. Only that one selection is confirmed |
| `temporal-23`, `second-12` | Unsupported interior/window visibility explanations | The generic explanation review is implemented, but these cases still need a fresh live check |

The model-assisted explanation review is not a correctness guarantee. It keeps an existing abstention classification and can still fail. Likewise, one corrected identity selection does not prove that opaque labels caused the improvement or that the final answer works. Do not combine focused outcomes into a fictional full-cohort score.

Pilot 07 stopped on Google's actual HTTP 429 daily free-tier limit of 500 requests for `gemini-3.5-flash-lite`; the response requested a wait of roughly 15 hours. No further requests were made. The latest source has **no complete live rerun**, so this follow-up remains a draft pending that check. The stronger selector diagnostic described below also produced no quality result.

The [committed trial summary](../experiments/live-answer-regression-summary.json) records answers, selected intervals, source revisions, fresh-response evidence, provider errors and audited artifact hashes. [Baseline reviews](../experiments/reviews/live-tfidf-baseline.jsonl) and [complete v1 reviews](../experiments/reviews/live-grounded-v1.jsonl) bind each judgment to its exact result bytes. Raw images and run folders stay local and outside Git.

With the preserved local runs available, reproduce the comparison without model calls:

```sh
python -m evaluation compare --manifest experiments/comparisons/live-answer-development.json --out runs/live-answer-comparison-recheck
```

After provider capacity is available, freeze the current source and run all 36 questions with the live command above using a new output directory and an explicit `--selection-model gemini-3.5-flash-lite`. Independently review answer content, target identities, time coverage and citations, then compare with both preserved complete conditions. Keep the two regressions, the truck case, all previous five successes, and every provider error in the cohort. Do not retry only successful cases or reuse cached answers.

`--selection-model` can explicitly select a different evidence-selection backend; the configured and provider-reported models remain recorded per request. A diagnostic requesting Gemini 3.5 Flash encountered two timeouts and then a daily quota error naming Gemini 3.6 Flash with a 20-request limit. It produced no quality result, and the remaining submissions were stopped. The identity-catalog condition uses the available Gemini 3.5 Flash-Lite endpoint for both stages; it does not silently substitute another backend after a quota error.
