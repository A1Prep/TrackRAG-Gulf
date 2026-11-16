# Fresh live answer investigation

The remaining 31 questions were submitted as fresh Gemini 3.5 Flash-Lite requests after the five scene-context cases. The initial condition used the explicitly selected TF-IDF development retriever and simple answer adapter. No gold answers were sent. Every attempt and processed input remains in `runs/task5-live-remaining-01`.

All 31 were attempted. Two questions had two HTTP 503 responses and no answer to grade. Independent AI-assisted development review found 15 content/status passes, five qualifications, eight failures, two unavailable answers and the predeclared manual entry diagnostic. Some correct words were supported only by incidental scene content while the highlighted vehicle was wrong. QA and target retrieval therefore need separate judgments. This condition had no previous complete live counterpart; these are newly measured baseline defects, not proof that the recent code change caused them.

## Corrected development condition

`tools.grounded_trial_adapters` is an explicit alternative for annotated development data. It does not replace the simple comparator or the teammates' trained-model interfaces.

- Use one stable T token per observed identity, its full query-bounded history and a privacy-processed identity gallery. This removes competing overlapping-window choices. Broad annotation categories remain separate from visible vehicle body shape. Original pixel areas/dimensions prevent gallery magnification from being mistaken for apparent size.
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

`--selection-model` can explicitly select a different evidence-selection backend; the configured and provider-reported models remain recorded per request. A diagnostic requesting Gemini 3.5 Flash encountered two timeouts and then a daily quota error naming Gemini 3.6 Flash with a 20-request limit. It produced no quality result, and the remaining submissions were stopped. The identity-catalog condition uses the available Gemini 3.5 Flash-Lite endpoint for both stages; it does not silently substitute another backend after a quota error.
