# TrackRAG-Gulf

The project combines MAAI 901 road-user detection/tracking with MAAI 902 question answering over visual evidence and passenger-instruction grounding. Preserve the existing cited-QA route while adding the approved destination-proposal, clarification and confirmation route.

## Work in this repository

- Current assignments use the previous task owners: Task 1 owns vision, scene handoff and independent evaluation; Task 2 owns data and splits; Task 3 owns QA and speech; Task 4 owns application integration, dialogue and release; Task 5 owns actual NLP grounding training/inference. See [Part 2](docs/final-completion-part2.md) and issues #1-#5 for deliverables and planned file boundaries. Preserve Part 1 authorship when editing the existing reviewer, retrieval and evaluation modules.
- The current NLP training milestone is learned passenger-instruction grounding, replacing the earlier separate Qwen-reranker milestone. Keep the lexical QA baseline accurately labelled; do not silently substitute a hosted answerer or static rules for the trained grounding route.
- Use descriptive task branch names such as `task-5-reviewer`.
- Import the records in `trackrag/contracts.py` so frames, boxes, questions and evidence agree across modules. Preserve original frame IDs and original-image pixel coordinates.
- Track IDs are local to a recording and source. Human labels and detector predictions assign identities independently.
- Keep retrieval and answering inputs bounded by the original query time. Gold answers are inputs to scoring or training only.
- Put recording/model locations in arguments or configuration. Keep raw footage, checkpoints and credentials outside version control.
- Select development adapters explicitly with `--adapters`. Keep their recorded results distinguishable from the normal task implementations.

## Local checks

Use Python 3.11 or newer and an isolated environment. Install reviewer dependencies with `python -m pip install -r requirements-review.txt`. For EMT integration and real-data trials, use `requirements-integration.txt`, which includes OpenCV and pytest. Part 2 NLP training uses its separately documented GPU environment; do not treat reviewer requirements as training dependencies.

Run `python -m pytest tests -q` with integration dependencies installed. CI also runs the Task 5 unittest commands and `tests/test_emt.py` separately. Check `python -m trackrag --help` and `python -m evaluation --help` when changing entry points. Run the relevant task's checks when editing another component.

Regression tests can generate temporary images and replace heavy adapters. In reports and PRs, distinguish those checks from runs using real EMT images, real detector predictions or a live answer model. Real integration uses at least two recordings through the same entry point.

Before completing a substantial Task 5 change, have an independent reviewer inspect adapter boundaries, future-frame handling, saved evidence and scoring. Address concrete findings and rerun the affected checks. Preserve accurately labeled failures in evaluation results.

For Codex configuration questions, consult current official OpenAI documentation using the configured documentation MCP or the official site. Keep project instructions concise and add guidance when it solves an observed problem.
