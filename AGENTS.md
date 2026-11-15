# TrackRAG-Gulf

The project combines MAAI 901 road-user detection/tracking with MAAI 902 question answering over visual evidence. A reviewer selects a recording and a query time, asks a question, then inspects the cited vehicle and frames.

## Work in this repository

- Task 1 owns `trackrag.vision`, Task 2 owns `trackrag.emt`, Task 3 owns `trackrag.questions` and `trackrag.answering`, Task 4 owns `trackrag.retrieval`, and Task 5 owns the reviewer command, evidence export and `evaluation`.
- Import the records in `trackrag/contracts.py` so frames, boxes, questions and evidence agree across modules. Preserve original frame IDs and original-image pixel coordinates.
- Track IDs are local to a recording and source. Human labels and detector predictions assign identities independently.
- Keep retrieval and answering inputs bounded by the original query time. Gold answers are inputs to scoring or training only.
- Put recording/model locations in arguments or configuration. Keep raw footage, checkpoints and credentials outside version control.

## Local checks

Use Python 3.11 or newer and an isolated environment. Install Task 5 dependencies with `python -m pip install -r requirements-review.txt`.

Run `python -m unittest discover -s tests -p "test_review*.py" -v` and `python -m unittest discover -s tests -p "test_score*.py" -v` for Task 5. Check `python -m trackrag --help` and `python -m evaluation --help` when changing entry points. Run the relevant task's checks when editing another component.

Regression tests can generate temporary images and replace heavy adapters. In reports and PRs, distinguish those checks from runs using real EMT images, real detector predictions or a live answer model. Real integration uses at least two recordings through the same entry point.

Before completing a substantial Task 5 change, have an independent reviewer inspect adapter boundaries, future-frame handling, saved evidence and scoring. Address concrete findings and rerun the affected checks. Preserve accurately labeled failures in evaluation results.

For Codex configuration questions, consult current official OpenAI documentation using the configured documentation MCP or the official site. Keep project instructions concise and add guidance when it solves an observed problem.
