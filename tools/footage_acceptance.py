"""Replay a bank through the running footage-review application and open its cited images."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(base, path, body=None):
    raw = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(base + path, data=raw, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=240) as response:
        return response.read()


def prepare(base, questions):
    catalog = json.loads(fetch(base, "/api/scenes"))
    scenes = [s for s in catalog["scenes"] if s["key"] in catalog["footage_scenes"]]
    anchors = {s["recording_id"]: s["key"] for s in scenes}
    keys = {(s["recording_id"], s["frame_id"]): s["key"] for s in scenes}
    for q in questions:
        identity = (q["recording_id"], q["query_frame"])
        if identity not in keys:
            frame = json.loads(fetch(base, "/api/frames", {
                "scene": anchors[identity[0]], "frame_id": identity[1], "for_footage": True,
            }))
            if (frame["recording_id"], frame["frame_id"]) != identity:
                raise ValueError("Application opened a different frame")
            keys[identity] = frame["key"]
    return keys


def review(base, key, question, hosted):
    response = json.loads(fetch(base, "/api/footage", {
        "scene": key, "question": question["question"], "hosted": hosted,
    }))
    observed = response["question"]
    for field in ("recording_id", "query_frame", "question"):
        if observed[field] != question[field]:
            raise ValueError(f"Application changed {field}")
    answer = response["answer"]
    cited = set(answer["cited_evidence_ids"])
    evidence = response["evidence"]
    if cited - {item["id"] for item in evidence}:
        raise ValueError("Answer cites an unavailable evidence item")
    if answer["answer_status"] == "answered" and not cited:
        raise ValueError("An answer was displayed without a citation")
    checked = {}
    for item in evidence:
        if item["interval"]["end_frame"] > question["query_frame"]:
            raise ValueError("Retrieved an interval after the question frame")
        if item["id"] in cited and not item["images"]:
            raise ValueError("A citation has no image to open")
        frames = []
        for image in item["images"]:
            if image["frame_id"] > question["query_frame"]:
                raise ValueError("Retrieved an image after the question frame")
            frame = {"frame_id": image["frame_id"]}
            if item["id"] in cited:
                pixels = fetch(base, image["url"])
                if not pixels.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")):
                    raise ValueError("A citation did not open a JPEG or PNG")
                frame["sha256"] = hashlib.sha256(pixels).hexdigest()
            frames.append(frame)
        checked[item["id"]] = frames
    return {"question": question, "run_id": observed["question_id"], "vision": response["vision"],
            "hosted_requested": response["hosted_requested"], "answer": answer, "evidence": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8775")
    parser.add_argument("--questions", type=Path, required=True, nargs="+")
    parser.add_argument("--hosted", action="store_true", help="Use the configured hosted answerer")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--out", type=Path, required=True, help="New directory for the per-question results")
    args = parser.parse_args()
    questions = [json.loads(line) for path in args.questions for line in path.read_bytes().splitlines() if line.strip()]
    ids = [q["question_id"] for q in questions]
    if not ids or len(set(ids)) != len(ids):
        parser.error("Supply a nonempty bank with unique question IDs")
    url = args.url.rstrip("/")
    keys = prepare(url, questions)
    args.out.mkdir(parents=True, exist_ok=False)

    def run(index_and_question):
        index, question = index_and_question
        try:
            key = keys[(question["recording_id"], question["query_frame"])]
            row = review(url, key, question, args.hosted)
        except Exception as error:
            row = {"question": question, "verification_error": str(error)}
        (args.out / f"{index:03d}.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        state = row.get("answer", {}).get("answer_status", "verification_error")
        print(question["question_id"], state, flush=True)
        return state

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        states = list(pool.map(run, enumerate(questions, 1)))
    return int(any(s in {"verification_error", "error"} for s in states))


if __name__ == "__main__":
    raise SystemExit(main())
