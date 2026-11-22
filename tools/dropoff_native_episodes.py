"""Collect typed, case-bound native Qwen clarification episodes for Task 1's frozen evaluator."""

import argparse
from base64 import b64encode
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

from PIL import Image

from evaluation import grounding as grader
from evaluation import grounding_inputs as converter
from tools.dropoff_final_input_inventory import ROOT, git, project
from tools.dropoff_predict import predict, read_requests
from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.data import file_sha256, inference_input_sha256
from trackrag.dropoff.formatting import build_messages, parse_model_output
from trackrag.dropoff.model import GroundingConfig, verify_adapter


TEMPLATE = Path(__file__).with_suffix(".html")
TASK5_MODEL_PAIR = ROOT / "experiments/dropoff/passenger-final-model-conditions-20261122.json"
ATTEST = "I consent to this review. I saw this original image and model turn, and entered my own decision."
FIXTURE_ATTEST = "Software fixture only. I am exercising the review controls."
REPLY_FIELDS = ("case_id", "reply_id", "kind", "text")
REFERENCE_FIELDS = {
    k: True
    for k in (
        "case_id",
        "partition",
        "scene_sha256",
        "recording_id",
        "frame_id",
        "image_sha256",
        "width",
        "height",
        "instruction",
        "dialogue",
    )
}
RUNTIME_FILES = (
    "evaluation/grounding.py",
    "evaluation/grounding_inputs.py",
    "tools/dropoff_predict.py",
    "tools/dropoff_native_episodes.py",
    "tools/dropoff_native_episodes.html",
    "trackrag/dropoff/model.py",
    "trackrag/dropoff/formatting.py",
    "trackrag/dropoff/streamed.py",
    "trackrag/dropoff/structured.py",
    "trackrag/dropoff/data.py",
    "trackrag/dropoff/contracts.py",
    "trackrag/dropoff/scenes.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def moment(text):
    time = datetime.fromisoformat(text.replace("Z", "+00:00"))
    require(time.tzinfo is not None, "A timezone is required")
    return time


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with path.open("xb") as f:
        f.write(canonical(value) + b"\n")


def write_rows(path, values):
    with path.open("xb") as f:
        for value in values:
            f.write(canonical(value) + b"\n")


@contextmanager
def staging(path):
    require(not path.exists(), f"Preserving existing output at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        new = Path(temp) / "content"
        new.mkdir()
        yield new
        new.rename(path)


@contextmanager
def locked(folder):
    lock = folder / ".operating"
    try:
        file = lock.open("x", encoding="utf8")
    except FileExistsError as exc:
        raise ValueError(
            "Another operation owns this session. Inspect its PID before recovering an interrupted lock."
        ) from exc
    with file:
        file.write(str(os.getpid()))
    try:
        yield
    finally:
        lock.unlink()


def assignments(path):
    data = read(path)
    require(
        isinstance(data, dict)
        and data
        and all(
            isinstance(k, str) and k.strip() and isinstance(v, str) and 0 < len(v) <= 80 and v.strip() == v
            for k, v in data.items()
        ),
        "Use a case-to-participant-code JSON object",
    )
    return data


def verify_evaluator_revision(revision):
    require(
        isinstance(revision, str) and re.fullmatch("[0-9a-f]{40}", revision),
        "Task 1 must specify the full frozen evaluator Git commit",
    )
    local = Path(grader.__file__).read_bytes().replace(b"\r\n", b"\n")
    require(
        git(revision, "evaluation/grounding.py").replace(b"\r\n", b"\n") == local,
        "The executing Task 1 scorer differs from the frozen evaluator commit",
    )


def freeze(
    references,
    replies,
    people,
    evaluator_commit,
    responder_rules,
    turn_budget,
    out,
    voice=None,
    model_conditions=None,
):
    require(
        type(turn_budget) is int and 1 <= turn_budget <= 9,
        "A native episode must fit the 16-message dialogue window",
    )
    require(bool(responder_rules.strip()), "Supply Task 1’s actual predeclared human responder rules")
    verify_evaluator_revision(evaluator_commit)
    people_map = assignments(people)
    official = converter.freeze_protocol(
        references,
        evaluator_commit=evaluator_commit,
        responder_rules=responder_rules,
        replies=replies,
        voice=voice,
    )
    official["native_clarification"] = {
        "version": 1,
        "decision_mode": "designated_human",
        "turn_budget": turn_budget,
        "assignments_sha256": digest(people_map),
    }
    if model_conditions:
        registered = read(model_conditions)
        pair = {
            role: {
                "config_sha256": registered[role]["config_sha256"],
                "adapter_sha256": registered[role].get("adapter_manifest_sha256"),
            }
            for role in grader.MODELS
        }
        require(
            all(
                isinstance(setting["config_sha256"], str)
                and re.fullmatch("[0-9a-f]{64}", setting["config_sha256"])
                for setting in pair.values()
            )
            and pair["pretrained"]["adapter_sha256"] is None
            and isinstance(pair["trained"]["adapter_sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", pair["trained"]["adapter_sha256"]),
            "Invalid pretrained or trained model condition registration",
        )
        official["native_clarification"].update(
            model_conditions=pair, model_conditions_source_sha256=file_sha256(model_conditions)
        )
    write(out, official)


def public_reference_headers(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row, end = project(line, REFERENCE_FIELDS)
        require(
            not line[end:].strip() and row["case_id"] not in result and row["partition"] == "final_evaluation",
            "Duplicate or nonfinal case identity in frozen references",
        )
        result[row["case_id"]] = row
    return result


def definition(folder, number, requests, incoming, parent):
    with staging(folder / f"round-{number:02}") as r:
        write_rows(r / "requests.jsonl", requests)
        write(r / "incoming.json", incoming)
        write(
            r / "plan.json",
            {
                "round": number,
                "parent_sha256": parent,
                "requests_sha256": file_sha256(r / "requests.jsonl"),
                "incoming_sha256": file_sha256(r / "incoming.json"),
            },
        )


def prepare(protocol, references, replies, people, scenes, requests, config, out, *, _fixture=False):
    official, protocol_sha = grader.load_protocol(protocol)
    verify_evaluator_revision(official["evaluator_commit"])
    require(
        moment(official["frozen_utc"]) <= datetime.now(timezone.utc)
        and official["references_sha256"] == grader.text_sha256(references)
        and official["replies_sha256"] == grader.text_sha256(replies),
        "Frozen reference/reply versions or time differ",
    )
    policy = read(protocol).get("native_clarification", {})
    require(
        _fixture or "model_conditions" in policy,
        "Freeze the selected pretrained and trained pair before production episodes",
    )
    roster = assignments(people)
    require(
        policy.get("version") == 1
        and policy.get("decision_mode") == "designated_human"
        and policy.get("assignments_sha256") == digest(roster)
        and type(policy.get("turn_budget")) is int
        and 1 <= policy["turn_budget"] <= 9,
        "Freeze the intended participant assignments and turn budget before collecting any output",
    )
    refs = public_reference_headers(references)
    inputs = read_requests(requests)
    require(
        set(refs) == set(roster) == {q["request_id"] for q in inputs},
        "The complete frozen case cohort is required",
    )
    scene_bank = grader.load_scenes(scenes, allow_stub=_fixture)
    reply_bank = grader.load_replies(replies, refs)
    conf = GroundingConfig.from_mapping(read(config))
    require(
        not conf.allow_annotated and conf.local_files_only,
        "Use the pinned local native model and predicted observations",
    )
    adapter = verify_adapter(conf)
    if "model_conditions" in policy:
        role = "trained" if adapter else "pretrained"
        require(
            policy["model_conditions"][role]
            == {
                "config_sha256": file_sha256(config),
                "adapter_sha256": adapter["manifest_sha256"] if adapter else None,
            },
            "Native config or adapter differs from Task 1’s registered model pair",
        )
    raw = {str(p.resolve()): file_sha256(p) for p in (protocol, references, replies, people, requests, config)}
    versions = {name: file_sha256(ROOT / name) for name in RUNTIME_FILES}
    with staging(out) as folder:
        (folder / "images").mkdir()
        cases = {}
        for req in inputs:
            case = req["request_id"]
            ref = refs[case]
            frame = req["scene"]["frame"]
            image = Path(frame["image_path"])
            scene = scene_bank.get(ref["scene_sha256"])
            photo = file_sha256(image)
            require(
                scene is not None
                and req["input_mode"] == "typed"
                and req["scene"]["source"] == "predicted"
                and all(
                    scene["model_input"][k] == ref[k]
                    for k in ("recording_id", "frame_id", "width", "height", "image_sha256")
                )
                and not converter.scene_problems(scene, frame, req["scene"]["tracks"], photo)
                and frame["timestamp_s"] == scene["model_input"]["timestamp_s"],
                "Wrong original scene, observations or time",
            )
            binding = grader.instruction_sha256(req["text"], req["dialogue"])
            require(
                binding == grader.instruction_sha256(ref["instruction"], ref["dialogue"])
                and photo == req["scene"].get("provenance", {}).get("image_sha256"),
                "Wrong typed instruction or source image",
            )
            require(
                len(req["dialogue"]) + 2 * (policy["turn_budget"] - 1) <= 16,
                "The registered turn budget and earlier conversation exceed the native dialogue limit",
            )
            with Image.open(image) as original:
                require(
                    original.size == (frame["width"], frame["height"])
                    and original.format in ("JPEG", "PNG", "WEBP"),
                    "Unsupported or incorrectly sized original photograph",
                )
                display_image = photo + {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[original.format]
            dest = folder / "images" / display_image
            if not dest.exists():
                shutil.copyfile(image, dest)
            require(file_sha256(dest) == photo, "Private display photo differs from the actual model input")
            cases[case] = {
                "participant": roster[case],
                "scene_sha256": ref["scene_sha256"],
                "image_sha256": photo,
                "display_image": display_image,
                "instruction_sha256": binding,
                "input_sha256": inference_input_sha256(req, photo),
            }
        # Task 1 accepts extra private columns in the original reply registry. The
        # model, review form and derived store receive only the four actual reply fields.
        write(folder / "replies.json", [{k: row[k] for k in REPLY_FIELDS} for row in reply_bank.values()])
        plan = {
            "version": 1,
            "session": uuid4().hex,
            "prepared_utc": stamp(),
            "protocol_sha256": protocol_sha,
            "frozen_utc": official["frozen_utc"],
            "responder_rules": official["responder_rules"],
            "turn_budget": policy["turn_budget"],
            "source_files": raw,
            "runtime_source_sha256": versions,
            "cases": cases,
            "configuration": asdict(conf),
            "model": "trained" if adapter else "pretrained",
            "adapter_sha256": adapter["manifest_sha256"] if adapter else None,
            "replies_sha256": file_sha256(folder / "replies.json"),
            "software_fixture": _fixture,
        }
        write(folder / "session.json", plan)
        definition(folder, 0, inputs, {}, file_sha256(folder / "session.json"))
    return plan


def load_session(folder):
    plan = read(folder / "session.json")
    require(
        all(file_sha256(p) == h for p, h in plan["source_files"].items())
        and all(file_sha256(ROOT / name) == sha for name, sha in plan["runtime_source_sha256"].items())
        and file_sha256(folder / "replies.json") == plan["replies_sha256"],
        "An original frozen file has changed",
    )
    return plan


def current(folder):
    rounds = sorted(p for p in folder.iterdir() if re.fullmatch(r"round-\d\d", p.name) and p.is_dir())
    require(
        rounds and [r.name for r in rounds] == [f"round-{i:02}" for i in range(len(rounds))],
        "An episode round disappeared",
    )
    return rounds


def load_definition(folder, round_dir):
    p = read(round_dir / "plan.json")
    n = p["round"]
    prior = folder / "session.json" if n == 0 else folder / f"round-{n - 1:02}" / "responses" / "decisions.json"
    require(
        round_dir.name == f"round-{n:02}"
        and file_sha256(prior) == p["parent_sha256"]
        and file_sha256(round_dir / "requests.jsonl") == p["requests_sha256"]
        and file_sha256(round_dir / "incoming.json") == p["incoming_sha256"],
        "A sealed turn input changed",
    )
    return p


def verify_native_receipt(q, o, rec, conf):
    """Bind the native prompt, raw model text, parsed pixels and any operational error."""
    source = o["provenance"]
    require(
        rec.get("provenance") == source and type(o.get("ok")) is bool,
        "Native receipt and outcome provenance disagree",
    )
    if o["ok"] or "messages" in rec or "messages_sha256" in source:
        messages = build_messages(
            q,
            task="interpret",
            allow_annotated=conf.allow_annotated,
            observation_format=conf.observation_format,
        )
        require(
            rec.get("messages") == messages and source.get("messages_sha256") == digest(messages),
            "Raw native messages differ from this registered model input",
        )
    if o["ok"]:
        require(
            rec.get("status") == "completed"
            and "error" not in rec
            and "error" not in o
            and isinstance(o.get("result"), dict)
            and o["result"] == rec.get("result"),
            "Successful native receipt and outcome disagree",
        )
        f = q["scene"]["frame"]
        parsed = parse_model_output(rec.get("raw_text"), f["width"], f["height"])
        require(parsed == o["result"], "Raw native text does not reproduce the saved interpretation")
    else:
        require(
            rec.get("status") == "error"
            and "result" not in rec
            and "result" not in o
            and isinstance(o.get("error"), dict)
            and o["error"] == rec.get("error"),
            "Failed native receipt and outcome disagree",
        )


def collect(folder, *, _predict=predict):
    with locked(folder):
        plan = load_session(folder)
        require(
            plan["software_fixture"] or _predict is predict,
            "Only the normal local native predictor can collect production episodes",
        )
        r = current(folder)[-1]
        d = load_definition(folder, r)
        require(
            not (folder / "final").exists() and not (r / "review").exists(), "This round is already collected"
        )
        rows = read_requests(r / "requests.jsonl")
        conf = GroundingConfig.from_mapping(plan["configuration"])
        for q in rows:
            image = plan["cases"][q["request_id"]]["image_sha256"]
            require(
                file_sha256(q["scene"]["frame"]["image_path"])
                == file_sha256(folder / "images" / plan["cases"][q["request_id"]]["display_image"])
                == image,
                "Original model or reviewer photograph changed",
            )
        native = r / "native"
        attempt = r / "attempt.json"
        identity = {
            "session": plan["session"],
            "request_plan_sha256": file_sha256(r / "plan.json"),
            "configuration_sha256": digest(plan["configuration"]),
        }
        if not attempt.exists():
            require(not native.exists(), "An existing native output was not collected within this frozen round")
            write(attempt, {**identity, "started_utc": stamp()})
            _predict(r / "requests.jsonl", conf, native)
        launched = read(attempt)
        require(
            all(launched.get(key) == value for key, value in identity.items()),
            "Native launch marker belongs to a different session or round",
        )
        require(
            (native / "run.json").is_file(),
            "An incomplete native attempt was preserved. Do not rerun or select outcomes in this session.",
        )
        run = read(native / "run.json")
        require(
            moment(plan["prepared_utc"])
            <= moment(launched["started_utc"])
            <= moment(run["created_at"])
            <= datetime.now(timezone.utc) + timedelta(minutes=5),
            "Native generation predates this registered round or has an invalid completion time",
        )
        require(
            run["requests_sha256"] == d["requests_sha256"] == file_sha256(native / "input-snapshot.jsonl")
            and run["configuration"] == asdict(conf)
            and run["requested_rows"] == len(rows),
            "Native run differs from this turn plan",
        )
        require(
            all(file_sha256(native / f) == run[k] for f, k in converter.TASK5_RUN_FILES),
            "Native run evidence changed",
        )
        supplied = grader.read_jsonl(native / "resolved-requests.jsonl")
        outcomes = grader.read_jsonl(native / "model-outcomes.jsonl")
        receipts = grader.read_jsonl(native / "receipts.jsonl")
        incoming = read(r / "incoming.json")
        require(
            supplied == rows and len(rows) == len(outcomes) == len(receipts),
            "Native request order or denominator differs",
        )
        require(
            run.get("semantic_results") == sum(o.get("ok") is True for o in outcomes)
            and run.get("operational_errors") == sum(o.get("ok") is False for o in outcomes),
            "Native success or error counts differ from the saved outcomes",
        )
        provenance = converter._run_provenance(run["configuration"], outcomes, plan["model"])
        if plan["adapter_sha256"] and provenance["adapter_sha256"] is None:
            require(all(o.get("ok") is False for o in outcomes), "A semantic output lacks its trained adapter")
            provenance["adapter_sha256"] = plan["adapter_sha256"]
        require(
            provenance["adapter_sha256"] == plan["adapter_sha256"]
            and (plan["software_fixture"] or not provenance["substitute"]),
            "Unexpected adapter or substituted native backend",
        )
        provenance.update(
            turn_budget=plan["turn_budget"],
            native_configuration_sha256=digest(asdict(conf)),
            decision_source="designated_human",
            protocol_sha256=plan["protocol_sha256"],
        )
        if plan["software_fixture"]:
            provenance["substitute"] = True
        turns, page_rows = [], []
        registered = read(folder / "replies.json")
        for q, o, rec in zip(rows, outcomes, receipts, strict=True):
            case = q["request_id"]
            attachment = plan["cases"][case]
            source = o["provenance"]
            require(
                rec["request_id"] == o["request_id"] == case
                and rec["request"] == q
                and source["request_sha256"] == digest(q)
                and source["requested_condition"] == grader.TASK5_REQUESTED[plan["model"]]
                and f"{source['model_id']}@{source['revision']}" == provenance["base_revision"],
                "A native call belongs to another input or model",
            )
            verify_native_receipt(q, o, rec, conf)
            actual_image = (source.get("image") or {}).get("sha256")
            require(
                actual_image == attachment["image_sha256"] or (actual_image is None and o.get("ok") is False),
                "The native model receipt identifies a different image",
            )
            if o["ok"] and plan["adapter_sha256"]:
                require(
                    source.get("adapter", {}).get("manifest_sha256") == plan["adapter_sha256"],
                    "Semantic output came from another trained adapter",
                )
            call, result, invalid = grader.from_task5_outcome(o)
            f = q["scene"]["frame"]
            status = (
                grader._classify(result, f["width"], f["height"], {"reasons": []}, invalid)
                if result
                else "operational_or_invalid"
            )
            binding = digest(
                {
                    "session": plan["session"],
                    "round": d["round"],
                    "case": case,
                    "input": source["request_sha256"],
                    "outcome": o,
                    "run": file_sha256(native / "run.json"),
                }
            )
            proposal = f"proposal-{binding}" if status == "proposed" else None
            turn = {
                "case_id": case,
                "turn": d["round"],
                "turn_id": binding,
                "scene_sha256": attachment["scene_sha256"],
                "native_input_sha256": source["request_sha256"],
                "native_image_receipt_available": bool(actual_image),
                "outcome": o,
                "raw_response": rec.get("raw_text"),
                **incoming.get(case, {}),
            }
            if proposal:
                turn["proposal_id"] = proposal
            turns.append(turn)
            if call["outcome"] == "ok" and status in grader.STATUSES:
                f = q["scene"]["frame"]
                kind = "rejection" if proposal else "answer"
                page_rows.append(
                    {
                        "case_id": case,
                        "turn_id": binding,
                        "proposal_id": proposal,
                        "status": status,
                        "result": result,
                        "image": "../../images/" + attachment["display_image"],
                        "image_sha256": attachment["image_sha256"],
                        "width": f["width"],
                        "height": f["height"],
                        "text": q["text"],
                        "dialogue": q["dialogue"],
                        "participant": attachment["participant"],
                        "reply_options": [v for v in registered if v["case_id"] == case and v["kind"] == kind]
                        if d["round"] + 1 < plan["turn_budget"]
                        else [],
                    }
                )
        with staging(r / "review") as review:
            write_rows(review / "turns.jsonl", turns)
            write(review / "provenance.json", provenance)
            page = {
                "session": plan["session"],
                "round": d["round"],
                "form": uuid4().hex,
                "attestation": FIXTURE_ATTEST if plan["software_fixture"] else ATTEST,
                "rules": plan["responder_rules"],
                "software_fixture": plan["software_fixture"],
                "created_utc": stamp(),
                "rows": page_rows,
            }
            write(review / "display.json", page)
            forms = review / "forms"
            forms.mkdir()
            index = []
            for person in sorted({row["participant"] for row in page_rows}):
                own = []
                name = digest({"session": plan["session"], "person": person})[:24] + ".html"
                for row in (x for x in page_rows if x["participant"] == person):
                    path = folder / "images" / plan["cases"][row["case_id"]]["display_image"]
                    raw = path.read_bytes()
                    require(
                        sha256(raw).hexdigest() == row["image_sha256"],
                        "Original display image changed before embedding",
                    )
                    mime = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[path.suffix]
                    own.append({**row, "image": f"data:{mime};base64," + b64encode(raw).decode("ascii")})
                embedded = (
                    canonical({**page, "rows": own})
                    .decode()
                    .replace("<", "\\u003c")
                    .replace(">", "\\u003e")
                    .replace("&", "\\u0026")
                )
                (forms / name).write_text(
                    TEMPLATE.read_text(encoding="utf8").replace("{{REVIEW_DATA}}", embedded),
                    encoding="utf8",
                    newline="\n",
                )
                index.append(f'<li><a href="forms/{name}">{escape(person)}</a></li>')
            (review / "review.html").write_text(
                '<!doctype html><meta charset="utf-8"><title>Participant forms</title><h1>Participant forms</h1>'
                "<p>Give each participant only their own HTML file. It contains their original images and the model responses.</p><ul>"
                + "".join(index)
                + "</ul>",
                encoding="utf8",
                newline="\n",
            )
            write(
                review / "receipt.json",
                {
                    "files": {
                        p.relative_to(review).as_posix(): file_sha256(p)
                        for p in review.rglob("*")
                        if p.is_file()
                    },
                    "native_run_sha256": file_sha256(native / "run.json"),
                    "round_plan_sha256": file_sha256(r / "plan.json"),
                    "attempt_sha256": file_sha256(attempt),
                },
            )
        return {"round": d["round"], "model_calls": len(rows), "awaiting_explicit_decisions": len(page_rows)}


def verified_review(folder, r):
    d = load_definition(folder, r)
    base = r / "review"
    receipt = read(base / "receipt.json")
    native = r / "native"
    run = read(native / "run.json")
    require(
        all(file_sha256(base / name) == expected for name, expected in receipt["files"].items())
        and file_sha256(native / "run.json") == receipt["native_run_sha256"]
        and all(file_sha256(native / name) == run[key] for name, key in converter.TASK5_RUN_FILES)
        and file_sha256(r / "plan.json") == receipt["round_plan_sha256"]
        and file_sha256(r / "attempt.json") == receipt["attempt_sha256"]
        and file_sha256(native / "input-snapshot.jsonl") == d["requests_sha256"],
        "A review or its raw model run changed",
    )
    display = read(base / "display.json")
    plan = read(folder / "session.json")
    for row in display["rows"]:
        info = plan["cases"][row["case_id"]]
        image = folder / "images" / info["display_image"]
        require(
            file_sha256(image) == info["image_sha256"] == row["image_sha256"]
            and (base / row["image"]).resolve() == image.resolve(),
            "The original displayed photograph changed",
        )
    return display, grader.read_jsonl(base / "turns.jsonl"), read(base / "provenance.json")


def checked_decisions(display, paths):
    eligible = {r["case_id"]: r for r in display["rows"]}
    selected = {}
    originals = {}
    for path in paths:
        require(path.stat().st_size <= 4 * 1024**2, "Review return is too large")
        raw = path.read_bytes()
        doc = json.loads(raw)
        token = sha256(raw).hexdigest()
        originals[token] = raw
        require(
            set(doc)
            == {
                "kind",
                "session",
                "round",
                "form",
                "participant",
                "exported_utc",
                "software_fixture",
                "consent",
                "attestation",
                "decisions",
            }
            and doc["kind"] == "native_clarification_human_decisions"
            and doc["session"] == display["session"]
            and doc["form"] == display["form"]
            and type(doc["round"]) is int
            and doc["round"] == display["round"]
            and doc["software_fixture"] is display["software_fixture"]
            and doc["consent"] is True
            and doc["attestation"] == display["attestation"]
            and isinstance(doc["decisions"], list)
            and doc["decisions"],
            "Wrong or unconsented review return",
        )
        when = moment(doc["exported_utc"])
        require(moment(display["created_utc"]) <= when, "Decision predates the displayed model turn")
        require(
            when <= datetime.now(timezone.utc) + timedelta(minutes=5),
            "Decision timestamp is later than import with the five-minute clock tolerance",
        )
        for decision in doc["decisions"]:
            require(
                isinstance(decision, dict)
                and set(decision) == {"case_id", "turn_id", "proposal_id", "action", "reply_id"},
                "Invalid decision fields",
            )
            case = decision["case_id"]
            source = eligible.get(case)
            require(
                source
                and case not in selected
                and doc["participant"] == source["participant"]
                and decision["turn_id"] == source["turn_id"]
                and decision["proposal_id"] == source["proposal_id"],
                "A decision is duplicate, foreign, stale, or belongs to another proposal or participant",
            )
            action = decision["action"]
            require(action in ("confirm", "reply", "finish"), "Unknown decision")
            require(
                action != "confirm" or bool(source["proposal_id"]),
                "Only this currently displayed proposal can be confirmed",
            )
            options = {v["reply_id"]: v for v in source["reply_options"]}
            require(
                (action == "reply" and decision["reply_id"] in options)
                or (action != "reply" and decision["reply_id"] is None),
                "Choose only a registered, case-bound reply for this turn type and remaining budget",
            )
            selected[case] = {
                **decision,
                "participant": doc["participant"],
                "returned_utc": doc["exported_utc"],
                "export_sha256": token,
                "reply_sha256": sha256(options[decision["reply_id"]]["text"].encode("utf8")).hexdigest()
                if action == "reply"
                else None,
            }
    require(
        set(selected) == set(eligible),
        "A designated participant decision is still missing. Keep the native turn unchanged.",
    )
    return selected, originals


def next_input(request, turn, reply):
    updated = deepcopy(request)
    f = request["scene"]["frame"]
    where = {"recording_id": f["recording_id"], "frame_id": f["frame_id"]}
    result = turn["outcome"]["result"]
    assistant = result.get("clarification_question") or canonical(result).decode()
    updated["dialogue"].extend(
        [{"role": "user", "text": request["text"], **where}, {"role": "assistant", "text": assistant, **where}]
    )
    updated["text"] = reply["text"]
    return validate_request(updated)


def saved_decisions(r):
    folder = r / "responses"
    registry = read(folder / "decisions.json")
    require(
        file_sha256(r / "review/receipt.json") == registry["review_receipt_sha256"],
        "Response registry belongs to another review",
    )
    require(
        all(file_sha256(folder / (h + ".json")) == h for h in registry["original_exports"]),
        "Original participant return changed",
    )
    display = read(r / "review/display.json")
    decisions, _ = checked_decisions(display, [folder / (h + ".json") for h in registry["original_exports"]])
    require(
        registry["decisions"] == decisions, "A stored participant decision differs from the actual human return"
    )
    return decisions


def finish(folder, plan):
    episodes = {case: [] for case in plan["cases"]}
    setup = None
    for r in current(folder):
        _, turns, provenance = verified_review(folder, r)
        decisions = saved_decisions(r)
        require(setup is None or setup == provenance, "Native model settings changed between turns")
        setup = provenance
        for t in turns:
            case = t["case_id"]
            require(t["turn"] == len(episodes[case]), "Missing earlier model turn for case")
            decision = decisions.get(case)
            if decision and decision["action"] == "confirm":
                require(
                    t.get("proposal_id") and decision["proposal_id"] == t["proposal_id"],
                    "Confirmation belongs to another proposal",
                )
                t["confirmed_proposal_id"] = decision["proposal_id"]
                t["confirmation_return_sha256"] = decision["export_sha256"]
            episodes[case].append(t)
    require(all(episodes.values()), "Missing initial cases")
    # The envelope records completion of episode collection. Native failures remain inside the exact turn that failed.
    records = [
        {
            "condition": {"model": plan["model"], "mode": "clarification", "transcript": "typed"},
            "case_id": case,
            "scene_sha256": info["scene_sha256"],
            "instruction_sha256": info["instruction_sha256"],
            "call": {"outcome": "ok", "error_code": None},
            "provenance": setup,
            "turns": episodes[case],
        }
        for case, info in plan["cases"].items()
    ]
    with staging(folder / "final") as final:
        result = final / "outputs.jsonl"
        write_rows(result, records)
        accepted, refused = grader.load_outputs([result], plan["cases"])
        if plan["software_fixture"]:
            require(
                len(refused) == len(records), "Software-test episodes must remain refused in formal scoring"
            )
        else:
            require(
                not refused and sum(len(v) for v in accepted.values()) == len(records),
                "The native evaluator refused an episode",
            )
        write(
            final / "receipt.json",
            {
                "protocol_sha256": plan["protocol_sha256"],
                "outputs_sha256": file_sha256(result),
                "cohort": len(records),
                "genuine_human_confirmations": 0
                if plan["software_fixture"]
                else sum("confirmed_proposal_id" in t for ts in episodes.values() for t in ts),
                "model_turns": sum(map(len, episodes.values())),
                "software_fixture": plan["software_fixture"],
                "scored_against_targets": False,
                "completed_utc": stamp(),
            },
        )
    return read(folder / "final/receipt.json")


def advance(folder, returns=()):
    with locked(folder):
        plan = load_session(folder)
        r = current(folder)[-1]
        d = load_definition(folder, r)
        require(not (folder / "final").exists(), "The native episodes have already been sealed")
        display, turns, _ = verified_review(folder, r)
        if (r / "responses").exists():
            existing = read(r / "responses/decisions.json")
            require(
                not returns or {file_sha256(p) for p in returns} == set(existing["original_exports"]),
                "This round already has sealed decisions",
            )
        else:
            selected, originals = checked_decisions(display, returns)
            with staging(r / "responses") as saved:
                for h, data in originals.items():
                    (saved / (h + ".json")).write_bytes(data)
                write(
                    saved / "decisions.json",
                    {
                        "decisions": selected,
                        "original_exports": sorted(originals),
                        "review_receipt_sha256": file_sha256(r / "review/receipt.json"),
                    },
                )
        selected = saved_decisions(r)
        registered = {(v["case_id"], v["reply_id"]): v for v in read(folder / "replies.json")}
        turn_by_case = {t["case_id"]: t for t in turns}
        later = []
        incoming = {}
        for req in read_requests(r / "requests.jsonl"):
            case = req["request_id"]
            decision = selected.get(case)
            if decision and decision["action"] == "reply":
                reply = registered[case, decision["reply_id"]]
                reply_hash = sha256(reply["text"].encode("utf8")).hexdigest()
                require(reply_hash == decision["reply_sha256"], "Registered reply changed")
                later.append(next_input(req, turn_by_case[case], reply))
                incoming[case] = {
                    "reply_id": reply["reply_id"],
                    "reply_sha256": reply_hash,
                    "passenger_return_sha256": decision["export_sha256"],
                }
        if later:
            require(d["round"] + 1 < plan["turn_budget"], "No model call remains under the frozen budget")
            definition(folder, d["round"] + 1, later, incoming, file_sha256(r / "responses/decisions.json"))
            return {"next_round": d["round"] + 1, "actual_registered_replies": len(later)}
        return finish(folder, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    registrar = commands.add_parser(
        "freeze",
        help="Freeze the actual Task 1 protocol, human assignments and budget before any final outputs",
    )
    for key in ("references", "replies", "participants", "responder-rules", "out"):
        registrar.add_argument("--" + key, type=Path, required=True)
    registrar.add_argument("--turn-budget", type=int, required=True)
    registrar.add_argument("--evaluator-commit", required=True)
    registrar.add_argument("--voice", type=Path)
    registrar.add_argument(
        "--model-conditions",
        type=Path,
        default=TASK5_MODEL_PAIR,
        help="Pre-output model registration (defaults to the published Task 5 pretrained/locally reviewed pair)",
    )
    init = commands.add_parser("prepare")
    for key in ("protocol", "references", "replies", "participants", "scenes", "requests", "config", "out"):
        init.add_argument("--" + key, type=Path, required=True)
    for name in ("collect", "advance"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--session", type=Path, required=True)
        if name == "advance":
            cmd.add_argument("--return", dest="returns", type=Path, action="append", default=[])
    a = parser.parse_args()
    if a.action == "freeze":
        freeze(
            a.references,
            a.replies,
            a.participants,
            a.evaluator_commit,
            a.responder_rules.read_text(encoding="utf8"),
            a.turn_budget,
            a.out,
            a.voice,
            a.model_conditions,
        )
        print("Frozen protocol SHA256", grader.text_sha256(a.out))
    elif a.action == "prepare":
        p = prepare(a.protocol, a.references, a.replies, a.participants, a.scenes, a.requests, a.config, a.out)
        print("Prepared", len(p["cases"]), "cases with", p["model"], "and a budget of", p["turn_budget"])
    else:
        print(
            json.dumps(collect(a.session) if a.action == "collect" else advance(a.session, a.returns), indent=2)
        )


if __name__ == "__main__":
    main()
