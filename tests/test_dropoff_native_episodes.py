"""Only synthetic road scenes, stand-in inference, artificial decisions and the unchanged Task 1 grader."""

from copy import deepcopy
from dataclasses import asdict
import json
import subprocess

import pytest

from evaluation import grounding as official
from tests import test_grounding_evaluation as example
from tools import dropoff_native_episodes as episodes
from tools.dropoff_predict import predict
from trackrag.dropoff.model import GroundingConfig, GroundingModel

# Register the evaluator's original synthetic-scene fixture in this test module.
world = example.world


def predictor(raw):
    def call(requests, config, out):
        return predict(
            requests,
            config,
            out,
            adapter_factory=lambda cfg: GroundingModel(cfg, backend=example.StandInBackend(raw)),
        )

    return call


@pytest.fixture()
def study(tmp_path, world):
    folder = tmp_path / "episode"
    roster = tmp_path / "participants.json"
    replies = tmp_path / "replies.jsonl"
    requests = tmp_path / "requests.jsonl"
    config = tmp_path / "config.json"
    protocol = tmp_path / "protocol.json"
    episodes.write(roster, {case: "fixture-" + case for case in example.INSTRUCTIONS})
    vals = []
    for case in example.INSTRUCTIONS:
        vals += [
            {"case_id": case, "reply_id": case + "-answer", "kind": "answer", "text": "I mean the far side."},
            {
                "case_id": case,
                "reply_id": case + "-rejection",
                "kind": "rejection",
                "text": "I meant the far side instead.",
            },
        ]
    for row in vals:
        row["private_review_note"] = "example-private-answer-key-never-displayed"
    episodes.write_rows(replies, vals)
    inputs = example.task5_requests(world)
    for q in inputs:
        ref = world.references[q["request_id"]]
        s = world.scenes[ref["scene_sha256"]]["model_input"]
        q["scene"].setdefault("provenance", {})["image_sha256"] = s["image_sha256"]
    episodes.write_rows(requests, inputs)
    episodes.write(config, asdict(GroundingConfig()))
    evaluator = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=episodes.ROOT, text=True).strip()
    episodes.freeze(
        world.references_path,
        replies,
        roster,
        evaluator,
        "Test human chooses the displayed proposal or an exact registered reply.",
        2,
        protocol,
    )
    episodes.prepare(
        protocol,
        world.references_path,
        replies,
        roster,
        world.scenes_dir,
        requests,
        config,
        folder,
        _fixture=True,
    )
    return folder, replies, protocol


def human_files(folder, r, actions):
    view = episodes.read(r / "review/display.json")
    out = []
    for row in view["rows"]:
        case = row["case_id"]
        action = actions[case]
        d = {
            "case_id": case,
            "turn_id": row["turn_id"],
            "proposal_id": row["proposal_id"],
            "action": action,
            "reply_id": None,
        }
        if action == "reply":
            d["reply_id"] = row["reply_options"][0]["reply_id"]
        result = {
            "kind": "native_clarification_human_decisions",
            "session": view["session"],
            "round": view["round"],
            "form": view["form"],
            "participant": row["participant"],
            "exported_utc": episodes.stamp(),
            "software_fixture": True,
            "consent": True,
            "attestation": view["attestation"],
            "decisions": [d],
        }
        path = folder / f"fixture-return-{view['round']}-{case}.json"
        episodes.write(path, result)
        out.append(path)
    return out


def test_real_stage_layout_and_unchanged_grader_with_controlled_decisions(world, study):
    folder, replies, protocol = study
    first = folder / "round-00"
    result = episodes.collect(folder, _predict=predictor(example.TASK5_REPLIES))
    assert result["awaiting_explicit_decisions"] == 4
    initial = episodes.read(first / "review/display.json")
    assert all((first / "review" / r["image"]).is_file() for r in initial["rows"])
    assert all(
        set(reply) == set(episodes.REPLY_FIELDS) for row in initial["rows"] for reply in row["reply_options"]
    )
    assert "example-private-answer-key-never-displayed" not in (folder / "replies.json").read_text()
    assert all(
        "example-private-answer-key-never-displayed" not in p.read_text(encoding="utf8")
        for p in (first / "review").glob("*.html")
    )
    assert not any(
        word in (first / "review/display.json").read_text()
        for word in ("mask_sha256", "expected_status", "target_box", "intended_mask")
    )
    returns = human_files(folder, first, {"C1": "confirm", "C2": "reply", "A1": "reply", "M1": "finish"})
    assert episodes.advance(folder, returns) == {"next_round": 1, "actual_registered_replies": 2}
    second = folder / "round-01"
    actual = episodes.read_requests(second / "requests.jsonl")
    assert [q["request_id"] for q in actual] == ["C2", "A1"]
    assert actual[0]["text"] == "I meant the far side instead." and actual[1]["text"] == "I mean the far side."
    assert actual[1]["dialogue"][-1]["text"] == "Before or after the bus?"
    assert json.loads(actual[0]["dialogue"][-1]["text"])["status"] == "proposed"
    assert actual[0]["dialogue"][0]["text"] == example.INSTRUCTIONS["C2"]
    episodes.collect(
        folder,
        _predict=predictor(
            [example.normalised([4, 44], [0, 40, 8, 48]), example.normalised([60, 44], [50, 40, 70, 48])]
        ),
    )
    assert all(not row["reply_options"] for row in episodes.read(second / "review/display.json")["rows"])
    done = episodes.advance(folder, human_files(folder, second, {"C2": "finish", "A1": "confirm"}))
    assert (
        done["cohort"] == 4
        and done["model_turns"] == 6
        and done["genuine_human_confirmations"] == 0
        and done["software_fixture"]
    )
    rows = official.read_jsonl(folder / "final/outputs.jsonl")
    by_case = {r["case_id"]: r for r in rows}
    registered = official.load_replies(replies, world.references)
    for case, row in by_case.items():
        assert row["provenance"]["substitute"] is True
        ref = world.references[case]
        score = official.score_output(
            row,
            ref,
            world.scenes[ref["scene_sha256"]],
            registered,
            official.PROTOCOL_DEFAULTS,
            "clarification",
            ref["instruction_sha256"],
        )
        assert score["outcome"] in official.STATUSES and not score["reasons"]
        assert score["completed"] is (case in ("C1", "A1"))
        if case == "A1":
            assert score["final_point_in_mask"] and score["turns_used"] == 2
    assert by_case["C2"]["turns"][-1].get("confirmed_proposal_id") is None
    reloaded, refused = official.load_outputs([folder / "final/outputs.jsonl"], world.references)
    assert (
        all(row is None for condition in reloaded.values() for row in condition.values()) and len(refused) == 4
    )
    report, _ = official.evaluate(
        world.references_path,
        world.scenes_dir,
        folder / "final/outputs.jsonl",
        replies_path=replies,
        protocol_path=protocol,
        allow_stub_scenes=True,
    )
    assert report["reserved_cases"] == 4 and len(report["refused_outputs"]) == 4


@pytest.mark.parametrize(
    "change,reason",
    [
        ("proposal", "stale"),
        ("person", "participant"),
        ("unattested", "unconsented"),
        ("opposite", "registered"),
        ("foreign_form", "Wrong"),
        ("timestamp", "predates"),
        ("future", "later than import"),
        ("declaration", "unconsented"),
    ],
)
def test_human_return_guards(study, change, reason):
    folder, _, _ = study
    r = folder / "round-00"
    episodes.collect(folder, _predict=predictor(example.TASK5_REPLIES))
    files = human_files(folder, r, {"C1": "confirm", "C2": "reply", "A1": "reply", "M1": "finish"})
    data = episodes.read(files[0])
    decision = data["decisions"][0]
    if change == "proposal":
        decision["proposal_id"] = "from-another-turn"
    elif change == "person":
        data["participant"] = "another-person"
    elif change == "unattested":
        data["consent"] = False
    elif change == "opposite":
        decision.update(action="reply", reply_id="C1-answer")
    elif change == "foreign_form":
        data["form"] = "another-form"
    elif change == "future":
        data["exported_utc"] = "2101-01-01T00:00:00Z"
    elif change == "declaration":
        data["attestation"] = episodes.ATTEST
    else:
        data["exported_utc"] = "2001-01-01T00:00:00Z"
    files[0].write_text(json.dumps(data))
    with pytest.raises(ValueError, match=reason):
        episodes.advance(folder, files)
    assert not (r / "responses").exists() and not (folder / "round-01").exists()


def test_keep_errors_and_require_actual_decision_for_every_other_case(study):
    folder, _, _ = study
    r = folder / "round-00"
    raw = deepcopy(example.TASK5_REPLIES)
    raw[-1] = "not json"
    episodes.collect(folder, _predict=predictor(raw))
    with pytest.raises(ValueError, match="decision is still missing"):
        episodes.advance(folder, [])
    items = human_files(folder, r, {"C1": "finish", "C2": "finish", "A1": "finish"})
    done = episodes.advance(folder, items)
    assert done["cohort"] == 4 and done["model_turns"] == 4
    records = {v["case_id"]: v for v in official.read_jsonl(folder / "final/outputs.jsonl")}
    assert records["M1"]["turns"][0]["outcome"]["error"]["code"] == "invalid_model_output"
    assert all("confirmed_proposal_id" not in x["turns"][0] for x in records.values())


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("provenance", "provenance disagree"),
        ("status", "Successful native"),
        ("result", "Successful native"),
        ("raw_result", "Raw native text"),
        ("prompt", "Raw native messages"),
        ("error", "Failed native"),
        ("backend_counts", "error counts"),
    ],
)
def test_authenticated_but_inconsistent_raw_native_files_cannot_reach_a_reviewer(study, mode, message):
    folder, _, _ = study

    def inconsistent(requests, config, out):
        raw = list(example.TASK5_REPLIES)
        raw[-1] = TimeoutError("controlled deadline")
        predictor(raw)(requests, config, out)
        receipt_path = out / "receipts.jsonl"
        recs = official.read_jsonl(receipt_path)
        if mode == "provenance":
            recs[0]["provenance"]["injected"] = False
        elif mode == "status":
            recs[0]["status"] = "error"
        elif mode == "result":
            recs[0]["result"] = recs[1]["result"]
        elif mode == "raw_result":
            recs[0]["raw_text"] = recs[1]["raw_text"]
        elif mode == "prompt":
            recs[0]["messages"] = recs[1]["messages"]
        elif mode == "error":
            recs[-1]["error"]["message"] = "inconsistent"
        receipt_path.write_bytes(b"".join(episodes.canonical(r) + b"\n" for r in recs))
        summary = episodes.read(out / "run.json")
        summary["receipts_sha256"] = episodes.file_sha256(receipt_path)
        if mode == "backend_counts":
            summary["operational_errors"] = 0
        (out / "run.json").write_bytes(episodes.canonical(summary))

    with pytest.raises(ValueError, match=message):
        episodes.collect(folder, _predict=inconsistent)
    assert not (folder / "round-00/review").exists() and (folder / "round-00/native/run.json").is_file()


def test_genuine_invalid_timeout_and_resource_outcomes_stay_in_episode_denominator(study):
    folder, _, _ = study
    raw = [
        example.TASK5_REPLIES[0],
        "unparseable",
        TimeoutError("fixture deadline"),
        RuntimeError("out of memory"),
    ]
    episodes.collect(folder, _predict=predictor(raw))
    first = folder / "round-00"
    receipt = episodes.advance(folder, human_files(folder, first, {"C1": "finish"}))
    assert receipt["cohort"] == 4 and receipt["model_turns"] == 4
    rows = official.read_jsonl(folder / "final/outputs.jsonl")
    assert [r["turns"][0]["outcome"].get("error", {}).get("code") for r in rows].count(
        "invalid_model_output"
    ) == 1
    assert {r["turns"][0]["outcome"].get("error", {}).get("code") for r in rows} == {
        None,
        "invalid_model_output",
        "timeout",
        "resource_unavailable",
    }


def test_changed_frozen_registry_or_native_run_is_refused(study):
    folder, replies, _ = study
    data = replies.read_bytes()
    replies.write_bytes(data + b"\n")
    with pytest.raises(ValueError, match="frozen file"):
        episodes.collect(folder, _predict=predictor(example.TASK5_REPLIES))
    replies.write_bytes(data)
    episodes.collect(folder, _predict=predictor(example.TASK5_REPLIES))
    r = folder / "round-00"
    source = r / "native" / "receipts.jsonl"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="raw model run changed"):
        episodes.advance(folder)


def test_display_photographs_are_content_bound_at_decision_time(study):
    folder, _, _ = study
    r = folder / "round-00"
    episodes.collect(folder, _predict=predictor(example.TASK5_REPLIES))
    display = episodes.read(r / "review/display.json")
    original = (r / "review" / display["rows"][0]["image"]).resolve()
    original.write_bytes(bytes(len(original.read_bytes())))
    with pytest.raises(ValueError, match="displayed photograph changed"):
        episodes.advance(folder)


def test_round_refuses_staged_preexisting_results_and_stale_generation_time(study):
    folder, _, _ = study
    r = folder / "round-00"
    (r / "native").mkdir()
    with pytest.raises(ValueError, match="not collected within"):
        episodes.collect(folder, _predict=predictor(example.TASK5_REPLIES))
    assert not (r / "attempt.json").exists()
    (r / "native").rmdir()

    def bad_clock(req, conf, out):
        predictor(example.TASK5_REPLIES)(req, conf, out)
        run = episodes.read(out / "run.json")
        run["created_at"] = "2000-01-01T00:00:00Z"
        (out / "run.json").write_text(json.dumps(run))

    with pytest.raises(ValueError, match="predates this registered round"):
        episodes.collect(folder, _predict=bad_clock)
    assert (r / "native/run.json").is_file() and not (r / "review").exists()


def test_freezer_binds_selected_config_and_trained_adapter_before_use(study, world, tmp_path):
    _, replies, original_protocol = study
    roster = tmp_path / "participants.json"
    config = tmp_path / "config.json"
    with pytest.raises(ValueError, match="selected pretrained and trained pair"):
        episodes.prepare(
            original_protocol,
            world.references_path,
            replies,
            roster,
            world.scenes_dir,
            tmp_path / "requests.jsonl",
            config,
            tmp_path / "production-without-pair",
        )
    registry = tmp_path / "models.json"
    registered = {
        "pretrained": {"config_sha256": "f" * 64},
        "trained": {"config_sha256": "e" * 64, "adapter_manifest_sha256": "a" * 64},
    }
    episodes.write(registry, registered)
    protocol = tmp_path / "pair-protocol.json"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=episodes.ROOT, text=True).strip()
    episodes.freeze(
        world.references_path,
        replies,
        roster,
        commit,
        "Controlled fixture rules.",
        2,
        protocol,
        model_conditions=registry,
    )
    with pytest.raises(ValueError, match="registered model pair"):
        episodes.prepare(
            protocol,
            world.references_path,
            replies,
            roster,
            world.scenes_dir,
            tmp_path / "requests.jsonl",
            config,
            tmp_path / "wrong-pair",
            _fixture=True,
        )
