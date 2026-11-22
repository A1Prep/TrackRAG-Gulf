"""Software-only ballots, artificial pictures and dates. No genuine reviewers or passenger labels."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from tools import dropoff_passenger_review_intake as intake
from tools.dropoff_passenger_references_pack import review_inputs
from trackrag.dropoff.data import inference_input_sha256, load_cases


A = "Software Fixture Alpha"
B = "Software Fixture Beta"
C = "Software Fixture Adjudicator"
WHEN = "2026-01-01T09:00:00+00:00"
LATER = "2026-01-01T10:00:00+00:00"
STATUSES = ("proposed", "needs_clarification", "insufficient_visual_evidence")


def target(status):
    return {"status": status, "destination_point_xy": [10, 20] if status == "proposed" else None,
            "destination_box_xyxy": [5, 15, 15, 25] if status == "proposed" else None,
            "clarification_question": "Which displayed fixture?" if status == "needs_clarification" else None}


def put(path, value):
    intake.write_json(path, value)
    return path


@pytest.fixture
def environment(tmp_path, monkeypatch):
    returns = tmp_path / "returns"
    returns.mkdir()
    banks = {}
    for partition, (recording, prefix, _) in intake.PARTITIONS.items():
        bank = sha256(partition.encode()).hexdigest()
        rows, requests = {}, {}
        for i, status in enumerate(STATUSES, 1):
            identity = f"software-fixture-{prefix}{i}"
            image = tmp_path / (identity + ".jpg")
            Image.new("RGB", (64, 40), (i * 40, 60 if partition == "train" else 180, 90)).save(image)
            image_sha = sha256(image.read_bytes()).hexdigest()
            frame = {"recording_id": recording, "frame_id": i, "image_path": str(image),
                     "timestamp_s": float(i), "width": 64, "height": 40}
            req = {"request_id": identity, "text": "Only a software fixture", "input_mode": "typed", "dialogue": [],
                   "scene": {"frame": frame, "source": "predicted", "provenance": {"mode": "software_fixture"}, "tracks": []}}
            rows[identity] = {"request_id": identity, "recording_id": recording, "frame_id": i, "image_sha256": image_sha,
                              "input_sha256": inference_input_sha256(req, image_sha)}
            requests[identity] = req
        banks[partition] = (bank, rows, requests)
    monkeypatch.setattr(intake, "review_inputs", lambda bank, path, *, partition: banks[partition])
    def export(partition="train", name=A, indexes=(0, 1, 2)):
        bank, rows, _ = banks[partition]
        selected = [list(rows.values())[i] for i in indexes]
        completed = [{**r, "reviewer": name, "reviewed_at": WHEN, "notes": "Generated software example",
                      "target": target(STATUSES[i])} for r, i in zip(selected, indexes)]
        return {"kind": "independent_passenger_review", "version": 2, "partition": partition,
                "source": "software_fixture", "human_attested": False, "reviewer": name,
                "input_bank_sha256": bank, "expected_cases": len(rows), "completed": completed}
    def run(name="snapshot", *, fixture=True, **kwargs):
        return intake.ingest(tmp_path, dict.fromkeys(banks, tmp_path), returns, tmp_path / name,
                             _software_fixture=fixture, **kwargs)
    return SimpleNamespace(root=tmp_path, returns=returns, banks=banks, export=export, run=run)


def test_zero_is_pending_and_immutable(environment):
    e = environment
    rec = e.run(fixture=False)
    assert rec["attested_human_exports"] == rec["eligible_human_cases"] == 0
    assert not rec["reviewed_files"] and not rec["training_loader_validated"]
    assert not (e.root / "snapshot/train.jsonl").exists()
    assert rec["partitions"]["development"]["resolutions"] == {"awaiting_review": 3}
    with pytest.raises(FileExistsError):
        e.run()


def test_frozen_pack_rejects_other_or_swapped_recordings(tmp_path):
    bank = Path(__file__).resolve().parents[1] / "annotations/dropoff/passenger-development"
    with pytest.raises(ValueError, match="nonfinal"):
        review_inputs(bank, None, partition="test")
    req = {"request_id": "software-fixture", "input_mode": "typed", "dialogue": [], "text": "Fixture",
           "scene": {"frame": {"recording_id": "software-fixture", "frame_id": 1, "image_path": "unused.jpg",
                               "timestamp_s": 0, "width": 64, "height": 40}, "tracks": [], "source": "predicted", "provenance": {}}}
    intake.write_rows(tmp_path / "requests.jsonl", [req])
    with pytest.raises(ValueError, match="complete development"):
        review_inputs(bank, tmp_path / "requests.jsonl", partition="development")


def test_full_software_fixture_through_actual_local_continuation_cpu_preflight(environment, monkeypatch):
    from tools import dropoff_adapt_local as adapt
    e = environment
    for partition in e.banks:
        for name in (A, B):
            put(e.returns / (name + partition + ".json"), e.export(partition, name))
    rec = e.run()
    assert rec["training_loader_validated"] and rec["eligible_human_cases"] == 0
    assert rec["attested_human_exports"] == 0 and rec["software_fixture"]
    assert rec["partitions"]["train"]["resolutions"] == {"exact_agreement": 3}
    train = e.root / "snapshot/train.jsonl"
    dev = e.root / "snapshot/development.jsonl"
    rows = load_cases(train)
    assert all(r["review"]["software_fixture"] and r["review"]["reviewers"] == [A, B] for r in rows)
    assert all(len(r["review"]["export_sha256"]) == 2 for r in rows)
    assert len(load_cases(dev, allowed_partitions=("development",))) == 3
    # Only parent-run artifact discovery is substituted. The continuation's real data checks, source
    # encoding, status/recording/parent separation and software-fixture counters are exercised.
    parent = {"case_artifacts": [], "train_scene_groups": [], "train_case_ids": [],
              "development_scene_groups": [], "development_case_ids": []}
    monkeypatch.setattr(adapt, "parent_context", lambda *_: (adapt.GroundingConfig(), parent, {"kind": "software_fixture"}))
    result = adapt.validate(SimpleNamespace(parent_run=None, config=None, train=train, development=dev))
    assert result["valid"] and result["software_fixture_cases"] == 6 and result["development_has_all_statuses"]
    assert not result["gpu_fit_verified"] and not result["local_quality_verified"]
    from tools import dropoff_train as generic
    monkeypatch.setattr(generic, "preflight", lambda *_: {"ready": True})
    with pytest.raises(ValueError, match="Software fixtures cannot be used"):
        generic.train(train, dev, generic.TrainConfig(run_kind="adaptation"), e.root / "never-trained", e.root / "unused-cache")
    assert not (e.root / "never-trained").exists()


def test_duplicates_partial_exports_and_one_review_are_not_two_reviewers(environment):
    e = environment
    first = e.export(indexes=(0, 1))
    put(e.returns / "part.json", first)
    put(e.returns / "copy.json", first)
    put(e.returns / "later.json", e.export(name="  SOFTWARE   Fixture Alpha  ", indexes=(1, 2)))
    rec = e.run()
    assert rec["returned_distinct_exports"] == 2
    resolutions = rec["partitions"]["train"]["resolutions"]
    assert resolutions == {"single_review": 2, "awaiting_second_clarification_review": 1}
    assert rec["partitions"]["train"]["eligible_cases"] == 2 and rec["conflicts_seen"] == 0
    assert not rec["training_loader_validated"]


def test_conflicting_boxes_remain_individual_until_a_signed_exact_choice(environment):
    e = environment
    one = e.export(indexes=(0,))
    two = e.export(name=B, indexes=(0,))
    two["completed"][0]["target"]["destination_point_xy"] = [12, 21]
    two["completed"][0]["target"]["destination_box_xyxy"] = [7, 17, 19, 26]
    put(e.returns / "a.json", one)
    put(e.returns / "b.json", two)
    rec = e.run()
    assert rec["conflicts_seen"] == 1 and not rec["reviewed_files"]
    ledger = [json.loads(l) for l in (e.root / "snapshot/individual-responses.jsonl").read_text().splitlines()]
    assert len(ledger[0]["individual_answers"]) == 2 and ledger[0]["selected_supervision_sha256"] is None
    sheet = json.loads((e.root / "snapshot/adjudication-template.json").read_bytes())
    sheet.update(source="software_fixture", human_attested=False, adjudicator=C, reviewed_at=LATER)
    row = sheet["decisions"][0]
    pick = next(c for c in row["choices"] if c["reviewer"] == B)
    row.update(chosen_supervision_sha256=pick["supervision_sha256"], reason="Explicit software choice")
    path = put(e.root / "choice.json", sheet)
    rec = e.run("resolved", decisions=path)
    assert rec["conflicts_adjudicated"] == 1
    case = load_cases(e.root / "resolved/train.jsonl")[0]
    assert case["target"] == two["completed"][0]["target"]
    assert set(case["review"]["reviewers"]) == {B, C} and A not in case["review"]["reviewers"]
    assert set(case["review"]["independent_reviewers"]) == {A, B}
    assert case["review"]["independent_supporters_before_adjudication"] == [B]
    assert case["review"]["resolution"] == "human_adjudicated" and case["review"]["software_fixture"]
    for name, changed in (("invented", {"chosen_supervision_sha256": "0" * 64}),
                          ("changed", {"review_set_sha256": "0" * 64}), ("reason-empty", {"reason": ""})):
        bad = deepcopy(sheet)
        bad["decisions"][0].update(changed)
        with pytest.raises(ValueError):
            e.run(name, decisions=put(e.root / (name + ".json"), bad))
        assert not (e.root / name).exists()


def test_disagreeing_clarification_wording_needs_second_supporter_for_chosen_text(environment):
    e = environment
    a, b = e.export(indexes=(1,)), e.export(name=B, indexes=(1,))
    b["completed"][0]["target"]["clarification_question"] = "Which synthetic reference on the right?"
    put(e.returns / "a.json", a)
    put(e.returns / "b.json", b)
    assert e.run()["conflicts_seen"] == 1
    sheet = json.loads((e.root / "snapshot/adjudication-template.json").read_bytes())
    row = sheet["decisions"][0]
    pick = next(c for c in row["choices"] if c["reviewer"] == A)
    row.update(chosen_supervision_sha256=pick["supervision_sha256"], reason="Signed software selection")
    sheet.update(source="software_fixture", human_attested=False, adjudicator=A, reviewed_at=LATER)
    rec = e.run("one-signer", decisions=put(e.root / "one.json", sheet))
    assert not rec["reviewed_files"]
    assert rec["partitions"]["train"]["resolutions"]["awaiting_second_clarification_review"] == 1
    sheet["adjudicator"] = C
    e.run("endorsed", decisions=put(e.root / "two.json", sheet))
    case = load_cases(e.root / "endorsed/train.jsonl")[0]
    assert set(case["review"]["reviewers"]) == {A, C} and case["target"] == a["completed"][0]["target"]


def test_legacy_export_archived_until_named_digest_attestation(environment):
    e = environment
    original = e.export(indexes=(0,))
    legacy = {key: original[key] for key in intake.EXPORT_KEYS}
    legacy["version"] = 1
    original_path = put(e.returns / "old.json", legacy)
    rec = e.run()
    assert rec["unattested_legacy_exports"] == 1 and rec["reviewed_files"] == {}
    sha = sha256(original_path.read_bytes()).hexdigest()
    assert (e.root / "snapshot/original-exports" / (sha + ".json")).read_bytes() == original_path.read_bytes()
    confirm = json.loads((e.root / "snapshot/legacy-confirmations-template.json").read_bytes())
    confirm["confirmations"][0].update(source="software_fixture", human_attested=False, reviewed_at=LATER)
    rec = e.run("signed", legacy=put(e.root / "legacy.json", confirm))
    assert rec["partitions"]["train"]["resolutions"]["single_review"] == 1
    confirm["confirmations"][0]["reviewer"] = B
    with pytest.raises(ValueError, match="original reviewer"):
        e.run("wrong", legacy=put(e.root / "wrong.json", confirm))


def test_adjudicator_is_not_a_second_independent_reader_for_two_versions_from_one_person(environment):
    e = environment
    first = e.export(indexes=(1,))
    revised = deepcopy(first)
    revised["completed"][0]["target"]["clarification_question"] = "Which artificial landmark?"
    put(e.returns / "first.json", first)
    put(e.returns / "revised.json", revised)
    assert e.run()["conflicts_seen"] == 1
    sheet = json.loads((e.root / "snapshot/adjudication-template.json").read_bytes())
    row = sheet["decisions"][0]
    assert len(row["choices"]) == 2
    row.update(chosen_supervision_sha256=row["choices"][0]["supervision_sha256"], reason="Software choice")
    sheet.update(source="software_fixture", human_attested=False, adjudicator=C, reviewed_at=LATER)
    final = e.run("same-person", decisions=put(e.root / "decision.json", sheet))
    assert final["reviewed_files"] == {}
    assert final["partitions"]["train"]["resolutions"]["awaiting_second_clarification_review"] == 1


@pytest.mark.parametrize("kind", ["recording", "frame", "bank", "partition", "image", "input", "duplicate",
                                  "nonfinite", "rightedge", "pending_geometry", "foreign_case", "ai", "wrong_signer"])
def test_malformed_misbound_and_nonhuman_inputs_fail_closed(environment, kind):
    e = environment
    ex = e.export(indexes=(0,))
    row = ex["completed"][0]
    if kind == "recording": row["recording_id"] = "foreign-fixture-recording"
    if kind == "frame": row["frame_id"] = True
    if kind == "bank": ex["input_bank_sha256"] = "0" * 64
    if kind == "partition": ex["partition"] = "development"
    if kind == "image": row["image_sha256"] = "0" * 64
    if kind == "input": row["input_sha256"] = "0" * 64
    if kind == "duplicate": ex["completed"].append(deepcopy(row))
    if kind == "nonfinite": row["target"]["destination_point_xy"][0] = float("nan")
    if kind == "rightedge": row["target"]["destination_point_xy"][0] = row["target"]["destination_box_xyxy"][2]
    if kind == "pending_geometry": row["target"]["status"] = "insufficient_visual_evidence"
    if kind == "foreign_case": row["request_id"] = "outside-nonfinal-pack"
    if kind == "ai": ex["source"] = "ai"
    if kind == "wrong_signer": row["reviewer"] = B
    (e.returns / "invalid.json").write_text(json.dumps(ex), encoding="utf-8")
    with pytest.raises(ValueError):
        e.run()
    assert not (e.root / "snapshot").exists()


def test_declared_fixtures_are_never_human_returns_and_fake_human_name_is_rejected(environment):
    e = environment
    ex = e.export(indexes=(0,))
    put(e.returns / "test.json", ex)
    with pytest.raises(ValueError, match="attestation"):
        e.run(fixture=False)
    ex.update(source="human", human_attested=True)
    put(e.returns / "test.json", ex)
    with pytest.raises(ValueError, match="cannot enter"):
        e.run(fixture=False)
