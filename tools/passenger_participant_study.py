"""Make a private, offline participant form and convert genuine pre-proposal intentions and replies."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from evaluation.grounding import instruction_sha256, load_mask, load_replies, mask_box, in_mask
from tools.dropoff_predict import read_requests
from trackrag.dropoff.data import inference_input_sha256


SOURCE_SHA = "e6006e632dd75c7d08866f860ee6fa4c6d806f519c1569aba683cb58aa96a0ef"
DISPLAY_SHA = "971af7ddf56ee695204b84cfb254e0d4ccb05d62a6376cfda5833fa106e9f105"
SOURCE_TOTAL = 44
DEFAULT_SOURCE = Path("runs/passenger-final-inputs-20261122/requests-with-saved-tracks.jsonl")
TEMPLATE = Path(__file__).with_suffix(".html")
ATTEST = ("I am the human participant identified here. I entered my own intention and replies, "
          "using only the original image and request, before seeing any model proposal or reference answer.")
FORM_VERSION = 1
BINDINGS = ("case_id", "recording_id", "frame_id", "image_sha256", "request_sha256", "instruction_sha256")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return sha256(data).hexdigest()


@contextmanager
def new_directory(out):
    out = Path(out)
    require(not out.exists(), "Choose a new output directory; previous forms and returns are preserved.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as temp:
        staged = Path(temp)/"output"
        staged.mkdir()
        yield staged
        require(not out.exists(), "The chosen output was created by another process.")
        staged.rename(out)


def json_file(path, data):
    path.write_bytes(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n")


def write_lines(path, rows):
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")


def plain_source(requests, *, _fixture=False):
    requests = Path(requests)
    bank_hash = digest(requests.read_bytes().replace(b"\r\n", b"\n"))
    require(_fixture or bank_hash == SOURCE_SHA, "Use the frozen input-only 44-case saved-tracks request bank.")
    rows, paths = {}, {}
    for req in read_requests(requests):
        scene, case = req["scene"], req["request_id"]
        frame = scene["frame"]
        original = Path(frame["image_path"])
        image_hash = scene["provenance"]["image_sha256"]
        if image_hash not in paths:
            raw = original.read_bytes()
            require(len(raw) <= 16 * 1024**2 and digest(raw) == image_hash, "Original image differs from the request.")
            with Image.open(original) as image:
                require(image.format == "JPEG" and image.size == (frame["width"], frame["height"]),
                        "Original JPEG dimensions differ from the request.")
            paths[image_hash] = original
        rows[case] = {"case_id":case, "recording_id":frame["recording_id"], "frame_id":frame["frame_id"],
                      "image_sha256":image_hash, "request_sha256":inference_input_sha256(req, image_hash),
                      "instruction_sha256":instruction_sha256(req["text"],req["dialogue"]),
                      "width":frame["width"], "height":frame["height"], "text":req["text"], "dialogue":req["dialogue"]}
    require(_fixture or len(rows) == len(paths) == SOURCE_TOTAL, "Incomplete original 44-case input bank.")
    require(_fixture or digest(canonical(list(rows.values())))==DISPLAY_SHA, "The blinded display inputs changed.")
    return bank_hash, rows, paths


def prepare(requests, out, *, _fixture=False):
    bank_hash, rows, paths = plain_source(requests, _fixture=_fixture)
    with new_directory(out) as directory:
        photos = directory / "images"
        photos.mkdir()
        for image_hash, path in paths.items():
            raw = path.read_bytes()
            require(digest(raw) == image_hash, "Original image changed before the form was prepared.")
            (photos / (image_hash+".jpg")).write_bytes(raw)
        data = {"version":FORM_VERSION, "input_bank_sha256":bank_hash, "rows":list(rows.values()),
                "declaration":ATTEST, "software_fixture":_fixture}
        payload = canonical(data).decode().replace("<","\\u003c").replace(">","\\u003e").replace("&","\\u0026")
        page = TEMPLATE.read_text(encoding="utf-8").replace("{{PARTICIPANT_STUDY}}", payload)
        (directory / "review.html").write_bytes(page.encode())
        receipt = {"kind":"blank_participant_intentions_and_replies", "prepared_utc":datetime.now(timezone.utc).isoformat(),
                   "input_bank_sha256":bank_hash, "form_sha256":digest(page.encode()), "version":FORM_VERSION,
                   "cases":len(rows), "original_jpegs":len(paths), "hidden_targets_read":False,
                   "model_proposals_shown":False,"detector_tracks_shown":False,"confirms_rebound_task2_context":False,
                   "participant_returns":0, "software_fixture":_fixture}
        json_file(directory / "blank-receipt.json", receipt)
    return receipt


def verify_pack(folder, *, _fixture=False):
    """Check a copied offline form, including the actual photographs, without private source paths."""
    directory = Path(folder)
    require({p.name for p in directory.iterdir()}=={"review.html","images","blank-receipt.json"},
            "A blind form folder must contain only review.html, images and blank-receipt.json.")
    page = (directory/"review.html").read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="study-data">(.*?)</script>',page,flags=re.S)
    require(match is not None, "The form's input registry is missing.")
    data=json.loads(match[1])
    require(data["version"]==FORM_VERSION and data["software_fixture"] is _fixture
            and data["declaration"]==ATTEST, "Unexpected form version, source or declaration.")
    require(_fixture or (data["input_bank_sha256"]==SOURCE_SHA and digest(canonical(data["rows"]))==DISPLAY_SHA),
            "Display inputs differ from the original registered 44.")
    require(TEMPLATE.read_text(encoding="utf-8").replace("{{PARTICIPANT_STUDY}}",match[1])==page,
            "Form controls differ from this checkout.")
    rows=data["rows"]
    images={r["image_sha256"]:(r["width"],r["height"]) for r in rows}
    require(all(re.fullmatch(r"[0-9a-f]{64}",k) for k in images), "Invalid original image digest.")
    require({p.name for p in (directory/"images").iterdir()}=={s+".jpg" for s in images}, "Unexpected or missing original JPEG.")
    for value, size in images.items():
        path=directory/"images"/(value+".jpg")
        require(digest(path.read_bytes())==value, "An original photograph changed after form preparation.")
        with Image.open(path) as image:
            require(image.format=="JPEG" and image.size==size, "Original image dimensions changed.")
    receipt=json.loads((directory/"blank-receipt.json").read_bytes())
    require(set(receipt)=={"kind","prepared_utc","input_bank_sha256","form_sha256","version","cases","original_jpegs",
                           "hidden_targets_read","model_proposals_shown","detector_tracks_shown","confirms_rebound_task2_context",
                           "participant_returns","software_fixture"}
            and receipt["kind"]=="blank_participant_intentions_and_replies" and receipt["input_bank_sha256"]==data["input_bank_sha256"]
            and receipt["version"]==FORM_VERSION and receipt["software_fixture"] is _fixture
            and receipt["cases"]==len(rows) and receipt["original_jpegs"]==len(images)
            and all(receipt[k] is False for k in ("hidden_targets_read","model_proposals_shown","detector_tracks_shown","confirms_rebound_task2_context"))
            and receipt["form_sha256"]==digest(page.encode()) and receipt["participant_returns"]==0,
            "Blank form differs from its construction receipt.")
    return {"verified_cases":len(rows),"verified_original_photographs":len(images),"form_sha256":receipt["form_sha256"],
            "input_bank_sha256":data["input_bank_sha256"],"display_inputs_sha256":digest(canonical(rows)),
            "software_fixture":_fixture}


def text_value(value, limit, what):
    require(isinstance(value,str) and 0 < len(value.strip()) <= limit and value == value.strip(),
            f"Missing or invalid {what}.")
    return value


def instant(value):
    require(isinstance(value,str) and len(value) <= 48, "Invalid review timestamp.")
    parsed = datetime.fromisoformat(value.replace("Z","+00:00"))
    require(parsed.tzinfo is not None, "Review timestamps must have a time zone.")
    return parsed


def coord(value, width, height):
    require(isinstance(value,list) and len(value)==2 and all(type(v) is int for v in value)
            and 0 <= value[0] < width and 0 <= value[1] < height, "Expected an original-image pixel within bounds.")
    return tuple(value)


def polygon_mask(answer, width, height):
    raw = answer.get("polygon_xy")
    require(isinstance(raw,list) and 3 <= len(raw) <= 128, "Draw a region using 3 to 128 vertices.")
    polygon = [coord(v,width,height) for v in raw]
    require(len(set(polygon))==len(polygon), "The polygon repeats a vertex.")
    edges = list(zip(polygon,polygon[1:]+polygon[:1]))
    twice_area = sum(a[0]*b[1]-b[0]*a[1] for a,b in edges)
    require(twice_area != 0, "The polygon encloses no area.")
    def cross(a,b,p): return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
    def intersects(a,b,c,d):
        return (max(min(a[0],b[0]),min(c[0],d[0])) <= min(max(a[0],b[0]),max(c[0],d[0]))
                and max(min(a[1],b[1]),min(c[1],d[1])) <= min(max(a[1],b[1]),max(c[1],d[1]))
                and cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0)
    for i,(a,b) in enumerate(edges):
        for j in range(i+2,len(edges)):
            if i != 0 or j != len(edges)-1:
                require(not intersects(a,b,*edges[j]), "The drawn polygon crosses itself.")
    point = coord(answer.get("point_xy"),width,height)
    mask = Image.new("L",(width,height),0)
    ImageDraw.Draw(mask).polygon(polygon,fill=255)
    require(mask.getpixel(point)==255, "The participant's selected point lies outside the actual drawing.")
    return mask


def load_return(path, rows, bank_hash, *, fixture):
    path = Path(path)
    require(path.stat().st_size <= 4*1024**2, "Participant return is too large.")
    raw = path.read_bytes()
    result = json.loads(raw)
    require(isinstance(result,dict) and set(result)=={"kind","version","input_bank_sha256","participant_id","exported_utc",
            "source","software_fixture","human_attested","declaration","completed"}, "Unexpected participant export fields.")
    require(result.get("kind")=="passenger_preproposal_return" and result.get("version")==FORM_VERSION
            and result.get("input_bank_sha256")==bank_hash, "Wrong form, version or request bank.")
    require(result.get("software_fixture") is fixture and result.get("human_attested") is (not fixture)
            and result.get("source")==("software_fixture" if fixture else "human_participant")
            and result.get("declaration")==ATTEST, "A production return needs the participant's explicit human declaration.")
    person = text_value(result.get("participant_id"),80,"participant name or code")
    submitted = instant(result.get("exported_utc"))
    entries = result.get("completed")
    require(isinstance(entries,list) and 0 < len(entries) <= len(rows), "Return has no reviewed cases or too many entries.")
    seen = set()
    for item in entries:
        require(isinstance(item,dict) and set(item)==set(BINDINGS)|{"participant_id","input_bank_sha256","reviewed_utc",
                "choice","polygon_xy","point_xy","answer","rejection"}, "Unexpected fields in a participant case.")
        case = item.get("case_id")
        require(isinstance(case,str) and case in rows and case not in seen, "Unknown or duplicate participant case.")
        seen.add(case)
        source = rows[case]
        require(item.get("participant_id")==person and item.get("input_bank_sha256")==bank_hash
                and all(type(item.get(k)) is type(source[k]) and item[k]==source[k] for k in BINDINGS),
                "Participant, instruction or original image binding differs.")
        require(instant(item.get("reviewed_utc")) <= submitted, "Review is dated after its export.")
        require(item.get("choice") in ("drawn_area","landmark_missing"), "State an intention or missing landmark.")
        text_value(item.get("answer"),2000,"natural clarification answer")
        text_value(item.get("rejection"),2000,"natural correction or rejection")
        if item["choice"]=="drawn_area":
            polygon_mask(item,source["width"],source["height"]).close()
        else:
            require(item.get("polygon_xy") is None and item.get("point_xy") is None,
                    "A missing landmark cannot carry an invented region or point.")
    return digest(raw), result


def unresolved_cases(path, known):
    if path is None:
        return None
    values = [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]
    require(all(isinstance(v,dict) and set(v)=={"case_id"} and isinstance(v["case_id"],str) for v in values),
            "Supply independent evaluator IDs only, one JSON object with case_id per line.")
    ids = {v["case_id"] for v in values}
    require(len(ids)==len(values) and ids <= set(known), "The unresolved selection contains repeated or unknown IDs.")
    return ids


def intake(requests, returns, out, *, unresolved_ids=None, _fixture=False):
    bank_hash, rows, _ = plain_source(requests, _fixture=_fixture)
    unresolved = unresolved_cases(unresolved_ids, rows)
    responses, consumed, attestations = {}, {}, {}
    for path in returns:
        export_sha, returned = load_return(path, rows, bank_hash, fixture=_fixture)
        attestations[export_sha]={"export_sha256":export_sha, **{k:returned[k] for k in
                ("participant_id","exported_utc","input_bank_sha256","source","human_attested","software_fixture","declaration")}}
        for entry in returned["completed"]:
            case = entry["case_id"]
            require(case not in responses or responses[case]==entry,
                    "Overlapping returns disagree. Keep the originals and import a pre-designated participant's submission per case.")
            responses[case]=entry
            consumed.setdefault(case,set()).add(export_sha)
    with new_directory(out) as directory:
        mask_dir = directory/"masks"
        mask_dir.mkdir()
        candidates, intended, replies, originals = [], [], [], []
        for case, source in rows.items():
            if case not in responses:
                continue
            entry = responses[case]
            region = None
            if entry["choice"]=="drawn_area":
                name = "masks/" + digest(case.encode())[:24] + ".png"
                mask = polygon_mask(entry,source["width"],source["height"])
                mask.save(directory/name,format="PNG")
                bounds=list(mask.getbbox())
                mask.close()
                rebound=load_mask(directory/name)
                require(bounds==mask_box(rebound,source["width"],source["height"])
                        and in_mask(rebound,entry["point_xy"],source["width"],source["height"]),
                        "Saved drawing failed the independent evaluator's box or point check.")
                region={"mask_path":name,"mask_sha256":digest((directory/name).read_bytes()),
                        "destination_box_xyxy":bounds,"destination_point_xy":entry["point_xy"]}
            provenance={"participant_id":entry["participant_id"],"reviewed_utc":entry["reviewed_utc"],
                        "input_bank_sha256":bank_hash,"export_sha256":sorted(consumed[case]),
                        "individual_entry_sha256":digest(canonical(entry)),"source":"software_fixture" if _fixture else "human_participant",
                        "human_self_attested":not _fixture, "detector_tracks_shown":False, "confirms_rebound_task2_context":False,
                        **{k:source[k] for k in BINDINGS}}
            originals.append({**provenance,"participant_response":entry,"drawing":region})
            for kind in ("answer","rejection"):
                replies.append({"case_id":case,"reply_id":case+"-"+kind,"kind":kind,"text":entry[kind]})
            if unresolved is not None and case in unresolved:
                intended.append({"case_id":case,"category":"ambiguous" if region else "missing_reference",
                                 "intended_mask":region["mask_path"] if region else None,
                                 "intended_mask_sha256":region["mask_sha256"] if region else None})
            elif region:
                candidates.append({**provenance, **region, "box_rule":"pixel_bbox", "status":"pending_independent_reviews",
                                   "required_independent_reviewers":2, "independent_reviewer_ids":[],
                                   "review_complete":False})
        prefix = "software-fixture-" if _fixture else ""
        write_lines(directory/(prefix+"participant-responses.jsonl"), originals)
        write_lines(directory/(prefix+"participant-attestations.jsonl"), (attestations[s] for s in sorted(attestations)))
        reply_path=directory/(prefix+"replies.jsonl")
        write_lines(reply_path,replies)
        if replies:
            require(len(load_replies(reply_path,rows))==len(replies), "Evaluator rejected a registered reply.")
        if unresolved is not None:
            write_lines(directory/(prefix+"intended.jsonl"),intended)
        write_lines(directory/(prefix+"task2-region-candidates.jsonl"),candidates)
        files={p.relative_to(directory).as_posix():digest(p.read_bytes()) for p in sorted(directory.rglob("*")) if p.is_file()}
        receipt={"kind":"participant_preproposal_intake", "created_utc":datetime.now(timezone.utc).isoformat(),
                 "input_bank_sha256":bank_hash,"expected_cases":len(rows),"returned_cases":len(responses),
                 "participants":len({e["participant_id"] for e in responses.values()}),"missing_cases":len(rows)-len(responses),
                 "genuine_human_cases":0 if _fixture else len(responses),
                 "detector_tracks_shown":False,"confirms_rebound_task2_context":False,
                 "registered_replies":len(replies),"drawn_pngs":sum(e["choice"]=="drawn_area" for e in responses.values()),
                 "independent_unresolved_ids_supplied":None if unresolved is None else len(unresolved),
                 "unresolved_id_file_sha256":digest(Path(unresolved_ids).read_bytes()) if unresolved_ids is not None else None,
                 "intended_rows":len(intended),"unresolved_awaiting_human":None if unresolved is None else sorted(unresolved-responses.keys()),
                 "task2_candidate_regions":len(candidates),"approved_task2_references":0,"software_fixture":_fixture,
                 "files":files}
        json_file(directory/(prefix+"receipt.json"),receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    actions=parser.add_subparsers(dest="action",required=True)
    check=actions.add_parser("verify",help="Verify a copied private form and all original images")
    check.add_argument("--form-dir",type=Path,required=True)
    for name in ("prepare","intake"):
        cmd=actions.add_parser(name)
        cmd.add_argument("--requests",type=Path,default=DEFAULT_SOURCE)
        cmd.add_argument("--out",type=Path,required=True)
        if name=="intake":
            cmd.add_argument("--return",dest="returns",action="append",default=[],type=Path)
            cmd.add_argument("--unresolved-ids",type=Path,help="Optional independent ID-only JSONL; one case_id per line")
    args=parser.parse_args()
    if args.action=="verify":
        result=verify_pack(args.form_dir)
    elif args.action=="prepare":
        result=prepare(args.requests,args.out)
    else:
        result=intake(args.requests,args.returns,args.out,unresolved_ids=args.unresolved_ids)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
