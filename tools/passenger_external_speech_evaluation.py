"""Reproduce the HCRC Map Task speech transport check on the released passenger application."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
from tempfile import NamedTemporaryFile
import threading
from urllib.request import Request, urlopen
import wave
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from tools.passenger_external_speech import place_matches
from tools.passenger_pursuit_speech import measures
from tools.passenger_speech_study import normalize, require, write_new


ANNOTATIONS = "https://groups.inf.ed.ac.uk/maptask/hcrcmaptask.nxtformatv2-2.zip"
ANNOTATIONS_SHA = "d6126b417d111233806c6319b0c3fbddf58fa0ccb9c80405ba87f760b9504416"
SIGNALS = "https://groups.inf.ed.ac.uk/maptask/signals/dialogues/"
PLAN_SHA = "f5cb4ef04e0738ba284c33c9acdce019598bce655b21f0ba08a538132f8eb380"
INTERRUPTION_SHA = "a957f3f611847feeef56cbd18ae3f9f50f1fa3e727ed95c6bde9b25728fc5efa"
BASE = "maptaskv2-2/Data/"
NITE = "{http://nite.sourceforge.net/}"
F781_SHA = "c411cf6760695905334f32e5590083cf144543df1ca98e68aae55fb345eb583e"


def hu(data):
    return sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def utc():
    return datetime.now(timezone.utc).isoformat()


def require_sha(path, expected):
    require(hu(path.read_bytes()) == expected, f"Changed registered file {path.name}")


def span(node, index):
    link = node.findall(NITE+"child")
    if len(link) != 1:
        return None
    ids = re.findall(r"id\(([^)]+)\)",link[0].get("href",""))
    if len(ids) not in (1,2) or not all(k in index for k in ids):
        return None
    return index[ids[0]],index[ids[-1]]


def get_candidates(z, plan):
    def xml(name):
        return ET.fromstring(z.read(BASE+name))
    corpus=xml("corpus-resources/maptask-corpus.xml")
    names={e.get("id"):e.get("name") for e in xml("corpus-resources/maptask-landmarks.xml").iter("landmark")}
    choices=[]
    for dialog in corpus.findall("conv"):
        did=dialog.attrib["id"]
        pointer=next(p for p in dialog.findall(NITE+"pointer") if p.get("role")=="g")
        speaker=re.search(r"#id\(([^)]+)\)",pointer.attrib["href"]).group(1)
        units=list(xml(f"timed-units/{did}.g.timed-units.xml"))
        index={e.attrib["id"]:i for i,e in enumerate(units)}
        follower=[(float(e.attrib["start"]),float(e.attrib["end"]))
                  for e in xml(f"timed-units/{did}.f.timed-units.xml") if e.tag=="tu" and e.text]
        mentioned=[]
        for item in xml(f"landmark-refs/{did}.g.landmark-refs.xml"):
            where=span(item,index)
            links=item.findall(NITE+"pointer")
            if not where or item.get("lit")!="l" or len(links)!=1:
                continue
            key=re.search(r"#id\(([^)]+)\)",links[0].get("href",""))
            name=names.get(key.group(1)) if key else None
            if not name or name in {"start","finish"}:
                continue
            words=" ".join(e.text for e in units[where[0]:where[1]+1] if e.tag=="tu" and e.text)
            if place_matches([name],words)[0]:
                mentioned.append((where[0],where[1],name))
        for move in xml(f"moves/{did}.g.moves.xml"):
            if move.get("label")!="instruct" or move.get("aban")!="false_aban" or move.get("meta")!="false_meta":
                continue
            where=span(move,index)
            if not where:
                continue
            spoken=[e for e in units[where[0]:where[1]+1] if e.tag=="tu" and e.text]
            if not spoken:
                continue
            raw=" ".join(e.text.strip() for e in spoken)
            words=normalize(raw)
            a,b=float(spoken[0].attrib["start"]),float(spoken[-1].attrib["end"])
            if not 4<=len(words)<=24 or not 2<=b-a<=10:
                continue
            if any(start < b+.12 and end > a-.12 for start,end in follower):
                continue
            landmarks=[n for left,right,n in sorted(mentioned) if where[0]<=left and right<=where[1]]
            if not landmarks or not(set(words)&set(plan["relations"])):
                continue
            mid=move.attrib["id"]
            choices.append({"dialogue":did,"move":mid,"speaker":speaker,"reference":raw,"landmarks":landmarks,
                "start":a,"end":b,"stratum":"between" if "between" in words else "other_location",
                "priority":hu(f"{plan['seed']}|{did}|{mid}".encode())})
    return choices


def freeze(work):
    require_sha(work/"protocol.json",PLAN_SHA)
    meta=work/"annotations.zip"
    if not meta.exists():
        with urlopen(ANNOTATIONS,timeout=60) as source:
            data=source.read(16*1024*1024)
        require(hu(data)==ANNOTATIONS_SHA,"Changed source annotation archive")
        meta.write_bytes(data)
    require_sha(meta,ANNOTATIONS_SHA)
    plan=load(work/"protocol.json")
    with ZipFile(meta) as z:
        choices=get_candidates(z,plan)
    selected=[];speakers=set();dialogs=set();sentences=set()
    for group,quota in plan["quotas"].items():
        for row in sorted((r for r in choices if r["stratum"]==group),key=lambda r:(r["priority"],r["move"])):
            sentence=tuple(normalize(row["reference"]))
            if row["speaker"] in speakers or row["dialogue"] in dialogs or sentence in sentences:
                continue
            selected.append(row);speakers.add(row["speaker"]);dialogs.add(row["dialogue"]);sentences.add(sentence)
            if sum(r["stratum"]==group for r in selected)==quota:
                break
    write_new(work/"cohort.json",{"sealed":utc(),"protocol_sha256":PLAN_SHA,"annotation_sha256":ANNOTATIONS_SHA,
        "eligible_candidates":dict(Counter(r["stratum"] for r in choices)),"selected":selected})
    return {"cohort_sha256":hu((work/"cohort.json").read_bytes()),"cases":len(selected),
            "distinct_speakers":len(speakers),"by_stratum":dict(Counter(r["stratum"] for r in selected))}


def source_range(url,start,end):
    request=Request(url,headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"})
    with urlopen(request,timeout=22) as response:
        actual=response.headers.get("Content-Range","")
        require(response.status==206 and re.fullmatch(fr"bytes {start}-{end}/\d+",actual),
                "Publisher did not honor the requested bounded range")
        payload=response.read(end-start+2)
        require(len(payload)==end-start+1,"Unexpected size for the audio byte range")
        return payload,int(actual.split("/")[-1])


def publish_once(path, payload):
    """Make a complete file visible without overwriting another acquisition."""
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".acquire-", delete=False) as out:
            temporary = Path(out.name)
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sealed_audio_summary(work, cohort_sha, expected_rows=None):
    """Recheck a completed acquisition offline and preserve its original timestamp and errors."""
    path = work / "bound.json"
    bound_sha = hu(path.read_bytes())
    cohort, bound = open_inputs(work, bound_sha)
    require(bound["cohort_sha256"] == cohort_sha, "Existing audio binding belongs to a different cohort")
    if expected_rows is not None:
        require(bound["rows"] == expected_rows, "Concurrent acquisitions differ; the original binding was preserved")
    for source, row in zip(cohort["selected"], bound["rows"], strict=True):
        name = source["move"] + ".wav"
        wav_path = work / "wavs" / name
        if "acquisition_error" in row:
            require("wav" not in row and not wav_path.exists(), "A failed source row conflicts with an existing WAV")
            continue
        require(row.get("wav") == name, "A sealed audio row does not identify its registered WAV")
        with wave.open(str(wav_path), "rb") as wav:
            frames = wav.getnframes()
            pcm = wav.readframes(frames)
            lo, hi = row["source_range"]
            require(wav.getnchannels() == row["channels"] == 2 and wav.getsampwidth() == 2
                    and wav.getframerate() == row["sample_rate"] and len(pcm) == frames * 4 == hi - lo + 1
                    and 0 <= lo <= hi < row["original_source_bytes"]
                    and frames / wav.getframerate() == row["duration_s"]
                    and hu(pcm) == row["source_pcm_sha256"], "Sealed PCM differs from its source-range binding")
    return {"cases": len(bound["rows"]), "audio_bound": sum("wav" in r for r in bound["rows"]),
            "bound_sha256": bound_sha}


def acquire(work, cohort_sha):
    """The frozen segment list cannot change on an individual asset failure."""
    cohort=work/"cohort.json";require_sha(cohort,cohort_sha)
    bound_path = work / "bound.json"
    if bound_path.exists():
        return sealed_audio_summary(work, cohort_sha)
    require_sha(work / "protocol.json", PLAN_SHA)
    selected = load(cohort)
    require(selected["protocol_sha256"] == PLAN_SHA and selected["annotation_sha256"] == ANNOTATIONS_SHA,
            "Frozen text rows do not belong to the registered sources")
    wavdir=work/"wavs";wavdir.mkdir(exist_ok=True)
    rows=[]
    for case in selected["selected"]:
        row={"move":case["move"]};rows.append(row)
        name = case["move"] + ".wav"
        require(Path(name).name == name and "\\" not in name, "Invalid registered WAV filename")
        destination = wavdir / name
        try:
            url=SIGNALS+case["dialogue"]+".mix.wav"
            header,total=source_range(url,0,255)
            require(header[:4]==b"RIFF" and header[8:16]==b"WAVEfmt ","Unexpected WAV header")
            fmt_size,encoding,channels,rate,_,align,bits=struct.unpack_from("<IHHIIHH",header,16)
            offset=20+fmt_size+(fmt_size%2)
            while header[offset:offset+4]!=b"data":
                require(offset+8<len(header),"Missing PCM data header")
                size=struct.unpack_from("<I",header,offset+4)[0]
                offset+=8+size+(size%2)
            nbytes=struct.unpack_from("<I",header,offset+4)[0];offset+=8
            require(encoding==1 and channels==2 and bits==16 and align==4 and 16000<=rate<=48000,"Unsupported source WAV")
            first=max(0,math.floor((case["start"]-.12)*rate));last=min(nbytes//align,math.ceil((case["end"]+.12)*rate))
            lo,hi=offset+first*align,offset+last*align-1
            require(offset+nbytes<=total and lo<=hi<total and hi-lo<2_500_000,"Invalid source extent")
            data,second_size=source_range(url,lo,hi)
            require(second_size==total,"The source file changed between byte ranges")
            buffer=BytesIO()
            with wave.open(buffer,"wb") as w:
                w.setparams((channels,2,rate,0,"NONE","not compressed"));w.writeframes(data)
            payload=buffer.getvalue()
        except (ValueError,OSError,struct.error) as exc:
            if destination.exists():
                raise ValueError("Cannot verify an existing WAV against the original source; acquisition remains unsealed") from exc
            row["acquisition_error"]=f"{type(exc).__name__}: {exc}"
            continue
        if destination.exists() or not publish_once(destination, payload):
            require(destination.is_file() and not destination.is_symlink() and destination.read_bytes() == payload,
                    "Existing WAV conflicts with the reconstructed source clip; the original was preserved")
        row.update(wav=name,header_sha256=hu(header),original_source_bytes=total,source_range=[lo,hi],
            source_pcm_sha256=hu(data),wav_sha256=hu(payload),duration_s=(last-first)/rate,sample_rate=rate,channels=channels)
    bound = {"sealed": utc(), "cohort_sha256": cohort_sha, "rows": rows}
    payload = (json.dumps(bound, indent=2, allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")
    publish_once(bound_path, payload)
    return sealed_audio_summary(work, cohort_sha, expected_rows=rows)


def open_inputs(work, bound_sha):
    require_sha(work/"protocol.json",PLAN_SHA)
    require_sha(work/"bound.json",bound_sha)
    bound=load(work/"bound.json")
    require_sha(work/"cohort.json",bound["cohort_sha256"])
    cohort=load(work/"cohort.json")
    require(cohort["protocol_sha256"]==PLAN_SHA and cohort["annotation_sha256"]==ANNOTATIONS_SHA,
            "Frozen text rows do not belong to the registered sources")
    for ref,clip in zip(cohort["selected"],bound["rows"],strict=True):
        require(ref["move"]==clip["move"],"Audio bindings reordered or missing")
        if "wav" in clip:
            name = ref["move"] + ".wav"
            require(clip["wav"] == name and Path(name).name == name and "\\" not in name,
                    "The sealed WAV filename differs from the registered move")
            path = work / "wavs" / name
            require(path.is_file() and not path.is_symlink(), "The registered WAV is missing or linked")
            require_sha(path,clip["wav_sha256"])
    return cohort,bound


def check_app(work, bound_sha, scene_bank, whisper_dir, bundle, vision_cache, resume_observer=False):
    from tools.passenger_release import verify_bundle
    from trackrag.dropoff.infrastructure import PassengerDialogueGrounder
    from trackrag.dropoff.passenger import load_passenger_model
    from trackrag.dropoff.passenger_pair_choice import PassengerSession
    from trackrag.passenger_app import PassengerSite, handler, load_scenes, load_speech_recognizer

    _,bindings=open_inputs(work,bound_sha)
    allowed=load_scenes([scene_bank])
    require(len(allowed)==1 and next(iter(allowed.values()))["provenance"]["image_sha256"]==F781_SHA,
            "Use only the public-team nonfinal F781 background, never a final test bank")
    manifest=verify_bundle(bundle)
    require(manifest["referent_policy"],"Expected released passenger model configuration")
    # The product does not start the lazy visual backend when audio is transcribed or cancelled.
    model=load_passenger_model(bundle/"grounding.json",geometry_path=bundle/"geometry.json",vision_cache=vision_cache,
                               verify_landmarks=manifest["verify_landmarks"],referent_path=bundle/"referent.json")
    asr=load_speech_recognizer(None,whisper_dir)
    if resume_observer:
        from trackrag.whisper_speech import WhisperTranscript, MODEL_CONTENT_SHA256
        fault=load(work/"observer-interruption.json")
        require(hu((work/"observer-interruption.json").read_bytes())==INTERRUPTION_SHA
                and bindings["rows"][0]["move"]==fault["move"], "Unexpected browser interruption")
        folder=work/"sessions"/fault["session"]
        stored=folder/"audio"/"transcription-2.json";require_sha(stored,fault["receipt_sha256"])
        verified=WhisperTranscript(**load(stored)["receipt"])
        require(verified.audio_sha256==bindings["rows"][0]["wav_sha256"] and
                verified.model_content_sha256==MODEL_CONTENT_SHA256,"Interrupted transcript has a different source")
        require_sha(folder/"audio"/(verified.audio_sha256+".wav"),verified.audio_sha256)
        original=asr
        class ContinueObserver:
            def __init__(self):
                self.count=0
            def transcribe_wav(self,data):
                if hu(data)==verified.audio_sha256:
                    self.count+=1
                    require(self.count==1,"Replayed the interrupted transcript twice")
                    return verified
                return original.transcribe_wav(data)
        asr=ContinueObserver()
    app=PassengerSite(allowed,PassengerDialogueGrounder(model),asr,work/"sessions",PassengerSession)
    if resume_observer:
        require(not list((work/"browser").iterdir()),"Observer recovery may only continue this cleanly interrupted batch")
    else:
        (work/"browser").mkdir(exist_ok=False)
    try:
        with ThreadingHTTPServer(("127.0.0.1",0),handler(app)) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                subprocess.run(["node",str(Path(__file__).with_suffix(".cjs")),
                    f"http://127.0.0.1:{server.server_port}",str(work.resolve())],check=True,timeout=1000)
            finally:
                server.shutdown();thread.join()
        require(model.resolver.backend is None and model.road.model is None,
                "A visual backend unexpectedly loaded during speech review")
        require(not resume_observer or asr.count==1,"First real ASR receipt was not replayed exactly once")
    finally:
        model.close()
    return {"browser_receipts":len(list((work/"browser").glob("*.json"))),"native_visual_backend_loaded":False}


def aggregate(rows):
    n=len(rows);words=sum(r["reference_words"] for r in rows);edits=sum(r["word_edits"] for r in rows)
    return {"recordings":n,"reference_words":words,"word_edits":edits,"wer":edits/words if words else None,
        "exact_utterances":sum(r["word_exact"] for r in rows),"reference_relation_tokens":sum(r["reference_directions"] for r in rows),
        "retained_relation_tokens":sum(r["retained_directions"] for r in rows),"extra_relation_tokens":sum(r["extra_direction_words"] for r in rows),
        "exact_relation_sequences":sum(r["all_directions_exact"] for r in rows),
        "named_landmark_mentions":sum(r["landmark_count"] for r in rows),"retained_named_landmarks":sum(r["landmark_hits"] for r in rows),
        "all_named_landmarks_retained":sum(r["landmark_count"]==r["landmark_hits"] for r in rows),
        "raw_transcripts_in_editable_ui":sum(r["editable"] for r in rows),"acquisition_or_asr_errors":sum(bool(r["error"]) for r in rows)}


def report(work, bound_sha, scene_bank, out):
    from trackrag.dropoff.contracts import validate_request
    from trackrag.passenger_app import load_scenes
    from trackrag import whisper_speech

    cohort,bound=open_inputs(work,bound_sha)
    plan=load(work/"protocol.json")
    scene=next(iter(load_scenes([scene_bank]).values()))
    require(scene["provenance"]["image_sha256"]==F781_SHA,"Changed nonfinal UI background")
    rows=[]
    interrupted=None
    if (work/"observer-interruption.json").exists():
        require_sha(work/"observer-interruption.json",INTERRUPTION_SHA)
        interrupted=load(work/"observer-interruption.json")
    for reference,clip in zip(cohort["selected"],bound["rows"],strict=True):
        text="";editable=False;seconds=None;display_hash=None;journal_hash=None;repair_hash=None
        error=clip.get("acquisition_error")
        if not error:
            ui_file=work/"browser"/(clip["move"]+".json");ui=load(ui_file)
            folder=work/"sessions"/ui["session"]
            journal,=list(folder.glob("*.jsonl"))
            actions=[json.loads(line)["action"] for line in journal.read_bytes().splitlines()]
            require(actions==["opened","cancelled"] and ui["no_turn_or_proposal_on_upload"]
                    and ui["explicitly_cancelled"] and ui["confirmation_without_proposal_http"]==409,
                    "Speech upload or editing produced an unregistered model turn or confirmation")
            require_sha(folder/"audio"/(clip["wav_sha256"]+".wav"),clip["wav_sha256"])
            recfile,=list((folder/"audio").glob("transcription-*.json"));raw=load(recfile)
            require(raw["audio_sha256"]==clip["wav_sha256"]==ui["audio_sha256"],"Wrong source WAV")
            display_hash=hu(ui_file.read_bytes());journal_hash=hu(journal.read_bytes())
            receipt=raw["receipt"]
            replayed=bool(interrupted and interrupted["move"]==reference["move"])
            if replayed:
                prior=work/"sessions"/interrupted["session"]
                prior_file=prior/"audio"/"transcription-2.json"
                require_sha(prior_file,interrupted["receipt_sha256"])
                prior_journal,=list(prior.glob("*.jsonl"));require_sha(prior_journal,interrupted["journal_sha256"])
                require(hu(prior_file.read_bytes())==hu(recfile.read_bytes()),"First replay changed the true original ASR output")
            if receipt:
                require(raw["error"] is None and receipt["device"]=="cpu" and receipt["audio_sha256"]==clip["wav_sha256"]
                      and receipt["model_content_sha256"]==whisper_speech.MODEL_CONTENT_SHA256
                      and receipt["transcript"]==ui["raw_displayed"] and ui["stage"]=="review" and
                      ui["scripted_edited_value"]==reference["reference"],"ASR, editable words and source differ")
                text=receipt["transcript"];seconds=round(receipt["elapsed_s"],4);editable=True
                # Validate the ordinary voice payload shape without submitting an unrelated map instruction on a road image.
                eligible=validate_request({"request_id":reference["move"],"scene":scene,"text":ui["scripted_edited_value"],
                      "original_transcript":text,"input_mode":"voice","dialogue":[]})
                require(eligible["text"]==reference["reference"] and eligible["original_transcript"]==text,"Lost original/edited text")
                repair_hash=hu(eligible["text"].encode())
            else:
                require(raw["error"] and ui["stage"]=="failed","A failed transcription appeared as usable speech")
                error=raw["error"]["code"]
        m=measures(reference["reference"],text,set(plan["relations"]))
        names=place_matches(reference["landmarks"],text)
        rows.append({**m,"move":reference["move"],"speaker_code":reference["speaker"],"stratum":reference["stratum"],
            "reference_text_sha256":hu(reference["reference"].encode()),"hypothesis_sha256":hu(text.encode()) if text else None,
            "scripted_ui_reference_edit_sha256":repair_hash,"source_wav_sha256":clip.get("wav_sha256"),
            "browser_receipt_sha256":display_hash,"app_journal_sha256":journal_hash,"elapsed_s":seconds,
            "landmark_count":len(names),"landmark_hits":sum(names),"editable":editable,"error":error,
            "display_replayed_from_original_real_asr":bool(interrupted and interrupted["move"]==reference["move"])})
    require(len(rows)==len({r["speaker_code"] for r in rows})==len(cohort["selected"]),"Speaker duplication")
    sel=load(work/"cohort.json")
    total_range=sum(256+r["source_range"][1]-r["source_range"][0]+1 for r in bound["rows"] if "wav" in r)
    result={"study":"prospective_HCRC_spontaneous_direction_giver_speech_to_local_passenger_review",
        "completed_utc":utc(),"attribution":"Human Communication Research Centre, University of Edinburgh and University of Glasgow, HCRC Map Task Corpus. A. H. Anderson et al. (1991).",
        "registration":plan,"plan_sha256":PLAN_SHA,"cohort_sealed_utc":sel["sealed"],"cohort_sha256":bound["cohort_sha256"],
        "audio_sealed_utc":bound["sealed"],"audio_binding_sha256":bound_sha,"source_audio_range_bytes":total_range,
        "source_audio_seconds":round(sum(r.get("duration_s",0) for r in bound["rows"]),5),
        "eligible_before_selection":sel["eligible_candidates"],"selected":len(rows),"distinct_corpus_speaker_ids":len({r["speaker_code"] for r in rows}),
        "model":{"id":whisper_speech.MODEL_ID,"revision":whisper_speech.REVISION,"verified_file_set_sha256":whisper_speech.MODEL_CONTENT_SHA256,
                 "engines":whisper_speech.VERSIONS,"device":"CPU","compute_type":"int8","decode_options":whisper_speech.OPTIONS},
        "overall":aggregate(rows),"by_stratum":{group:aggregate([r for r in rows if r["stratum"]==group]) for group in plan["quotas"]},
        "raw_page_and_task5_contract":{"original_audio_and_transcript_checked":sum(bool(r["browser_receipt_sha256"]) for r in rows),
              "fresh_with_complete_browser_observer":sum(r["editable"] and not r["display_replayed_from_original_real_asr"] for r in rows),
              "browser_displays_replayed_from_first_genuine_asr":sum(r["display_replayed_from_original_real_asr"] for r in rows),
              "scripted_reference_edits_preserving_original_validated":sum(r["editable"] for r in rows),
              "sessions_with_only_open_then_cancel_actions":sum(bool(r["app_journal_sha256"]) for r in rows),
              "visual_model_requests":0,"actual_destination_proposals":0,"human_confirmations":0},
        "source_license":{"audio":"CC BY 4.0","annotations":"CC BY-SA 4.0 v2.2. Derived source annotations shared under the same terms.",
            "source_publication":"https://groups.inf.ed.ac.uk/maptask/maptasknxt.html","audio_legal_code":"https://groups.inf.ed.ac.uk/maptask/signals/dialogues/00LICENSE.html"},
        "scope":"Spontaneous English direction-givers speaking about fictional map landmarks. This selected sample is separate from earlier SLURP and PURSUIT material. It tests recognizer wording, including landmark and spatial predicates, and the passenger page's editable transcript handoff. No autonomous scene inference, road-location score, in-car noise, Gulf-accent claim or genuine project volunteer response is measured. Whisper training exposure is unknown. Adjacent speech and manual boundaries can influence recognition. The reference-text edit is a software-driven check.",
        "reproduce":["Run python -m tools.passenger_external_speech_evaluation freeze --work runs/my-hcrc from the checkout; the stored registration verifies the annotation release and repeats the fixed selection.",
            "Use audio --work runs/my-hcrc --sha <returned cohort_sha256> to fetch the range-bound clips. Timestamps make the new provenance files distinct; original speaker, segment and WAV identities are fixed.",
            "Use app --work runs/my-hcrc --sha <returned bound_sha256> --model <verified faster-whisper-small.en folder> --vision-cache <segformer folder>. The normal release bundle and nonfinal F781 bookmark are required. Set TRACKRAG_PLAYWRIGHT to the installed Node Playwright module and TRACKRAG_CHROMIUM to its headless Chromium executable as needed.",
            "Use score --work runs/my-hcrc --sha <returned bound_sha256> --out runs/my-hcrc/results.json. Edited text is a contract check and earns no ASR credit."],
        "rows":rows}
    if (work/"study-intake.json").is_file():
        result["formal_team_intake_separate"]=load(work/"study-intake.json")
    if interrupted:
        result["technical_interruption"]={"detail":interrupted["failure"],"recovery":interrupted["registered_recovery"],
             "primary_audio_sha256":interrupted["source_wav_sha256"],"original_wav_to_asr_receipt_sha256":interrupted["receipt_sha256"],
             "preserved_before_recovery_sha256":INTERRUPTION_SHA}
    write_new(out,result)
    return result["overall"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["freeze","audio","app","score"])
    parser.add_argument("--work",type=Path,required=True)
    parser.add_argument("--published",type=Path,default=Path("experiments/dropoff/passenger-external-speech-20261122.json"),
        help="Previously registered plan for reconstruction on a fresh checkout")
    parser.add_argument("--sha",help="For audio, the exact frozen text selection SHA. For app/score, the frozen audio binding SHA.")
    parser.add_argument("--scene-bank",type=Path,default=Path("runs/passenger-pair-scene/requests.jsonl"))
    parser.add_argument("--bundle",type=Path,default=Path("models/passenger-feature"))
    parser.add_argument("--model",type=Path)
    parser.add_argument("--vision-cache",type=Path)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--resume-observer",action="store_true",help="For the one recorded, SHA-bound historical observer interruption only")
    args=parser.parse_args()
    if args.action=="freeze":
        args.work.mkdir(parents=True,exist_ok=True)
        if not (args.work/"protocol.json").exists():
            write_new(args.work/"protocol.json",load(args.published)["registration"])
        result=freeze(args.work)
    else:
        require(bool(args.sha),"Supply the previously frozen input digest")
        if args.action=="audio":
            result=acquire(args.work,args.sha)
        elif args.action=="app":
            require(args.model is not None and args.vision_cache is not None,"Supply the two local model cache paths")
            result=check_app(args.work,args.sha,args.scene_bank,args.model,args.bundle,args.vision_cache,args.resume_observer)
        else:
            require(args.out is not None,"Choose a new score JSON path")
            result=report(args.work,args.sha,args.scene_bank,args.out)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
