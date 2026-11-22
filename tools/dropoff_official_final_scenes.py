"""Stream the official EMT archive, validate real prefix JPEGs and export canonical input scenes without model inference."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
from tempfile import gettempdir
import time
from urllib.request import Request,urlopen
from uuid import uuid4

from evaluation.grounding import load_scenes
from tools.dropoff_final_input_inventory import ROOT,TASK1,RELEASE,TEST_ARCHIVE_SHA,git,require,write
from trackrag.dropoff.data import file_sha256
from trackrag.dropoff.scenes import load_scene,validate_scene


ARCHIVE_SIZE=8720080037
SOURCE="https://huggingface.co/datasets/KuAvLab/EMT/resolve/"+RELEASE+"/test_images.tar.gz"
MEMBER=re.compile(r"(?:^|/)(video_[0-9]+)/([0-9]+)\.(?:jpg|jpeg)$",re.I)
MIB=1024**2
GIB=1024**3


class Ranges:
    def __init__(self,size,workers):
        self.block=size;self.pool=ThreadPoolExecutor(max_workers=workers);self.workers=workers
        self.futures={};self.scheduled=self.consumed=self.processed=0;self.buffer=b"";self.at=0
        self.sha=sha256();self.start=time.monotonic()
        self.fill()

    def fetch(self,start):
        end=min(ARCHIVE_SIZE,start+self.block)-1
        for attempt in range(5):
            try:
                req=Request(SOURCE+f"?trackrag_original_prefix={start}-{end}",
                            headers={"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"})
                with urlopen(req,timeout=75) as response:
                    require(response.status==206 and response.headers.get("Content-Range")==f"bytes {start}-{end}/{ARCHIVE_SIZE}",
                            "Source refused exact original archive range")
                    data=response.read(end-start+2)
                require(len(data)==end-start+1,"Incomplete official archive range")
                return data
            except (ValueError,OSError) as exc:
                if attempt==4:
                    raise OSError(f"Original archive range {start}-{end} failed after five attempts") from exc
                time.sleep(2**attempt)

    def fill(self):
        while self.scheduled<ARCHIVE_SIZE and len(self.futures)<self.workers:
            self.futures[self.scheduled]=self.pool.submit(self.fetch,self.scheduled)
            self.scheduled=min(ARCHIVE_SIZE,self.scheduled+self.block)

    def read(self,n):
        require(n>=0,"Read length must be bounded")
        parts=[]
        while n:
            if self.at==len(self.buffer):
                if self.consumed==ARCHIVE_SIZE:
                    break
                self.buffer=self.futures.pop(self.consumed).result()
                self.consumed+=len(self.buffer)
                self.at=0
                self.fill()
            count=min(n,len(self.buffer)-self.at)
            segment=self.buffer[self.at:self.at+count]
            self.sha.update(segment)
            parts.append(segment)
            self.at+=count;self.processed+=count;n-=count
        return b"".join(parts)

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def run_configuration(args):
    native = ('trackrag/dropoff/scenes.py', 'evaluation/grounding.py', 'configs/vision/finetuned.json')
    files = ('tools/dropoff_official_final_scenes.py', *native)
    pinned = {name: (ROOT / name).read_bytes().replace(b'\r\n', b'\n') == git(TASK1, name).replace(b'\r\n', b'\n')
              for name in native}
    require(all(pinned.values()), 'Native Task 1 source differs from the pinned evaluator')
    return {'documented_utc': now(), 'worker_source_sha256': {name: file_sha256(ROOT / name) for name in files},
            'unmodified_pinned_task1_source_lf_equivalence': pinned, 'task1_revision': TASK1,
            'ranged_download_bytes': args.chunk_mib * MIB, 'range_workers': args.workers,
            'transient_source_layout': 'images/test/<recording>/<5-digit original frame>.jpg',
            'original_predictions': args.predictions.name + '/predictions-finetuned',
            'cleanup': 'Only generated authentic JPEG copies in the single run-specific temporary root are removed after all original validations for a recording succeed.',
            'minimum_free_space_gib': {'D': args.min_d_gib, 'C': args.min_c_gib}, 'model_inference': False}


def run(args):
    state_dir=args.out.resolve()
    state_dir.mkdir(parents=True,exist_ok=False)
    scenes=state_dir/"scenes";scenes.mkdir()
    write(state_dir / 'run-configuration.json', run_configuration(args))
    frame_manifest=json.loads(git(TASK1,"experiments/vision/emt_frame_sha256_manifest.json"))
    run_manifest=json.loads(git(TASK1,"experiments/vision/shared_prediction_runs.json"))
    bank=json.loads((args.bank/"receipt.json").read_bytes())
    require(file_sha256(args.bank/"task1-scene-ingredients.jsonl")==bank["files"]["task1-scene-ingredients.jsonl"]
            and bank["confirmed_final_inputs"]==44,"Different source input inventory")
    target=defaultdict(dict)
    for line in (args.bank/"task1-scene-ingredients.jsonl").read_bytes().splitlines():
        row=json.loads(line);inp=row["model_input"]
        target[inp["recording_id"]][inp["frame_id"]]=row
    prefix_counts = {rec: sum(int(fid) <= max(cases) for fid in frame_manifest['recordings'][rec])
                     for rec, cases in target.items()}
    unique="trackrag-verified-originals-"+uuid4().hex
    scratch=[(p.resolve()/unique,floor) for p,floor in [(args.scratch_d,args.min_d_gib*GIB),(args.scratch_c,args.min_c_gib*GIB)]]
    native={"profile":ROOT/"configs/vision/finetuned.json","predictions_dir":args.predictions/"predictions-finetuned",
            "frame_manifest":frame_manifest,"run_manifest":run_manifest}
    queue=Ranges(args.chunk_mib*MIB,args.workers)
    written={};done={};exports=[];active=None;loc=None;floor=0;retained=0
    start=time.monotonic()
    status={"started_utc":now(),"pid":os.getpid(),"source":SOURCE,"expected_whole_archive_sha256":TEST_ARCHIVE_SHA,
            "declared_archive_bytes":ARCHIVE_SIZE,"scratch_run_directory_name":unique,"no_models_called":True,
            "label_files_opened":False,"source_input_manifest_sha256":file_sha256(args.bank/"receipt.json")}
    def report(state):
        seconds=time.monotonic()-start
        data={**status,"state":state,"updated_utc":now(),"compressed_bytes_streamed":queue.processed,
              "mib_per_s":round(queue.processed/MIB/max(seconds,.1),2),"active_recording":active,
              "active_prefix_verified_count":len(written),"active_prefix_bytes":retained,
              "active_scratch_drive":loc.drive if loc else None,"complete_groups":done,"canonical_scene_exports":len(exports)}
        partial=state_dir/"status.partial"
        partial.write_text(json.dumps(data,indent=2)+"\n",encoding="utf8")
        os.replace(partial,state_dir/"status.json")
    def free(p):
        while not p.exists():
            p=p.parent
        return shutil.disk_usage(p).free
    def begin(rec):
        maximum=max(target[rec])
        sizes=[(args.bank/"source/images/test"/rec/f"{i:05d}.jpg").stat().st_size for i in target[rec]]
        estimate=int(sum(sizes)/len(sizes)*maximum*1.12)
        while True:
            selected=next(((p,margin) for p,margin in scratch if free(p)-margin>=estimate),None)
            if selected:
                root,margin=selected
                break
            report("waiting_for_space_for_next_original_recording")
            time.sleep(12)
        (root/"images/test"/rec).mkdir(parents=True,exist_ok=False)
        return root,margin
    def finish(rec):
        nonlocal retained,written,loc
        expected={int(i):v for i,v in frame_manifest["recordings"][rec].items() if int(i)<=max(target[rec])}
        require(set(written)==set(expected),"Missing genuine frame in the physical prefix")
        folder=(loc/"images/test"/rec).resolve()
        require(folder.parents[2]==loc and loc.name==unique,"Scratch directory escaped its generated root")
        for i,hash_value in sorted(expected.items()):
            image=folder/f"{i:05d}.jpg"
            require(image.resolve().parent==folder and not image.is_symlink() and file_sha256(image)==hash_value,
                    "On-disk physical prefix differs from the known original")
        for i,prior in sorted(target[rec].items()):
            actual=load_scene(loc,rec,i,**native)
            require(actual["scene_sha256"]==prior["expected_task1_scene_sha256"]
                    and actual["model_input"]==prior["model_input"] and actual["provenance"]==prior["provenance"],
                    "The canonical source exporter disagrees with the sparse derivation")
            validate_scene(actual,data_root=loc,**native)
            label=f"{rec}_f{i}.json"
            write(scenes/label,actual)
            validate_scene(json.loads((scenes/label).read_bytes()),frame_manifest=frame_manifest,run_manifest=run_manifest)
            exports.append({"file":label,"scene_sha256":actual["scene_sha256"]})
        done[rec]={"prefix_files_verified_individually_after_writing":len(written),"authentic_image_bytes":retained,
                   "first_id":min(written),"last_id":max(written),"canonical_scene_count":len(target[rec]),
                   "full_data_root_and_shared_detector_validation":True,"scratch_originals_removed":False}
        report("removing_this_verified_temporary_duplicate_prefix")
        for i in sorted(written):
            (folder/f"{i:05d}.jpg").unlink()
        for directory in (folder,folder.parent,folder.parent.parent,loc):
            directory.rmdir()
        done[rec]["scratch_originals_removed"]=True
        written={};retained=0;loc=None
    report("reading_original_archive")
    try:
        # 128 KiB balances decoder buffer copies and sequential stream throughput.
        with tarfile.open(fileobj=queue,mode="r|gz",bufsize=128*1024) as tar:
            for member in tar:
                match=MEMBER.search(member.name)
                if not match or not member.isfile():
                    continue
                rec,fid=match.group(1),int(match.group(2))
                if rec not in target or rec in done or fid>max(target[rec]):
                    continue
                if rec!=active:
                    require(loc is None,"Archive recording groups interleave before a prefix is complete")
                    active=rec
                    loc,floor=begin(rec)
                expected=frame_manifest["recordings"][rec].get(str(fid))
                require(expected and fid not in written and 0<member.size<16*MIB,"Unknown or oversized source JPEG")
                while free(loc)<floor+member.size:
                    report("waiting_with_real_prefix_for_free_space")
                    time.sleep(10)
                payload=tar.extractfile(member).read(member.size+1)
                require(len(payload)==member.size and sha256(payload).hexdigest()==expected,"Source image differs from original frame manifest")
                fname=loc/"images/test"/rec/f"{fid:05d}.jpg"
                with fname.open("xb") as f:
                    f.write(payload)
                retained+=len(payload);written[fid]=True
                if len(written)%25==0:
                    report("hashing_and_materializing_authentic_originals")
                if len(written)==prefix_counts[rec]:
                    finish(rec);active=None
                    report("verified_and_released_recording_prefix")
        while queue.read(MIB):
            pass
        require(queue.processed==ARCHIVE_SIZE and queue.sha.hexdigest()==TEST_ARCHIVE_SHA,"Whole compressed source archive differs")
        require(set(done)==set(target) and len(exports)==44,"Missing canonical recording or scene output")
        write(scenes/"index.json",sorted(exports,key=lambda r:r["file"]))
        require(len(load_scenes(scenes))==44,"Unmodified evaluator cannot reload all 44 genuine canonical scenes")
        status.update(archive_sha256=queue.sha.hexdigest(),full_archive_sha_verified=True,ending_utc=now(),
                      scene_index_sha256=file_sha256(scenes/"index.json"),unmodified_task1_evaluator_reloaded=44)
        report("complete")
    except BaseException as exc:
        status["failure"]={"type":type(exc).__name__,"message":str(exc)}
        report("failed_preserving_verified_evidence")
        raise
    finally:
        queue.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--predictions",type=Path,default=ROOT/"data/task1-shared-20261119")
    parser.add_argument("--scratch-d",type=Path,default=ROOT/"runs")
    parser.add_argument("--scratch-c",type=Path,default=Path(gettempdir()))
    parser.add_argument("--min-c-gib",type=float,default=1.0)
    parser.add_argument("--min-d-gib",type=float,default=.22)
    parser.add_argument("--workers",type=int,choices=range(1,7),default=3)
    parser.add_argument("--chunk-mib",type=int,choices=(4,8,16,24,32),default=16)
    run(parser.parse_args())
