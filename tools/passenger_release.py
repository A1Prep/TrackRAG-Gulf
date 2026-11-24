"""Verify and start a passenger application handoff from the project checkout."""
import argparse
from dataclasses import asdict, replace
from importlib.util import find_spec
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ASR_ID = "vosk-model-small-en-us-0.15"
ASR_URL = f"https://alphacephei.com/vosk/models/{ASR_ID}.zip"
ASR_ARCHIVE_SHA = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"
SEG_REV = "21b3847fae21ddee674abd31129307b6a1235bd9"
SEG_HASHES = {
    "config.json":"649ba4afbaa11ee730b14ad5e31bd50f2e2d5be661a291e31a91512510dba8d9",
    "preprocessor_config.json":"d96ab2cb112985c19d5ea32f1d1f2369104fc78b1379f5ce26e33ca8646202a7",
    "pytorch_model.bin":"027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c",
}


PUBLIC_PHOTO_LISTS = ("passenger-uae-parking-photos-20261123.json", "passenger-place-public-photos-20261123.json",
                      "passenger-frontage-photos-20261123.json", "passenger-frontage-extended-photos-20261123.json")
EXTRA_PHOTO_BANK = "data/uae-parking-local-extra/passenger-requests.jsonl"
EXTRA_PHOTO_RESULTS = "experiments/dropoff/passenger-additional-local-photos-20261124"
EXTRA_MODEL_FILES = {
    "local_surface_index": ("passenger-local-image-surfaces-v1.json",
                            "41a0c2e929523fb2220e04027203c2f863b8e9653a3ba7c1d21e675f92b8e4e8"),
    "local_observations": ("observations.json",
                           "3c11f54f49eb0f5db47e8715e252e8b091b20830930d75e61eb1fb6eb0f75e6f"),
}
REAR_PHOTO_BANK = 'data/uae-parking-local-rear-20261124/passenger-requests.jsonl'
REAR_PHOTO_RESULTS = 'experiments/dropoff/passenger-rear-local-photos-20261124'
REAR_MODEL_FILES = (
    ('local_surface_index', 'passenger-local-image-surfaces-v1.json',
     'c9ead6f19929510cf46b9645166ca064e0cbe887158ff17e4e95f3806a7722bf'),
    ('local_vehicle_observations', 'vehicle-observations.json',
     '7b12523b429e064d2b650e4e3fa6958c2a865e4e659e6959adf756ef124a19d1'),
)
RESERVE_PHOTO_BANK = 'data/uae-parking-local-rear-reserve-20261124/passenger-requests.jsonl'
RESERVE_PHOTO_RESULTS = 'experiments/dropoff/passenger-rear-reserve-photos-20261124'
RESERVE_MODEL_FILES = (
    ('local_surface_index', 'passenger-local-image-surfaces-v1.json',
     '9dd806e609645fe51e49ead8c1d35db908fdf98217dbb43f4e620f9e3662d452'),
    ('local_vehicle_observations', 'vehicle-observations.json',
     'b03aa392eb8631683ff58cfa9d4cc69a63c5e2997515a6a7bd94a56d4e2e1828'),
)
PHOTO_PACKS = (
    (EXTRA_PHOTO_BANK, EXTRA_PHOTO_RESULTS, tuple((kind, *pin) for kind, pin in EXTRA_MODEL_FILES.items())),
    (REAR_PHOTO_BANK, REAR_PHOTO_RESULTS, REAR_MODEL_FILES),
    (RESERVE_PHOTO_BANK, RESERVE_PHOTO_RESULTS, RESERVE_MODEL_FILES),
)


def file_hash(path):
    with Path(path).open("rb") as src:
        import hashlib
        return hashlib.file_digest(src,"sha256").hexdigest()


def verify_bundle(folder):
    from trackrag.dropoff.anchor_geometry import AnchorGeometry
    from trackrag.dropoff.model import GroundingConfig, _load_json, verify_adapter
    from trackrag.dropoff.referent import ReferentRanker

    root = Path(folder).resolve()
    manifest = _load_json(root/"manifest.json")
    if manifest.get("format") != "passenger-app-handoff-v1":
        raise ValueError("Unknown passenger bundle")
    for flag in ("referent_policy", "verify_landmarks"):
        if type(manifest.get(flag, False)) is not bool:
            raise ValueError(f"{flag} must be a JSON boolean")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Passenger bundle requires a file-to-SHA256 manifest")
    required = {"geometry.json","grounding.json","adapter/adapter_model.safetensors",
                "adapter/adapter_config.json","adapter/grounding_adapter_manifest.json"}
    if manifest.get("referent_policy"):
        required.add("referent.json")
    if set(files) != required or (root/"referent.json").exists() != ("referent.json" in required):
        raise ValueError("Passenger bundle files differ from its declared model components")
    for name,expected in files.items():
        path = (root/name).resolve()
        if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or not path.is_relative_to(root) or not path.is_file() or file_hash(path) != expected):
            raise ValueError(f"Passenger artifact unavailable or changed: {name}")
    cfg = GroundingConfig.from_mapping(_load_json(root/"grounding.json"))
    if ((cfg.model_id, cfg.revision) != (manifest.get("base_model"), manifest.get("revision"))
            or cfg.adapter_dir != "adapter" or cfg.runtime != "streamed" or cfg.allow_annotated):
        raise ValueError("Packaged runtime or base identity differs from the passenger model")
    verify_adapter(replace(cfg, adapter_dir=str(root/"adapter")))
    AnchorGeometry.load(root/"geometry.json", expected_sha=files["geometry.json"])
    if manifest.get("referent_policy"):
        ReferentRanker.load(root/"referent.json", expected_sha256=files["referent.json"])
    if root == (ROOT/"models/passenger-feature").resolve():
        protocol = ROOT/"experiments/dropoff/passenger-release-protocol-20261122.json"
        if manifest != _load_json(protocol).get("components"):
            raise ValueError("Installed passenger model differs from the frozen release protocol")
    return manifest


def pack(config_path, geometry_path, referent_path, destination):
    """Export only redistributable learned parameters and a portable inference configuration."""
    from trackrag.dropoff.anchor_geometry import AnchorGeometry
    from trackrag.dropoff.model import GroundingConfig, verify_adapter
    from trackrag.dropoff.referent import ReferentRanker
    cfg = GroundingConfig.from_mapping(json.loads(config_path.read_bytes()))
    proof = verify_adapter(cfg)
    if cfg.runtime != "streamed" or cfg.allow_annotated or proof is None:
        raise ValueError("The passenger release needs a verified local trained model and predicted scenes")
    AnchorGeometry.load(geometry_path)
    if referent_path:
        ReferentRanker.load(referent_path)
    destination = destination.resolve()
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Choose unused package and archive paths")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as td:
        staging = Path(td)/"bundle"
        (staging/"adapter").mkdir(parents=True)
        for name in ("adapter_model.safetensors","adapter_config.json","grounding_adapter_manifest.json"):
            shutil.copyfile(Path(cfg.adapter_dir)/name,staging/"adapter"/name)
        shutil.copyfile(geometry_path,staging/"geometry.json")
        if referent_path:
            shutil.copyfile(referent_path,staging/"referent.json")
        portable = replace(cfg,adapter_dir="adapter",model_dir="models/qwen3-vl-8b-local-nf4",
                           cache_dir="models/dropoff-processor-cache/hub")
        (staging/"grounding.json").write_bytes((json.dumps(asdict(portable),indent=2)+"\n").encode())
        files = {p.relative_to(staging).as_posix():file_hash(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        manifest = {"format":"passenger-app-handoff-v1","files":files,"verify_landmarks":False,
                    "referent_policy":bool(referent_path),"base_model":cfg.model_id,"revision":cfg.revision}
        (staging/"manifest.json").write_bytes((json.dumps(manifest,indent=2)+"\n").encode())
        verify_bundle(staging)
        with zipfile.ZipFile(Path(td)/"bundle.zip","w") as output:
            for p in sorted(staging.rglob("*")):
                if p.is_file():
                    info = zipfile.ZipInfo(p.relative_to(staging).as_posix())
                    output.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
        staging.rename(destination)
        (Path(td)/"bundle.zip").rename(archive)
    return {"folder":str(destination),"archive":str(archive),"sha256":file_hash(archive)}


def download(url, output, expected, max_bytes):
    output = Path(output)
    if output.exists():
        if file_hash(output) != expected:
            raise ValueError(f"Existing support file has different bytes: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent,delete=False) as file:
            temporary = Path(file.name)
            total = 0
            with urllib.request.urlopen(url,timeout=60) as response:
                while block := response.read(512*1024):
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError("Support download exceeded expected size")
                    file.write(block)
        if file_hash(temporary) != expected:
            raise ValueError(f"Official support download failed its SHA256 check: {output.name}")
        os.replace(temporary,output)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def support(folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in SEG_HASHES.items():
        url = f"https://huggingface.co/nvidia/segformer-b0-finetuned-cityscapes-1024-1024/resolve/{SEG_REV}/{name}"
        download(url,folder/"segformer-b0"/name,value,16*1024**2)
    from trackrag.local_speech import _verify_model
    dest = folder/ASR_ID
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            temp = Path(temp)
            archive = temp/"voice.zip"
            download(ASR_URL,archive,ASR_ARCHIVE_SHA,42*1024**2)
            with zipfile.ZipFile(archive) as bundle:
                for entry in bundle.infolist():
                    target = (temp/entry.filename).resolve()
                    if not target.is_relative_to(temp/ASR_ID):
                        raise ValueError("Unexpected path in speech bundle")
                bundle.extractall(temp)
            _verify_model(temp/ASR_ID)
            os.replace(temp/ASR_ID,dest)
    _verify_model(dest)
    return {"vision":str(folder/"segformer-b0"),"speech":str(dest)}


def speech_arguments(caches, *, text_only=False, vosk=None, whisper=None):
    if sum((bool(text_only), vosk is not None, whisper is not None)) > 1:
        raise ValueError("Choose one of text only, a Vosk model, or a Whisper model.")
    if text_only:
        return []
    if whisper is not None:
        return ["--whisper-model", str(Path(whisper).resolve())]
    if vosk is not None:
        return ["--asr-model", str(Path(vosk).resolve())]
    selected = os.environ.get("TRACKRAG_PASSENGER_WHISPER")
    if selected:
        return ["--whisper-model", str(Path(selected).resolve())]
    if find_spec("faster_whisper") is not None:
        installed = next((p/"faster-whisper-small.en" for p in caches if (p/"faster-whisper-small.en").is_dir()), None)
        if installed is not None:
            return ["--whisper-model", str(installed.resolve())]
    installed = next((p/ASR_ID for p in caches if (p/ASR_ID).is_dir()), None)
    if installed is None:
        raise FileNotFoundError("Install a local speech model using the passenger release instructions, or use --text-only.")
    return ["--asr-model", str(installed.resolve())]


def start(args):
    folder = args.bundle.resolve()
    manifest = verify_bundle(folder)
    caches = [folder, ROOT/"models"/"passenger-support"]
    if os.environ.get("LOCALAPPDATA"):
        local = Path(os.environ["LOCALAPPDATA"])/"TrackRAG-method-cache"
        caches.extend([local,local/"asr-assets"])
    def locate(explicit, name):
        if explicit is not None:
            return explicit.resolve()
        found = next((d/name for d in caches if (d/name).is_dir()),None)
        if found is None:
            raise FileNotFoundError(f"Missing {name}. Run python -m tools.passenger_release fetch-support.")
        return found
    road = locate(args.vision_cache,"segformer-b0")
    voice = speech_arguments(caches, text_only=args.text_only, vosk=args.asr_model,
                             whisper=getattr(args, "whisper_model", None))
    cfg = json.loads((folder/"grounding.json").read_text(encoding="utf8"))
    cfg.update(adapter_dir=str(folder/"adapter"),model_dir=str(args.base_dir.resolve()),
               cache_dir=str(args.processor_cache.resolve()))
    banks = args.requests or [ROOT/"runs/passenger-inputs/requests.jsonl"]
    if args.requests is None:
        for extra in ("runs/passenger-pair-scene/requests.jsonl", "data/uae-parking-study/passenger-requests.jsonl",
                      "data/uae-parking-local/passenger-requests.jsonl", *(bank for bank, _, _ in PHOTO_PACKS)):
            if (ROOT/extra).is_file():
                banks.append(ROOT/extra)
    banks = list(dict.fromkeys(Path(p).resolve() for p in [*banks, *(getattr(args,"additional_requests",None) or ())]))
    for path in banks:
        if not path.is_file():
            raise FileNotFoundError(f"Prepared scene bank unavailable: {path}")
    from trackrag.passenger_app import main as serve
    os.environ["GEMINI_MODEL"] = args.qa_model
    with tempfile.TemporaryDirectory(prefix="passenger-config-") as td:
        config = Path(td)/"model.json"
        config.write_text(json.dumps(cfg),encoding="utf8")
        argv = ["--config",str(config),"--geometry",str(folder/"geometry.json"),"--vision-cache",str(road),
                "--port",str(args.port),"--out",str(args.out)]
        argv.extend(["--between-road-mode", getattr(args, "between_road_mode", "edge")])
        if getattr(args, "parking_backend", None):
            argv.extend(["--parking-backend", args.parking_backend])
        sources = list(getattr(args, "place_public_sources", None) or ())
        if getattr(args, "public_places", False):
            sources.extend(ROOT / "experiments/dropoff" / name for name in PUBLIC_PHOTO_LISTS)
        for path in dict.fromkeys(Path(source).resolve() for source in sources):
            argv.extend(["--place-public-sources",str(path)])
        for option in ("local_surface_index", "local_observations", "local_vehicle_observations"):
            pairs = [(Path(path).resolve(), expected) for path, expected in getattr(args,option,None) or ()]
            for bank, study, files in PHOTO_PACKS:
                if ROOT / bank not in banks:
                    continue
                for kind, name, digest in files:
                    if kind == option and all(expected != digest for _, expected in pairs):
                        pairs.append((ROOT / study / name, digest))
            for path, expected in dict.fromkeys(pairs):
                argv.extend(["--"+option.replace("_","-"),str(path),expected])
        if getattr(args, "low_memory", False):
            argv.append("--low-memory")
        for bank in banks:
            argv.extend(["--requests",str(bank)])
        if manifest.get("verify_landmarks"):
            argv.append("--verify-landmarks")
        if manifest.get("referent_policy"):
            argv.extend(["--referent",str(folder/"referent.json")])
        argv.extend(voice)
        if bool(args.qa_data_root) != bool(args.qa_predictions):
            raise ValueError("Supply both original recordings and their saved predictions for footage search")
        if args.qa_frame_rate:
            argv.extend(["--qa-frame-rate",str(args.qa_frame_rate)])
        qa_data, qa_predictions = args.qa_data_root, args.qa_predictions
        if qa_data is None:
            for image_dir, track_dir in (
                    ("data/passenger-demo-images", "models/passenger-demo-predictions"),
                    ("data/task1-image-only-20261119", "runs/task1-artifact-verification-20261119/finetuned-live-01")):
                if (ROOT/image_dir).is_dir() and (ROOT/track_dir).is_dir():
                    qa_data, qa_predictions = ROOT/image_dir, ROOT/track_dir
                    break
        if qa_data is not None:
            argv.extend(["--qa-data-root",str(qa_data),"--qa-predictions",str(qa_predictions),
                         "--qa-vision-config",str(args.qa_vision_config)])
        minimum = getattr(args, "require_footage", 0)
        if minimum:
            argv.extend(["--require-footage", str(minimum)])
        return serve(argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    fetch = sub.add_parser("fetch-support",help="Install the pinned road and English speech models")
    fetch.add_argument("--out",type=Path,default=ROOT/"models/passenger-support")
    export = sub.add_parser("pack",help="Package a verified trained model for another checkout")
    for name in ("config","geometry","out"):
        export.add_argument("--"+name,type=Path,required=True)
    export.add_argument("--referent",type=Path)
    for cmd in ("verify","start"):
        p = sub.add_parser(cmd)
        p.add_argument("--bundle",type=Path,default=ROOT/"models/passenger-feature")
        if cmd == "start":
            p.add_argument("--requests",type=Path,action="append")
            p.add_argument("--additional-requests",type=Path,action="append",metavar="JSONL",
                           help="Add a prepared scene bank while keeping the normal gallery and recordings")
            p.add_argument("--local-surface-index",nargs=2,action="append",metavar=("JSON","SHA256"),
                           help="Surface maps prepared offline for an additional photo bank")
            p.add_argument("--local-vehicle-observations",nargs=2,action="append",metavar=("JSON","SHA256"),
                           help="Vehicle crop readings prepared on the original local photos")
            p.add_argument("--local-observations",nargs=2,action="append",metavar=("JSON","SHA256"),
                           help="Verified offline 8B observations for additional registered surfaces")
            p.add_argument("--base-dir",type=Path,default=ROOT/"models/qwen3-vl-8b-local-nf4")
            p.add_argument("--processor-cache",type=Path,default=ROOT/"models/dropoff-processor-cache/hub")
            p.add_argument("--vision-cache",type=Path)
            p.add_argument("--between-road-mode", choices=("edge", "edge-or-bay"), default="edge",
                           help="Check the visible road edge, or also check angled and facing parking bays")
            p.add_argument("--parking-backend", choices=("modal",),
                           help="Use the hosted two-vehicle parking model in the ordinary passenger form")
            p.add_argument("--public-places", action="store_true",
                           help="Verify named places on the published UAE demo photos using Modal")
            p.add_argument("--place-public-sources",type=Path,action="append",metavar="JSON",
                           help="Enable GPU verification of places on these public KartaView source lists")
            p.add_argument("--asr-model",type=Path)
            p.add_argument("--whisper-model",type=Path,help="Use the pinned English Whisper Small model on CPU")
            p.add_argument("--text-only",action="store_true")
            p.add_argument("--qa-data-root",type=Path)
            p.add_argument("--qa-predictions",type=Path)
            p.add_argument("--qa-frame-rate",type=float)
            p.add_argument("--require-footage", type=int, metavar="RECORDINGS", default=0,
                           help="Start only if at least this many distinct recordings have verified, searchable footage")
            p.add_argument("--low-memory", action="store_true", help="Unload the 8B worker between requests")
            p.add_argument("--qa-model",default=os.environ.get("GEMINI_MODEL","gemini-3.5-flash-lite"),
                           help="Hosted answerer for the separate footage search")
            p.add_argument("--qa-vision-config",type=Path,default=ROOT/"configs/vision/finetuned.json")
            p.add_argument("--port",type=int,default=8765)
            p.add_argument("--out",type=Path,default=ROOT/"runs/passenger-sessions")
    args = parser.parse_args()
    if args.command == "start":
        return start(args)
    if args.command == "pack":
        data = pack(args.config,args.geometry,args.referent,args.out)
    else:
        data = verify_bundle(args.bundle) if args.command == "verify" else support(args.out)
    print(json.dumps(data,indent=2))


if __name__ == "__main__":
    main()
