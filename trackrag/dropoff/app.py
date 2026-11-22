"""Local recorded-scene application for cited QA and passenger drop-off grounding.

Run ``python -m trackrag.dropoff.app --help``. No hosted grounding fallback,
annotation loader or reference-answer file is used by this application.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
import json
import mimetypes
from pathlib import Path
import re
from threading import RLock
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from trackrag.local_speech import SpeechInputError

from .contracts import error_outcome, proposal_fingerprint
from .inputs import MAX_IMAGE_BYTES, build_instruction_request
from .interaction import InteractionController, InvalidTransition


WEB_ROOT = Path(__file__).with_name("web")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
MAX_AUDIO_BYTES = 16 * 1024 * 1024
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".mp4"}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_json(payload):
    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(payload, object_pairs_hook=_unique, parse_constant=reject)


def _fields(value, required=(), optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError("request has missing or unknown fields")


def _config(path):
    if path is None:
        return None
    payload = Path(path).read_bytes()
    if len(payload) > 1024 * 1024:
        raise ValueError("configuration exceeds 1 MiB")
    value = parse_json(payload.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be an object")
    return value


@dataclass(frozen=True)
class AppConfig:
    data_root: Path
    output_dir: Path
    vision_config: Path | None = None
    predictions: Path | None = None
    vision_mode: str = "saved"
    grounding_config: Path | None = None
    grounding_backend: str = "transformers"
    qa_answerer: str = "gemini"
    speech_adapter: str | None = None
    asr_model: Path | None = None
    whisper_model: Path | None = None
    adapters: str | None = None
    frame_rate: float | None = None
    split: str | None = None


class RecordingPrefix:
    """Give every module the actual available-frame prefix, ending at the cutoff."""

    def __init__(self, recording, cutoff):
        self.recording_id = recording.recording_id
        self._recording = recording
        self.frame_ids = [fid for fid in recording.frame_ids if fid <= cutoff]
        if cutoff not in self.frame_ids:
            raise KeyError(f"Original frame {cutoff} is unavailable")

    def get_frame(self, frame_id):
        if frame_id not in self.frame_ids:
            raise KeyError("frame lies outside the selected prefix")
        return self._recording.get_frame(frame_id)

    def iter_frames(self):
        for frame_id in self.frame_ids:
            yield self.get_frame(frame_id)


class _UnavailableGrounding:
    def interpret(self, request):
        return error_outcome("configuration_unavailable", "No grounding configuration was supplied",
                             request["request_id"], {"component": "local_application"})


class LocalApplication:
    """One local passenger session. Slow adapters run outside the UI/state lock."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._lock = RLock()
        self._close_lock = RLock()
        self._closed = False
        self._recordings = {}
        self._media = {}
        self._selected = None
        self._selection_token = None
        self._audio_intent = None
        self._audio_current = None
        self._jobs = []
        self._qa_calls = []
        self._audio_calls = []
        self._selection_calls = []
        self._api_errors = []
        self._adapters = import_module(config.adapters) if config.adapters else None
        self._vision_data = _config(config.vision_config)
        self._grounding_data = _config(config.grounding_config)
        if config.vision_mode not in ("saved", "fresh"):
            raise ValueError("vision_mode must be saved or fresh")
        if config.grounding_backend not in ("transformers", "ollama"):
            raise ValueError("unknown grounding backend")
        if config.qa_answerer not in ("none", "gemini"):
            raise ValueError("unknown QA answerer")
        if sum(value is not None for value in (config.speech_adapter, config.asr_model, config.whisper_model)) > 1:
            raise ValueError("select only one of --speech-adapter, --asr-model and --whisper-model")
        self._vision = None
        if self._vision_data is not None:
            from trackrag.vision.config import config_from_dict
            self._vision = config_from_dict(self._vision_data)
        if self._adapters is None and (self._vision is None or
                (config.vision_mode == "saved" and config.predictions is None)):
            raise ValueError("normal vision requires --vision-config and saved mode requires --predictions")
        if self._adapters is not None:
            for function in ("predict_tracks_with_receipt", "create_grounding_adapter"):
                if not callable(getattr(self._adapters, function, None)):
                    raise ValueError(f"development adapter must supply {function}")
            model = self._adapters.create_grounding_adapter(self._grounding_data)
        elif self._grounding_data is None:
            model = _UnavailableGrounding()
        elif config.grounding_backend == "ollama":
            from .ollama import OllamaGroundingModel
            model = OllamaGroundingModel(self._grounding_data)
        else:
            from .model import GroundingModel
            model = GroundingModel(self._grounding_data)
        # The application owns adapters created for this session, including an
        # explicitly selected development factory. Older adapters need no close.
        self._owned_model = model
        speech_module = import_module(config.speech_adapter) if config.speech_adapter else self._adapters
        if config.speech_adapter and getattr(speech_module, "DEVELOPMENT_ONLY", False) and self._adapters is None:
            raise ValueError("development speech adapters require explicit --adapters mode")
        self._transcribe = getattr(speech_module, "transcribe_audio", None)
        if (config.speech_adapter or self._transcribe is not None) and not callable(self._transcribe):
            raise ValueError("speech adapter transcribe_audio must be callable")
        self._native_speech = None
        if config.asr_model is not None or config.whisper_model is not None:
            from .speech import NativeSpeechAdapter
            self._native_speech = NativeSpeechAdapter(asr_model=config.asr_model, whisper_model=config.whisper_model)
            self._transcribe = self._native_speech.transcribe_audio
        self.controller = InteractionController(model)
        self._catalog = self._discover_recordings()
        if not self._catalog:
            raise ValueError("No EMT recording image directories found under --data-root")
        self.output_dir = Path(config.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / "images").mkdir()
        (self.output_dir / "audio").mkdir()
        (self.output_dir / "qa").mkdir()
        for name, value in (("vision", self._vision_data), ("grounding", self._grounding_data)):
            if value is not None:
                (self.output_dir / f"{name}-config.json").write_text(
                    json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trackrag-app")
        self._persist()

    def _discover_recordings(self):
        found = {}
        root = Path(self.config.data_root)
        for split in ([self.config.split] if self.config.split else ["train", "test"]):
            for base in (root / "images" / split, root / "images" / split / "images"):
                if base.is_dir():
                    for folder in base.iterdir():
                        if folder.is_dir() and folder.name != "images" and NAME.fullmatch(folder.name):
                            found.setdefault(folder.name, set()).add(split)
        return [{"recording_id": name, "splits": sorted(splits)} for name, splits in sorted(found.items())]

    def recording(self, recording_id):
        if not isinstance(recording_id, str) or not any(
            item["recording_id"] == recording_id for item in self._catalog
        ):
            raise ValueError("unknown recording")
        with self._lock:
            if recording_id not in self._recordings:
                from trackrag.emt import open_recording
                self._recordings[recording_id] = open_recording(
                    self.config.data_root, recording_id, frame_rate=self.config.frame_rate,
                    split=self.config.split,
                )
            return self._recordings[recording_id]

    def frames(self, recording_id):
        recording = self.recording(recording_id)
        return {"recording_id": recording_id, "frame_ids": list(recording.frame_ids),
                "frame_rate": recording.frame_rate}

    def _condition(self):
        trained = bool((self._grounding_data or {}).get("adapter_dir"))
        return {
            "development_adapters": self.config.adapters,
            "vision": "development" if self._adapters else self.config.vision_mode,
            "grounding": "development" if self._adapters else
                "not_configured" if self._grounding_data is None else "trained_configuration" if trained else "pretrained",
            "model_id": (self._grounding_data or {}).get("model_id", "Development adapter" if self._adapters else "Not configured"),
            "qa_answerer": "development" if self._adapters and callable(getattr(self._adapters, "answer_question", None))
                else self.config.qa_answerer,
            "speech": self._native_speech.engine if self._native_speech else
                self.config.speech_adapter or (self.config.adapters if self._transcribe else "unavailable"),
            "speech_model_path": str(self._native_speech.model_path) if self._native_speech else None,
        }

    def register_media(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.output_dir) or not path.is_file():
            raise ValueError("media must be a saved file in this application run")
        token = sha256(str(path).encode("utf-8")).hexdigest()
        self._media[token] = path
        return "/media/" + token

    def media(self, token):
        with self._lock:
            if token not in self._media:
                raise KeyError("unknown media")
            return self._media[token]

    def view(self):
        with self._lock:
            interaction = self.controller.snapshot()
            state = interaction["state"]
            binding = None
            if state is not None and state["proposal"] is not None:
                proposal = state["proposal"]
                binding = {"proposal_id": proposal["proposal_id"], "request_id": proposal["request_id"],
                           "scene": deepcopy(proposal["scene"]),
                           "proposal_sha256": proposal_fingerprint(proposal, state["request"])}
            last_outcome = next((call["outcome"] for call in reversed(interaction["calls"])
                                 if state is not None and call["request"]["request_id"] == state["request"]["request_id"]), None)
            selected = None
            if self._selected is not None:
                scene = self._selected["scene"]
                selected = {"token": self._selection_token, "frame": deepcopy(scene["frame"]),
                            "tracks": deepcopy(scene["tracks"]), "source": scene["source"],
                            "image_url": self._selected["image_url"],
                            "provenance": deepcopy(scene.get("provenance", {}))}
            return {"session_id": self.controller.session_id, "recordings": deepcopy(self._catalog),
                    "condition": self._condition(), "selected": selected,
                    "selection": deepcopy(next((call for call in reversed(self._selection_calls)
                                                if call["scene_token"] == self._selection_token), None)),
                    "state": state, "proposal_binding": binding, "last_outcome": deepcopy(last_outcome),
                    "audio": deepcopy(self._audio_current),
                    "qa": deepcopy([call for call in self._qa_calls if call["scene_token"] == self._selection_token]),
                    "jobs": deepcopy(self._jobs[-20:]),
                    "speech_available": callable(self._transcribe)}

    def audit(self):
        with self._lock:
            return {"schema_version": "trackrag.application.v1", "condition": self._condition(),
                    "grounding": self.controller.snapshot(), "scene_calls": deepcopy(self._selection_calls),
                    "qa_calls": deepcopy(self._qa_calls), "audio_calls": deepcopy(self._audio_calls),
                    "api_errors": deepcopy(self._api_errors)}

    def _persist(self):
        payload = json.dumps(self.audit(), indent=2, allow_nan=False) + "\n"
        temporary = self.output_dir / "session.json.tmp"
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.output_dir / "session.json")

    def record_api_error(self, message):
        with self._lock:
            self._api_errors.append({"code": "invalid_application_request", "message": str(message)[:4096]})
            self._persist()

    def _job(self, kind, function):
        job = {"job_id": str(uuid4()), "kind": kind, "status": "pending", "error": None}
        self._jobs.append(job)

        def run():
            try:
                function()
            except Exception as error:
                with self._lock:
                    job["error"] = {"code": f"{kind}_failed", "message": f"{type(error).__name__}: {error}"[:4096]}
                    job["status"] = "error"
            else:
                with self._lock:
                    job["status"] = "completed"
            finally:
                with self._lock:
                    self._persist()

        self._executor.submit(run)
        return deepcopy(job)

    def _check_scene(self, token):
        if self._selected is None or token != self._selection_token:
            raise InvalidTransition("scene changed; select the scene again")
        return self._selected

    def select(self, payload):
        _fields(payload, ("recording_id", "frame_id"))
        rid, cutoff = payload["recording_id"], payload["frame_id"]
        with self._lock:
            self.controller.reset_scene()
            self._selected, self._audio_current, self._audio_intent = None, None, None
            token = self._selection_token = str(uuid4())
            receipt = {"recording_id": rid, "frame_id": cutoff, "scene_token": token,
                       "status": "pending", "error": None}
            self._selection_calls.append(receipt)

            def prepare():
                try:
                    if type(cutoff) is not int or cutoff < 0:
                        raise ValueError("frame_id must be an original nonnegative integer")
                    recording = self.recording(rid)
                    bounded = RecordingPrefix(recording, cutoff)
                    if self._adapters:
                        rows, provenance = self._adapters.predict_tracks_with_receipt(
                            bounded, config=self._vision, directory=self.config.predictions)
                    elif self.config.vision_mode == "saved":
                        from tools.vision_saved_adapters import predict_tracks_with_receipt
                        rows, provenance = predict_tracks_with_receipt(
                            bounded, config=self._vision, directory=self.config.predictions)
                    else:
                        from trackrag.vision import predict_tracks
                        rows = predict_tracks(bounded, config=self._vision)
                        provenance = {"kind": "fresh_detector_inference", "fresh_detector_inference": True,
                                      "vision_identity_hash": self._vision.identity_hash()}
                    rows = list(rows)
                    # Neither QA nor grounding may receive future or foreign rows.
                    for row in rows:
                        if (row.get("recording_id") != rid or row.get("source") != "predicted" or
                                type(row.get("frame_id")) is not int or row["frame_id"] > cutoff):
                            raise ValueError("vision returned a foreign source/recording or future observation")
                    provenance = {**provenance, "application_condition": self._condition(),
                                  "prediction_image_hash_binding": "development_outputs_not_image_grounded"
                                  if self._adapters else "not_verified_in_historical_artifact"
                                  if self.config.vision_mode == "saved" else "image_snapshotted_after_vision_call"}
                    prepared = build_instruction_request(
                        recording, rows, frame_id=cutoff, request_id="scene-" + token,
                        text="Scene selection", provenance=provenance,
                    )
                    scene = prepared["scene"]
                    original_path = Path(scene["frame"]["image_path"])
                    with original_path.open("rb") as stream:
                        image = stream.read(MAX_IMAGE_BYTES + 1)
                    if len(image) > MAX_IMAGE_BYTES or sha256(image).hexdigest() != scene["provenance"]["image_sha256"]:
                        raise ValueError("selected image changed after validation")
                    destination = self.output_dir / "images" / (scene["provenance"]["image_sha256"] + ".image")
                    with self._lock:
                        if destination.exists():
                            with destination.open("rb") as stream:
                                cached_image = stream.read(MAX_IMAGE_BYTES + 1)
                            if sha256(cached_image).hexdigest() != scene["provenance"]["image_sha256"]:
                                raise ValueError("saved scene image snapshot changed")
                        else:
                            destination.write_bytes(image)
                        scene["frame"]["image_path"] = str(destination)
                        receipt.update(status="prepared", provenance=deepcopy(provenance),
                                       image_sha256=scene["provenance"]["image_sha256"])
                        if token != self._selection_token:
                            receipt["status"] = "discarded"
                            return
                        self._selected = {"scene": scene, "recording": bounded, "rows": deepcopy(rows),
                                          "source_image_path": original_path,
                                          "image_url": self.register_media(destination)}
                except Exception as error:
                    with self._lock:
                        receipt.update(status="error", error={"code": "scene_unavailable",
                                       "message": f"{type(error).__name__}: {error}"[:4096]})
                    raise

            job = self._job("scene", prepare)
            self._persist()
            return job

    def reset_selection(self, payload):
        _fields(payload)
        with self._lock:
            self.controller.reset_scene()
            self._selected, self._audio_current, self._audio_intent = None, None, None
            self._selection_token = str(uuid4())
            self._persist()

    def _audio_fields(self, audio_id):
        if audio_id is None:
            return {"input_mode": "typed"}
        audio = next((item for item in self._audio_calls if item["audio_id"] == audio_id), None)
        if audio is None or audio["status"] != "transcribed" or audio["scene_token"] != self._selection_token:
            raise ValueError("audio transcript is unavailable or belongs to another scene")
        return {"input_mode": "voice", "original_transcript": audio["text"]}

    def interpret(self, payload, action="instruction"):
        _fields(payload, ("scene_token", "text"), ("audio_id", "request_id"))
        with self._lock:
            selected = self._check_scene(payload["scene_token"])
            audio = self._audio_fields(payload.get("audio_id"))
            if action == "instruction":
                request = {"request_id": str(uuid4()), "session_id": self.controller.session_id,
                           "scene": deepcopy(selected["scene"]), "text": payload["text"],
                           "dialogue": [], **audio}
                pending = self.controller.begin(request)
            else:
                current = self.controller.state
                if current is None or current["request"]["request_id"] != payload.get("request_id"):
                    raise InvalidTransition("instruction changed; refresh before replying or correcting")
                pending = (self.controller.begin_reply(payload["text"], **audio) if action == "reply"
                           else self.controller.begin_revision(payload["text"]))
            self._audio_intent = None
            self._audio_current = None
            job = self._job("grounding", lambda: self.controller.run_pending(pending["request"]["request_id"]))
            self._persist()
            return job

    def confirm(self, payload):
        _fields(payload, ("scene_token", "confirmation"))
        with self._lock:
            self._check_scene(payload["scene_token"])
            self.controller.confirm(payload["confirmation"])
            self._persist()

    def cancel(self, payload):
        _fields(payload, ("scene_token",), ("discard_audio",))
        if type(payload.get("discard_audio", False)) is not bool:
            raise ValueError("discard_audio must be a boolean")
        with self._lock:
            self._check_scene(payload["scene_token"])
            if self.controller.state is not None:
                self.controller.cancel()
            self._audio_intent = None
            if payload.get("discard_audio"):
                self._audio_current = None
            self._persist()

    def qa(self, payload):
        _fields(payload, ("scene_token", "text"))
        with self._lock:
            selected = self._check_scene(payload["scene_token"])
            text = payload["text"]
            if not isinstance(text, str) or not text.strip() or len(text) > 8192:
                raise ValueError("question must be nonempty text of at most 8192 characters")
            frame = selected["scene"]["frame"]
            question = {"question_id": str(uuid4()), "recording_id": frame["recording_id"],
                        "query_frame": frame["frame_id"], "question": text}
            call = {"question": question, "scene_token": self._selection_token, "status": "pending",
                    "result": None, "error": None, "run_dir": None}
            self._qa_calls.append(call)

            def review():
                try:
                    from trackrag.retrieval import build_index, rank_evidence
                    from trackrag.review import review_recording
                    answerer = None
                    if self._adapters and callable(getattr(self._adapters, "answer_question", None)):
                        answerer = self._adapters.answer_question
                    elif self.config.qa_answerer == "gemini":
                        from trackrag.answering import answer_question
                        answerer = answer_question
                    run_dir = self.output_dir / "qa" / question["question_id"]
                    with selected["source_image_path"].open("rb") as stream:
                        current_image = stream.read(MAX_IMAGE_BYTES + 1)
                    if (len(current_image) > MAX_IMAGE_BYTES or
                            sha256(current_image).hexdigest() != selected["scene"]["provenance"]["image_sha256"]):
                        raise ValueError("Original image changed; reopen the scene before requesting QA")
                    result_path = review_recording(
                        selected["recording"], [question], source="predicted",
                        load_tracks=lambda recording: deepcopy(selected["rows"]),
                        build_index=build_index, rank_evidence=rank_evidence,
                        answer_question=answerer, output_dir=run_dir,
                        run_metadata={"application_condition": self._condition(),
                                      "vision_provenance": selected["scene"]["provenance"]},
                    )
                    result = parse_json(result_path.read_text(encoding="utf-8").strip())
                    with self._lock:
                        for item in result["evidence"]:
                            for image in item["images"]:
                                image["image_url"] = self.register_media(run_dir / image["image_path"])
                        call.update(status="completed", result=result,
                                    run_dir=str(run_dir.relative_to(self.output_dir)))
                except Exception as error:
                    with self._lock:
                        call.update(status="error", error={"code": "qa_pipeline_error",
                                    "message": f"{type(error).__name__}: {error}"[:4096]})
                    raise

            job = self._job("qa", review)
            self._persist()
            return job

    def audio(self, token, filename, content):
        with self._lock:
            self._check_scene(token)
            self.controller.reset_scene()
            self._audio_current, self._audio_intent = None, None
            suffix = Path(filename).suffix.lower()
            if suffix not in AUDIO_SUFFIXES or not content or len(content) > MAX_AUDIO_BYTES:
                self._persist()
                raise ValueError("supply a supported nonempty audio file of at most 16 MiB")
            identity = self._audio_intent = str(uuid4())
            self._audio_current = None
            path = self.output_dir / "audio" / (identity + suffix)
            path.write_bytes(content)
            call = {"audio_id": identity, "scene_token": token, "status": "pending", "text": None,
                    "error": None, "audio_path": str(path.relative_to(self.output_dir)),
                    "sha256": sha256(content).hexdigest(), "provenance": None}
            self._audio_calls.append(call)

            def recognize():
                try:
                    if not callable(self._transcribe):
                        raise RuntimeError("Task 3 speech adapter is unavailable; supply --speech-adapter, --asr-model or --whisper-model")
                    result = self._transcribe(path)
                    _fields(result, ("text",), ("provenance",))
                    text = result["text"]
                    if not isinstance(text, str) or not text.strip() or len(text) > 8192:
                        raise ValueError("speech adapter returned an empty or invalid transcript")
                    if not isinstance(result.get("provenance", {}), dict):
                        raise ValueError("speech provenance must be an object")
                    declared_audio = result.get("provenance", {}).get("audio_sha256")
                    if declared_audio is not None and declared_audio != call["sha256"]:
                        raise ValueError("speech receipt belongs to different audio bytes")
                    json.dumps(result, allow_nan=False)
                    with self._lock:
                        call.update(status="transcribed", text=text, provenance=deepcopy(result.get("provenance", {})))
                        if token == self._selection_token and identity == self._audio_intent:
                            self._audio_current = deepcopy(call)
                        else:
                            call["status"] = "discarded"
                except Exception as error:
                    with self._lock:
                        call.update(status="error", error={"code": error.code if isinstance(error, SpeechInputError)
                                    else "transcription_error",
                                    "message": f"{type(error).__name__}: {error}"[:4096]})
                    raise

            job = self._job("speech", recognize)
            self._persist()
            return job

    def close(self):
        # A streamed model owns a persistent GPU worker. Drain application jobs
        # before releasing it, and make repeated/concurrent shutdown harmless.
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            try:
                with self._lock:
                    self._persist()
            finally:
                try:
                    self._executor.shutdown(wait=True)
                finally:
                    try:
                        close = getattr(self._owned_model, "close", None)
                        if callable(close):
                            close()
                    except Exception as error:
                        with self._lock:
                            self._api_errors.append({
                                "code": "grounding_shutdown_failed",
                                "message": f"{type(error).__name__}: {error}"[:4096],
                            })
                        raise
                    finally:
                        with self._lock:
                            self._persist()


def make_server(application, port=8765):
    """Serve registered assets and explicit actions on loopback only."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, payload, content_type="application/json; charset=utf-8"):
            if not isinstance(payload, bytes):
                payload = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            try:
                if path == "/api/state":
                    return self.send(200, application.view())
                if path == "/api/audit":
                    return self.send(200, application.audit())
                if path.startswith("/api/frames/"):
                    return self.send(200, application.frames(path.removeprefix("/api/frames/")))
                if path.startswith("/media/"):
                    file = application.media(path.removeprefix("/media/"))
                    # Image snapshots retain exact bytes independent of their name.
                    kind = mimetypes.guess_type(file)[0]
                    if file.suffix == ".image":
                        from PIL import Image
                        with Image.open(file) as image:
                            kind = Image.MIME.get(image.format, "application/octet-stream")
                    return self.send(200, file.read_bytes(), kind or "application/octet-stream")
                names = {"/": "index.html", "/app.js": "app.js", "/app.css": "app.css", "/capture.js": "capture.js"}
                if path not in names:
                    return self.send(404, {"error": "not_found"})
                file = WEB_ROOT / names[path]
                return self.send(200, file.read_bytes(), mimetypes.guess_type(file)[0] or "text/plain")
            except (ValueError, KeyError, OSError) as error:
                return self.send(400, {"error": str(error)})

        def do_POST(self):
            path = urlsplit(self.path).path
            origin = self.headers.get("Origin")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_AUDIO_BYTES:
                    raise ValueError("invalid or oversized request body")
                payload = self.rfile.read(length)
                # Drain a bounded request before rejecting it. Closing a socket
                # with unread request bytes can reset it on Windows, hiding the
                # intended HTTP error from the browser/client.
                if origin is not None and origin != "http://" + self.headers.get("Host", ""):
                    return self.send(403, {"error": "origin_mismatch"})
                if path == "/api/audio":
                    result = application.audio(self.headers.get("X-Scene-Token"),
                                               self.headers.get("X-Filename", "recording.webm"), payload)
                else:
                    if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                        raise ValueError("expected application/json")
                    value = parse_json(payload.decode("utf-8"))
                    actions = {"/api/select": application.select, "/api/instruction": application.interpret,
                               "/api/reset": application.reset_selection,
                               "/api/reply": lambda item: application.interpret(item, "reply"),
                               "/api/revise": lambda item: application.interpret(item, "revise"),
                               "/api/qa": application.qa, "/api/confirm": application.confirm,
                               "/api/cancel": application.cancel}
                    if path not in actions:
                        return self.send(404, {"error": "not_found"})
                    result = actions[path](value)
                self.send(202 if result else 200, result or {"ok": True})
            except (ValueError, KeyError, OSError) as error:
                application.record_api_error(error)
                self.send(409 if isinstance(error, InvalidTransition) else 400, {"error": str(error)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="new run directory; defaults to a unique folder under runs/")
    parser.add_argument("--vision-config", type=Path)
    parser.add_argument("--vision-mode", choices=("saved", "fresh"), default="saved")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--grounding-config", type=Path)
    parser.add_argument("--grounding-backend", choices=("transformers", "ollama"), default="transformers")
    parser.add_argument("--qa-answerer", choices=("none", "gemini"), default="gemini")
    speech = parser.add_mutually_exclusive_group()
    speech.add_argument("--speech-adapter", help="Task 3 module exposing transcribe_audio(path)")
    speech.add_argument("--asr-model", type=Path, help="verified local Vosk English model; accepts 16-bit PCM WAV")
    speech.add_argument("--whisper-model", type=Path, help="verified local Whisper Small English model on CPU; accepts PCM WAV")
    parser.add_argument("--adapters", help="explicit development adapters; labelled in UI and receipts")
    parser.add_argument("--frame-rate", type=float)
    parser.add_argument("--split", choices=("train", "test"))
    parser.add_argument("--port", type=int, default=8765)
    args = vars(parser.parse_args(argv))
    port = args.pop("port")
    args["output_dir"] = args.pop("out") or Path("runs") / ("dropoff-app-" + str(uuid4())[:12])
    application = None
    server = None
    try:
        try:
            application = LocalApplication(AppConfig(**args))
            server = make_server(application, port)
        except (ValueError, OSError, ImportError, SpeechInputError) as error:
            parser.error(str(error))
        print(f"TrackRAG-Gulf local application: http://127.0.0.1:{server.server_port}", flush=True)
        print(f"Run receipts: {application.output_dir}", flush=True)
        print(f"Execution condition: {json.dumps(application._condition())}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        try:
            if server is not None:
                server.server_close()
        finally:
            if application is not None:
                application.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
