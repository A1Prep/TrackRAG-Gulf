"""Run the passenger drop-off demonstration locally with a prepared scene bank."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import secrets
import tempfile
import threading
from contextlib import nullcontext
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.inputs import MAX_IMAGE_BYTES, _inspect_image
from trackrag.speech import LocalSpeechRecognizer, SpeechInputError
from trackrag.local_speech import MAX_WAV_BYTES


ASSETS = Path(__file__).with_name("passenger_ui")
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/footage": ("footage.html", "text/html; charset=utf-8"),
          "/footage.js": ("footage.js", "text/javascript"),
          "/frames.js": ("frames.js", "text/javascript"),
          "/pair-choices.js": ("pair-choices.js", "text/javascript"),
          "/app.css": ("app.css", "text/css"), "/app.js": ("app.js", "text/javascript"),
          "/capture.js": ("capture.js", "text/javascript")}


def scene_key(recording_id, frame_id):
    return hashlib.sha256(json.dumps((recording_id, frame_id)).encode()).hexdigest()[:24]


def public_frame(key, scene):
    frame = scene["frame"]
    return {"key":key, **{field:frame[field] for field in ("recording_id","frame_id","timestamp_s","width","height")}}


def load_scenes(banks):
    """Read only the scene input from each row and verify its declared original bytes."""
    catalog = {}
    seen = {}
    for bank in banks:
        with Path(bank).open(encoding="utf-8-sig") as rows:
            for line in rows:
                if not line.strip():
                    continue
                scene = json.loads(line)["scene"]
                scene = validate_request({"scene": scene, "text": "Open frame", "input_mode": "typed",
                                          "dialogue": [], "request_id": "catalog"})["scene"]
                frame = scene["frame"]
                image_path = Path(frame["image_path"]).expanduser()
                if not image_path.is_absolute():
                    image_path = Path(bank).resolve().parent / image_path
                frame["image_path"] = str(image_path.resolve())
                key = (frame["recording_id"], frame["frame_id"])
                if key in seen:
                    if seen[key] != scene:
                        raise ValueError(f"Conflicting scene inputs supplied for {key}.")
                    continue
                path, digest = _inspect_image(frame)
                if scene.get("provenance", {}).get("image_sha256") != digest:
                    raise ValueError(f"Prepared image hash is missing or does not match {key}.")
                # Keep a stable, resolved path for the model and later image requests.
                frame["image_path"] = str(path)
                catalog[scene_key(*key)] = scene
                seen[key] = scene
    if not catalog:
        raise ValueError("Supply a prepared request bank with at least one recorded scene.")
    return catalog


class PassengerRequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def archive_wav(directory, data):
    """Publish the complete input bytes before recording any speech or dialogue receipt."""
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    target = directory / f"{digest}.wav"
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
        return digest
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as temp:
        partial = Path(temp.name)
        try:
            temp.write(data)
            temp.flush()
        except BaseException:
            temp.close()
            partial.unlink(missing_ok=True)
            raise
    try:
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return digest


class BrowserSession:
    def __init__(self, identity, scene_key, dialogue, directory):
        self.identity = identity
        self.scene_key = scene_key
        self.dialogue = dialogue
        self.directory = directory
        self.lock = threading.Lock()
        self.revision = 0
        self.voice_stage = None
        self.receipt = None
        self.speech_error = None
        self.retired = False

    def snapshot(self):
        view = dict(self.dialogue.view())
        if self.voice_stage:
            # Replacing a recording invalidates any previously displayed confirmation.
            view.update(proposal_id=None, result=None)
            if "pair_choices" in view:
                view["pair_choices"] = []
        return {"session": self.identity, "scene": self.scene_key, "revision": self.revision,
                "view": view, "voice_stage": self.voice_stage, "speech_error": self.speech_error,
                "audio_sha256": self.receipt["audio_sha256"] if self.receipt else None,
                "transcript": self.receipt["transcript"] if self.receipt else None}

    def clear_audio(self):
        self.voice_stage = self.receipt = self.speech_error = None


class PassengerSite:
    def __init__(self, scenes, model, recognizer, out, session_factory, footage=None):
        self.scenes = scenes
        self.footage = footage
        self.footage_scenes = {key for key, scene in scenes.items() if footage is not None and footage.covers(scene)}
        self.model = model
        self.recognizer = recognizer
        self.out = Path(out)
        self.session_factory = session_factory
        self.sessions = {}
        self.session_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.scene_lock = threading.Lock()

    def catalog(self):
        with self.scene_lock:
            frames = [public_frame(key, s) for key, s in self.scenes.items()]
            covered = sorted(self.footage_scenes)
        return {"scenes": sorted(frames, key=lambda f: (f["recording_id"], f["frame_id"])),
                "speech_available": self.recognizer is not None, "footage_available": bool(covered),
                "footage_scenes": covered}

    def open_recording_frame(self, origin, frame_id, *, for_footage=False):
        if self.footage is None:
            raise PassengerRequestError(503,"Attach the original recording and saved predictions to open other frames.")
        if not isinstance(origin,str) or origin not in self.scenes or type(frame_id) is not int or frame_id < 0:
            raise PassengerRequestError(400,"Choose a loaded recording and a nonnegative whole frame number.")
        if type(for_footage) is not bool:
            raise PassengerRequestError(400,"Invalid frame selection.")
        template = self.scenes[origin]
        key = scene_key(template["frame"]["recording_id"], frame_id)
        with self.scene_lock:
            scene = self.scenes.get(key)
            if scene is not None and for_footage and key not in self.footage_scenes:
                raise PassengerRequestError(400, "Only a passenger photo is available at that frame. Choose a frame in the attached recording.")
            if scene is None and origin not in self.footage_scenes:
                template = next((s for k, s in self.scenes.items() if k in self.footage_scenes and
                                 s["frame"]["recording_id"] == template["frame"]["recording_id"]), template)
        if scene is None:
            scene = self.footage.frame_scene(template, frame_id)
            with self.scene_lock:
                existing = self.scenes.setdefault(key, scene)
                if existing != scene:
                    raise PassengerRequestError(409,"The recording changed while this frame was opening.")
                self.footage_scenes.add(key)
        return public_frame(key, scene)

    def open(self, scene_key, previous=None):
        if not isinstance(scene_key, str) or scene_key not in self.scenes:
            raise PassengerRequestError(400, "Choose an available frame.")
        old = self.lookup(previous) if previous else None
        transition_lock = old.lock if old else nullcontext()
        with transition_lock:
            if old and old.retired:
                raise PassengerRequestError(409, "This frame has already been replaced.")
            sid = secrets.token_urlsafe(24)
            directory = self.out / sid
            # The dialogue verifies the next original frame before the current one is retired.
            dialogue = self.session_factory(self.scenes[scene_key], self.model, audit_dir=directory)
            session = BrowserSession(sid, scene_key, dialogue, directory)
            with self.session_lock:
                if old:
                    # Navigation retires the old link. Only an explicit passenger action cancels a request.
                    old.clear_audio()
                    old.retired = True
                    self.sessions.pop(old.identity, None)
                self.sessions[sid] = session
        return session.snapshot()

    def lookup(self, sid):
        if not isinstance(sid, str):
            raise PassengerRequestError(400, "Invalid session.")
        with self.session_lock:
            session = self.sessions.get(sid)
        if session is None:
            raise PassengerRequestError(404, "This passenger session is no longer open.")
        return session

    def image(self, sid):
        s = self.lookup(sid)
        scene = self.scenes[s.scene_key]
        with Path(scene["frame"]["image_path"]).open("rb") as source:
            data = source.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES or hashlib.sha256(data).hexdigest() != scene["provenance"]["image_sha256"]:
            raise PassengerRequestError(409, "The original image changed. Prepare this scene again.")
        with Image.open(io.BytesIO(data)) as image:
            mime = Image.MIME.get(image.format)
        if mime not in ("image/png", "image/jpeg", "image/webp"):
            raise PassengerRequestError(415, "This frame format cannot be shown in the browser.")
        return mime, data

    def change(self, sid, revision, action, payload):
        s = self.lookup(sid)
        with s.lock:
            if s.retired or type(revision) is not int or revision != s.revision:
                raise PassengerRequestError(409, "This action is out of date. Use the current frame and proposal.")
            if action == "begin-audio":
                s.clear_audio()
                s.voice_stage = "recording"
            elif action == "audio":
                if s.voice_stage != "recording":
                    raise PassengerRequestError(409, "Start a new audio input first.")
                s.voice_stage = "failed"
                s.receipt = None
                if not isinstance(payload, bytes):
                    raise PassengerRequestError(400, "Upload WAV audio.")
                digest = None
                try:
                    if self.recognizer is None:
                        raise SpeechInputError("asr_unavailable", "The speech model was not supplied at startup.")
                    if not payload or len(payload) > MAX_WAV_BYTES:
                        raise SpeechInputError("invalid_audio_size", "Record or upload a WAV file up to 12 MiB.")
                    digest = archive_wav(s.directory / "audio", payload)
                    receipt = asdict(self.recognizer.transcribe_wav(payload))
                    if receipt["audio_sha256"] != digest:
                        raise SpeechInputError("speech_receipt_mismatch", "The speech receipt belongs to different audio.")
                    s.receipt = receipt
                    s.voice_stage = "review"
                except SpeechInputError as error:
                    s.speech_error = {"code": error.code, "message": str(error)}
                folder = s.directory / "audio"
                folder.mkdir(exist_ok=True)
                with (folder / f"transcription-{s.revision + 1}.json").open("x", encoding="utf-8") as report:
                    json.dump({"audio_sha256": digest, "receipt": s.receipt, "error": s.speech_error},
                              report, allow_nan=False, indent=2)
            elif action == "submit":
                text = payload.get("text")
                mode = payload.get("mode")
                if not isinstance(text, str) or not text.strip() or len(text) > 4096:
                    raise PassengerRequestError(400, "Enter a passenger instruction.")
                if mode == "voice" and (s.voice_stage != "review" or s.receipt is None):
                    raise PassengerRequestError(409, "Record or upload audio and review its transcript first.")
                if mode not in ("typed", "voice"):
                    raise PassengerRequestError(400, "Unknown input mode.")
                pointing = payload.get("point_to_vehicles", False)
                if type(pointing) is not bool:
                    raise PassengerRequestError(400, "Invalid vehicle selection mode.")
                if pointing and not s.dialogue.view().get("can_request_pointing"):
                    raise PassengerRequestError(409, "Vehicle selection is unavailable for this scene.")
                receipt = s.receipt if mode == "voice" else None
                with self.model_lock:
                    kwargs = {"point_to_vehicles":True} if pointing else {}
                    s.dialogue.submit(text, original_transcript=receipt["transcript"] if receipt else None, speech=receipt, **kwargs)
                s.clear_audio()
            elif action == "area":
                if s.voice_stage:
                    raise PassengerRequestError(409, "Finish your new instruction before marking its area.")
                with self.model_lock:
                    s.dialogue.select_area(payload.get("box"))
            elif action == "select-pair":
                if s.voice_stage:
                    raise PassengerRequestError(409, "Finish your new instruction before selecting its vehicles.")
                with self.model_lock:
                    s.dialogue.select_pair(payload.get("track_ids"), payload.get("placement"))
            elif action == "confirm":
                token = payload.get("proposal_id")
                current = s.dialogue.view()
                if (s.voice_stage or current["state"] != "proposed" or not isinstance(token, str)
                        or not token or token != current.get("proposal_id")):
                    raise PassengerRequestError(409, "Confirm the currently displayed proposal.")
                s.dialogue.confirm(token)
            elif action == "cancel":
                if s.dialogue.view()["state"] in ("confirmed", "cancelled"):
                    if not s.voice_stage:
                        raise PassengerRequestError(409, "This request has already finished.")
                else:
                    s.dialogue.cancel()
                s.clear_audio()
            else:
                raise PassengerRequestError(404, "Unknown passenger action.")
            s.revision += 1
            return s.snapshot()


def handler(site):
    class Handler(BaseHTTPRequestHandler):
        def send_data(self, status, mime, data):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def respond(self, value, status=200):
            self.send_data(status, "application/json", json.dumps(value, allow_nan=False).encode("utf-8"))

        def read_body(self, limit):
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= limit:
                raise PassengerRequestError(413, "Upload exceeds the allowed size or has no length.")
            data = self.rfile.read(length)
            if len(data) != length:
                raise PassengerRequestError(400, "Incomplete upload.")
            return data

        def serve(self, method):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if host not in allowed or self.headers.get("Origin") not in (None, f"http://{host}"):
                raise PassengerRequestError(403, "Use this application from its local address.")
            path = urlsplit(self.path).path
            if method == "GET":
                if path in STATIC:
                    filename, mime = STATIC[path]
                    return self.send_data(200, mime, (ASSETS / filename).read_bytes())
                if path == "/api/scenes":
                    return self.respond(site.catalog())
                match = re.fullmatch(r"/api/footage/([0-9a-f]{32})/(\d{1,2})/(\d{1,2})",path)
                if match and site.footage:
                    try:
                        mime, payload = site.footage.evidence_image(match[1],int(match[2]),int(match[3]))
                    except (FileNotFoundError, ValueError):
                        raise PassengerRequestError(404,"Evidence image unavailable.")
                    return self.send_data(200,mime,payload)
                m = re.fullmatch(r"/api/sessions/([A-Za-z0-9_-]+)(/image)?", path)
                if m:
                    if m[2]:
                        mime, data = site.image(m[1])
                        return self.send_data(200, mime, data)
                    s = site.lookup(m[1])
                    with s.lock:
                        return self.respond(s.snapshot())
            else:
                m = re.fullmatch(r"/api/sessions/([A-Za-z0-9_-]+)/([a-z-]+)", path)
                is_audio = m and m[2] == "audio"
                expected = "audio/wav" if is_audio else "application/json"
                if self.headers.get_content_type() != expected:
                    raise PassengerRequestError(415, f"Expected {expected}.")
                payload = self.read_body(MAX_WAV_BYTES if is_audio else 32768)
                if not is_audio:
                    payload = json.loads(payload)
                    if not isinstance(payload, dict):
                        raise PassengerRequestError(400, "A JSON object is required.")
                if path == "/api/sessions":
                    return self.respond(site.open(payload.get("scene"), payload.get("previous")))
                if path == "/api/frames":
                    return self.respond(site.open_recording_frame(payload.get("scene"),payload.get("frame_id"),
                                                                  for_footage=payload.get("for_footage", False)))
                if path == "/api/footage":
                    if not site.footage:
                        raise PassengerRequestError(503,"Start the app with recording and prediction folders to enable footage search.")
                    if payload.get("scene") not in site.footage_scenes:
                        raise PassengerRequestError(400,"Choose a frame in the attached footage recording.")
                    return self.respond(site.footage.ask(site.scenes[payload["scene"]],payload.get("question"),
                                                         hosted=payload.get("hosted",False)))
                if m:
                    revision = int(self.headers.get("X-Session-Revision", "-1"))
                    return self.respond(site.change(m[1], revision, m[2], payload))
            raise PassengerRequestError(404, "Page not found.")

        def dispatch(self):
            try:
                self.serve(self.command)
            except PassengerRequestError as error:
                self.respond({"error": str(error)}, error.status)
            except (ValueError, KeyError, TypeError) as error:
                self.respond({"error": str(error)}, 400)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                pass
            except Exception:
                import traceback
                traceback.print_exc()
                self.respond({"error": "Could not finish the operation. Check the application console."}, 500)

        do_GET = do_POST = dispatch

        def log_message(self, *args):
            # Request paths hold opaque session tokens and do not belong in the console.
            pass

    return Handler


def load_speech_recognizer(vosk_model, whisper_model=None):
    if whisper_model:
        from trackrag.whisper_speech import WhisperSpeechRecognizer
        return WhisperSpeechRecognizer(whisper_model)
    return LocalSpeechRecognizer(vosk_model) if vosk_model is not None else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", action="append", type=Path, required=True, help="Prepared JSONL scene bank; repeat for additional banks")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--referent",type=Path,help="Use the learned candidate ranker for clarification")
    parser.add_argument("--qa-data-root",type=Path)
    parser.add_argument("--qa-predictions",type=Path)
    parser.add_argument("--qa-frame-rate",type=float,help="Used if the prepared EMT frames used an explicit rate")
    parser.add_argument("--qa-vision-config",type=Path,default=Path("configs/vision/finetuned.json"))
    parser.add_argument("--vision-cache", type=Path, required=True)
    parser.add_argument("--between-road-mode", choices=("edge", "edge-or-bay"), default="edge",
                        help="Check the visible road edge, or also check angled and facing parking bays")
    parser.add_argument("--verify-landmarks", action="store_true", help="Enable the model's additional landmark check")
    parser.add_argument("--asr-model", type=Path, help="Unpacked local Vosk English model")
    parser.add_argument("--whisper-model", type=Path, help="Use this verified local English Whisper Small model on CPU")
    parser.add_argument("--out", type=Path, required=True, help="Local session receipts and source WAVs")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    from trackrag.dropoff.passenger_pair_choice import PassengerSession
    from trackrag.dropoff.passenger import load_passenger_model

    scenes = load_scenes(args.requests)
    asr = load_speech_recognizer(args.asr_model, args.whisper_model)
    model = load_passenger_model(args.config, geometry_path=args.geometry, vision_cache=args.vision_cache,
                                 verify_landmarks=args.verify_landmarks, referent_path=args.referent,
                                 between_road_mode=args.between_road_mode)
    dialogue_model = model
    if args.referent:
        from trackrag.dropoff.infrastructure import PassengerDialogueGrounder
        dialogue_model = PassengerDialogueGrounder(model)
    footage = None
    if args.qa_data_root or args.qa_predictions:
        if not (args.qa_data_root and args.qa_predictions):
            parser.error("Footage search needs both --qa-data-root and --qa-predictions.")
        from trackrag.footage_app import FootageReview
        footage = FootageReview(args.qa_data_root,args.qa_predictions,args.qa_vision_config,args.out/"footage",args.qa_frame_rate)
    site = PassengerSite(scenes, dialogue_model, asr, args.out, PassengerSession,footage)
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler(site)) as server:
        print(f"Open http://127.0.0.1:{server.server_port}/  ({len(scenes)} prepared frames)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            model.close()


if __name__ == "__main__":
    main()
