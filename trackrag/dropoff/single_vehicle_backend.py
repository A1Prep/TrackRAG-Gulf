"""Reuse signed, public-source vehicle evidence while binding each decision to the current passenger."""
import base64
from collections import OrderedDict
from concurrent.futures import Future, TimeoutError as FutureTimeout
import json
import logging
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from threading import Lock, Thread

from . import single_vehicle as vehicle
from .parking_evidence import MAX_IMAGE_BYTES, signed_bytes, source_input
from .passenger import digest

LOG = logging.getLogger(__name__)
INTERACTIVE_WAIT_SECONDS = 4.0
MAX_PENDING = 3
MEMORY_PACKETS = 12


class SingleVehicleBackend:
    def __init__(self, places, cache=None):
        self.places = places
        self.cache = Path(cache) if cache else places.root / "runs" / "model-cache" / "single-vehicle-rear-v1"
        self._index_lock = Lock()
        self._pending = {}
        self._memory = OrderedDict()

    def eligible(self, request):
        from .local_surfaces import local_only
        scene = request["scene"]
        frame = scene["frame"]
        source = (scene["provenance"].get("image_sha256"), frame["recording_id"], frame["frame_id"])
        return (not local_only(scene) and scene["source"] == "predicted" and source in self.places.allowed
                and vehicle.request_reference(request) is not None)

    def _save(self, path, packet):
        staged = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
                staged = Path(temp.name)
                temp.write(signed_bytes(packet))
            os.replace(staged, path)
        except OSError:
            # The current verified result and journal are independent of this optional cache.
            LOG.warning("Could not cache verified single-vehicle evidence on disk")
        finally:
            if staged is not None:
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    pass

    def _cached(self, key, path, request, data):
        with self._index_lock:
            packet = self._memory.get(key)
            if packet is not None:
                self._memory.move_to_end(key)
        try:
            if packet is None:
                if path.stat().st_size > 3_000_000:
                    return None
                packet = json.loads(path.read_bytes())
            return vehicle.outcome(request, data, packet)
        except (KeyError, OSError, TypeError, ValueError):
            return None

    def interpret(self, request):
        if not self.eligible(request):
            raise ValueError("This instruction and original are outside the public single-vehicle scope")
        original = Path(request["scene"]["frame"]["image_path"])
        if original.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Photograph is too large")
        data = original.read_bytes()
        request, _, _ = source_input(request, data)
        ctx = vehicle.context(request)
        key = digest({"recipe":vehicle.RECIPE, "context":ctx})
        saved = self.cache / (key + ".json")
        result = self._cached(key, saved, request, data)
        if result is not None:
            return result
        from . import manual_surfaces
        surface=manual_surfaces.get_packet(ctx["image_sha256"])
        if surface is not None:
            try:
                empty=vehicle.empty_visible_space(request,data,surface)
            except (TypeError,ValueError,KeyError):
                empty=None
            if empty is not None:
                return empty
        # Only the short in-memory registry operation holds this lock, never a GPU call.
        with self._index_lock:
            packet = self._memory.get(key)
            pending = self._pending.get(key)
            if packet is None and pending is None and len(self._pending) < MAX_PENDING:
                pending = self._pending[key] = Future()
                Thread(target=self._collect, args=(request,data,ctx,key,saved,pending), daemon=True).start()
        if packet is not None:
            return vehicle.outcome(request, data, packet)
        if pending is None:
            return vehicle.outcome(request, data, waiting="busy")
        try:
            packet = pending.result(timeout=INTERACTIVE_WAIT_SECONDS)
        except FutureTimeout:
            return vehicle.outcome(request, data, waiting="pending")
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, subprocess.SubprocessError):
            return vehicle.outcome(request, data)
        return vehicle.outcome(request, data, packet)

    def _collect(self, request, data, ctx, key, saved, pending):
        try:
            payload = {"context":ctx, "key":key, "photo":base64.b64encode(data).decode("ascii")}
            job = subprocess.run([self.places.python, "-m", "tools.passenger_single_vehicle_call"],
                    input=signed_bytes(payload), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=self.places.root, timeout=165, creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            if job.returncode or len(job.stdout)>3_000_000:
                raise RuntimeError("Private vehicle inspection did not return a complete response")
            packet = json.loads(job.stdout)
            vehicle.outcome(request, data, packet)
            with self._index_lock:
                self._memory[key] = packet
                if len(self._memory) > MEMORY_PACKETS:
                    self._memory.popitem(last=False)
            self._save(saved, packet)
            pending.set_result(packet)
        except Exception as exc:
            LOG.warning("Could not obtain verified single-vehicle evidence (%s)", type(exc).__name__)
            pending.set_exception(exc)
        finally:
            with self._index_lock:
                self._pending.pop(key,None)
