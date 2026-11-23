"""Keep a public place lookup running when the passenger's first response is due."""
from collections import OrderedDict
from concurrent.futures import Future, wait
import logging
from threading import Lock, Thread
from time import monotonic

LOG = logging.getLogger(__name__)


class PlaceLookups:
    def __init__(self, capacity=3, remembered=12):
        self.capacity = capacity
        self.remembered = remembered
        self._lock = Lock()
        self._running = {}
        self._finished = OrderedDict()

    def lookup(self, key, fetch, deadline):
        with self._lock:
            future = self._finished.get(key)
            if future is not None:
                self._finished.move_to_end(key)
            elif key in self._running:
                return "pending", None
            elif len(self._running) >= self.capacity:
                return "busy", None
            else:
                future = self._running[key] = Future()
                worker = Thread(target=self._collect, args=(key,future,fetch), daemon=True,
                                name="public-place-lookup")
                try:
                    worker.start()
                except Exception:
                    del self._running[key]
                    raise
        if not wait((future,), timeout=max(0., deadline-monotonic())).done:
            return "pending", None
        try:
            return "complete", future.result()
        except Exception:
            # A retry after a late failure sees the genuine error. A subsequent attempt can run afresh.
            with self._lock:
                if self._finished.get(key) is future:
                    del self._finished[key]
            raise

    def _collect(self, key, future, fetch):
        try:
            result = fetch()
            error = None
        except Exception as exc:
            LOG.warning("Public place lookup %s failed (%s)",key[:12],type(exc).__name__)
            error = exc.with_traceback(None)
        with self._lock:
            del self._running[key]
            self._finished[key] = future
            if len(self._finished) > self.remembered:
                self._finished.popitem(last=False)
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)
