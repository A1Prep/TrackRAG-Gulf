"""Application ownership checks with lightweight models, without a GPU backend."""

import json
from pathlib import Path
from threading import Event, Thread

import pytest

import tools.dropoff_app_demo as demo
from trackrag.dropoff import app as app_module


class OwnedModel:
    def __init__(self, events, *, fail_close=False):
        self.events = events
        self.fail_close = fail_close
        self.close_calls = 0

    def interpret(self, request):
        raise AssertionError("Lifecycle checks must not invoke model inference")

    def close(self):
        self.close_calls += 1
        self.events.append("model closed")
        if self.fail_close:
            raise RuntimeError("worker cleanup failed")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    corpus = tmp_path / "source"
    (corpus / "images" / "train" / "recording_a").mkdir(parents=True)
    events = []
    model = OwnedModel(events)
    monkeypatch.setattr(demo, "create_grounding_adapter", lambda config: model)
    config = app_module.AppConfig(data_root=corpus, output_dir=tmp_path / "application",
                                  adapters="tools.dropoff_app_demo")
    return config, model, events


def test_shutdown_drains_jobs_before_closing_owned_model_and_is_idempotent(setup, monkeypatch):
    config, model, events = setup
    application = app_module.LocalApplication(config)
    working, release, draining = Event(), Event(), Event()
    original_shutdown = application._executor.shutdown

    def shutdown(*args, **kwargs):
        draining.set()
        return original_shutdown(*args, **kwargs)

    monkeypatch.setattr(application._executor, "shutdown", shutdown)

    def work():
        working.set()
        assert release.wait(5), "test worker was not released"
        events.append("work completed")

    with application._lock:
        application._job("lifecycle", work)
    assert working.wait(5)
    errors = []

    def close():
        try:
            application.close()
        except Exception as error:
            errors.append(error)

    closer = Thread(target=close)
    closer.start()
    concurrent_closer = None
    try:
        assert draining.wait(5)
        assert model.close_calls == 0
        concurrent_closer = Thread(target=close)
        concurrent_closer.start()
    finally:
        release.set()
        closer.join(5)
        if concurrent_closer is not None:
            concurrent_closer.join(5)
    assert not closer.is_alive() and not errors
    assert concurrent_closer is not None and not concurrent_closer.is_alive()
    application.close()
    assert model.close_calls == 1
    assert events == ["work completed", "model closed"]
    assert application.view()["jobs"][0]["status"] == "completed"
    saved = json.loads((config.output_dir / "session.json").read_text(encoding="utf-8"))
    assert saved == application.audit()


def test_failed_model_cleanup_still_persists_final_audit_and_is_not_repeated(setup):
    config, model, events = setup
    model.fail_close = True
    application = app_module.LocalApplication(config)
    application.record_api_error("earlier rejected request")
    with pytest.raises(RuntimeError, match="worker cleanup failed"):
        application.close()
    application.close()
    assert model.close_calls == 1
    saved = json.loads((config.output_dir / "session.json").read_text(encoding="utf-8"))
    assert saved == application.audit()
    assert saved["api_errors"][-1] == {
        "code": "grounding_shutdown_failed", "message": "RuntimeError: worker cleanup failed",
    }
    assert saved["api_errors"][0]["message"] == "earlier rejected request"


def test_legacy_development_adapter_without_close_still_shuts_down(setup, monkeypatch):
    config, _, _ = setup
    monkeypatch.setattr(demo, "create_grounding_adapter", lambda config: demo.DemoGrounding())
    application = app_module.LocalApplication(config)
    application.close()
    application.close()
    assert (config.output_dir / "session.json").is_file()


def test_normal_task5_model_is_owned_without_loading_heavy_backend(setup, monkeypatch, tmp_path):
    config, model, _ = setup
    from dataclasses import replace
    from trackrag.dropoff import model as model_module

    supplied = []

    def create(config):
        supplied.append(config)
        return model

    monkeypatch.setattr(model_module, "GroundingModel", create)
    grounding = tmp_path / "grounding.json"
    grounding.write_text("{}", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    normal = replace(config, adapters=None, grounding_config=grounding,
                     vision_config=root / "configs" / "vision" / "pretrained.json",
                     predictions=tmp_path / "saved-predictions")
    application = app_module.LocalApplication(normal)
    application.close()
    assert supplied == [{}]
    assert model.close_calls == 1


def test_cli_bind_failure_releases_application_model(setup, monkeypatch):
    config, model, _ = setup

    def fail_bind(application, port):
        raise OSError("address already in use")

    monkeypatch.setattr(app_module, "make_server", fail_bind)
    with pytest.raises(SystemExit) as result:
        app_module.main(["--data-root", str(config.data_root), "--out", str(config.output_dir),
                         "--adapters", "tools.dropoff_app_demo"])
    assert result.value.code == 2
    assert model.close_calls == 1
    assert (config.output_dir / "session.json").is_file()


@pytest.mark.parametrize("close_failure", [False, True])
def test_cli_server_failure_still_releases_application_model(setup, monkeypatch, close_failure):
    config, model, events = setup

    class Server:
        server_port = 8765

        def serve_forever(self):
            raise RuntimeError("serving failed")

        def server_close(self):
            events.append("server closed")
            if close_failure:
                raise RuntimeError("server close failed")

    monkeypatch.setattr(app_module, "make_server", lambda application, port: Server())
    message = "server close failed" if close_failure else "serving failed"
    with pytest.raises(RuntimeError, match=message):
        app_module.main(["--data-root", str(config.data_root), "--out", str(config.output_dir),
                         "--adapters", "tools.dropoff_app_demo"])
    assert events == ["server closed", "model closed"]
    assert model.close_calls == 1
    assert (config.output_dir / "session.json").is_file()
