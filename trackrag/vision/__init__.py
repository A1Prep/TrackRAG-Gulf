"""Task 1: detect road users and follow them through a recording.

Importing this package must stay cheap: model libraries (torch, rfdetr, ultralytics,
supervision, OpenCV) are imported only inside the functions that run a model. Task 5's
adapter check imports this module without any model environment installed.
"""


def predict_tracks(recording, *, config=None, detector=None, tracker_factory=None, sink=None):
    """Return one ``TrackObservation`` per visible road user per frame (source ``"predicted"``).

    RF-DETR Large finds boxes in each frame (pre-resized to a 1280 px long side); BoT-SORT links
    them over time. Boxes are the detector's boxes in original-image pixels; class names are EMT
    names; track IDs restart at 1 for every recording. Output is online: rows for frame t depend
    only on frames <= t.

    Configuration comes from ``config`` or the file named by ``TRACKRAG_VISION_CONFIG`` (default
    configs/vision/pretrained.json); the weights folder from ``TRACKRAG_VISION_WEIGHTS_DIR``.
    ``detector``/``tracker_factory`` replace the real models (tests, stubs); ``sink`` receives
    per-frame detections and the final provenance (used by ``python -m trackrag.vision predict``).
    """
    from .config import load_config
    from .pipeline import run_recording

    config = config or load_config()
    if detector is None:
        from .detector import RFDETRDetector

        detector = RFDETRDetector(config)
    if tracker_factory is None:
        from .tracker import BotSortTracker

        tracker_factory = BotSortTracker
    return run_recording(recording, config, detector, tracker_factory, sink=sink)
