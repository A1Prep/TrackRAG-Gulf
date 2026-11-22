"""Explicit local UI wiring adapters. Not detector, ASR or learned NLP inference.

Select ONLY with --adapters tools.dropoff_app_demo. Real EMT images can be used,
but all observations, transcripts and model outputs below are generated fixtures.
"""

from copy import deepcopy


DEVELOPMENT_ONLY = True


def predict_tracks_with_receipt(recording, *, config=None, directory=None):
    rows = []
    for frame in recording.iter_frames():
        width, height = frame["width"], frame["height"]
        rows.append({"recording_id": recording.recording_id, "frame_id": frame["frame_id"],
                     "track_id": 1, "source": "predicted", "class_name": "Car",
                     "bbox_xyxy": [width * .60, height * .45, width * .78, height * .70]})
    return rows, {"kind": "generated_development_observations", "genuine_detector_predictions": False,
                  "fresh_detector_inference": False, "adapter": __name__}


class DemoGrounding:
    def interpret(self, request):
        frame = request["scene"]["frame"]
        if not request["dialogue"]:
            result = {"status": "needs_clarification", "destination_point_xy": None,
                      "destination_box_xyxy": None,
                      "clarification_question": "For this wiring example, do you mean the car on the right?"}
        else:
            width, height = frame["width"], frame["height"]
            result = {"status": "proposed", "destination_point_xy": [width * .52, height * .80],
                      "destination_box_xyxy": [width * .44, height * .73, width * .59, height * .90],
                      "clarification_question": None,
                      "anchor_boxes_xyxy": [[width * .60, height * .45, width * .78, height * .70]]}
        provenance = {"backend_kind": "development_demo", "model_id": "generated-ui-fixture",
                      "trained_model": False, "adapter": __name__}
        self.last_receipt = {"request_id": request["request_id"], "request": deepcopy(request),
                             "result": deepcopy(result), "provenance": provenance}
        return {"ok": True, "request_id": request["request_id"], "result": result, "provenance": provenance}


def create_grounding_adapter(config):
    return DemoGrounding()


def answer_question(question, evidence, run_dir):
    if not evidence:
        return {"answer_status": "insufficient_evidence", "answer": None, "cited_evidence_ids": []}
    return {"answer_status": "answered",
            "answer": "Development fixture: the highlighted evidence can be opened below. This is not a live model answer.",
            "cited_evidence_ids": [evidence[0]["evidence_id"]]}


def transcribe_audio(path):
    return {"text": "Drop me off beside the white car.",
            "provenance": {"kind": "generated_development_transcript", "genuine_asr": False,
                           "adapter": __name__}}
