"""A matched request hash alone cannot certify the prompt that reached the model."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from tests.test_dropoff_ai_experiment import request
from tools.dropoff_compare_reviewed_models import verify_prompt
from trackrag.dropoff.formatting import build_messages


SETTINGS = {"allow_annotated": False, "observation_format": "compact_rows_v1"}


def test_late_prompt_edit_cannot_pass_by_rehashing_it():
    original = request()
    messages = build_messages(original, **SETTINGS)
    def stamped(content):
        return sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {"request": original, "messages": messages, "status": "completed",
               "provenance": {"messages_sha256": stamped(messages)}}
    assert verify_prompt(receipt, SETTINGS) == stamped(messages)
    changed = deepcopy(receipt)
    changed["messages"][-1]["content"] += " Use the reference answer."
    changed["provenance"]["messages_sha256"] = stamped(changed["messages"])
    with pytest.raises(ValueError, match="Recorded prompt differs"):
        verify_prompt(changed, SETTINGS)


def test_failure_before_prompt_has_no_claimed_prompt():
    receipt = {"request": request(), "status": "error", "provenance": {}}
    assert verify_prompt(receipt, SETTINGS) is None
    receipt["provenance"]["messages_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Recorded prompt differs"):
        verify_prompt(receipt, SETTINGS)
