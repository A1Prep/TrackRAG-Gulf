"""Structured Gemini requests with local receipts and exact-request caching."""

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from hashlib import sha256
from pathlib import Path


def _save_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def request_json(parts, schema, role, receipt_path=None, metadata=None, *, validate=None, repair_once=False):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is required for the semantic trial")
    model = os.environ.get("TRACKRAG_TRIAL_MODEL", "gemini-3.5-flash-lite")
    timeout = float(os.environ.get("TRACKRAG_TRIAL_TIMEOUT_SECONDS", "90"))
    if not 1 <= timeout <= 600:
        raise ValueError("TRACKRAG_TRIAL_TIMEOUT_SECONDS must be between 1 and 600")
    max_output = int(os.environ.get("TRACKRAG_TRIAL_MAX_OUTPUT_TOKENS", "3072"))
    if not 256 <= max_output <= 65536:
        raise ValueError("TRACKRAG_TRIAL_MAX_OUTPUT_TOKENS must be between 256 and 65536")
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": {
        "temperature": 0, "maxOutputTokens": max_output,
        "responseMimeType": "application/json", "responseSchema": schema,
    }}
    signature = sha256((model + json.dumps(body, sort_keys=True)).encode()).hexdigest()
    cache_dir = Path(os.environ.get("TRACKRAG_TRIAL_CACHE", "runs/temporal-request-cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{signature}.json"
    receipt = {"model": model, "role": role, "request_sha256": signature, "max_output_tokens": max_output, **(metadata or {})}
    if cached.is_file():
        try:
            stored = json.loads(cached.read_text(encoding="utf-8"))
            response = stored["response"]
            if (not isinstance(response, dict) or stored.get("request_sha256") != signature
                    or stored.get("model") != model or stored.get("request_status") != "completed"):
                raise ValueError("Expected a cached JSON object")
            if validate:
                validate(response)
        except (ValueError, KeyError, TypeError):
            cached.unlink(missing_ok=True)
            receipt["cache_rejected"] = True
        else:
            receipt = {**stored, **receipt, "cached": True}
            if receipt_path:
                _save_json(receipt_path, receipt)
            return response
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        failed_attempts = []
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as stream:
                    raw = json.load(stream)
                break
            except urllib.error.HTTPError as error:
                try:
                    provider = json.loads(error.read(16384)).get("error", {})
                except (ValueError, OSError):
                    provider = {}
                details = provider.get("details", [])
                retry_delay = 2 * (attempt + 1)
                for detail in details:
                    if detail.get("@type", "").endswith("RetryInfo"):
                        try:
                            retry_delay = float(detail.get("retryDelay", "2s").rstrip("s"))
                        except (ValueError, TypeError):
                            pass
                failure = {"http_status": error.code, "message": str(provider.get("message", ""))[:1200]}
                failed_attempts.append(failure)
                receipt["failed_attempts"] = failed_attempts
                daily_limit = "perday" in json.dumps(details).lower()
                if error.code not in {429, 500, 502, 503, 504} or attempt == 2 or daily_limit or retry_delay > 60:
                    raise
                retry_delay = max(1.0, retry_delay)
                print(f"Provider returned {error.code}; retrying in {retry_delay:.1f}s", flush=True)
                time.sleep(retry_delay)
        receipt.update(reported_model=raw.get("modelVersion"), response_id=raw.get("responseId"),
                       usage=raw.get("usageMetadata"), finish_reason=raw["candidates"][0].get("finishReason"))
        content = raw["candidates"][0]["content"]["parts"]
        response = json.loads("".join(part["text"] for part in content if "text" in part and not part.get("thought")))
        if not isinstance(response, dict):
            raise ValueError("Expected a JSON object from the model")
        receipt["response"] = response
        if validate:
            validate(response)
    except Exception as error:
        receipt.update(request_status="error", error_type=type(error).__name__)
        errors = cache_dir / "errors"
        errors.mkdir(exist_ok=True)
        _save_json(errors / f"{signature}.json", receipt)
        if receipt_path:
            _save_json(receipt_path, receipt)
        # One output-validation correction keeps rejected model output visible in its
        # error receipt. No evidence or reference answer is added to the request.
        if (repair_once and isinstance(error, (ValueError, KeyError, TypeError))
                and receipt.get("finish_reason") in {None, "STOP"}):
            correction = {
                "text": "The previous response failed output validation. Return a complete JSON object following the original schema and constraints. "
                        "An insufficient_evidence answer must use an empty cited_evidence_ids list. "
                        "Do not add facts or change the supplied evidence. "
                        f"Validation error: {error}. Previous response: {json.dumps(receipt.get('response'))}",
            }
            return request_json(
                [*parts, correction], schema, role, receipt_path,
                {**(metadata or {}), "repair_of_request_sha256": signature}, validate=validate,
            )
        raise
    receipt.update(request_status="completed", cached=False, response=response, usage=raw.get("usageMetadata"),
                   reported_model=raw.get("modelVersion"), response_id=raw.get("responseId"))
    _save_json(cached, receipt)
    if receipt_path:
        _save_json(receipt_path, receipt)
    return response
