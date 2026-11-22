"""Offline research: parking segmentation -> marked choices -> real Qwen selection.

This comparator does not change the application or issue a vehicle command. Models
must already be present locally; this command never downloads weights. Run with
``python -m tools.passenger_parking_study --help``. Model calls use the settings of
the measured Mask2Former and Qwen3.5 trials, including the original marked prompt.
Optional torch/transformers imports happen only when their stage actually runs.
"""
from __future__ import annotations

import argparse
from hashlib import sha1, sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from time import monotonic

SEG_MODEL = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
SEG_REVISION = "4772b6bf101d91f2534c106dc524d906aeb3c68a"
SEG_WEIGHTS = "72721afb6894c0b1239b9259c428996d9ca0aecbebc4efb4a2deaae0403e2ea7"
QWEN_MODEL = "Qwen/Qwen3.5-9B"
QWEN_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SEG_FILES = {
    "model.safetensors": ("sha256", SEG_WEIGHTS),
    "config.json": ("git_blob", "8ca727ca392af2fd61b6ac8ad0d23be296c7d61a"),
    "preprocessor_config.json": ("git_blob", "f6e96122552341b4b879d23f88dff22c1beef25e"),
}
# SHA256 for LFS objects, Git blob SHA1 for regular files, from the public
# Hugging Face file metadata at QWEN_REVISION. No mutable branch/tag is used.
QWEN_FILES = {
    "model.safetensors-00001-of-00004.safetensors": ("sha256", "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e"),
    "model.safetensors-00002-of-00004.safetensors": ("sha256", "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330"),
    "model.safetensors-00003-of-00004.safetensors": ("sha256", "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec"),
    "model.safetensors-00004-of-00004.safetensors": ("sha256", "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6"),
    "model.safetensors.index.json": ("git_blob", "e4c1cb7dba5096b43b9d92bc781aba5e3aa8acd8"),
    "config.json": ("git_blob", "273ce437e01baf96a07cd9eb3d5f48bac8d7c657"),
    "preprocessor_config.json": ("git_blob", "2ea84a437d448ff71b08df68fdd949d5cc4ebb64"),
    "video_preprocessor_config.json": ("git_blob", "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe"),
    "tokenizer_config.json": ("git_blob", "eda48d3e75a8e59a8479ee4ec8b37f76e711d9c1"),
    "tokenizer.json": ("sha256", "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"),
    "chat_template.jinja": ("git_blob", "a585dec894e63da457d9440ec6aa7caa16d20860"),
    "merges.txt": ("git_blob", "a494e019ca1502219fd0128658b979e5f05ae8e8"),
    "vocab.json": ("git_blob", "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176"),
}


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("nonfinite JSON value")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _versions():
    result = {}
    for name in ("torch", "transformers", "numpy", "Pillow", "scipy", "accelerate", "tokenizers"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def verify_checkpoint(path, kind):
    """Verify actual model bytes offline before constructing either heavy model."""
    path = Path(path)
    if kind not in ("qwen", "segment"):
        raise ValueError("Unknown model kind")
    expected = QWEN_FILES if kind == "qwen" else SEG_FILES
    # These optional auto-loader files could override the pinned processors,
    # generation defaults or model even while every expected file still matches.
    for name in ("adapter_config.json", "generation_config.json", "processor_config.json",
                 "chat_template.json", "added_tokens.json", "special_tokens_map.json"):
        if (path / name).exists() and name not in expected:
            raise ValueError(f"Unpinned auto-loader override: {name}")
    if kind == "qwen" and (path / "model.safetensors").exists():
        raise ValueError("Unpinned monolithic Qwen weights could override the verified shards")
    files = {}
    for name, (algorithm, wanted) in expected.items():
        target = path / name
        if not target.is_file():
            raise ValueError(f"Missing pinned {kind} checkpoint file: {name}")
        actual = file_hash(target)
        if algorithm == "git_blob":
            data = target.read_bytes()
            pin = sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        else:
            pin = actual
        if pin != wanted:
            raise ValueError(f"Pinned {kind} checkpoint digest mismatch: {name}")
        files[name] = {"sha256": actual, "bytes": target.stat().st_size,
                       "verified_by": algorithm, "expected_digest": wanted}
    for name in ("config.json", "preprocessor_config.json"):
        if name not in files:
            files[name] = {"sha256": file_hash(path / name), "bytes": (path / name).stat().st_size,
                           "verified_by": "recorded_only"}
    return {"model": QWEN_MODEL if kind == "qwen" else SEG_MODEL,
            "revision": QWEN_REVISION if kind == "qwen" else SEG_REVISION,
            "files": files, "local_files_only": True, "trust_remote_code": False}


def load_inputs(requests_path):
    """Use shared gold-free contracts and check actual original pixels/identity."""
    from PIL import Image
    from tools.dropoff_predict import read_requests
    requests_path = Path(requests_path)
    if requests_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Request snapshot exceeds64MiB")
    payload = requests_path.read_bytes()
    rows = read_requests(requests_path, payload=payload)
    for row in rows:
        scene = row["scene"]
        frame = scene["frame"]
        if any(t["frame_id"] != frame["frame_id"] for t in scene["tracks"]):
            raise ValueError("Research candidates require current-frame observations")
        if file_hash(frame["image_path"]) != scene.get("provenance", {}).get("image_sha256"):
            raise ValueError("Original photograph SHA256 differs from the request")
        with Image.open(frame["image_path"]) as image:
            if image.size != (frame["width"], frame["height"]):
                raise ValueError("Original photograph dimensions differ from the request")
    return rows, payload


class ParkingSegmenter:
    def __init__(self, checkpoint, device="cuda:0"):
        self.receipt = verify_checkpoint(checkpoint, "segment")
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        self.torch, self.device = torch, device
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        ).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
        self.labels = {str(k): v for k, v in self.model.config.id2label.items()}
        if self.labels.get("10", "").lower() != "parking" or self.labels.get("13", "").lower() != "road":
            raise ValueError("Pinned parking class IDs changed")

    def predict(self, image, shape):
        with self.torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            output = self.model(**inputs)
            classes = self.processor.post_process_semantic_segmentation(output, target_sizes=[shape])[0]
            classes = classes.to("cpu", dtype=self.torch.uint8).numpy()
        return classes, list(inputs.pixel_values.shape[-2:])

    def close(self):
        del self.model
        import gc
        gc.collect()
        if str(self.device).startswith("cuda"):
            self.torch.cuda.empty_cache()


def segment_requests(requests_path, checkpoint, out, device="cuda:0", *, segmenter_factory=ParkingSegmenter):
    """Create original-bound 65-class PNGs and the exporter's receipt format."""
    from PIL import Image
    import numpy as np
    rows, snapshot = load_inputs(requests_path)
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    segmenter = segmenter_factory(checkpoint, device)
    out.mkdir(parents=True, exist_ok=False)
    receipts, seen = [], set()
    try:
        for row in rows:
            image_sha = row["scene"]["provenance"]["image_sha256"]
            if image_sha in seen:
                continue
            seen.add(image_sha)
            with Image.open(row["scene"]["frame"]["image_path"]) as original:
                image = original.convert("RGB")
            scale = min(1, 900 / max(image.size))
            shape = (round(image.height * scale), round(image.width * scale))
            classes, input_hw = segmenter.predict(image, shape)
            if classes.dtype != np.uint8 or classes.shape != shape or int(classes.max()) > 64:
                raise ValueError("Segmenter returned an invalid 65-class map")
            path = out / (image_sha + ".png")
            Image.fromarray(classes).save(path, format="PNG")
            receipts.append({"id": row["request_id"], "sha256": image_sha,
                             "size": list(image.size), "mask_shape": list(shape), "input_hw": input_hw,
                             "mask_sha": file_hash(path), "parking_fraction": float(np.mean(classes == 10)),
                             "revision": SEG_REVISION, "model": SEG_MODEL, "labels": segmenter.labels,
                             "file": path.name})
            _write(out / "receipt.json", receipts)
        manifest = {"kind": "research_parking_segmentation", "complete": True,
                    "requests_sha256": sha256(snapshot).hexdigest(),
                    "source_sha256": file_hash(__file__), "checkpoint": segmenter.receipt,
                    "processor_input": "original RGB without external resizing",
                    "postprocess": "official semantic aggregation; output max dimension900",
                    "device": device, "versions": _versions(), "receipts_sha256": file_hash(out / "receipt.json"),
                    "labels_used": False, "unique_photographs": len(receipts)}
        _write(out / "manifest.json", manifest)
        return receipts
    finally:
        close = getattr(segmenter, "close", None)
        if callable(close):
            close()
        del segmenter


def parse_choice(text, allowed):
    """Accept one JSON object, a known letter or null; never repair a response."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    value = _strict_json(text)
    if not isinstance(value, dict) or set(value) != {"choice"}:
        raise ValueError("Expected exactly one choice field")
    if value["choice"] is not None and (not isinstance(value["choice"], str) or value["choice"] not in allowed):
        raise ValueError("The returned letter was not offered")
    return value


def final_answer_tokens(tokens, close_tokens, thinking):
    if not thinking:
        return tokens
    for index in range(len(tokens) - len(close_tokens) + 1):
        if tokens[index:index + len(close_tokens)] == close_tokens:
            return tokens[index + len(close_tokens):]
    return []


def content_tokens(tokens, stop_ids):
    """Remove EOS and rectangular-batch padding for one sequence's own count."""
    for index, token in enumerate(tokens):
        if token in stop_ids:
            return tokens[:index]
    return tokens


class QwenChooser:
    """Exact non-quantized, greedy Qwen3.5 marked-image trial settings."""
    def __init__(self, checkpoint, device="cuda:0", thinking=False):
        self.receipt = verify_checkpoint(checkpoint, "qwen")
        self.checkpoint = checkpoint
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        self.torch, self.device, self.thinking = torch, device, thinking
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
            device_map={"": device}, attn_implementation="sdpa",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False, min_pixels=65536, max_pixels=1280 * 720,
        )
        self.processor.tokenizer.padding_side = "left"
        self.anchor_processor = None

    def infer_batch(self, rows):
        from tools.passenger_road_points import MARKED
        return self._infer(rows, processor=self.processor, thinking=self.thinking, limit=1800 if self.thinking else 85,
                           system_for=lambda row: MARKED + " Offered letters: " + ", ".join(o["choice"] for o in row["options"]),
                           image_key="marked", hash_key="marked_sha256", maximum_batch=3, mode="marked")

    def infer_anchors(self, rows):
        from transformers import AutoProcessor
        from tools.passenger_parking_anchors import ANCHOR_PROMPT
        if self.anchor_processor is None:
            self.anchor_processor = AutoProcessor.from_pretrained(
                self.checkpoint, local_files_only=True, trust_remote_code=False, min_pixels=65536, max_pixels=1440 * 900,
            )
            self.anchor_processor.tokenizer.padding_side = "left"
        return self._infer(rows, processor=self.anchor_processor, thinking=False, limit=144,
                           system_for=lambda _: ANCHOR_PROMPT, image_key="original", hash_key="sha256",
                           maximum_batch=2, mode="anchors")

    def _infer(self, rows, *, processor, thinking, limit, system_for, image_key, hash_key, maximum_batch, mode):
        from PIL import Image
        if not 1 <= len(rows) <= maximum_batch:
            raise ValueError("Batch size differs from the measured inference setting")
        texts, photos, systems = [], [], []
        for row in rows:
            if file_hash(row[image_key]) != row[hash_key]:
                raise ValueError("Photograph changed before inference")
            with Image.open(row[image_key]) as original:
                image = original.convert("RGB")
            system = system_for(row)
            chat = [{"role": "system", "content": system}, {"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": row["request"]}]}]
            prompt = processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
            )
            if not prompt.endswith("<think>\n" if thinking else "<think>\n\n</think>\n\n"):
                raise ValueError("Pinned chat template changed")
            texts.append(prompt)
            photos.append(image)
            systems.append(system)
        inputs = processor(text=texts, images=photos, padding=True, return_tensors="pt").to(self.device)
        inputs.pop("token_type_ids", None)
        begin = monotonic()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=limit, use_cache=True)
        seconds = round(monotonic() - begin, 2)
        close = processor.tokenizer.encode("</think>", add_special_tokens=False)
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        stop_ids = set(eos if isinstance(eos, list) else [eos])
        stop_ids.add(getattr(processor.tokenizer, "pad_token_id", None))
        convert = getattr(processor.tokenizer, "convert_tokens_to_ids", None)
        if callable(convert):
            stop_ids.add(convert("<|im_end|>"))
        stop_ids.discard(None)
        records = []
        for index, row in enumerate(rows):
            raw_tokens = generated[index, inputs.input_ids.shape[1]:].tolist()
            tokens = content_tokens(raw_tokens, stop_ids)
            final_tokens = final_answer_tokens(tokens, close, thinking)
            answer = processor.tokenizer.decode(final_tokens, skip_special_tokens=True).strip()
            record = {"id": row["id"], "mode": mode, "thinking": thinking,
                      "sent_sha": row[hash_key], "original_sha": row["sha256"], "request": row["request"],
                      "decoded_final_answer": answer, "generated_text": processor.tokenizer.decode(tokens, skip_special_tokens=True),
                      "generated_tokens": len(tokens), "generation_limit": limit,
                      "raw_generated_token_ids": raw_tokens,
                      "generated_tokens_including_end_and_padding": len(raw_tokens),
                      "generated_count_excludes_end_and_padding": True,
                      "stop_token_present": len(tokens) < len(raw_tokens),
                      "reached_generation_limit_without_stop": len(tokens) >= limit,
                      "completed_thinking": any(tokens[j:j + len(close)] == close
                                                for j in range(len(tokens) - len(close) + 1)) if thinking else None,
                      "prompt_sha256": sha256(systems[index].encode("utf8")).hexdigest(),
                      "rendered_chat_sha256": sha256(texts[index].encode("utf8")).hexdigest(),
                      "input_tokens": int(inputs.attention_mask[index].sum().item()),
                      "model_revision": QWEN_REVISION, "seconds_for_batch": seconds, "batch_size": len(rows)}
            try:
                if mode == "marked":
                    record["result"] = parse_choice(answer, [o["choice"] for o in row["options"]])
                # Anchor schema/identity resolution is independently owned by
                # passenger_parking_anchors. Preserve unsanitized final text.
                else:
                    record.update(source_sha=row["sha256"], raw=answer, revision=QWEN_REVISION)
            except ValueError as error:
                record["error"] = "no usable final answer"
                record["error_detail"] = str(error)
            records.append(record)
        return records


def anchor_requests(requests_path, checkpoint, out, device="cuda:0", *, chooser_factory=QwenChooser):
    """First learned pass: the same original photograph -> two reference boxes."""
    from tools.passenger_parking_anchors import ANCHOR_PROMPT
    requests, snapshot = load_inputs(requests_path)
    rows = [{"id": req["request_id"], "original": req["scene"]["frame"]["image_path"],
             "sha256": req["scene"]["provenance"]["image_sha256"], "request": req["text"]} for req in requests]
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    chooser = chooser_factory(checkpoint, device, False)
    out.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for start in range(0, len(rows), 2):
            group = rows[start:start + 2]
            try:
                batch = chooser.infer_anchors(group)
                if [r["id"] for r in batch] != [r["id"] for r in group]:
                    raise ValueError("Anchor inference changed request identity/order")
            except Exception as error:
                batch = [{"id": r["id"], "source_sha": r["sha256"], "request": r["request"],
                          "prompt_sha256": sha256(ANCHOR_PROMPT.encode()).hexdigest(), "revision": QWEN_REVISION,
                          "raw": "", "error": type(error).__name__ + ": " + str(error)} for r in group]
            records.extend(batch)
            _write(out / "anchor-records.json", records)
        _write(out / "manifest.json", {"kind": "research_original_image_reference_identification",
                    "complete": True, "requests_sha256": sha256(snapshot).hexdigest(),
                    "source_sha256": file_hash(__file__), "checkpoint": chooser.receipt,
                    "device": device, "max_pixels": 1440 * 900, "min_pixels": 65536,
                    "batch_size": 2, "max_new_tokens": 144, "thinking": False,
                    "versions": _versions(), "labels_used": False,
                    "anchor_records_sha256": file_hash(out / "anchor-records.json")})
        return records
    finally:
        del chooser


def validate_choices(requests_path, choices_dir, class_maps):
    """Replay current candidate geometry before trusting a stored letter mapping."""
    from PIL import Image
    import numpy as np
    from tools import passenger_parking_candidates as candidates
    rows, snapshot = load_inputs(requests_path)
    folder, maps = Path(choices_dir), Path(class_maps)
    raw_choices = (folder / "choices.json").read_bytes()
    manifest = _strict_json((folder / "manifest.json").read_text(encoding="utf8"))
    raw_receipts = (maps / "receipt.json").read_bytes()
    if (manifest.get("requests_sha256") != sha256(snapshot).hexdigest()
            or manifest.get("class_receipts_sha256") != sha256(raw_receipts).hexdigest()
            or manifest.get("source_sha256") != file_hash(candidates.__file__)
            or manifest.get("choices_sha256") != sha256(raw_choices).hexdigest()):
        raise ValueError("Choices snapshot differs from the current requests, maps or candidate source")
    choices = _strict_json(raw_choices.decode("utf8"))
    receipts = _strict_json(raw_receipts.decode("utf8"))
    anchor_resolutions = None
    if manifest.get("geometry_mode") == "model-anchors":
        from tools.passenger_parking_anchors import read_anchor_records
        raw_anchors = (folder / "anchor-records.json").read_bytes()
        if manifest.get("anchor_receipts_sha256") != sha256(raw_anchors).hexdigest():
            raise ValueError("Stored anchor inference receipts changed")
        anchor_resolutions = read_anchor_records(rows, raw_anchors)
    elif manifest.get("geometry_mode") not in (None, "all-pairs"):
        raise ValueError("Unsupported research geometry mode")
    by_sha = {}
    for receipt in receipts:
        old = by_sha.get(receipt["sha256"])
        if old is not None and old != receipt:
            raise ValueError("Conflicting receipts for one original photograph")
        by_sha[receipt["sha256"]] = receipt
    if not isinstance(choices, list) or len(choices) != len(rows):
        raise ValueError("Choices must retain every request exactly once")
    for request, choice in zip(rows, choices):
        frame = request["scene"]["frame"]
        image_sha = request["scene"]["provenance"]["image_sha256"]
        receipt = by_sha.get(image_sha, {})
        mask_path = maps / (image_sha + ".png")
        size = [frame["width"], frame["height"]]
        if (choice["id"] != request["request_id"] or choice["request"] != request["text"]
                or choice["sha256"] != image_sha or choice["original_size"] != size
                or Path(choice["original"]).resolve() != Path(frame["image_path"]).resolve()
                or receipt.get("model") != SEG_MODEL or receipt.get("revision") != SEG_REVISION
                or receipt.get("size") != size or receipt.get("file") != image_sha + ".png"
                or receipt.get("labels", {}).get("10", "").lower() != "parking"
                or receipt.get("labels", {}).get("13", "").lower() != "road"
                or receipt.get("mask_sha") != file_hash(mask_path)
                or choice.get("mask_sha256") != receipt.get("mask_sha")
                or choice.get("surface") != manifest.get("surface")
                or file_hash(choice["marked"]) != choice["marked_sha256"]):
            raise ValueError("Candidate identity, original pixels or class map changed")
        with Image.open(mask_path) as image:
            classes = np.array(image)
        if list(classes.shape) != receipt["mask_shape"]:
            raise ValueError("Class-map dimensions changed")
        if anchor_resolutions is None:
            expected_options = candidates.options_for(request, classes, choice["surface"])
        else:
            resolution = anchor_resolutions[request["request_id"]]
            if (choice.get("anchor_state") != resolution["state"] or choice.get("pair_ids") != resolution["track_ids"]
                    or choice.get("anchor_iou") != resolution["ious"]
                    or choice.get("resolved_boxes_xyxy") != resolution["boxes"] or choice["surface"] != "parking"):
                raise ValueError("Anchor resolution or its matched current track identities changed")
            expected_options = candidates.options_for(request, classes, "parking", resolution["track_ids"]) if resolution["state"] == "matched" else []
        if choice["options"] != expected_options:
            raise ValueError("Candidate point, track IDs, letter order or geometry changed")
    return choices, snapshot


def decision_for(row, record):
    """Map only a validated letter to its exact original-pixel research point."""
    base = {"id": row["id"], "image_sha256": row["sha256"],
            "marked_sha256": row["marked_sha256"], "research_only": True,
            "vehicle_control": False, "passenger_confirmed": False}
    if row.get("anchor_state") == "error":
        return {**base, "status": "operational_error", "error": "invalid_or_failed_anchor_model_output",
                "model_called": False, "anchor_model_called": True}
    if row.get("anchor_state") == "ask":
        return {**base, "status": "ask", "reason": "anchor_references_unresolved", "model_called": False,
                "anchor_model_called": True}
    if not row["options"]:
        return {**base, "status": "unresolved", "reason": "no_candidates", "model_called": False}
    if "error" in record:
        return {**base, "status": "operational_error", "error": record["error"], "model_called": True}
    result = parse_choice(json.dumps(record["result"], allow_nan=False), [o["choice"] for o in row["options"]])
    if result["choice"] is None:
        return {**base, "status": "ask", "reason": "model_returned_null", "model_called": True}
    option = next(o for o in row["options"] if o["choice"] == result["choice"])
    return {**base, "status": "proposed_location", "model_called": True,
            "choice": option["choice"], "point_original_xy": option["point"],
            "anchor_track_ids": option["tracks"], "requires_confirmation": True}


def select_choices(requests_path, choices_dir, checkpoint, out, *, class_maps, device="cuda:0",
                   thinking=False, chooser_factory=QwenChooser):
    """Run real inference; invalid outputs and failed batches remain explicit errors."""
    rows, snapshot = validate_choices(requests_path, choices_dir, class_maps)
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    active = [r for r in rows if r["options"]]
    chooser = chooser_factory(checkpoint, device, thinking) if active else None
    out.mkdir(parents=True, exist_ok=False)
    records, by_id = [], {}
    try:
        for start in range(0, len(active), 3):
            group = active[start:start + 3]
            try:
                batch = chooser.infer_batch(group)
                if [r["id"] for r in batch] != [r["id"] for r in group]:
                    raise ValueError("Inference did not retain the input identity/order")
            except Exception as error:
                batch = [{"id": r["id"], "error": type(error).__name__ + ": " + str(error),
                          "original_sha": r["sha256"], "sent_sha": r["marked_sha256"]} for r in group]
            records.extend(batch)
            by_id.update({r["id"]: r for r in batch})
            _write(out / "raw-model-results.json", records)
        decisions = [decision_for(row, by_id.get(row["id"], {})) for row in rows]
        if not active:
            _write(out / "raw-model-results.json", [])
        _write(out / "decisions.json", decisions)
        from tools.passenger_road_points import MARKED
        (out / "selection-system-prompt.txt").write_text(MARKED + "\n", encoding="utf8")
        _write(out / "manifest.json", {
            "kind": "research_parking_then_marked_qwen", "complete": True, "research_only": True,
            "normal_app_route_changed": False, "requests_sha256": sha256(snapshot).hexdigest(),
            "choices_sha256": file_hash(Path(choices_dir) / "choices.json"),
            "candidate_manifest_sha256": file_hash(Path(choices_dir) / "manifest.json"),
            "source_sha256": file_hash(__file__), "checkpoint": chooser.receipt if chooser else None,
            "device": device, "thinking": thinking, "batch_size": 3,
            "max_new_tokens": 1800 if thinking else 85, "min_pixels": 65536, "max_pixels": 1280 * 720,
            "dtype": "bfloat16", "attn_implementation": "sdpa", "do_sample": False,
            "versions": _versions(), "labels_used": False, "model_requests": len(active),
            "generation_batches": (len(active) + 2) // 3,
            "requests": len(rows), "no_candidates": len(rows) - len(active),
            "decision_counts": {name: sum(d["status"] == name for d in decisions)
                                for name in ("proposed_location", "ask", "unresolved", "operational_error")},
            "raw_results_sha256": file_hash(out / "raw-model-results.json"),
            "decisions_sha256": file_hash(out / "decisions.json"),
        })
        return decisions
    finally:
        del chooser


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=Path, required=True, help="Annotation-free requests JSONL, original-image paths")
    parser.add_argument("--out", type=Path, required=True, help="New research output directory")
    parser.add_argument("--stage", choices=("all", "segment", "anchors", "candidates", "select"), default="all")
    parser.add_argument("--segment-checkpoint", type=Path, help=f"Local {SEG_MODEL} at {SEG_REVISION}")
    parser.add_argument("--qwen-checkpoint", type=Path, help=f"Local {QWEN_MODEL} at {QWEN_REVISION}")
    parser.add_argument("--class-maps", type=Path, help="Existing segment-stage output when reusing maps")
    parser.add_argument("--choices", type=Path, help="Existing candidates-stage output for select-only")
    parser.add_argument("--anchor-first", action="store_true", help="First identify two vehicle boxes with Qwen, then offer only that pair's parking options")
    parser.add_argument("--anchor-records", type=Path, help="Saved original-image model receipts for candidates --anchor-first")
    parser.add_argument("--surface", choices=("parking", "all-road"), default="parking")
    parser.add_argument("--font-file", type=Path, help="Original trial used Arial Bold; glyph/font choice is recorded")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--thinking", action="store_true", help="Measured optional1800-token reasoning condition; default85-token nonthinking")
    args = parser.parse_args(argv)
    if args.stage in ("all", "segment") and not args.segment_checkpoint:
        parser.error("--segment-checkpoint is required for segmentation")
    if args.stage in ("all", "select", "anchors") and not args.qwen_checkpoint:
        parser.error("--qwen-checkpoint is required for model selection")
    if args.stage in ("candidates", "select") and not args.class_maps:
        parser.error("--class-maps is required for this stage")
    if args.stage == "select" and not args.choices:
        parser.error("--choices is required for select-only")
    if args.stage == "candidates" and args.anchor_first and not args.anchor_records:
        parser.error("--anchor-records is required for candidates --anchor-first")
    if args.anchor_first and args.surface != "parking":
        parser.error("The measured anchor-first condition uses the parking class only")
    if args.stage == "all":
        args.out.mkdir(parents=True, exist_ok=False)
        segment_requests(args.requests, args.segment_checkpoint, args.out / "maps", args.device)
        if args.anchor_first:
            shared_chooser = QwenChooser(args.qwen_checkpoint, args.device, args.thinking)
            factory = lambda *_: shared_chooser
            anchor_requests(args.requests, args.qwen_checkpoint, args.out / "anchors", args.device, chooser_factory=factory)
            from tools.passenger_parking_anchors import export
            export(args.requests, args.out / "maps", args.out / "candidates", args.out / "anchors" / "anchor-records.json", args.font_file)
        else:
            factory = QwenChooser
            from tools.passenger_parking_candidates import export
            export(args.requests, args.out / "maps", args.out / "candidates", args.surface, args.font_file)
        select_choices(args.requests, args.out / "candidates", args.qwen_checkpoint, args.out / "selection",
                       class_maps=args.out / "maps", device=args.device, thinking=args.thinking, chooser_factory=factory)
    elif args.stage == "segment":
        segment_requests(args.requests, args.segment_checkpoint, args.out, args.device)
    elif args.stage == "candidates":
        load_inputs(args.requests)
        if args.anchor_first:
            from tools.passenger_parking_anchors import export
            export(args.requests, args.class_maps, args.out, args.anchor_records, args.font_file)
        else:
            from tools.passenger_parking_candidates import export
            export(args.requests, args.class_maps, args.out, args.surface, args.font_file)
    elif args.stage == "anchors":
        anchor_requests(args.requests, args.qwen_checkpoint, args.out, args.device)
    else:
        select_choices(args.requests, args.choices, args.qwen_checkpoint, args.out, class_maps=args.class_maps,
                       device=args.device, thinking=args.thinking)
    print(f"Research {args.stage} stage saved to {args.out}")


if __name__ == "__main__":
    main()
