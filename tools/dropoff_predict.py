"""Run the real grounding adapter, or export annotation-free request inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.data import file_sha256, load_cases
from trackrag.dropoff.model import GroundingConfig, GroundingModel


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_requests(path, *, allow_annotated=False, payload=None):
    path = Path(path)
    rows, seen = [], set()
    if payload is None:
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("request file exceeds the snapshot byte budget")
        payload = path.read_bytes()
    if not isinstance(payload, bytes) or len(payload) > 64 * 1024 * 1024:
        raise ValueError("request file exceeds the snapshot byte budget")
    for number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        if len(line) > 2_000_000 or len(rows) >= 50000:
            raise ValueError("request file exceeds bounded input limits")
        try:
            row = validate_request(json.loads(line, object_pairs_hook=_unique_object,
                                              parse_constant=_reject_constant),
                                   allow_annotated=allow_annotated)
        except ValueError as error:
            raise ValueError(f"{path.name}:{number}: {error}") from error
        identity = row["request_id"]
        if identity in seen:
            raise ValueError(f"duplicate request_id: {identity}")
        seen.add(identity)
        image = Path(row["scene"]["frame"]["image_path"])
        if not image.is_absolute():
            image = path.parent / image
        row["scene"]["frame"]["image_path"] = str(image.resolve())
        rows.append(row)
    if not rows:
        raise ValueError("request file is empty")
    return rows


def export_requests(case_file, partition, output):
    """Preparation only: never construct a model or include targets in its input."""
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise FileExistsError("choose a new request output path")
    case_file = Path(case_file)
    if case_file.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("case file exceeds the snapshot byte budget")
    case_snapshot = case_file.read_bytes()
    rows = load_cases(case_file, allowed_partitions={partition}, payload=case_snapshot)
    for row in rows:
        row["request"]["scene"].setdefault("provenance", {})["image_sha256"] = row["source"]["image_sha256"]
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row["request"], sort_keys=True, allow_nan=False) + "\n" for row in rows)
    output.write_bytes(payload.encode("utf-8"))
    manifest = {
        "schema_version": 1, "kind": "annotation_free_grounding_requests",
        "source_cases_sha256": hashlib.sha256(case_snapshot).hexdigest(), "partition": partition,
        "requests_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "request_ids": [row["case_id"] for row in rows],
        "image_sha256": {row["case_id"]: row["source"]["image_sha256"] for row in rows},
        "gold_fields_exported": False,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def predict(requests_path, config, output_dir, *, adapter_factory=GroundingModel):
    requests_path = Path(requests_path)
    if requests_path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("request file exceeds the snapshot byte budget")
    snapshot = requests_path.read_bytes()
    rows = read_requests(requests_path, allow_annotated=config.allow_annotated, payload=snapshot)
    snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "input-snapshot.jsonl").write_bytes(snapshot)
    (output_dir / "resolved-requests.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    model = adapter_factory(config)
    try:
        outcomes, receipts = [], []
        for row in rows:
            atomic = getattr(model, "interpret_with_receipt", None)
            if callable(atomic):
                outcome, receipt = atomic(row)
            else:
                # Preserve the existing sequential injected-adapter test protocol.
                outcome = model.interpret(row)
                receipt = deepcopy(model.last_receipt)
            outcomes.append(outcome)
            receipts.append(receipt)
        for name, values in (("model-outcomes.jsonl", outcomes), ("receipts.jsonl", receipts)):
            (output_dir / name).write_text(
                "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in values),
                encoding="utf-8",
            )
        summary = {
            "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "requests_sha256": snapshot_sha256, "requests_snapshot": "input-snapshot.jsonl",
            "resolved_requests_sha256": file_sha256(output_dir / "resolved-requests.jsonl"),
            "configuration": asdict(config),
            "requested_rows": len(rows), "semantic_results": sum(item["ok"] is True for item in outcomes),
            "operational_errors": sum(item["ok"] is False for item in outcomes),
            "model_quality_measured": False,
            "outcomes_sha256": file_sha256(output_dir / "model-outcomes.jsonl"),
            "receipts_sha256": file_sha256(output_dir / "receipts.jsonl"),
        }
        (output_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        original_error = sys.exc_info()[1]
        try:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        except BaseException:
            # Preserve the exception that interrupted interpretation/export. A
            # cleanup failure after an otherwise successful run remains visible.
            if original_error is None:
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-requests", help="Strip targets/review fields before inference")
    export.add_argument("--cases", type=Path, required=True)
    export.add_argument("--partition", choices=("train", "development", "test"), required=True)
    export.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("predict", help="Run a pinned local/self-hosted model adapter")
    run.add_argument("--requests", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--backend", choices=("transformers", "ollama"), default="transformers",
                     help="Explicit local pretrained Ollama route; the trained HF route remains the default")
    run.add_argument("--adapter-dir", type=Path)
    run.add_argument("--cache-dir", type=Path)
    run.add_argument("--allow-download", action="store_true",
                     help="Explicitly allow base downloads after storage checks; use on the GPU host")
    args = parser.parse_args(argv)
    try:
        if args.command == "export-requests":
            result = export_requests(args.cases, args.partition, args.out)
        else:
            raw = json.loads(args.config.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
            if args.backend == "ollama":
                from trackrag.dropoff.ollama import OllamaGroundingConfig, OllamaGroundingModel
                if args.adapter_dir or args.cache_dir or args.allow_download:
                    raise ValueError("The separate local pretrained route accepts no adapters, caches or downloads")
                config = OllamaGroundingConfig.from_mapping(raw)
                result = predict(args.requests, config, args.out, adapter_factory=OllamaGroundingModel)
            else:
                config = GroundingConfig.from_mapping(raw)
                overrides = {}
                if args.adapter_dir:
                    overrides["adapter_dir"] = str(args.adapter_dir.resolve())
                if args.cache_dir:
                    overrides["cache_dir"] = str(args.cache_dir.resolve())
                if args.allow_download:
                    overrides["local_files_only"] = False
                config = replace(config, **overrides)
                result = predict(args.requests, config, args.out)
        print(json.dumps(result, indent=2))
        return 1 if result.get("operational_errors", 0) else 0
    except (ValueError, OSError) as error:
        print(f"Grounding command failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
