#!/usr/bin/env python3
"""Build the Task 2 drop-off data release without hiding data limitations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trackrag.dropoff import (
    parse_regseg_annotations, prepare_public_cases, read_local_cases,
    sha256, write_json, write_release,
)


def source_receipt(annotation_dir: Path, output: Path, commit: str) -> None:
    counts, files = {}, {}
    for split in ("train", "val"):
        path = annotation_dir / f"annotation_new_{split}.txt"
        counts[split] = len(parse_regseg_annotations(path, split))
        files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    write_json(output, {
        "repository": "https://github.com/kanji95/Talk2car-Refseg",
        "commit": commit,
        "download": "gdrive 1kNEtbmV_777l8FxYig9WnxYH_E6W-4jH",
        "counts_observed": counts,
        "total_observed": sum(counts.values()),
        "supplied_audit": {"train": 170, "val": 179, "total": 349},
        "audit_matches_observed_source": counts == {"train": 170, "val": 179},
        "note": "Observed source counts are reported as-is; examples are never padded or silently truncated.",
        "files": files,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    receipt = sub.add_parser("source-receipt", help="Audit small public annotation files")
    receipt.add_argument("--annotation-dir", type=Path, required=True)
    receipt.add_argument("--commit", required=True)
    receipt.add_argument("--output", type=Path, required=True)

    public = sub.add_parser("public", help="Validate actual public images and masks")
    public.add_argument("--data-root", type=Path, required=True)
    public.add_argument("--annotation-dir", type=Path, required=True)
    public.add_argument("--output-dir", type=Path, required=True)

    local = sub.add_parser("local", help="Build local manifests and review forms")
    local.add_argument("--annotation-root", type=Path, required=True)
    local.add_argument("--emt-root", type=Path)
    local.add_argument("--output-dir", type=Path, required=True)

    release = sub.add_parser("release", help="Build the combined public/local release")
    release.add_argument("--data-root", type=Path, required=True)
    release.add_argument("--public-annotation-dir", type=Path, required=True)
    release.add_argument("--local-annotation-root", type=Path, required=True)
    release.add_argument("--emt-root", type=Path)
    release.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "source-receipt":
        source_receipt(args.annotation_dir, args.output, args.commit)
        return
    if args.command == "public":
        valid, excluded, summary = prepare_public_cases(args.data_root, args.annotation_dir)
        manifest = write_release(args.output_dir, [], [], valid, excluded, summary)
    elif args.command == "local":
        valid, excluded = read_local_cases(args.annotation_root, args.emt_root)
        manifest = write_release(args.output_dir, valid, excluded)
    else:
        public, public_excluded, summary = prepare_public_cases(
            args.data_root, args.public_annotation_dir
        )
        local, local_excluded = read_local_cases(args.local_annotation_root, args.emt_root)
        manifest = write_release(
            args.output_dir, local, local_excluded, public, public_excluded, summary
        )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Release written to {args.output_dir}")
    if manifest["local_excluded_count"] or manifest["public_excluded_count"]:
        print("Exclusions were recorded; inspect the exclusion JSON receipts.")


if __name__ == "__main__":
    main()
