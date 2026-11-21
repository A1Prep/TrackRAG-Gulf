"""Compare two Task 1 metrics files number by number (for example archived results and a reproduction).

    python -m tools.vision_compare_metrics ARCHIVED.json REPRODUCED.json [--tolerance 1e-9] [--ignore tracking_only]

Every numeric value present in both files is compared; differences above the tolerance, values that differ in kind,
and keys present in only one file are listed. Keys named with --ignore are skipped wherever they occur (for example a
flag that a newer evaluator adds). Exit status 0 means no difference beyond the tolerance and no missing key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


def compare(archived, reproduced, *, tolerance: float = 1e-9, ignore=()) -> dict:
    """Walk both JSON values together; return counts, differences and keys found in one file only."""
    differences, only_in, compared = [], [], [0]
    largest = [0.0]

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key in ignore:
                    continue
                if key not in a or key not in b:
                    only_in.append({"path": f"{path}/{key}", "only_in": "archived" if key in a else "reproduced"})
                else:
                    walk(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append({"path": path, "archived": f"list of {len(a)}", "reproduced": f"list of {len(b)}"})
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{index}]")
        elif _is_number(a) and _is_number(b):
            compared[0] += 1
            if a == b or (math.isnan(a) and math.isnan(b)):
                return
            gap = abs(a - b)
            largest[0] = max(largest[0], gap if math.isfinite(gap) else math.inf)
            if not gap <= tolerance:
                differences.append({"path": path, "archived": a, "reproduced": b, "difference": gap})
        elif type(a) is not type(b) or a != b:  # True and 1 are equal in Python but not in a metrics file
            differences.append({"path": path, "archived": a, "reproduced": b})

    walk(archived, reproduced, "")
    return {"numbers_compared": compared[0], "tolerance": tolerance, "largest_difference": largest[0],
            "differences": differences, "keys_in_one_file_only": only_in,
            "identical_within_tolerance": not differences and not only_in}


def _is_number(value) -> bool:
    return type(value) in (int, float)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("archived")
    parser.add_argument("reproduced")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--ignore", nargs="*", default=[], help="Keys to skip wherever they occur")
    parser.add_argument("--out", help="Write the comparison as JSON")
    args = parser.parse_args(argv)
    files = {name: Path(path) for name, path in (("archived", args.archived), ("reproduced", args.reproduced))}
    data = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in files.items()}
    result = compare(data["archived"], data["reproduced"], tolerance=args.tolerance, ignore=set(args.ignore))
    result = {"archived_file": files["archived"].name, "reproduced_file": files["reproduced"].name,
              "archived_sha256": hashlib.sha256(files["archived"].read_bytes()).hexdigest(),
              "reproduced_sha256": hashlib.sha256(files["reproduced"].read_bytes()).hexdigest(),
              "ignored_keys": sorted(args.ignore), **result}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{result['numbers_compared']} numbers compared; {len(result['differences'])} differences above "
          f"{args.tolerance}; {len(result['keys_in_one_file_only'])} keys in one file only; largest difference "
          f"{result['largest_difference']}")
    for item in result["differences"][:20] + result["keys_in_one_file_only"][:20]:
        print("  ", item)
    return 0 if result["identical_within_tolerance"] else 1


if __name__ == "__main__":
    sys.exit(main())
