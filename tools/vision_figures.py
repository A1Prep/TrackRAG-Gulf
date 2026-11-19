"""Report figures and tables for Task 1 (run in the evaluation environment with matplotlib).

    python -m tools.vision_figures curves --metrics-csv RUN/metrics.csv --out fig_learning.png
    python -m tools.vision_figures compare --model "Pretrained=PRE/metrics.json" --model "Fine-tuned=FT/metrics.json" --out-dir FIGS
    python -m tools.vision_figures height --model "Pretrained=PRE_DIR" --model "Fine-tuned=FT_DIR" \
        --labels-root LABELS --split test --recordings ... --out fig_recall_by_height.png

The first model is always blue and the second orange, a pair that colour-blind readers can also
tell apart. Every chart has one y-axis, a legend and direct labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

SERIES = ["#2a78d6", "#eb6834"]          # blue, orange
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HEIGHT_BINS = [0, 8, 16, 32, 64, 128, 10_000]


def _style(ax, title: str, ylabel: str):
    import matplotlib.ticker as ticker

    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))


def _figure(width=7.0, height=3.4):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def learning_curves(metrics_csv: Path, out: Path) -> dict:
    """Two separate panels (no dual axis): training loss, and validation EMA mAP per epoch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss, val50, val5095 = {}, {}, {}
    with Path(metrics_csv).open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            epoch = int(row["epoch"])
            if row.get("train/loss"):
                loss[epoch] = float(row["train/loss"])
            if row.get("val/ema_mAP_50_95"):
                val5095[epoch] = float(row["val/ema_mAP_50_95"])
                val50[epoch] = float(row["val/ema_mAP_50"])
    best = max(val5095, key=val5095.get)
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    epochs = sorted(loss)
    left.plot([e + 1 for e in epochs], [loss[e] for e in epochs], color=SERIES[0], linewidth=2, marker="o", markersize=4)
    _style(left, "Training loss", "loss")
    left.set_xlabel("epoch", color=INK_2, fontsize=9)
    epochs = sorted(val5095)
    for index, (label, values) in enumerate((("mAP@0.5", val50), ("mAP@[.5:.95]", val5095))):
        xs, ys = [e + 1 for e in epochs], [values[e] for e in epochs]
        right.plot(xs, ys, color=SERIES[index], linewidth=2, marker="o", markersize=4, label=label)
        right.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                       fontsize=8, color=INK_2)
    import matplotlib.ticker as ticker

    right.axvline(best + 1, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    # Mid-height, right of the line: the gap between the two series, clear of points and direct labels.
    right.annotate(f"selected: epoch {best + 1}", (best + 1, 0.5), xycoords=("data", "axes fraction"),
                   xytext=(4, 0), textcoords="offset points", ha="left", va="center", fontsize=8, color=INK_2)
    _style(right, "Validation (dev recordings, EMA weights)", "mAP")
    right.set_xlabel("epoch", color=INK_2, fontsize=9)
    right.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                 ncol=2)
    right.margins(x=0.15)
    last = max(max(loss), max(val5095)) + 1
    for axis in (left, right):
        axis.xaxis.set_major_locator(ticker.FixedLocator([1] + list(range(5, last + 1, 5))))
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    return {"best_epoch": best + 1, "best_val_ema_map_50_95": val5095[best], "best_val_ema_map_50": val50[best]}


def _grouped_bars(categories, series: dict, title: str, ylabel: str, out: Path, note: str | None = None):
    fig, ax = _figure(max(6.0, 1.1 * len(categories) + 2), 3.4)
    names = list(series)
    width = 0.8 / len(names)
    for index, name in enumerate(names):
        values = series[name]
        xs = [i + (index - (len(names) - 1) / 2) * width for i in range(len(categories))]
        bars = ax.bar(xs, [v if v is not None else 0 for v in values], width=width * 0.92, color=SERIES[index],
                      label=name, edgecolor=SURFACE, linewidth=1)
        for bar, value in zip(bars, values):
            text = "n/a" if value is None else f"{value:.2f}"
            ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=8, color=INK_2)
    ax.set_ylim(0, 1.0)
    _style(ax, title, ylabel)
    if len(names) > 1:  # a single series is named by the title; no legend box
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, ncol=len(names), loc="upper right")
    if note:
        fig.text(0.01, 0.01, note, fontsize=7, color=INK_2)
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    fig.savefig(out, facecolor=SURFACE)


def compare(models: dict[str, Path], out_dir: Path) -> dict:
    """Per-recording detection AP50 and tracking IDF1, and per-group AP50, for each model; plus a markdown table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {name: json.loads(Path(path).read_text(encoding="utf-8")) for name, path in models.items()}
    recordings = next(iter(data.values()))["recordings"]
    short = [r.replace("video_", "") for r in recordings]
    _grouped_bars(short, {n: [d["detection_per_recording"][r]["AP50"] for r in recordings] for n, d in data.items()},
                  "Detection AP@0.5 per test recording (road users, class-agnostic)", "AP@0.5",
                  out_dir / "fig_ap50_per_recording.png")
    _grouped_bars(short, {n: [d["tracking"]["road_user"][r]["idf1"] for r in recordings] for n, d in data.items()},
                  "Tracking IDF1 per test recording (road users)", "IDF1", out_dir / "fig_idf1_per_recording.png")
    groups = ["Pedestrian", "Cyclist", "Motorbike", "Vehicle"]
    _grouped_bars(groups, {n: [d["detection"]["group"]["per_class"][g]["AP50"] for g in groups] for n, d in data.items()},
                  "Detection AP@0.5 by EMT group", "AP@0.5", out_dir / "fig_ap50_by_group.png",
                  note="n/a: no ground truth of that group in the evaluated frames")
    lines = ["| Metric | " + " | ".join(data) + " |", "|---|" + "---|" * len(data)]

    def row(label, getter, fmt="{:.3f}"):
        values = []
        for d in data.values():
            try:
                value = getter(d)
                values.append("n/a" if value is None else fmt.format(value))
            except (KeyError, TypeError):
                values.append("n/a")
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    for granularity, label in (("agnostic", "road user"), ("group", "4 groups"), ("class", "9 classes")):
        row(f"Detection AP@[.5:.95] ({label})", lambda d, g=granularity: d["detection"][g]["summary"]["AP"])
        row(f"Detection AP@0.5 ({label})", lambda d, g=granularity: d["detection"][g]["summary"]["AP50"])
    for size in ("small", "medium", "large"):
        row(f"Detection AP {size} (road user, 1280-px scale)", lambda d, s=size: d["detection"]["agnostic"]["summary"][f"AP_{s}"])
    row("Precision @ score>=0.4, IoU>=0.5", lambda d: d["operating_point"]["groups"]["road_user"]["precision"])
    row("Recall @ score>=0.4, IoU>=0.5", lambda d: d["operating_point"]["groups"]["road_user"]["recall"])
    row("F1 @ score>=0.4, IoU>=0.5", lambda d: d["operating_point"]["groups"]["road_user"]["f1"])
    for metric, label in (("idf1", "IDF1"), ("mota", "MOTA"), ("recall", "Tracking recall"),
                          ("precision", "Tracking precision")):
        row(f"{label} (road users, all evaluated recordings)", lambda d, m=metric: d["tracking"]["road_user"]["OVERALL"][m])
    row("ID switches (road users)", lambda d: d["tracking"]["road_user"]["OVERALL"]["num_switches"], "{:.0f}")
    (out_dir / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return {"figures": sorted(p.name for p in out_dir.glob("*.png")), "table": "results_table.md"}


def recall_by_height(models: dict[str, Path], labels_root: Path, split: str, recordings: list[str], out: Path,
                     score: float = 0.4) -> dict:
    """Recall (IoU >= 0.5, score >= 0.4) by ground-truth box height after scaling to a 1280-px long side."""
    from tools.vision_evaluate import _iou, load_recording

    labels = [f"{a}-{b}" if b < 10_000 else f">={a}" for a, b in zip(HEIGHT_BINS, HEIGHT_BINS[1:])]
    results = {}
    for name, directory in models.items():
        hits, totals = [0] * len(labels), [0] * len(labels)
        for recording in recordings:
            data = load_recording(Path(directory), labels_root, split, recording)
            scale = 1280 / max(data["size"])
            for frame in data["frames"]:
                preds = [box for box, s, _ in data["detections"].get(frame, []) if s >= score]
                for _, _, box in data["gt"].get(frame, []):
                    height = (box[3] - box[1]) * scale
                    index = next(i for i, (a, b) in enumerate(zip(HEIGHT_BINS, HEIGHT_BINS[1:])) if a <= height < b)
                    totals[index] += 1
                    hits[index] += any(_iou(box, p) >= 0.5 for p in preds)
        results[name] = {"recall": [h / t if t else None for h, t in zip(hits, totals)], "gt_boxes": totals}
    fig, ax = _figure(7.0, 3.4)
    for index, (name, values) in enumerate(results.items()):
        xs = [i for i, v in enumerate(values["recall"]) if v is not None]
        ys = [values["recall"][i] for i in xs]
        ax.plot(xs, ys, color=SERIES[index], linewidth=2, marker="o", markersize=5, label=name)
        if len(results) > 1:
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8,
                        color=INK_2, va="center")
    counts = next(iter(results.values()))["gt_boxes"]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{l}\n(n={c:,})" for l, c in zip(labels, counts)], fontsize=7, color=INK_2)
    ax.set_ylim(0, 1.0)
    _style(ax, "Recall by object height (px at 1280-px scale; score >= 0.4, IoU >= 0.5)", "recall")
    if len(results) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    return {"bins": labels, **results}


def _models(values: list[str]) -> dict[str, Path]:
    pairs = {}
    for value in values:
        name, _, path = value.partition("=")
        if not path:
            raise SystemExit(f"--model needs NAME=PATH, got {value!r}")
        pairs[name] = Path(path)
    if len(pairs) > len(SERIES):
        raise SystemExit(f"At most {len(SERIES)} models per figure")
    return pairs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    curves = commands.add_parser("curves")
    curves.add_argument("--metrics-csv", type=Path, required=True)
    curves.add_argument("--out", type=Path, required=True)
    comp = commands.add_parser("compare")
    comp.add_argument("--model", action="append", required=True, help="NAME=metrics.json (repeat)")
    comp.add_argument("--out-dir", type=Path, required=True)
    height = commands.add_parser("height")
    height.add_argument("--model", action="append", required=True, help="NAME=prediction folder (repeat)")
    height.add_argument("--labels-root", type=Path, required=True)
    height.add_argument("--split", required=True)
    height.add_argument("--recordings", nargs="+", required=True)
    height.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "curves":
        print(json.dumps(learning_curves(args.metrics_csv, args.out)))
    elif args.command == "compare":
        print(json.dumps(compare(_models(args.model), args.out_dir)))
    else:
        print(json.dumps(recall_by_height(_models(args.model), args.labels_root, args.split, args.recordings, args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
