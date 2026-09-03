"""Evaluate a trained checkpoint on the held-out TEST split and quantify the
false-positive rate the OVERRIDING REQUIREMENT cares about.

The test split (built by training/prepare_dataset.py) is untouched by
training and by any threshold selection here, so these numbers are the
honest ones -- not the inflated ones you'd get from the original archive's
leaky train/valid split.

For each class this reports:
  - precision/recall/AP50 at the model's own max-F1 operating point
  - a confidence-threshold sweep (precision & recall at each step)
  - the LOWEST confidence threshold that achieves precision >= --target-precision
    (default 0.95), or an explicit "not reachable" if no threshold does

`spaghetti` and `warping` are flagged as CATASTROPHIC: these are the two
classes allowed to stop a print, so their precision at the chosen threshold
is what actually determines how often a real print gets killed for nothing.

All of this is computed once via a single low-confidence model.val() pass
(conf=0.001) using Ultralytics' own IoU=0.5 precision/recall-vs-confidence
curves (ap_per_class), not re-implemented matching logic -- so the numbers
match what Ultralytics would report at any single operating point.

Usage:
    python training/evaluate.py --weights runs/train/argus_yolov8n/weights/best.pt
    python training/evaluate.py --weights ... --target-precision 0.95 --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "datasets" / "argus" / "data.yaml"
DEFAULT_OUT_PATH = REPO_ROOT / "runs" / "evaluation.json"

CATASTROPHIC_CLASSES = ("spaghetti", "warping")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained best.pt")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help=f"Path to data.yaml (default: {DEFAULT_DATA_YAML})")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="IoU threshold used for NMS during val (default: 0.7, Ultralytics default)")
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.05,
        help=(
            "A candidate threshold must also reach at least this much recall to count as "
            "'reaching' --target-precision (default: 0.05). Ultralytics' precision curve reports "
            "precision=1.0 by convention wherever a class has zero surviving predictions (0/0), "
            "which is a statistically meaningless 'perfect' score, not a usable operating point -- "
            "a detector that never fires offers no protection at all. This floor rejects that "
            "vacuous case so the recommended threshold is backed by real detections."
        ),
    )
    parser.add_argument("--sweep-start", type=float, default=0.05)
    parser.add_argument("--sweep-end", type=float, default=0.95)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help=f"Where to write the full JSON report (default: {DEFAULT_OUT_PATH})")
    return parser.parse_args(argv)


def run_validation(weights: Path, data: Path, imgsz: int, batch: int, device: str, nms_iou: float):
    """Run Ultralytics val() on the TEST split at low confidence (0.001) so
    the full precision/recall-vs-confidence curve is available afterward."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=0.001,
        iou=nms_iou,
        plots=False,
        save_json=False,
        verbose=False,
    )
    return metrics


def sweep_thresholds(start: float, end: float, step: float) -> list[float]:
    n_steps = int(round((end - start) / step)) + 1
    return [round(start + i * step, 10) for i in range(n_steps) if start + i * step <= end + 1e-9]


def find_lowest_threshold_for_precision(
    px: np.ndarray,
    p_curve: np.ndarray,
    r_curve: np.ndarray,
    target_precision: float,
    min_recall: float,
) -> tuple[Optional[dict[str, float]], Optional[dict[str, float]]]:
    """Scan confidence thresholds ascending and return the first (lowest) one
    whose precision meets `target_precision` AND whose recall is at least
    `min_recall`, paired with the recall at that same threshold.

    The `min_recall` floor matters: Ultralytics' precision curve reports
    precision=1.0 by convention wherever zero predictions survive for a
    class at that confidence (a vacuous 0/0-style "perfect" score -- see
    `ap_per_class`'s `left=1` fill). Without this floor, a barely-trained or
    poorly-performing class can "achieve" the target precision purely by
    never firing at all, which is a statistically meaningless operating
    point (a detector that never fires can't ever be a false positive, but
    it also can't ever catch a real failure -- exactly the kind of
    inflated-looking-but-dishonest number this project explicitly wants to
    avoid).

    Returns `(result, vacuous_example)`:
      - `result` is the first point meeting both conditions, or None.
      - `vacuous_example` is set (only when `result` is None) to the first
        point that met `target_precision` but NOT `min_recall`, so callers
        can report *why* it wasn't accepted rather than silently saying
        "unreachable".
    """
    idxs = np.argsort(px)  # px is already ascending (linspace) but be defensive
    vacuous_example: Optional[dict[str, float]] = None
    for i in idxs:
        if p_curve[i] >= target_precision:
            if r_curve[i] >= min_recall:
                return {"threshold": float(px[i]), "precision": float(p_curve[i]), "recall": float(r_curve[i])}, None
            if vacuous_example is None:
                vacuous_example = {"threshold": float(px[i]), "precision": float(p_curve[i]), "recall": float(r_curve[i])}
    return None, vacuous_example


def best_supported_point(sweep: list[dict], min_recall: float) -> dict:
    """Pick the highest-precision sweep point that still clears `min_recall`
    (i.e. backed by a meaningful number of real detections, not the vacuous
    0-predictions/precision=1.0 artifact). Falls back to the single
    highest-recall point if the class never clears `min_recall` anywhere in
    the sweep -- i.e. the class is effectively non-functional at every
    threshold tried."""
    supported = [pt for pt in sweep if pt["recall"] >= min_recall]
    if supported:
        return max(supported, key=lambda pt: pt["precision"])
    return max(sweep, key=lambda pt: pt["recall"])


def build_report(metrics, class_names: tuple[str, ...], args: argparse.Namespace) -> dict:
    box = metrics.box
    names: dict[int, str] = metrics.names  # {idx: name}, full class set from data.yaml
    ap_class_index = list(box.ap_class_index)  # which classes had instances, row order into p/r/ap50/ap/p_curve/r_curve
    px = np.asarray(box.px)

    thresholds = sweep_thresholds(args.sweep_start, args.sweep_end, args.sweep_step)

    per_class: dict[str, dict] = {}
    for row, cls_idx in enumerate(ap_class_index):
        cname = names.get(int(cls_idx), f"class_{cls_idx}")
        p_curve_row = np.asarray(box.p_curve[row])
        r_curve_row = np.asarray(box.r_curve[row])

        sweep = [
            {
                "threshold": t,
                "precision": float(np.interp(t, px, p_curve_row)),
                "recall": float(np.interp(t, px, r_curve_row)),
            }
            for t in thresholds
        ]

        target_result, vacuous_example = find_lowest_threshold_for_precision(
            px, p_curve_row, r_curve_row, args.target_precision, args.min_recall
        )

        per_class[cname] = {
            "class_index": int(cls_idx),
            "num_test_instances": int(metrics.nt_per_class[cls_idx]) if metrics.nt_per_class is not None else None,
            "precision_at_max_f1": float(box.p[row]),
            "recall_at_max_f1": float(box.r[row]),
            "ap50": float(box.ap50[row]),
            "ap50_95": float(box.ap[row]),
            "confidence_sweep": sweep,
            "target_precision": args.target_precision,
            "min_recall": args.min_recall,
            "lowest_threshold_for_target_precision": target_result,  # None => unreachable (with meaningful recall)
            "target_reachable": target_result is not None,
            # Set only when target_precision is met SOLELY by near-zero-recall points (the
            # vacuous "0 predictions -> precision=1.0" artifact) -- i.e. the number looks
            # perfect but the class effectively never fires there.
            "vacuous_precision_only": vacuous_example,
            "is_catastrophic": cname in CATASTROPHIC_CLASSES,
        }

    # Any configured class with zero test instances (shouldn't happen with a
    # reasonable split, but call it out rather than silently omitting it).
    missing_classes = [c for c in class_names if c not in per_class]

    report = {
        "weights": str(args.weights),
        "data_yaml": str(args.data),
        "test_split": "test",
        "overall": {
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "mean_precision_at_max_f1": float(box.mp),
            "mean_recall_at_max_f1": float(box.mr),
        },
        "target_precision": args.target_precision,
        "min_recall": args.min_recall,
        "per_class": per_class,
        "classes_with_no_test_instances": missing_classes,
        "catastrophic_classes": list(CATASTROPHIC_CLASSES),
    }
    return report


def format_yaml_block(report: dict, class_names: tuple[str, ...]) -> str:
    lines = ["class_thresholds:"]
    for cname in class_names:
        entry = report["per_class"].get(cname)
        key = f'"{cname}"' if " " in cname else cname
        if entry is None:
            lines.append(f"  {key}: 0.95  # NO TEST INSTANCES for this class -- cannot evaluate, do not trust")
            continue
        target = report["target_precision"]
        result = entry["lowest_threshold_for_target_precision"]
        if result is not None:
            lines.append(
                f"  {key}: {result['threshold']:.2f}  "
                f"# precision={result['precision']:.3f} recall={result['recall']:.3f} "
                f"(target precision {target:.2f} MET)"
            )
        elif entry["vacuous_precision_only"] is not None:
            # Target precision is only "met" where the model made essentially no
            # predictions at all (recall < --min-recall) -- Ultralytics reports
            # precision=1.0 by convention for 0 surviving predictions. That's a
            # statistically meaningless number, not a usable threshold: a detector
            # that never fires can't be a false positive, but it also can't ever
            # catch a real failure. Recommend the best REAL (recall-backed) point
            # instead and say so loudly.
            v = entry["vacuous_precision_only"]
            best_p = best_supported_point(entry["confidence_sweep"], report["min_recall"])
            lines.append(
                f"  {key}: {best_p['threshold']:.2f}  "
                f"# WARNING: target precision {target:.2f} only reachable at near-zero recall "
                f"(precision={v['precision']:.3f} but recall={v['recall']:.3f} at conf={v['threshold']:.2f} -- "
                f"model effectively never fires there). Falling back to the best REAL operating point: "
                f"precision={best_p['precision']:.3f} recall={best_p['recall']:.3f}. "
                f"Do NOT deploy this class's threshold as-is; needs more data/training."
            )
        else:
            # Precision never reaches target at any threshold, real or vacuous:
            # fall back to the best recall-backed sweep point and say so loudly
            # rather than silently emitting a threshold that looks fine but isn't.
            best_p = best_supported_point(entry["confidence_sweep"], report["min_recall"])
            lines.append(
                f"  {key}: 0.95  "
                f"# WARNING: target precision {target:.2f} NOT reachable at any threshold -- "
                f"best real (recall>={report['min_recall']:.2f}) precision={best_p['precision']:.3f} "
                f"at conf={best_p['threshold']:.2f}. Do NOT deploy this class's threshold as-is; "
                f"needs more data/training."
            )
    return "\n".join(lines)


def print_report(report: dict, class_names: tuple[str, ...]) -> None:
    print()
    print("=" * 78)
    print("EVALUATION -- TEST SPLIT (held out from training and threshold selection)")
    print("=" * 78)
    print(f"Weights: {report['weights']}")
    print(f"mAP@50:    {report['overall']['map50']:.4f}")
    print(f"mAP@50-95: {report['overall']['map50_95']:.4f}")
    print()

    header = f"{'class':<20}{'AP50':>8}{'AP50-95':>10}{'P(maxF1)':>10}{'R(maxF1)':>10}{'#test':>8}"
    print(header)
    print("-" * len(header))
    for cname in class_names:
        entry = report["per_class"].get(cname)
        tag = "  [CATASTROPHIC]" if cname in CATASTROPHIC_CLASSES else ""
        if entry is None:
            print(f"{cname:<20}{'--':>8}{'--':>10}{'--':>10}{'--':>10}{0:>8}{tag}")
            continue
        print(
            f"{cname:<20}{entry['ap50']:>8.3f}{entry['ap50_95']:>10.3f}"
            f"{entry['precision_at_max_f1']:>10.3f}{entry['recall_at_max_f1']:>10.3f}"
            f"{entry['num_test_instances'] or 0:>8}{tag}"
        )

    target = report["target_precision"]
    print()
    print(f"Confidence threshold needed for precision >= {target:.2f} (per class):")
    for cname in class_names:
        entry = report["per_class"].get(cname)
        if entry is None:
            print(f"  {cname:<20} NO TEST INSTANCES -- cannot evaluate")
            continue
        result = entry["lowest_threshold_for_target_precision"]
        tag = " [CATASTROPHIC -- governs false-positive rate]" if cname in CATASTROPHIC_CLASSES else ""
        if result is not None:
            print(
                f"  {cname:<20} threshold={result['threshold']:.2f}  "
                f"precision={result['precision']:.3f}  recall={result['recall']:.3f}{tag}"
            )
        elif entry["vacuous_precision_only"] is not None:
            v = entry["vacuous_precision_only"]
            best_p = best_supported_point(entry["confidence_sweep"], report["min_recall"])
            print(
                f"  {cname:<20} TARGET ONLY MET AT NEAR-ZERO RECALL (vacuous: precision={v['precision']:.3f} "
                f"recall={v['recall']:.3f} @ conf={v['threshold']:.2f} -- model effectively never fires there). "
                f"Best real point: precision={best_p['precision']:.3f} recall={best_p['recall']:.3f} "
                f"@ conf={best_p['threshold']:.2f}{tag}"
            )
        else:
            best_p = best_supported_point(entry["confidence_sweep"], report["min_recall"])
            print(
                f"  {cname:<20} TARGET NOT REACHABLE at any threshold "
                f"(best real precision={best_p['precision']:.3f} @ conf={best_p['threshold']:.2f}){tag}"
            )

    print()
    print("-- Catastrophic classes (spaghetti, warping) are the ONLY classes allowed to stop a")
    print("   print. Their precision at the deployed threshold is the actual false-positive rate")
    print("   the system exposes to a running print. A low-epoch smoke-train run is expected to")
    print("   score poorly here; do not treat smoke-train numbers as production-ready.")
    print()
    print("Paste into config.example.yaml / config.yaml:")
    print("-" * 78)
    print(format_yaml_block(report, class_names))
    print("-" * 78)
    print("=" * 78)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(f"data.yaml not found: {args.data}")

    import yaml

    with open(args.data, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    class_names = tuple(data_cfg["names"])

    print(f"[evaluate] Running model.val() on TEST split of '{args.data}' with weights '{args.weights}' ...")
    metrics = run_validation(args.weights, args.data, args.imgsz, args.batch, args.device, args.nms_iou)

    report = build_report(metrics, class_names, args)
    print_report(report, class_names)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[evaluate] Full report written to: {args.out}")


if __name__ == "__main__":
    main()
