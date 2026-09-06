"""Trains either the detection model (datasets/argus/data.yaml) or the classification model (datasets/argus_cls/{train,val,test}/<class>/*.jpg) with a single shared CLI. ``--task`` picks the trainer; if omitted, it's auto-detected from ``--weights`` (a ``-cls`` substring means classify).

``flipud=0.0``: a print is never upside down, so vertical flips would manufacture impossible images; ``fliplr`` stays on since left/right framing is arbitrary.

``--optimizer`` defaults to ``AdamW`` (not Ultralytics' ``auto``): at some batch/imgsz combos, ``auto`` selects a MuSGD/Muon implementation that crashes on a non-contiguous `.view()`, and the choice silently varies with batch size. ``--lr0`` defaults to ``0.001``, the right scale for AdamW (not SGD's 0.01).

Usage: python training/train.py [--epochs N] [--batch N] [--imgsz N] [--device 0]
       python training/train.py --data datasets/argus_cls --weights yolo26s-cls.pt --task classify
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "datasets" / "argus" / "data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "train"


# argparse.Namespace parse_args(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: argparse.Namespace - parsed training options (data, weights, task, epochs, batch,
#          imgsz, device, seed, patience, project, name, fliplr, optimizer, lr0, exist_ok).
#          Notable defaults: --fliplr 0.5 (horizontal flip kept), --optimizer AdamW and
#          --lr0 0.001 (pinned to avoid the upstream Muon crash under optimizer=auto, see
#          module docstring), --seed 1337.
# Description: Defines and parses the shared CLI for detection and classification training.
# Side Effects: None (argparse may print usage/help and call sys.exit on bad input, but no
#               filesystem or network activity)
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help=f"Path to data.yaml (detect) or a classification dataset directory (classify) "
        f"(default: {DEFAULT_DATA_YAML})",
    )
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Starting weights (default: yolov8n.pt, pretrained on COCO)")
    parser.add_argument(
        "--task",
        type=str,
        choices=["detect", "classify"],
        default=None,
        help="Training task. Default: auto-detected from --weights -- a filename containing "
        "'-cls' resolves to classify, otherwise detect. An explicit --task always wins.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0", help="CUDA device index, list ('0,1'), or 'cpu'")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience (epochs with no val improvement)")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help=f"Ultralytics project dir (default: {DEFAULT_PROJECT})")
    parser.add_argument("--name", type=str, default="argus_yolov8n", help="Run name (subdir of --project)")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal-flip augmentation probability (default: 0.5)")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Optimizer passed to model.train() (default: AdamW). Pinned instead of Ultralytics' "
        "'auto' -- see module docstring for the upstream Muon crash this avoids.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001, appropriate for AdamW; see module docstring).",
    )
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing run dir of the same name")
    return parser.parse_args(argv)


# str resolve_task(str | None explicit_task, str weights)
# Inputs: str | None explicit_task - the --task value if the user passed one explicitly
#         str weights - the --weights filename, used for auto-detection when explicit_task
#         is None
# Outputs: str - "classify" or "detect"
# Description: Resolves the training task. An explicit --task always wins; otherwise
#              auto-detects from the weights filename ("-cls" substring means classify).
# Side Effects: None
def resolve_task(explicit_task: str | None, weights: str) -> str:
    if explicit_task is not None:
        return explicit_task
    return "classify" if "-cls" in Path(weights).name else "detect"


# None validate_data(str task, Path data)
# Inputs: str task - resolved task, "detect" or "classify"
#         Path data - the --data path to validate (a data.yaml file for detect, a dataset
#         directory for classify)
# Outputs: None
# Description: Validates --data against the requirements of the resolved task, delegating to
#              _validate_classify_data for the classify case.
# Side Effects: Raises FileNotFoundError with an actionable message if the dataset referenced
#               by data isn't in the shape the task needs. No filesystem writes.
def validate_data(task: str, data: Path) -> None:
    if task == "classify":
        _validate_classify_data(data)
        return

    if not data.is_file():
        raise FileNotFoundError(
            f"data.yaml not found at '{data}'. Run training/ingest_dataset.py and "
            "training/prepare_dataset.py first."
        )


# None _validate_classify_data(Path data)
# Inputs: Path data - candidate classification dataset directory
# Outputs: None
# Description: Checks that data is a directory containing train/ and val/ subdirectories
#              that each have at least one class subfolder (the layout
#              build_classification_dataset.py produces).
# Side Effects: Raises FileNotFoundError with an actionable message (naming the missing pieces
#               and pointing at training/build_classification_dataset.py) if the layout is
#               invalid. No filesystem writes.
def _validate_classify_data(data: Path) -> None:
    build_hint = "Run training/build_classification_dataset.py first."

    if not data.is_dir():
        raise FileNotFoundError(f"Classification dataset directory not found at '{data}'. {build_hint}")

    problems: list[str] = []
    for split in ("train", "val"):
        split_dir = data / split
        if not split_dir.is_dir():
            problems.append(f"missing '{split}/' subdirectory")
        elif not any(p.is_dir() for p in split_dir.iterdir()):
            problems.append(f"'{split}/' has no class subfolders")

    if problems:
        raise FileNotFoundError(
            f"Classification dataset at '{data}' is invalid: {'; '.join(problems)}. Expected the "
            f"Ultralytics classification layout '{data}/train/<class>/*.jpg' and "
            f"'{data}/val/<class>/*.jpg'. {build_hint}"
        )


# None main(list[str] | None argv)
# Inputs: list[str] | None argv - command-line arguments to parse, default None (uses sys.argv)
# Outputs: None
# Description: CLI entry point. Resolves the task, validates the dataset, loads the starting
#              weights, runs Ultralytics model.train() with flipud=0.0 (vertical flip disabled
#              because a print is never upside down -- see module docstring) and the pinned
#              AdamW optimizer/lr0, then reports the resulting run directory and weights paths.
# Side Effects: Imports ultralytics.YOLO lazily; loads model weights (may download the base
#               checkpoint if not cached locally); runs a full GPU/CPU training job that writes
#               checkpoints, logs, and run artifacts under --project/--name on disk; prints
#               progress and a completion summary to stdout.
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    task = resolve_task(args.task, args.weights)

    validate_data(task, args.data)

    # Imported lazily so --help works without ultralytics' startup cost/side effects.
    from ultralytics import YOLO

    print(f"[train] Loading base weights: {args.weights}")
    model = YOLO(args.weights, task=task)

    print(
        f"[train] Starting training: task={task} data={args.data} epochs={args.epochs} batch={args.batch} "
        f"imgsz={args.imgsz} device={args.device} seed={args.seed} patience={args.patience} "
        f"optimizer={args.optimizer} lr0={args.lr0} "
        f"flipud=0.0 (disabled -- a print is never upside down) fliplr={args.fliplr}"
    )
    results = model.train(
        data=str(args.data),
        task=task,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        flipud=0.0,  # 3D prints are never upside down -- see module docstring.
        fliplr=args.fliplr,
        optimizer=args.optimizer,
        lr0=args.lr0,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(args.project) / args.name
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Run dir:      {save_dir}")
    if task == "classify" and results is not None and hasattr(results, "top1"):
        print(f"top1 accuracy: {results.top1:.4f}")
    print(f"Best weights: {best_weights}  (exists: {best_weights.is_file()})")
    print(f"Last weights: {last_weights}  (exists: {last_weights.is_file()})")
    print("=" * 72)


if __name__ == "__main__":
    main()
