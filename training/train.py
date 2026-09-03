"""Train either the detection model (training/prepare_dataset.py's
datasets/argus/data.yaml) or the classification model (training/
build_classification_dataset.py's datasets/argus_cls/{train,val,test}/
<class>/*.jpg) with a single shared CLI.

Task selection: `--task {detect,classify}` picks which Ultralytics trainer
runs. If `--task` is omitted it is auto-detected from `--weights`: a
filename containing `-cls` (e.g. `yolo26s-cls.pt`) resolves to `classify`,
anything else resolves to `detect`. An explicit `--task` always overrides
the auto-detection. `--data` is validated according to the resolved task:
detection requires an existing `data.yaml` file; classification requires an
existing directory with `train/` and `val/` subdirectories that each
contain at least one class subfolder (the layout
training/build_classification_dataset.py produces).

Key choice: `flipud=0.0` (vertical-flip augmentation disabled). A camera
looking at a 3D printer never sees an upside-down print, so vertical flips
manufacture physically-impossible training images -- that's capacity spent
learning to recognize something that can't happen. `fliplr=0.5` (horizontal
flip) is kept since the camera's left/right framing is arbitrary. This
applies to both tasks. All other Ultralytics augmentation/hyperparameters
are left at their defaults.

Key choice: `--optimizer` defaults to `AdamW` instead of Ultralytics'
`auto`. With `optimizer=auto`, Ultralytics picks the optimizer via an
internal heuristic that depends on batch size and other run settings --
it is not a stable, reproducible choice. At `--batch 32 --imgsz 512` the
heuristic selects MuSGD (Muon), and the Muon implementation shipped in the
installed ultralytics 8.4.137 (ultralytics/optim/muon.py, muon_update) calls
`.view()` on a non-contiguous tensor, which crashes with:
    RuntimeError: view size is not compatible with input tensor's size and
    stride (at least one dimension spans across two contiguous subspaces).
    Use .reshape(...) instead.
This is an upstream bug in that Muon code path. A different batch/imgsz
combination can make `auto` pick a different (working) optimizer, which is
exactly the problem: the same script can crash or not depending on flags,
with no code change. Pinning `optimizer=AdamW` sidesteps the buggy code
path entirely and makes runs reproducible. AdamW is also the well-tested
default for fine-tuning YOLOv8n.

Pinning the optimizer disables Ultralytics' automatic LR selection, so
`--lr0` is also explicit here and defaults to `0.001` -- the appropriate
starting LR for AdamW fine-tuning (Ultralytics' own auto-selection uses
~0.001 for AdamW; the library-wide default of 0.01 is an SGD-scale LR and
would be far too hot for AdamW).

Usage:
    # Detection (default task; datasets/argus/data.yaml)
    python training/train.py
    python training/train.py --epochs 100 --batch 32 --imgsz 512 --device 0
    python training/train.py --epochs 3 --name smoke_test   # quick smoke test

    # Classification (auto-detected from a '-cls' weights filename)
    python training/train.py --data datasets/argus_cls --weights yolo26s-cls.pt \
        --epochs 100 --batch 64 --imgsz 512 --device 0
    # ...or force the task explicitly regardless of the weights filename:
    python training/train.py --data datasets/argus_cls --weights best.pt --task classify
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "datasets" / "argus" / "data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "train"


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


def resolve_task(explicit_task: str | None, weights: str) -> str:
    """Resolve the training task.

    An explicit `--task` always wins. Otherwise, auto-detect from the
    weights filename: Ultralytics classification checkpoints are named like
    `yolo26s-cls.pt` / `yolov8n-cls.pt`, so a `-cls` substring in the
    filename means `classify`; anything else means `detect`.
    """
    if explicit_task is not None:
        return explicit_task
    return "classify" if "-cls" in Path(weights).name else "detect"


def validate_data(task: str, data: Path) -> None:
    """Validate `--data` against the requirements of the resolved task.

    Raises FileNotFoundError with an actionable message if the dataset
    referenced by `data` isn't in the shape the task needs.
    """
    if task == "classify":
        _validate_classify_data(data)
        return

    if not data.is_file():
        raise FileNotFoundError(
            f"data.yaml not found at '{data}'. Run training/ingest_dataset.py and "
            "training/prepare_dataset.py first."
        )


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
