"""Fine-tune YOLOv8n on the leak-free dataset built by
training/prepare_dataset.py (datasets/argus/data.yaml).

Key choice: `flipud=0.0` (vertical-flip augmentation disabled). A camera
looking at a 3D printer never sees an upside-down print, so vertical flips
manufacture physically-impossible training images -- that's capacity spent
learning to recognize something that can't happen. `fliplr=0.5` (horizontal
flip) is kept since the camera's left/right framing is arbitrary. All other
Ultralytics augmentation/hyperparameters are left at their defaults.

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
    python training/train.py
    python training/train.py --epochs 100 --batch 32 --imgsz 512 --device 0
    python training/train.py --epochs 3 --name smoke_test   # quick smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "datasets" / "argus" / "data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "runs" / "train"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML, help=f"Path to data.yaml (default: {DEFAULT_DATA_YAML})")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Starting weights (default: yolov8n.pt, pretrained on COCO)")
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.data.is_file():
        raise FileNotFoundError(
            f"data.yaml not found at '{args.data}'. Run training/ingest_dataset.py and "
            "training/prepare_dataset.py first."
        )

    # Imported lazily so --help works without ultralytics' startup cost/side effects.
    from ultralytics import YOLO

    print(f"[train] Loading base weights: {args.weights}")
    model = YOLO(args.weights)

    print(
        f"[train] Starting training: data={args.data} epochs={args.epochs} batch={args.batch} "
        f"imgsz={args.imgsz} device={args.device} seed={args.seed} patience={args.patience} "
        f"optimizer={args.optimizer} lr0={args.lr0} "
        f"flipud=0.0 (disabled -- a print is never upside down) fliplr={args.fliplr}"
    )
    results = model.train(
        data=str(args.data),
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
    print(f"Best weights: {best_weights}  (exists: {best_weights.is_file()})")
    print(f"Last weights: {last_weights}  (exists: {last_weights.is_file()})")
    print("=" * 72)


if __name__ == "__main__":
    main()
