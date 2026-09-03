"""Unit tests for training/train.py's pure CLI logic: `parse_args` defaults,
`resolve_task`'s detect/classify auto-detection, and `validate_data`'s
per-task dataset checks.

These tests never import `ultralytics` or run any training -- `main()`
imports ultralytics lazily (after argument parsing and dataset validation)
specifically so that logic stays testable without the dependency's startup
cost/side effects. Only `tmp_path` directory trees and plain function calls
are used here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.train import DEFAULT_DATA_YAML, parse_args, resolve_task, validate_data


# --------------------------------------------------------------------------
# parse_args
# --------------------------------------------------------------------------


def test_parse_args_defaults():
    args = parse_args([])
    assert args.data == DEFAULT_DATA_YAML
    assert args.weights == "yolov8n.pt"
    assert args.task is None
    assert args.epochs == 100
    assert args.batch == 32
    assert args.imgsz == 640
    assert args.fliplr == 0.5
    assert args.optimizer == "AdamW"
    assert args.lr0 == 0.001


def test_parse_args_task_explicit():
    args = parse_args(["--task", "classify"])
    assert args.task == "classify"


def test_parse_args_task_rejects_unknown_value():
    with pytest.raises(SystemExit):
        parse_args(["--task", "segment"])


def test_parse_args_data_and_weights_override():
    args = parse_args(["--data", "datasets/argus_cls", "--weights", "yolo26s-cls.pt"])
    assert args.data == Path("datasets/argus_cls")
    assert args.weights == "yolo26s-cls.pt"


# --------------------------------------------------------------------------
# resolve_task
# --------------------------------------------------------------------------


def test_resolve_task_explicit_always_wins():
    assert resolve_task("detect", "yolo26s-cls.pt") == "detect"
    assert resolve_task("classify", "yolov8n.pt") == "classify"


def test_resolve_task_auto_detects_classify_from_weights_filename():
    assert resolve_task(None, "yolo26s-cls.pt") == "classify"
    assert resolve_task(None, "yolov8n-cls.pt") == "classify"


def test_resolve_task_auto_detects_detect_by_default():
    assert resolve_task(None, "yolov8n.pt") == "detect"
    assert resolve_task(None, "yolo26s.pt") == "detect"


def test_resolve_task_auto_detects_from_basename_of_a_path():
    # A full path to a '-cls' checkpoint should still resolve to classify,
    # not just a bare filename.
    assert resolve_task(None, "runs/train/cls_smoke/weights/best-cls.pt") == "classify"


def test_resolve_task_no_false_positive_on_unrelated_substring():
    # 'cls' without the leading hyphen must not trigger classify.
    assert resolve_task(None, "myclsmodel.pt") == "detect"


# --------------------------------------------------------------------------
# validate_data -- detect task (must be byte-for-byte unchanged behavior)
# --------------------------------------------------------------------------


def test_validate_data_detect_passes_when_yaml_exists(tmp_path: Path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("train: train\nval: val\n")
    validate_data("detect", data_yaml)  # must not raise


def test_validate_data_detect_raises_original_message_when_missing(tmp_path: Path):
    missing = tmp_path / "data.yaml"
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_data("detect", missing)
    message = str(excinfo.value)
    assert f"data.yaml not found at '{missing}'" in message
    assert "training/ingest_dataset.py" in message
    assert "training/prepare_dataset.py" in message


def test_validate_data_detect_raises_when_data_is_a_directory_not_a_file(tmp_path: Path):
    data_dir = tmp_path / "data.yaml"
    data_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        validate_data("detect", data_dir)


# --------------------------------------------------------------------------
# validate_data -- classify task
# --------------------------------------------------------------------------


def test_validate_data_classify_passes_with_valid_layout(tmp_path: Path):
    (tmp_path / "train" / "normal").mkdir(parents=True)
    (tmp_path / "train" / "spaghetti").mkdir(parents=True)
    (tmp_path / "val" / "normal").mkdir(parents=True)
    validate_data("classify", tmp_path)  # must not raise


def test_validate_data_classify_raises_when_directory_missing(tmp_path: Path):
    missing = tmp_path / "argus_cls"
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_data("classify", missing)
    message = str(excinfo.value)
    assert str(missing) in message
    assert "training/build_classification_dataset.py" in message


def test_validate_data_classify_raises_when_data_is_a_file_not_a_directory(tmp_path: Path):
    data_file = tmp_path / "argus_cls"
    data_file.write_text("not a directory")
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_data("classify", data_file)
    assert "training/build_classification_dataset.py" in str(excinfo.value)


def test_validate_data_classify_raises_when_train_subdir_missing(tmp_path: Path):
    (tmp_path / "val" / "normal").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_data("classify", tmp_path)
    message = str(excinfo.value)
    assert "train" in message
    assert "training/build_classification_dataset.py" in message


def test_validate_data_classify_raises_when_val_subdir_missing(tmp_path: Path):
    (tmp_path / "train" / "normal").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_data("classify", tmp_path)
    assert "val" in str(excinfo.value)


def test_validate_data_classify_raises_when_train_has_no_class_subfolders(tmp_path: Path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val" / "normal").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_data("classify", tmp_path)
    message = str(excinfo.value)
    assert "train" in message
    assert "no class subfolders" in message


def test_validate_data_classify_raises_when_val_has_no_class_subfolders(tmp_path: Path):
    (tmp_path / "train" / "normal").mkdir(parents=True)
    (tmp_path / "val").mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_data("classify", tmp_path)
    message = str(excinfo.value)
    assert "val" in message
    assert "no class subfolders" in message


def test_validate_data_classify_reports_both_missing_pieces_together(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_data("classify", tmp_path)
    message = str(excinfo.value)
    assert "train" in message
    assert "val" in message


def test_validate_data_classify_ignores_stray_files_in_split_dirs(tmp_path: Path):
    # A non-directory entry (e.g. a stray README) inside train/ or val/
    # shouldn't count as a class subfolder.
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "README.txt").write_text("not a class")
    (tmp_path / "val" / "normal").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_data("classify", tmp_path)
    assert "no class subfolders" in str(excinfo.value)
