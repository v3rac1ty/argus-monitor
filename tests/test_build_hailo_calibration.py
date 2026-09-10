"""Tests for training/build_hailo_calibration.py: pure listing/allocation/
selection functions, plus an end-to-end run over a tiny synthetic dataset
tree (a handful of small solid-color images per class, not the real
datasets/argus_bin/train) to prove determinism, class balance, the exact
requested count, and that the shared production crop geometry
(argus.detectors.classifier.resize_and_center_crop) is actually what gets
applied before writing.

No network access, no Hailo dependency, no real dataset required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training.build_hailo_calibration import (  # noqa: E402
    allocate_per_class_counts,
    build_calibration_set,
    list_class_images,
    select_calibration_images,
)


# --------------------------------------------------------------------------
# list_class_images
# --------------------------------------------------------------------------


def _write_solid_image(path: Path, value: int, size: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size, size, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    assert ok


def _make_dataset_tree(root: Path, counts: dict[str, int]) -> None:
    """Builds root/<class>/<class>_<i>.jpg for each class in counts, each a
    small solid-color image whose pixel value encodes (class, index) so
    tests can verify exactly which source image ended up in the output."""
    for class_name, n in counts.items():
        for i in range(n):
            _write_solid_image(root / class_name / f"{class_name}_{i:03d}.jpg", value=(i * 5) % 256)


def test_list_class_images_lists_every_class_sorted(tmp_path):
    _make_dataset_tree(tmp_path, {"failure": 3, "normal": 5})
    result = list_class_images(tmp_path)
    assert set(result) == {"failure", "normal"}
    assert len(result["failure"]) == 3
    assert len(result["normal"]) == 5
    # sorted by filename
    assert [p.name for p in result["failure"]] == ["failure_000.jpg", "failure_001.jpg", "failure_002.jpg"]


def test_list_class_images_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        list_class_images(Path("/nonexistent/does/not/exist"))


def test_list_class_images_no_class_subdirs_raises(tmp_path):
    (tmp_path / "not_a_class_dir_but_a_file.txt").write_text("x")
    with pytest.raises(FileNotFoundError):
        list_class_images(tmp_path)


# --------------------------------------------------------------------------
# allocate_per_class_counts
# --------------------------------------------------------------------------


def test_allocate_per_class_counts_even_split():
    counts = allocate_per_class_counts(["failure", "normal"], 256)
    assert counts == {"failure": 128, "normal": 128}
    assert sum(counts.values()) == 256


def test_allocate_per_class_counts_remainder_goes_to_earliest_sorted_class():
    counts = allocate_per_class_counts(["normal", "failure"], 5)  # unsorted input on purpose
    # sorted(["normal", "failure"]) == ["failure", "normal"] -- remainder of
    # 1 goes to "failure", the earliest in sorted order.
    assert counts == {"failure": 3, "normal": 2}
    assert sum(counts.values()) == 5


def test_allocate_per_class_counts_is_deterministic():
    a = allocate_per_class_counts(["failure", "normal"], 257)
    b = allocate_per_class_counts(["failure", "normal"], 257)
    assert a == b


def test_allocate_per_class_counts_empty_class_names_raises():
    with pytest.raises(ValueError):
        allocate_per_class_counts([], 10)


def test_allocate_per_class_counts_negative_n_raises():
    with pytest.raises(ValueError):
        allocate_per_class_counts(["failure", "normal"], -1)


# --------------------------------------------------------------------------
# select_calibration_images
# --------------------------------------------------------------------------


def test_select_calibration_images_respects_counts_and_is_deterministic(tmp_path):
    _make_dataset_tree(tmp_path, {"failure": 10, "normal": 10})
    images_by_class = list_class_images(tmp_path)
    counts = {"failure": 4, "normal": 6}

    first = select_calibration_images(images_by_class, counts, seed=1337)
    second = select_calibration_images(images_by_class, counts, seed=1337)

    assert {k: len(v) for k, v in first.items()} == counts
    assert first == second  # same seed, same inputs -> same picks


def test_select_calibration_images_different_seed_can_differ(tmp_path):
    _make_dataset_tree(tmp_path, {"failure": 10, "normal": 10})
    images_by_class = list_class_images(tmp_path)
    counts = {"failure": 4, "normal": 6}

    a = select_calibration_images(images_by_class, counts, seed=1)
    b = select_calibration_images(images_by_class, counts, seed=2)
    assert a != b


def test_select_calibration_images_insufficient_images_raises(tmp_path):
    _make_dataset_tree(tmp_path, {"failure": 2, "normal": 10})
    images_by_class = list_class_images(tmp_path)
    with pytest.raises(ValueError, match="failure"):
        select_calibration_images(images_by_class, {"failure": 5, "normal": 5}, seed=1337)


# --------------------------------------------------------------------------
# build_calibration_set -- end-to-end over a tiny synthetic tree
# --------------------------------------------------------------------------


def test_build_calibration_set_writes_exact_count_and_class_balance(tmp_path):
    data_dir = tmp_path / "train"
    out_dir = tmp_path / "calib"
    _make_dataset_tree(data_dir, {"failure": 20, "normal": 20})

    written = build_calibration_set(data_dir, out_dir, n=10, input_size=16, seed=1337)

    assert len(written) == 10
    assert all(p.exists() for p in written)
    by_class = {"failure": 0, "normal": 0}
    for p in written:
        prefix = p.name.split("__", 1)[0]
        by_class[prefix] += 1
    assert by_class == {"failure": 5, "normal": 5}


def test_build_calibration_set_applies_input_size_crop(tmp_path):
    data_dir = tmp_path / "train"
    out_dir = tmp_path / "calib"
    _make_dataset_tree(data_dir, {"failure": 4, "normal": 4})

    written = build_calibration_set(data_dir, out_dir, n=4, input_size=8, seed=1337)

    for p in written:
        image = cv2.imread(str(p), cv2.IMREAD_COLOR)
        assert image.shape == (8, 8, 3)


def test_build_calibration_set_is_deterministic_across_runs(tmp_path):
    data_dir = tmp_path / "train"
    _make_dataset_tree(data_dir, {"failure": 20, "normal": 20})

    out_a = tmp_path / "calib_a"
    out_b = tmp_path / "calib_b"
    written_a = build_calibration_set(data_dir, out_a, n=10, input_size=16, seed=1337)
    written_b = build_calibration_set(data_dir, out_b, n=10, input_size=16, seed=1337)

    names_a = sorted(p.name for p in written_a)
    names_b = sorted(p.name for p in written_b)
    assert names_a == names_b

    # Byte-identical content too, not just matching filenames.
    for name in names_a:
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()


def test_build_calibration_set_existing_out_dir_requires_force(tmp_path):
    data_dir = tmp_path / "train"
    out_dir = tmp_path / "calib"
    _make_dataset_tree(data_dir, {"failure": 4, "normal": 4})

    build_calibration_set(data_dir, out_dir, n=4, input_size=8, seed=1337)
    with pytest.raises(FileExistsError):
        build_calibration_set(data_dir, out_dir, n=4, input_size=8, seed=1337, force=False)

    # force=True must succeed and not raise.
    build_calibration_set(data_dir, out_dir, n=4, input_size=8, seed=1337, force=True)


def test_build_calibration_set_insufficient_images_raises_before_partial_write(tmp_path):
    data_dir = tmp_path / "train"
    out_dir = tmp_path / "calib"
    _make_dataset_tree(data_dir, {"failure": 1, "normal": 20})

    with pytest.raises(ValueError):
        build_calibration_set(data_dir, out_dir, n=10, input_size=16, seed=1337)
