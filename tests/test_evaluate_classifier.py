"""Unit tests for training/evaluate_classifier.py's pure metrics logic:
confusion-matrix construction, per-class precision/recall/F1, the binary
normal-vs-defect collapse, the catastrophic-path (spaghetti) threshold
sweep, its near-zero-recall "vacuous precision" guard, the source-confound
recall grouping, and filename-based source inference.

Everything here operates on synthetic class-name lists / hand-built
CatastrophicRecord objects -- no model file, no GPU, no dataset on disk.
"""

from __future__ import annotations

import pytest

from training.evaluate_classifier import (
    REPORT_CLASS_ORDER,
    CatastrophicRecord,
    best_supported_point,
    binary_labels,
    binary_metrics,
    build_confusion_matrix,
    check_class_order_matches_model,
    evaluate_catastrophic_threshold,
    find_lowest_threshold_for_precision,
    group_recall_by_source,
    infer_source,
    per_class_prf1,
    sweep_catastrophic,
    sweep_thresholds,
    top1_accuracy,
)

CLASSES = ("normal", "spaghetti", "cracking")


# --------------------------------------------------------------------------
# build_confusion_matrix / top1_accuracy
# --------------------------------------------------------------------------


def test_build_confusion_matrix_all_correct():
    y_true = ["normal", "spaghetti", "cracking"]
    y_pred = ["normal", "spaghetti", "cracking"]
    cm = build_confusion_matrix(y_true, y_pred, CLASSES)
    assert cm.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert top1_accuracy(cm) == pytest.approx(1.0)


def test_build_confusion_matrix_mixed():
    # 2 normal correct, 1 normal predicted spaghetti (false positive);
    # 3 spaghetti correct, 1 spaghetti predicted cracking (missed).
    y_true = ["normal", "normal", "normal", "spaghetti", "spaghetti", "spaghetti", "spaghetti"]
    y_pred = ["normal", "normal", "spaghetti", "spaghetti", "spaghetti", "spaghetti", "cracking"]
    cm = build_confusion_matrix(y_true, y_pred, CLASSES)
    # rows/cols ordered per CLASSES = (normal, spaghetti, cracking)
    assert cm[0].tolist() == [2, 1, 0]  # true normal -> {normal:2, spaghetti:1}
    assert cm[1].tolist() == [0, 3, 1]  # true spaghetti -> {spaghetti:3, cracking:1}
    assert cm[2].tolist() == [0, 0, 0]  # no true cracking instances
    assert top1_accuracy(cm) == pytest.approx(5 / 7)


def test_build_confusion_matrix_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        build_confusion_matrix(["normal"], ["normal", "spaghetti"], CLASSES)


def test_top1_accuracy_empty_matrix_is_zero():
    cm = build_confusion_matrix([], [], CLASSES)
    assert top1_accuracy(cm) == 0.0


# --------------------------------------------------------------------------
# per_class_prf1
# --------------------------------------------------------------------------


def test_per_class_prf1_basic():
    y_true = ["normal", "normal", "normal", "spaghetti", "spaghetti", "spaghetti", "spaghetti"]
    y_pred = ["normal", "normal", "spaghetti", "spaghetti", "spaghetti", "spaghetti", "cracking"]
    cm = build_confusion_matrix(y_true, y_pred, CLASSES)
    result = per_class_prf1(cm, CLASSES)

    # normal: 2 tp, support 3 (recall 2/3), predicted_count 2 (precision 2/2=1.0)
    assert result["normal"]["support"] == 3
    assert result["normal"]["predicted_count"] == 2
    assert result["normal"]["precision"] == pytest.approx(1.0)
    assert result["normal"]["recall"] == pytest.approx(2 / 3)

    # spaghetti: 3 tp, support 4 (recall 3/4), predicted_count 4 (1 fp from normal) -> precision 3/4
    assert result["spaghetti"]["support"] == 4
    assert result["spaghetti"]["predicted_count"] == 4
    assert result["spaghetti"]["precision"] == pytest.approx(3 / 4)
    assert result["spaghetti"]["recall"] == pytest.approx(3 / 4)
    expected_f1 = 2 * 0.75 * 0.75 / (0.75 + 0.75)
    assert result["spaghetti"]["f1"] == pytest.approx(expected_f1)

    # cracking: 0 support, 1 predicted (the false negative from spaghetti) -> precision 0, recall 0
    assert result["cracking"]["support"] == 0
    assert result["cracking"]["predicted_count"] == 1
    assert result["cracking"]["precision"] == 0.0
    assert result["cracking"]["recall"] == 0.0
    assert result["cracking"]["f1"] == 0.0


def test_per_class_prf1_zero_predictions_is_zero_not_error():
    # A class with real support that the model never predicts at all.
    y_true = ["cracking", "cracking"]
    y_pred = ["normal", "normal"]
    cm = build_confusion_matrix(y_true, y_pred, CLASSES)
    result = per_class_prf1(cm, CLASSES)
    assert result["cracking"]["support"] == 2
    assert result["cracking"]["predicted_count"] == 0
    assert result["cracking"]["precision"] == 0.0
    assert result["cracking"]["recall"] == 0.0


# --------------------------------------------------------------------------
# binary_labels / binary_metrics
# --------------------------------------------------------------------------


def test_binary_labels_collapses_defects():
    names = ["normal", "spaghetti", "cracking", "layer_shifting", "stringing", "warping"]
    assert binary_labels(names) == ["normal", "defect", "defect", "defect", "defect", "defect"]


def test_binary_labels_case_and_whitespace_insensitive():
    assert binary_labels([" Normal ", "NORMAL"]) == ["normal", "normal"]


def test_binary_metrics_false_positive_rate():
    # 10 true normal: 9 correctly called normal, 1 wrongly called defect.
    # 5 true defect: 4 correctly called defect, 1 wrongly called normal.
    y_true = ["normal"] * 10 + ["defect"] * 5
    y_pred = ["normal"] * 9 + ["defect"] * 1 + ["defect"] * 4 + ["normal"] * 1
    result = binary_metrics(y_true, y_pred)

    cm = result["confusion_matrix"]
    assert cm == {"tp": 4, "fp": 1, "fn": 1, "tn": 9}
    assert result["false_positive_rate"] == pytest.approx(1 / 10)
    assert result["precision"] == pytest.approx(4 / 5)
    assert result["recall"] == pytest.approx(4 / 5)
    assert result["num_normal"] == 10
    assert result["num_defect"] == 5


def test_binary_metrics_perfect_classifier_zero_false_positive_rate():
    y_true = ["normal", "normal", "defect", "defect"]
    y_pred = ["normal", "normal", "defect", "defect"]
    result = binary_metrics(y_true, y_pred)
    assert result["false_positive_rate"] == 0.0
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_binary_metrics_all_zero_when_no_data():
    result = binary_metrics([], [])
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["false_positive_rate"] == 0.0


# --------------------------------------------------------------------------
# sweep_thresholds
# --------------------------------------------------------------------------


def test_sweep_thresholds_matches_evaluate_py_convention():
    thresholds = sweep_thresholds(0.05, 0.95, 0.05)
    assert thresholds[0] == pytest.approx(0.05)
    assert thresholds[-1] == pytest.approx(0.95)
    assert len(thresholds) == 19  # (0.95-0.05)/0.05 + 1


def test_sweep_thresholds_single_step():
    assert sweep_thresholds(0.5, 0.5, 0.05) == [0.5]


# --------------------------------------------------------------------------
# evaluate_catastrophic_threshold / sweep_catastrophic
# --------------------------------------------------------------------------


def _records(rows: list[tuple[str, bool, float]]) -> list[CatastrophicRecord]:
    """rows: (true_name, is_argmax_spaghetti, spaghetti_prob)"""
    return [CatastrophicRecord(true_name=t, is_argmax=a, catastrophic_prob=p) for t, a, p in rows]


def test_evaluate_catastrophic_threshold_basic_counts():
    records = _records(
        [
            ("spaghetti", True, 0.9),  # tp at any threshold <= 0.9
            ("spaghetti", True, 0.3),  # tp only at low threshold
            ("spaghetti", False, 0.6),  # never a positive prediction: not argmax -> fn always
            ("normal", True, 0.8),  # fp at threshold <= 0.8 (impossible in practice, but exercises the guard)
            ("cracking", False, 0.1),  # tn (not argmax spaghetti, not spaghetti)
        ]
    )
    result = evaluate_catastrophic_threshold(records, threshold=0.5)
    # positive iff is_argmax and prob >= 0.5: rows 0 (0.9) and 3 (0.8) qualify.
    assert result["tp"] == 1  # row 0
    assert result["fp"] == 1  # row 3 (true normal, predicted positive)
    assert result["fn"] == 2  # rows 1 and 2 (true spaghetti, not predicted positive)
    assert result["tn"] == 1  # row 4
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1 / 3)


def test_evaluate_catastrophic_threshold_non_argmax_never_counts_as_positive():
    # A very high spaghetti probability that ISN'T the argmax (some other
    # class won) must never count as a positive prediction -- this is the
    # real runtime rule (postprocess_classify only ever emits a detection
    # for the argmax class).
    records = _records([("spaghetti", False, 0.99)])
    result = evaluate_catastrophic_threshold(records, threshold=0.05)
    assert result["tp"] == 0
    assert result["fn"] == 1
    assert result["fp"] == 0


def test_evaluate_catastrophic_threshold_zero_predictions_is_vacuous_precision_one():
    # No record clears the threshold -> tp+fp == 0 -> precision reported as
    # 1.0 by convention (mirrors training/evaluate.py / Ultralytics).
    records = _records([("spaghetti", True, 0.1), ("normal", False, 0.0)])
    result = evaluate_catastrophic_threshold(records, threshold=0.99)
    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 0.0


def test_sweep_catastrophic_is_monotonic_non_increasing_recall():
    records = _records([("spaghetti", True, 0.3), ("spaghetti", True, 0.6), ("spaghetti", True, 0.9)])
    sweep = sweep_catastrophic(records, [0.1, 0.5, 0.7, 0.95])
    recalls = [pt["recall"] for pt in sweep]
    assert recalls == sorted(recalls, reverse=True)
    assert recalls[0] == pytest.approx(1.0)  # threshold 0.1: all 3 qualify
    assert recalls[-1] == pytest.approx(0.0)  # threshold 0.95: none qualify


# --------------------------------------------------------------------------
# find_lowest_threshold_for_precision -- the vacuous-target guard
# --------------------------------------------------------------------------


def test_find_lowest_threshold_for_precision_finds_real_operating_point():
    sweep = [
        {"threshold": 0.1, "precision": 0.80, "recall": 0.9, "tp": 9, "fp": 2, "fn": 1, "tn": 0},
        {"threshold": 0.2, "precision": 0.97, "recall": 0.5, "tp": 5, "fp": 0, "fn": 5, "tn": 0},
        {"threshold": 0.3, "precision": 0.99, "recall": 0.2, "tp": 2, "fp": 0, "fn": 8, "tn": 0},
    ]
    result, vacuous = find_lowest_threshold_for_precision(sweep, target_precision=0.95, min_recall=0.05)
    assert result is not None
    assert result["threshold"] == 0.2  # lowest threshold meeting both conditions
    assert vacuous is None


def test_find_lowest_threshold_for_precision_vacuous_guard_fires():
    # Target precision is only "met" at a threshold where almost nothing
    # fires (recall below the min_recall floor) -- must be rejected and
    # reported as the vacuous example, not accepted as a real result.
    sweep = [
        {"threshold": 0.1, "precision": 0.60, "recall": 0.90, "tp": 9, "fp": 6, "fn": 1, "tn": 0},
        {"threshold": 0.5, "precision": 0.70, "recall": 0.40, "tp": 4, "fp": 2, "fn": 6, "tn": 0},
        {"threshold": 0.9, "precision": 1.00, "recall": 0.01, "tp": 1, "fp": 0, "fn": 99, "tn": 0},
    ]
    result, vacuous = find_lowest_threshold_for_precision(sweep, target_precision=0.95, min_recall=0.05)
    assert result is None
    assert vacuous is not None
    assert vacuous["threshold"] == 0.9
    assert vacuous["recall"] == pytest.approx(0.01)


def test_find_lowest_threshold_for_precision_unreachable_anywhere():
    sweep = [
        {"threshold": 0.1, "precision": 0.5, "recall": 0.9, "tp": 9, "fp": 9, "fn": 1, "tn": 0},
        {"threshold": 0.5, "precision": 0.6, "recall": 0.4, "tp": 4, "fp": 3, "fn": 6, "tn": 0},
    ]
    result, vacuous = find_lowest_threshold_for_precision(sweep, target_precision=0.95, min_recall=0.05)
    assert result is None
    assert vacuous is None


def test_best_supported_point_prefers_highest_precision_with_recall_floor():
    sweep = [
        {"threshold": 0.1, "precision": 0.5, "recall": 0.9, "tp": 9, "fp": 9, "fn": 1, "tn": 0},
        {"threshold": 0.5, "precision": 0.6, "recall": 0.4, "tp": 4, "fp": 3, "fn": 6, "tn": 0},
        {"threshold": 0.9, "precision": 1.0, "recall": 0.01, "tp": 1, "fp": 0, "fn": 99, "tn": 0},  # vacuous, excluded
    ]
    best = best_supported_point(sweep, min_recall=0.05)
    assert best["threshold"] == 0.5  # highest precision among points with recall >= 0.05


def test_best_supported_point_falls_back_to_highest_recall_when_none_clear_floor():
    sweep = [
        {"threshold": 0.5, "precision": 0.3, "recall": 0.02, "tp": 1, "fp": 2, "fn": 49, "tn": 0},
        {"threshold": 0.9, "precision": 1.0, "recall": 0.01, "tp": 1, "fp": 0, "fn": 99, "tn": 0},
    ]
    best = best_supported_point(sweep, min_recall=0.05)
    assert best["threshold"] == 0.5  # neither clears the floor -> highest recall wins


# --------------------------------------------------------------------------
# group_recall_by_source (source-confound diagnostic)
# --------------------------------------------------------------------------


def test_group_recall_by_source_basic():
    sources = ["fdm", "fdm", "argus_v2", "argus_v2", "argus_v2"]
    correct = [True, False, True, True, True]
    result = group_recall_by_source(sources, correct)
    assert result["fdm"] == {"support": 2, "correct": 1, "recall": pytest.approx(0.5)}
    assert result["argus_v2"] == {"support": 3, "correct": 3, "recall": pytest.approx(1.0)}


def test_group_recall_by_source_large_gap_reproduces_real_finding_shape():
    # Shape of the actual measured result on this project's test split:
    # near-perfect on argus_v2, near-total failure on fdm.
    sources = ["fdm"] * 22 + ["argus_v2"] * 61
    correct = [False] * 20 + [True] * 2 + [True] * 61
    result = group_recall_by_source(sources, correct)
    assert result["fdm"]["recall"] == pytest.approx(2 / 22)
    assert result["argus_v2"]["recall"] == pytest.approx(1.0)


def test_group_recall_by_source_empty_source_is_none_not_zero():
    result = group_recall_by_source([], [])
    assert result == {}


def test_group_recall_by_source_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        group_recall_by_source(["fdm"], [])


# --------------------------------------------------------------------------
# infer_source
# --------------------------------------------------------------------------


def test_infer_source_fdm_prefix():
    assert infer_source("fdm_off_platform_Image_20231228112802894.jpg") == "fdm"


def test_infer_source_argus_v2_prefix():
    assert infer_source("argus_v2_00107_error_dataset_jpeg.rf.d4a23052.jpg") == "argus_v2"


def test_infer_source_hf_prefix():
    assert infer_source("hf_normal_train_000123.jpg") == "hf"


def test_infer_source_unknown_prefix():
    assert infer_source("some_random_filename.jpg") == "unknown"


# --------------------------------------------------------------------------
# check_class_order_matches_model
# --------------------------------------------------------------------------


def test_check_class_order_matches_model_no_warning_when_matching():
    names = {i: name for i, name in enumerate(REPORT_CLASS_ORDER)}
    assert check_class_order_matches_model(names, REPORT_CLASS_ORDER) is None


def test_check_class_order_matches_model_warns_on_alphabetical_mismatch():
    # Reproduces the real, measured mismatch: Ultralytics assigns indices
    # alphabetically from folder names, not the human-readable config order.
    alphabetical = sorted(REPORT_CLASS_ORDER)
    names = {i: name for i, name in enumerate(alphabetical)}
    warning = check_class_order_matches_model(names, REPORT_CLASS_ORDER)
    assert warning is not None
    assert "CLASS ORDER MISMATCH" in warning
    assert "ClassifierDetector" in warning
