"""Unit tests for training/evaluate.py's pure logic: per-source-photo dedup
grouping and deterministic representative selection (the fix for the
augmentation-weighted mAP bug), per-class effective sample size, source
provenance resolution for the source-confound diagnostic, the deduped-vs-raw
gap note, and the preserved threshold-sweep / min-recall "vacuous precision"
guard.

Everything here runs on synthetic filename lists, hand-built label maps and a
fake metrics stub -- no model file, no GPU, no network, no real dataset on disk.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from training.evaluate import (
    CATASTROPHIC_CLASSES,
    COSMETIC_CLASSES,
    LOW_SUPPORT_PHOTOS,
    best_supported_point,
    build_report,
    class_group_support,
    class_support,
    find_lowest_threshold_for_precision,
    format_yaml_block,
    list_image_filenames,
    load_manifest_groups,
    load_manifest_sources,
    map_gap_note,
    metrics_summary,
    parse_args,
    read_split_labels,
    resolve_image_sources,
    resolve_split_images_dir,
    select_representatives,
    shared_class_recall_gaps,
    source_confound_conclusion,
    source_recall_gap,
    sweep_thresholds,
    variant_count_stats,
    write_subset_dataset,
)

CLASS_NAMES = (
    "spaghetti",
    "layer_separation",
    "bed_adhesion",
    "blob_of_death",
    "warping",
    "stringing",
    "error_extrusion",
)


# str _rf(str stem, str h)
# Inputs: str stem - the source-photo stem, e.g. "atco__imag-102-_jpg"
#         str h - the Roboflow variant hex hash
# Outputs: str - a Roboflow-style filename "<stem>.rf.<h>.jpg"
# Description: Builds one augmentation-variant filename in the exact shape
#              ingest_roboflow_multi.py emits, so tests exercise the real regex.
# Side Effects: None
def _rf(stem: str, h: str) -> str:
    return f"{stem}.rf.{h}.jpg"


# --------------------------------------------------------------------------
# select_representatives -- dedup grouping + representative determinism
# --------------------------------------------------------------------------


class TestSelectRepresentatives:
    def test_groups_augmentation_variants_of_one_photo_together(self):
        files = [_rf("atco__imag-102-_jpg", f"{i:032x}") for i in range(60)]
        reps, groups, no_rf = select_representatives(files, seed=1337)
        assert len(groups) == 1
        assert set(groups["atco__imag-102-_jpg"]) == set(files)
        assert len(reps) == 1
        assert no_rf == 0

    def test_one_representative_per_source_photo_regardless_of_variant_count(self):
        # The whole point of the fix: a 60-variant photo and a 1-variant photo
        # must contribute exactly one evaluated image each.
        heavy = [_rf("atco__heavy", f"{i:032x}") for i in range(60)]
        light = [_rf("atco__light", "aa" * 16)]
        reps, groups, _ = select_representatives(heavy + light, seed=1337)
        assert len(groups) == 2
        assert len(reps) == 2
        assert sum(1 for r in reps if r.startswith("atco__heavy")) == 1
        assert sum(1 for r in reps if r.startswith("atco__light")) == 1

    def test_representative_is_always_one_of_that_photos_variants(self):
        files = [_rf(f"src__photo{p}", f"{v:032x}") for p in range(7) for v in range(5)]
        reps, groups, _ = select_representatives(files, seed=7)
        by_group = {gid: set(v) for gid, v in groups.items()}
        assert len(reps) == len(by_group)
        for rep in reps:
            owner = [gid for gid, variants in by_group.items() if rep in variants]
            assert len(owner) == 1

    def test_deterministic_across_calls_with_same_seed(self):
        files = [_rf(f"src__photo{p}", f"{v:032x}") for p in range(20) for v in range(9)]
        first = select_representatives(files, seed=1337)[0]
        second = select_representatives(files, seed=1337)[0]
        assert first == second

    def test_independent_of_input_order(self):
        # Filesystem iteration order must never change the evaluated subset.
        files = [_rf(f"src__photo{p}", f"{v:032x}") for p in range(15) for v in range(6)]
        forward = select_representatives(files, seed=1337)[0]
        backward = select_representatives(list(reversed(files)), seed=1337)[0]
        shuffled = select_representatives(sorted(files, key=lambda s: s[::-1]), seed=1337)[0]
        assert forward == backward == shuffled

    def test_choice_for_a_photo_does_not_depend_on_which_other_photos_are_present(self):
        target = [_rf("src__target", f"{v:032x}") for v in range(11)]
        others = [_rf(f"src__other{p}", f"{v:032x}") for p in range(5) for v in range(4)]
        alone = select_representatives(target, seed=99)[0]
        together = select_representatives(target + others, seed=99)[0]
        assert [r for r in together if r.startswith("src__target")] == alone

    def test_different_seeds_select_a_different_subset(self):
        files = [_rf(f"src__photo{p}", f"{v:032x}") for p in range(40) for v in range(8)]
        a = select_representatives(files, seed=1)[0]
        b = select_representatives(files, seed=2)[0]
        assert a != b
        assert len(a) == len(b) == 40

    def test_first_strategy_picks_lexicographically_first_variant(self):
        files = [_rf("src__photo", "cc" * 16), _rf("src__photo", "aa" * 16), _rf("src__photo", "bb" * 16)]
        reps, _, _ = select_representatives(files, strategy="first")
        assert reps == [_rf("src__photo", "aa" * 16)]

    def test_first_strategy_is_also_order_independent(self):
        files = [_rf(f"src__p{p}", f"{v:032x}") for p in range(6) for v in range(4)]
        assert (
            select_representatives(files, strategy="first")[0]
            == select_representatives(list(reversed(files)), strategy="first")[0]
        )

    def test_returned_representatives_are_sorted(self):
        files = [_rf(f"src__p{p}", f"{v:032x}") for p in range(10) for v in range(3)]
        reps, _, _ = select_representatives(files, seed=5)
        assert reps == sorted(reps)

    def test_filenames_without_rf_suffix_are_their_own_group_and_counted(self):
        files = ["plain_photo.jpg", "another_plain.png", _rf("src__p", "ab" * 16)]
        reps, groups, no_rf = select_representatives(files, seed=1)
        assert no_rf == 2
        assert len(groups) == 3
        assert sorted(reps) == sorted(files)

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="strategy must be one of"):
            select_representatives([_rf("a__b", "aa" * 16)], strategy="whichever-the-fs-gave")

    def test_empty_input(self):
        reps, groups, no_rf = select_representatives([])
        assert (reps, groups, no_rf) == ([], {}, 0)


# --------------------------------------------------------------------------
# variant_count_stats
# --------------------------------------------------------------------------


class TestVariantCountStats:
    def test_reports_the_spread_that_makes_per_image_means_wrong(self):
        groups = {"a": ["x"] * 60, "b": ["x"] * 13, "c": ["x"], "d": ["x"] * 12}
        stats = variant_count_stats(groups)
        assert stats["num_images"] == 86
        assert stats["num_source_photos"] == 4
        assert stats["min"] == 1
        assert stats["max"] == 60
        assert stats["median"] == pytest.approx(12.5)
        assert stats["mean"] == pytest.approx(21.5)
        assert stats["num_photos_over_10_variants"] == 3

    def test_uniform_groups(self):
        stats = variant_count_stats({k: ["x", "y", "z"] for k in "abcde"})
        assert stats["min"] == stats["max"] == 3
        assert stats["mean"] == pytest.approx(3.0)
        assert stats["num_photos_over_10_variants"] == 0

    def test_empty(self):
        stats = variant_count_stats({})
        assert stats["num_source_photos"] == 0
        assert stats["num_images"] == 0
        assert stats["mean"] == 0.0


# --------------------------------------------------------------------------
# class_support -- effective sample size in unique source photographs
# --------------------------------------------------------------------------


class TestClassSupport:
    def test_counts_photos_not_augmentation_variants(self):
        labels = {
            _rf("atco__p1", "aa" * 16): [0, 0],
            _rf("atco__p1", "bb" * 16): [0],
            _rf("atco__p1", "cc" * 16): [0],
            _rf("atco__p2", "dd" * 16): [0],
        }
        support = class_support(labels, CLASS_NAMES)
        assert support["spaghetti"]["instances"] == 5
        assert support["spaghetti"]["images"] == 4
        # Only TWO independent photographs back this class, not four images.
        assert support["spaghetti"]["source_photos"] == 2

    def test_background_images_contribute_no_support(self):
        labels = {_rf("s__bg", "aa" * 16): [], _rf("s__p", "bb" * 16): [4]}
        support = class_support(labels, CLASS_NAMES)
        assert support["warping"]["source_photos"] == 1
        assert support["spaghetti"] == {"instances": 0, "images": 0, "source_photos": 0}

    def test_multiple_classes_in_one_image_count_that_image_once_per_class(self):
        labels = {_rf("s__p", "aa" * 16): [0, 0, 4]}
        support = class_support(labels, CLASS_NAMES)
        assert support["spaghetti"] == {"instances": 2, "images": 1, "source_photos": 1}
        assert support["warping"] == {"instances": 1, "images": 1, "source_photos": 1}

    def test_every_configured_class_is_present_even_at_zero(self):
        support = class_support({}, CLASS_NAMES)
        assert set(support) == set(CLASS_NAMES)
        assert all(v == {"instances": 0, "images": 0, "source_photos": 0} for v in support.values())

    def test_out_of_range_class_index_is_ignored_not_fatal(self):
        labels = {_rf("s__p", "aa" * 16): [0, 99, -1]}
        support = class_support(labels, CLASS_NAMES)
        assert support["spaghetti"]["instances"] == 1


class TestClassGroupSupport:
    def test_counts_independent_scene_groups_not_photos(self):
        # Three different photographs, but the ingest clustered two of them as
        # near-duplicates/timelapse siblings of one scene.
        labels = {
            _rf("s__p1", "aa" * 16): [0],
            _rf("s__p2", "bb" * 16): [0],
            _rf("s__p3", "cc" * 16): [0],
        }
        groups = {
            _rf("s__p1", "aa" * 16): "g1",
            _rf("s__p2", "bb" * 16): "g1",
            _rf("s__p3", "cc" * 16): "g2",
        }
        assert class_support(labels, CLASS_NAMES)["spaghetti"]["source_photos"] == 3
        assert class_group_support(labels, CLASS_NAMES, groups)["spaghetti"] == 2

    def test_images_missing_from_the_group_map_are_skipped(self):
        labels = {_rf("s__p1", "aa" * 16): [0], _rf("s__p2", "bb" * 16): [0]}
        assert class_group_support(labels, CLASS_NAMES, {_rf("s__p1", "aa" * 16): "g1"})["spaghetti"] == 1

    def test_every_class_present_at_zero(self):
        result = class_group_support({}, CLASS_NAMES, {})
        assert set(result) == set(CLASS_NAMES)
        assert all(v == 0 for v in result.values())


# --------------------------------------------------------------------------
# Provenance for the source-confound diagnostic
# --------------------------------------------------------------------------


class TestResolveImageSources:
    def test_manifest_is_authoritative_over_the_filename_prefix(self):
        names = ["atco__p.rf.aa.jpg"]
        assert resolve_image_sources(names, {"atco__p.rf.aa.jpg": "stereovision"}) == {
            "atco__p.rf.aa.jpg": "stereovision"
        }

    def test_falls_back_to_the_source_tag_filename_prefix(self):
        names = ["atco__p.rf.aa.jpg", "stereovision__q.rf.bb.jpg"]
        assert resolve_image_sources(names, {}) == {
            "atco__p.rf.aa.jpg": "atco",
            "stereovision__q.rf.bb.jpg": "stereovision",
        }

    def test_unknown_when_neither_manifest_nor_prefix_is_available(self):
        assert resolve_image_sources(["bare.jpg"], {}) == {"bare.jpg": "unknown"}


class TestLoadManifestSources:
    def test_reads_output_basename_to_source(self, tmp_path):
        p = tmp_path / "manifest.jsonl"
        p.write_text(
            json.dumps({"output": "test/images/atco__a.rf.aa.jpg", "source": "atco"})
            + "\n"
            + json.dumps({"output": "test/images/stereovision__b.rf.bb.jpg", "source": "stereovision"})
            + "\n",
            encoding="utf-8",
        )
        assert load_manifest_sources(p) == {
            "atco__a.rf.aa.jpg": "atco",
            "stereovision__b.rf.bb.jpg": "stereovision",
        }

    def test_malformed_and_incomplete_lines_are_skipped_not_fatal(self, tmp_path):
        p = tmp_path / "manifest.jsonl"
        p.write_text(
            "not json\n\n"
            + json.dumps({"output": "x/atco__a.jpg"})
            + "\n"
            + json.dumps({"output": "x/atco__b.jpg", "source": "atco"})
            + "\n",
            encoding="utf-8",
        )
        assert load_manifest_sources(p) == {"atco__b.jpg": "atco"}

    def test_missing_manifest_yields_empty_map(self, tmp_path):
        assert load_manifest_sources(tmp_path / "nope.jsonl") == {}
        assert load_manifest_sources(None) == {}

    def test_group_ids_load_from_the_same_manifest(self, tmp_path):
        p = tmp_path / "manifest.jsonl"
        p.write_text(
            json.dumps({"output": "test/images/atco__a.rf.aa.jpg", "source": "atco", "group": "g000001"}) + "\n",
            encoding="utf-8",
        )
        assert load_manifest_groups(p) == {"atco__a.rf.aa.jpg": "g000001"}
        assert load_manifest_groups(None) == {}


class TestSourceConfound:
    def test_gap_is_max_minus_min(self):
        assert source_recall_gap({"atco": 0.6, "stereovision": 0.2}) == pytest.approx(0.4)

    def test_gap_is_none_with_fewer_than_two_measurable_sources(self):
        assert source_recall_gap({"atco": 0.6}) is None
        assert source_recall_gap({"atco": 0.6, "stereovision": None}) is None

    def test_conclusion_is_inconclusive_with_one_source(self):
        assert "INCONCLUSIVE" in source_confound_conclusion({"atco": 0.4}, None, 0)

    def test_conclusion_reports_unrecoverable_provenance(self):
        shared = shared_class_recall_gaps({"atco": {"spaghetti": 0.4}, "stereovision": {"spaghetti": 0.39}})
        text = source_confound_conclusion({"atco": 0.4, "stereovision": 0.39}, 0.01, 12, shared)
        assert "12 test image(s) had unrecoverable provenance" in text


class TestSharedClassRecallGaps:
    def test_keeps_only_classes_measured_on_both_origins(self):
        shared = shared_class_recall_gaps(
            {
                "atco": {"spaghetti": 0.295, "warping": 0.215, "stringing": 0.237},
                "stereovision": {"spaghetti": 0.577, "warping": 0.228, "blob_of_death": 0.965},
            }
        )
        assert set(shared) == {"spaghetti", "warping"}
        assert shared["spaghetti"]["gap"] == pytest.approx(0.282)
        assert shared["warping"]["gap"] == pytest.approx(0.013)
        assert shared["spaghetti"]["by_source"] == {"atco": 0.295, "stereovision": 0.577}

    def test_disjoint_class_coverage_yields_nothing(self):
        assert shared_class_recall_gaps({"a": {"x": 0.1}, "b": {"y": 0.9}}) == {}

    def test_empty_input(self):
        assert shared_class_recall_gaps({}) == {}


class TestSourceConfoundConclusion:
    # A source contributing only easy classes outscores one contributing only hard classes,
    # for reasons that have nothing to do with dataset origin. The verdict must not read that
    # aggregate difference as a confound.
    def test_disjoint_coverage_refuses_to_call_it_a_confound(self):
        text = source_confound_conclusion({"atco": 0.239, "stereovision": 0.447}, 0.208, 0, {})
        assert "NOT A CLEAN CONFOUND TEST" in text
        assert "CONFOUND LIKELY" not in text
        assert "class mixes" in text

    def test_verdict_rests_on_the_shared_class_gap_not_the_aggregate(self):
        # Big aggregate gap, but the one comparable class matches closely -> not a confound.
        shared = shared_class_recall_gaps({"atco": {"warping": 0.215}, "stereovision": {"warping": 0.228}})
        text = source_confound_conclusion({"atco": 0.239, "stereovision": 0.447}, 0.208, 0, shared)
        assert "NO STRONG CONFOUND" in text
        assert "0.013" in text
        assert "only 1 class is scored on both origins" in text

    def test_flags_a_large_shared_class_gap(self):
        shared = shared_class_recall_gaps(
            {"atco": {"spaghetti": 0.295, "warping": 0.215}, "stereovision": {"spaghetti": 0.577, "warping": 0.228}}
        )
        text = source_confound_conclusion({"atco": 0.239, "stereovision": 0.447}, 0.208, 0, shared)
        assert "CONFOUND LIKELY" in text
        assert "spaghetti" in text
        assert "0.282" in text

    def test_aggregate_gap_is_still_reported(self):
        shared = shared_class_recall_gaps({"atco": {"warping": 0.21}, "stereovision": {"warping": 0.22}})
        text = source_confound_conclusion({"atco": 0.239, "stereovision": 0.447}, 0.208, 0, shared)
        assert "0.239" in text and "0.447" in text and "0.208" in text


# --------------------------------------------------------------------------
# map_gap_note -- deduped vs raw, never hidden
# --------------------------------------------------------------------------


class TestMapGapNote:
    def test_flags_an_optimistic_raw_number(self):
        note = map_gap_note(0.200, 0.260)
        assert note["significant"] is True
        assert note["raw_is_optimistic"] is True
        assert note["absolute"] == pytest.approx(0.06)
        assert note["relative"] == pytest.approx(0.30)
        assert "OPTIMISTIC" in note["message"]

    def test_flags_a_pessimistic_raw_number(self):
        note = map_gap_note(0.300, 0.240)
        assert note["significant"] is True
        assert note["raw_is_optimistic"] is False
        assert "PESSIMISTIC" in note["message"]

    def test_small_gap_is_not_flagged(self):
        note = map_gap_note(0.300, 0.303)
        assert note["significant"] is False
        assert "agree within" in note["message"]

    def test_zero_deduped_map_has_no_relative_gap(self):
        note = map_gap_note(0.0, 0.05)
        assert note["relative"] is None
        assert note["significant"] is False


# --------------------------------------------------------------------------
# Preserved behaviour: sweep + min_recall vacuous-precision floor
# --------------------------------------------------------------------------


class TestThresholdSelection:
    def test_sweep_thresholds_inclusive(self):
        assert sweep_thresholds(0.05, 0.25, 0.05) == [0.05, 0.1, 0.15, 0.2, 0.25]

    def test_returns_lowest_threshold_meeting_both_floors(self):
        px = np.array([0.1, 0.2, 0.3, 0.4])
        p = np.array([0.50, 0.96, 0.97, 1.00])
        r = np.array([0.80, 0.40, 0.20, 0.00])
        result, vacuous = find_lowest_threshold_for_precision(px, p, r, 0.95, 0.05)
        assert vacuous is None
        assert result["threshold"] == pytest.approx(0.2)
        assert result["recall"] == pytest.approx(0.4)

    def test_rejects_vacuous_precision_at_zero_recall(self):
        px = np.array([0.1, 0.2, 0.3])
        p = np.array([0.40, 0.60, 1.00])
        r = np.array([0.90, 0.50, 0.00])
        result, vacuous = find_lowest_threshold_for_precision(px, p, r, 0.95, 0.05)
        assert result is None
        assert vacuous["threshold"] == pytest.approx(0.3)
        assert vacuous["recall"] == pytest.approx(0.0)

    def test_best_supported_point_prefers_precision_above_the_recall_floor(self):
        sweep = [
            {"threshold": 0.1, "precision": 0.30, "recall": 0.90},
            {"threshold": 0.5, "precision": 0.70, "recall": 0.20},
            {"threshold": 0.9, "precision": 0.99, "recall": 0.01},
        ]
        assert best_supported_point(sweep, 0.05)["threshold"] == pytest.approx(0.5)

    def test_best_supported_point_falls_back_to_max_recall(self):
        sweep = [
            {"threshold": 0.5, "precision": 0.70, "recall": 0.02},
            {"threshold": 0.9, "precision": 0.99, "recall": 0.01},
        ]
        assert best_supported_point(sweep, 0.05)["threshold"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


class TestParseArgs:
    def test_dedup_is_on_by_default(self):
        args = parse_args(["--weights", "w.pt"])
        assert args.dedup_sources is True
        assert args.representative == "seeded"
        assert args.seed == 1337
        assert args.min_recall == pytest.approx(0.05)
        assert args.target_precision == pytest.approx(0.95)

    def test_escape_hatch_disables_dedup(self):
        assert parse_args(["--weights", "w.pt", "--no-dedup-sources"]).dedup_sources is False

    def test_rejects_an_unknown_representative_strategy(self):
        with pytest.raises(SystemExit):
            parse_args(["--weights", "w.pt", "--representative", "filesystem"])


# --------------------------------------------------------------------------
# Report assembly against a fake metrics object
# --------------------------------------------------------------------------


class _FakeBox:
    # None __init__(self, list[int] ap_class_index, dict[str, list[float]] values)
    # Inputs: list[int] ap_class_index - class indices that had test instances, in row order
    #         dict[str, list[float]] values - per-row p/r/ap50/ap values
    # Outputs: None
    # Description: Stands in for Ultralytics' Metric object with hand-chosen precision/recall
    #              curves, so report assembly is testable without running a model.
    # Side Effects: None
    def __init__(self, ap_class_index, values):
        self.ap_class_index = ap_class_index
        self.px = np.linspace(0.0, 1.0, 11)
        n = len(ap_class_index)
        self.p = np.array(values["p"])
        self.r = np.array(values["r"])
        self.ap50 = np.array(values["ap50"])
        self.ap = np.array(values["ap"])
        # Precision rises with confidence, recall falls -- the usual shape.
        self.p_curve = np.tile(np.linspace(0.1, 1.0, 11), (n, 1))
        self.r_curve = np.tile(np.linspace(0.9, 0.0, 11), (n, 1))
        self.map50 = float(np.mean(self.ap50))
        self.map = float(np.mean(self.ap))
        self.mp = float(np.mean(self.p))
        self.mr = float(np.mean(self.r))


class _FakeMetrics:
    # None __init__(self, tuple[str, ...] class_names, list[int] ap_class_index, dict values, dict nt)
    # Inputs: tuple[str, ...] class_names - class names in data.yaml order
    #         list[int] ap_class_index - class indices that had instances
    #         dict values - per-row p/r/ap50/ap values for _FakeBox
    #         dict nt - class index -> instance count in the evaluated subset
    # Outputs: None
    # Description: Stands in for Ultralytics' DetMetrics so build_report/metrics_summary can be
    #              unit tested offline.
    # Side Effects: None
    def __init__(self, class_names, ap_class_index, values, nt):
        self.names = {i: c for i, c in enumerate(class_names)}
        self.box = _FakeBox(ap_class_index, values)
        self.nt_per_class = nt


# argparse.Namespace _args()
# Inputs: none
# Outputs: argparse.Namespace - the evaluation options build_report reads
# Description: Minimal CLI namespace for report-assembly tests.
# Side Effects: None
def _args():
    return parse_args(["--weights", "best.pt", "--data", "d.yaml"])


class TestBuildReport:
    # tuple[dict, dict] _report(self)
    # Inputs: none
    # Outputs: tuple[dict, dict] - (report, support) for a two-class fake evaluation
    # Description: Builds one report from fake deduped + raw metrics, with spaghetti resting on
    #              400 source photos and layer_separation on only 12.
    # Side Effects: None
    def _report(self):
        idx = [0, 1]
        deduped = _FakeMetrics(
            CLASS_NAMES, idx, {"p": [0.6, 0.4], "r": [0.5, 0.2], "ap50": [0.42, 0.10], "ap": [0.20, 0.04]}, {0: 500, 1: 14}
        )
        raw = _FakeMetrics(
            CLASS_NAMES, idx, {"p": [0.7, 0.5], "r": [0.6, 0.3], "ap50": [0.55, 0.19], "ap": [0.28, 0.08]}, {0: 2436, 1: 693}
        )
        support = class_support({}, CLASS_NAMES)
        support["spaghetti"] = {"instances": 2436, "images": 1500, "source_photos": 400, "independent_groups": 250}
        support["layer_separation"] = {"instances": 693, "images": 300, "source_photos": 12, "independent_groups": 4}
        dedup = {"applied": True, "num_raw_images": 5328, "num_source_photos": 855}
        report = build_report(
            deduped, CLASS_NAMES, _args(), dedup, metrics_summary(raw), support, {"by_source": {}}
        )
        return report, support

    def test_headline_is_the_deduped_number(self):
        report, _ = self._report()
        assert report["headline_metric"] == "per_source_photo_deduped"
        assert report["overall"] == report["overall_deduped_per_source_photo"]
        assert report["overall_deduped_per_source_photo"]["map50"] == pytest.approx(0.26)

    def test_raw_per_image_is_reported_beside_it_not_instead_of_it(self):
        report, _ = self._report()
        assert report["overall_raw_per_image"]["map50"] == pytest.approx(0.37)
        assert report["deduped_vs_raw"]["significant"] is True
        assert report["deduped_vs_raw"]["raw_is_optimistic"] is True

    def test_per_class_carries_both_deduped_and_raw_numbers(self):
        report, _ = self._report()
        entry = report["per_class"]["spaghetti"]
        assert entry["ap50"] == pytest.approx(0.42)
        assert entry["raw_per_image"]["ap50"] == pytest.approx(0.55)

    def test_effective_sample_size_distinguishes_400_photos_from_12(self):
        report, _ = self._report()
        big = report["per_class"]["spaghetti"]["effective_sample_size"]
        small = report["per_class"]["layer_separation"]["effective_sample_size"]
        assert big["source_photos_with_class"] == 400
        assert big["low_support"] is False
        assert small["source_photos_with_class"] == 12
        assert small["low_support"] is True
        assert small["low_support_floor"] == LOW_SUPPORT_PHOTOS
        # The inflated counts are kept, but clearly separated from the honest one.
        assert small["raw_instances"] == 693

    def test_independent_group_count_is_reported_beside_the_photo_count(self):
        report, _ = self._report()
        big = report["per_class"]["spaghetti"]["effective_sample_size"]
        small = report["per_class"]["layer_separation"]["effective_sample_size"]
        assert big["independent_groups_with_class"] == 250
        assert big["low_support_groups"] is False
        assert small["independent_groups_with_class"] == 4
        assert small["low_support_groups"] is True

    def test_group_count_flags_low_n_even_when_the_photo_count_looks_fine(self):
        # 200 photographs of only 9 print jobs is 9 observations, not 200.
        idx = [0]
        m = _FakeMetrics(CLASS_NAMES, idx, {"p": [0.8], "r": [0.9], "ap50": [0.96], "ap": [0.44]}, {0: 200})
        support = class_support({}, CLASS_NAMES)
        support["spaghetti"] = {"instances": 200, "images": 200, "source_photos": 200, "independent_groups": 9}
        report = build_report(m, CLASS_NAMES, _args(), {"applied": True}, None, support, {"by_source": {}})
        ess = report["per_class"]["spaghetti"]["effective_sample_size"]
        assert ess["low_support"] is False
        assert ess["low_support_groups"] is True

    def test_group_count_is_none_when_no_manifest_was_available(self):
        idx = [0]
        m = _FakeMetrics(CLASS_NAMES, idx, {"p": [0.8], "r": [0.9], "ap50": [0.96], "ap": [0.44]}, {0: 200})
        report = build_report(
            m, CLASS_NAMES, _args(), {"applied": True}, None, class_support({}, CLASS_NAMES), {"by_source": {}}
        )
        ess = report["per_class"]["spaghetti"]["effective_sample_size"]
        assert ess["independent_groups_with_class"] is None
        assert ess["low_support_groups"] is False

    def test_severity_policy_matches_the_seven_class_model(self):
        report, _ = self._report()
        assert report["severity_policy"]["catastrophic"] == list(CATASTROPHIC_CLASSES)
        assert report["severity_policy"]["cosmetic"] == list(COSMETIC_CLASSES)
        assert report["per_class"]["spaghetti"]["is_catastrophic"] is True
        assert CATASTROPHIC_CLASSES == ("spaghetti", "layer_separation", "bed_adhesion", "blob_of_death")
        assert "warping" in COSMETIC_CLASSES

    def test_classes_without_instances_are_named_not_dropped(self):
        report, _ = self._report()
        assert "bed_adhesion" in report["classes_with_no_test_instances"]

    def test_report_is_json_serializable(self):
        report, _ = self._report()
        assert json.loads(json.dumps(report))["headline_metric"] == "per_source_photo_deduped"

    def test_yaml_block_annotates_low_n_classes(self):
        report, _ = self._report()
        block = format_yaml_block(report, CLASS_NAMES)
        assert "class_thresholds:" in block
        assert "LOW-N: only 12 source photos / 4 independent scenes" in block
        assert "NO TEST INSTANCES" in block  # bed_adhesion etc.

    def test_yaml_block_flags_low_n_from_the_group_count_alone(self):
        # 200 photographs of 9 print jobs: the photo count looks fine, the scene count does not.
        idx = [0]
        m = _FakeMetrics(CLASS_NAMES, idx, {"p": [0.8], "r": [0.9], "ap50": [0.96], "ap": [0.44]}, {0: 200})
        support = class_support({}, CLASS_NAMES)
        support["spaghetti"] = {"instances": 200, "images": 200, "source_photos": 200, "independent_groups": 9}
        report = build_report(m, CLASS_NAMES, _args(), {"applied": True}, None, support, {"by_source": {}})
        block = format_yaml_block(report, CLASS_NAMES)
        assert "LOW-N: only 200 source photos / 9 independent scenes" in block

    def test_dedup_disabled_reports_raw_as_headline_and_no_gap(self):
        idx = [0]
        m = _FakeMetrics(CLASS_NAMES, idx, {"p": [0.7], "r": [0.6], "ap50": [0.55], "ap": [0.28]}, {0: 2436})
        args = parse_args(["--weights", "w.pt", "--no-dedup-sources"])
        report = build_report(
            m, CLASS_NAMES, args, {"applied": False}, None, class_support({}, CLASS_NAMES), {"by_source": {}}
        )
        assert report["headline_metric"].startswith("raw_per_image")
        assert report["overall_raw_per_image"] is None
        assert report["deduped_vs_raw"] is None


# --------------------------------------------------------------------------
# Filesystem helpers (tmp_path only -- never the real dataset)
# --------------------------------------------------------------------------


class TestSplitIo:
    # tuple[Path, dict] _dataset(self, tmp_path)
    # Inputs: Path tmp_path - pytest temporary directory
    # Outputs: tuple[Path, dict] - (data.yaml path, parsed config) for a tiny synthetic split
    # Description: Lays out a miniature test split (2 photos x 2 variants, one background)
    #              on disk so the split-resolution helpers can be exercised offline.
    # Side Effects: Creates directories and files under tmp_path.
    def _dataset(self, tmp_path):
        import yaml

        root = tmp_path / "ds"
        (root / "test" / "images").mkdir(parents=True)
        (root / "test" / "labels").mkdir(parents=True)
        files = {
            _rf("atco__p1", "aa" * 16): "0 0.5 0.5 0.2 0.2\n",
            _rf("atco__p1", "bb" * 16): "0 0.5 0.5 0.2 0.2\n4 0.1 0.1 0.1 0.1\n",
            _rf("stereovision__p2", "cc" * 16): "",
        }
        for name, label in files.items():
            (root / "test" / "images" / name).write_bytes(b"x")
            (root / "test" / "labels" / (name.rsplit(".", 1)[0] + ".txt")).write_text(label, encoding="utf-8")
        cfg = {
            "path": str(root),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(CLASS_NAMES),
            "names": list(CLASS_NAMES),
        }
        yaml_path = root / "data.yaml"
        yaml_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return yaml_path, cfg

    def test_resolve_and_list(self, tmp_path):
        yaml_path, cfg = self._dataset(tmp_path)
        images_dir = resolve_split_images_dir(cfg, yaml_path, "test")
        assert images_dir.name == "images"
        assert len(list_image_filenames(images_dir)) == 3

    def test_missing_split_raises(self, tmp_path):
        yaml_path, cfg = self._dataset(tmp_path)
        with pytest.raises(KeyError):
            resolve_split_images_dir({k: v for k, v in cfg.items() if k != "test"}, yaml_path, "test")

    def test_read_labels_keeps_background_images(self, tmp_path):
        yaml_path, cfg = self._dataset(tmp_path)
        images_dir = resolve_split_images_dir(cfg, yaml_path, "test")
        names = list_image_filenames(images_dir)
        labels = read_split_labels(images_dir, names)
        assert len(labels) == 3
        assert labels[_rf("stereovision__p2", "cc" * 16)] == []
        support = class_support(labels, CLASS_NAMES)
        assert support["spaghetti"] == {"instances": 2, "images": 2, "source_photos": 1}
        assert support["warping"]["source_photos"] == 1

    def test_write_subset_dataset_emits_a_list_and_yaml(self, tmp_path):
        import yaml

        yaml_path, cfg = self._dataset(tmp_path)
        images_dir = resolve_split_images_dir(cfg, yaml_path, "test")
        names = list_image_filenames(images_dir)
        reps, _, _ = select_representatives(names, seed=1337)
        out = write_subset_dataset(cfg, images_dir, reps, tmp_path / "work", "dedup")
        subset = yaml.safe_load(out.read_text(encoding="utf-8"))
        listed = (tmp_path / "work" / "dedup_images.txt").read_text(encoding="utf-8").strip().splitlines()
        assert subset["names"] == list(CLASS_NAMES)
        assert subset["test"].endswith("dedup_images.txt")
        assert len(listed) == len(reps) == 2
        assert all(line.startswith(images_dir.as_posix()) for line in listed)

    def test_write_subset_dataset_refuses_an_empty_subset(self, tmp_path):
        yaml_path, cfg = self._dataset(tmp_path)
        images_dir = resolve_split_images_dir(cfg, yaml_path, "test")
        with pytest.raises(ValueError, match="empty evaluation subset"):
            write_subset_dataset(cfg, images_dir, [], tmp_path / "work", "dedup")


# --------------------------------------------------------------------------
# metrics_summary
# --------------------------------------------------------------------------


class TestMetricsSummary:
    def test_flattens_a_metrics_object(self):
        m = _FakeMetrics(
            CLASS_NAMES, [0, 4], {"p": [0.6, 0.3], "r": [0.5, 0.2], "ap50": [0.4, 0.1], "ap": [0.2, 0.05]}, {0: 10, 4: 3}
        )
        summary = metrics_summary(m)
        assert set(summary["per_class"]) == {"spaghetti", "warping"}
        assert summary["per_class"]["warping"]["num_instances"] == 3
        assert summary["map50"] == pytest.approx(0.25)
