"""Unit tests for training/prepare_dataset.py's pure dataset-splitting logic.

These tests build a small synthetic fixture directory tree under `tmp_path`
and never touch the real (300MB+, gitignored) dataset. They focus on the
one thing that must never be wrong: source-level splitting with zero
train/val/test leakage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.prepare_dataset import (
    CLASS_NAMES,
    apply_class_filter,
    assert_no_source_overlap,
    extract_source_id,
    group_by_source,
    main,
    materialize_split,
    normalize_label_text,
    parse_yolo_label_classes,
    pool_raw_images,
    resolve_class_filter,
    select_files_for_split,
    split_sources,
)


# --------------------------------------------------------------------------
# extract_source_id
# --------------------------------------------------------------------------


class TestExtractSourceId:
    def test_matches_rf_pattern(self) -> None:
        src, matched = extract_source_id(
            "00001_error_dataset_jpeg.rf.549c891f051b38ac8300be431e761f8c.jpg"
        )
        assert matched is True
        assert src == "00001_error_dataset_jpeg"

    def test_matches_case_insensitively_and_other_extensions(self) -> None:
        for ext in ("jpg", "JPG", "jpeg", "PNG", "png"):
            filename = f"photo123.rf.deadbeef.{ext}"
            src, matched = extract_source_id(filename)
            assert matched is True
            assert src == "photo123"

    def test_source_id_with_dots_and_underscores(self) -> None:
        src, matched = extract_source_id("-_jpg.rf.55a277b312ca8411e6e69da53576b90f.jpg")
        assert matched is True
        assert src == "-_jpg"

    def test_fallback_when_no_rf_marker(self) -> None:
        src, matched = extract_source_id("plain_photo_no_marker.jpg")
        assert matched is False
        assert src == "plain_photo_no_marker"

    def test_fallback_uses_stem_without_extension(self) -> None:
        src, matched = extract_source_id("weird.name.without.rf.marker.png")
        assert matched is False
        # Path.stem only strips the final extension.
        assert src == "weird.name.without.rf.marker"


# --------------------------------------------------------------------------
# group_by_source
# --------------------------------------------------------------------------


class TestGroupBySource:
    def test_groups_all_variants_of_a_source_together(self) -> None:
        filenames = [
            "img001.rf.aaaa1111.jpg",
            "img001.rf.bbbb2222.jpg",
            "img001.rf.cccc3333.jpg",
            "img002.rf.dddd4444.jpg",
        ]
        groups, fallback_count = group_by_source(filenames)
        assert fallback_count == 0
        assert set(groups.keys()) == {"img001", "img002"}
        assert sorted(groups["img001"]) == sorted(
            ["img001.rf.aaaa1111.jpg", "img001.rf.bbbb2222.jpg", "img001.rf.cccc3333.jpg"]
        )
        assert groups["img002"] == ["img002.rf.dddd4444.jpg"]

    def test_counts_fallback_filenames(self) -> None:
        filenames = ["img001.rf.aaaa1111.jpg", "no_marker_here.jpg", "also_no_marker.png"]
        groups, fallback_count = group_by_source(filenames)
        assert fallback_count == 2
        assert "no_marker_here" in groups
        assert "also_no_marker" in groups

    def test_within_group_order_is_sorted(self) -> None:
        # Suffixes must be valid hex (matching real Roboflow hashes) so they
        # actually hit the .rf.<hex> pattern instead of the fallback path.
        filenames = ["img001.rf.3333.jpg", "img001.rf.1111.jpg", "img001.rf.2222.jpg"]
        groups, _ = group_by_source(filenames)
        assert groups["img001"] == sorted(filenames)


# --------------------------------------------------------------------------
# split_sources
# --------------------------------------------------------------------------


class TestSplitSources:
    def _sources(self, n: int) -> list[str]:
        return [f"source_{i:04d}" for i in range(n)]

    def test_deterministic_for_fixed_seed(self) -> None:
        sources = self._sources(200)
        result_a = split_sources(sources, seed=1337)
        result_b = split_sources(sources, seed=1337)
        assert result_a == result_b

    def test_different_seeds_can_differ(self) -> None:
        sources = self._sources(200)
        result_a = split_sources(sources, seed=1337)
        result_b = split_sources(sources, seed=42)
        assert result_a != result_b

    def test_input_order_does_not_matter(self) -> None:
        sources = self._sources(200)
        reversed_sources = list(reversed(sources))
        result_forward = split_sources(sources, seed=7)
        result_reversed = split_sources(reversed_sources, seed=7)
        assert result_forward == result_reversed

    def test_ratios_approximately_70_15_15(self) -> None:
        sources = self._sources(1000)
        train, val, test = split_sources(sources, seed=1337)
        assert len(train) + len(val) + len(test) == 1000
        assert 690 <= len(train) <= 710
        assert 140 <= len(val) <= 160
        assert 140 <= len(test) <= 160

    def test_no_overlap_between_splits(self) -> None:
        sources = self._sources(357)  # not evenly divisible, exercises rounding
        train, val, test = split_sources(sources, seed=99)
        assert set(train) & set(val) == set()
        assert set(train) & set(test) == set()
        assert set(val) & set(test) == set()
        # every source accounted for exactly once
        assert set(train) | set(val) | set(test) == set(sources)

    def test_all_sources_preserved_for_small_n(self) -> None:
        sources = self._sources(3)
        train, val, test = split_sources(sources, seed=1337)
        assert set(train) | set(val) | set(test) == set(sources)
        assert len(train) + len(val) + len(test) == 3

    def test_rejects_ratios_not_summing_to_one(self) -> None:
        with pytest.raises(ValueError):
            split_sources(self._sources(10), seed=1, ratios=(0.5, 0.5, 0.5))


# --------------------------------------------------------------------------
# select_files_for_split
# --------------------------------------------------------------------------


class TestSelectFilesForSplit:
    @pytest.fixture()
    def groups(self) -> dict[str, list[str]]:
        filenames = [
            "img001.rf.aaaa.jpg",
            "img001.rf.bbbb.jpg",
            "img001.rf.cccc.jpg",
            "img002.rf.dddd.jpg",
            "img002.rf.eeee.jpg",
        ]
        groups, _ = group_by_source(filenames)
        return groups

    def test_train_keeps_all_variants(self, groups: dict[str, list[str]]) -> None:
        selected = select_files_for_split(groups, ["img001", "img002"], "train")
        assert len(selected["img001"]) == 3
        assert len(selected["img002"]) == 2

    def test_val_keeps_exactly_one_variant_per_source(self, groups: dict[str, list[str]]) -> None:
        selected = select_files_for_split(groups, ["img001", "img002"], "val")
        assert len(selected["img001"]) == 1
        assert len(selected["img002"]) == 1

    def test_test_keeps_exactly_one_variant_per_source(self, groups: dict[str, list[str]]) -> None:
        selected = select_files_for_split(groups, ["img001", "img002"], "test")
        assert len(selected["img001"]) == 1
        assert len(selected["img002"]) == 1

    def test_val_test_pick_is_deterministic_first_sorted(self, groups: dict[str, list[str]]) -> None:
        selected_val = select_files_for_split(groups, ["img001"], "val")
        selected_test = select_files_for_split(groups, ["img001"], "test")
        # Same source -> same deterministic pick regardless of split name.
        assert selected_val["img001"] == selected_test["img001"] == [sorted(groups["img001"])[0]]

    def test_only_requested_sources_are_included(self, groups: dict[str, list[str]]) -> None:
        selected = select_files_for_split(groups, ["img001"], "train")
        assert set(selected.keys()) == {"img001"}


# --------------------------------------------------------------------------
# assert_no_source_overlap
# --------------------------------------------------------------------------


class TestAssertNoSourceOverlap:
    def test_passes_on_disjoint_splits(self) -> None:
        assert_no_source_overlap({"train": ["a", "b"], "val": ["c"], "test": ["d"]})  # no raise

    def test_raises_on_overlap(self) -> None:
        with pytest.raises(AssertionError):
            assert_no_source_overlap({"train": ["a", "b"], "val": ["b"], "test": ["d"]})

    def test_raises_on_train_test_overlap(self) -> None:
        with pytest.raises(AssertionError):
            assert_no_source_overlap({"train": ["a"], "val": ["b"], "test": ["a"]})


# --------------------------------------------------------------------------
# parse_yolo_label_classes
# --------------------------------------------------------------------------


class TestParseYoloLabelClasses:
    def test_parses_class_ids(self) -> None:
        text = "1 0.5 0.5 0.2 0.2\n3 0.1 0.1 0.05 0.05\n"
        assert parse_yolo_label_classes(text) == [1, 3]

    def test_empty_file_means_no_objects(self) -> None:
        assert parse_yolo_label_classes("") == []
        assert parse_yolo_label_classes("   \n\n  ") == []


# --------------------------------------------------------------------------
# End-to-end-ish: synthetic raw dataset fixture on disk
# --------------------------------------------------------------------------


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture()
def synthetic_raw_dir(tmp_path: Path) -> Path:
    """Build a small raw/ tree mimicking the real archive's defects:
    - source 'imgA' has variants split across train and valid (leakage)
    - source 'imgB' lives entirely in train
    - source 'imgC' lives entirely in valid
    - 'imgB' variant #2 has no label file (missing-label case)
    - one filename doesn't match the .rf. pattern (fallback case)
    """
    raw = tmp_path / "raw"

    label_txt = "1 0.5 0.5 0.2 0.2\n"  # class 1 = spaghetti

    # imgA: 2 variants in train, 1 variant in valid -> classic leakage shape
    _write(raw / "train" / "images" / "imgA.rf.aaaa1111.jpg", "fakejpg")
    _write(raw / "train" / "labels" / "imgA.rf.aaaa1111.txt", label_txt)
    _write(raw / "train" / "images" / "imgA.rf.aaaa2222.jpg", "fakejpg")
    _write(raw / "train" / "labels" / "imgA.rf.aaaa2222.txt", label_txt)
    _write(raw / "valid" / "images" / "imgA.rf.aaaa3333.jpg", "fakejpg")
    _write(raw / "valid" / "labels" / "imgA.rf.aaaa3333.txt", label_txt)

    # imgB: 2 variants, both in train; second has NO label file
    _write(raw / "train" / "images" / "imgB.rf.bbbb1111.jpg", "fakejpg")
    _write(raw / "train" / "labels" / "imgB.rf.bbbb1111.txt", "0 0.5 0.5 0.1 0.1\n")
    _write(raw / "train" / "images" / "imgB.rf.bbbb2222.jpg", "fakejpg")
    # no label file written for imgB.rf.bbbb2222 on purpose

    # imgC: 1 variant, in valid only
    _write(raw / "valid" / "images" / "imgC.rf.cccc1111.jpg", "fakejpg")
    _write(raw / "valid" / "labels" / "imgC.rf.cccc1111.txt", "3 0.5 0.5 0.1 0.1\n")  # warping

    # fallback filename with no .rf. marker
    _write(raw / "train" / "images" / "no_marker_photo.jpg", "fakejpg")
    _write(raw / "train" / "labels" / "no_marker_photo.txt", "4 0.5 0.5 0.1 0.1\n")  # zits

    return raw


class TestSyntheticEndToEnd:
    def test_pool_raw_images_merges_train_and_valid(self, synthetic_raw_dir: Path) -> None:
        pooled, duplicates = pool_raw_images(synthetic_raw_dir)
        assert duplicates == 0
        assert len(pooled) == 7  # 3 imgA + 2 imgB + 1 imgC + 1 fallback

    def test_full_pipeline_no_leakage_and_variant_counts(self, synthetic_raw_dir: Path, tmp_path: Path) -> None:
        pooled, _ = pool_raw_images(synthetic_raw_dir)
        groups, fallback_count = group_by_source(pooled.keys())

        assert fallback_count == 1
        assert set(groups.keys()) == {"imgA", "imgB", "imgC", "no_marker_photo"}
        assert len(groups["imgA"]) == 3  # all 3 variants grouped despite train/valid split
        assert len(groups["imgB"]) == 2
        assert len(groups["imgC"]) == 1

        # Force every source into a distinct split to directly exercise the
        # leakage-prevention guarantee end to end. imgB goes to TRAIN so all
        # its variants (including the one with a missing label file) get
        # pulled in -- val/test only ever pick one deterministic variant, so
        # that's the only way this fixture's missing-label case gets hit.
        split_source_ids = {
            "train": ["imgB"],
            "val": ["imgA"],
            "test": ["imgC", "no_marker_photo"],
        }

        # No exception -> no overlap.
        assert_no_source_overlap(split_source_ids)

        out_dir = tmp_path / "argus"
        stats = {}
        for split, ids in split_source_ids.items():
            selected = select_files_for_split(groups, ids, split)
            stats[split] = materialize_split(split, selected, pooled, out_dir)

        # train keeps ALL of imgB's variants (one of which has no label file).
        assert stats["train"].n_images == 2
        assert (out_dir / "train" / "images").exists()
        assert len(list((out_dir / "train" / "images").iterdir())) == 2
        assert stats["train"].n_missing_labels == 1

        # val keeps exactly ONE of imgA's variants (deterministic: first sorted).
        assert stats["val"].n_images == 1
        expected_pick = sorted(groups["imgA"])[0]
        picked = list((out_dir / "val" / "images").iterdir())[0].name
        assert picked == expected_pick
        assert stats["val"].n_missing_labels == 0

        # test keeps one variant of imgC + the fallback file (which has no
        # variants to begin with).
        assert stats["test"].n_images == 2
        assert stats["test"].n_missing_labels == 0

        # No source-identity overlap when reconstructed from what's on disk.
        written_source_ids = {}
        for split in ("train", "val", "test"):
            images_dir = out_dir / split / "images"
            filenames = [p.name for p in images_dir.iterdir()]
            written_groups, _ = group_by_source(filenames)
            written_source_ids[split] = list(written_groups.keys())
        assert_no_source_overlap(written_source_ids)  # no raise

        # Missing-label file was counted, not silently dropped.
        total_missing = sum(s.n_missing_labels for s in stats.values())
        assert total_missing == 1

        # Sanity: class instance tally lines up with what we wrote (imgB's
        # labeled variant -> "error extrusion" (class 0), imgA -> spaghetti
        # (class 1), imgC -> warping (class 3), fallback -> zits (class 4)).
        assert stats["train"].instances_per_class["error extrusion"] == 1
        assert stats["val"].instances_per_class["spaghetti"] == 1
        assert stats["test"].instances_per_class["warping"] == 1
        assert stats["test"].instances_per_class["zits"] == 1

    def test_missing_label_does_not_crash_and_is_counted(self, synthetic_raw_dir: Path, tmp_path: Path) -> None:
        pooled, _ = pool_raw_images(synthetic_raw_dir)
        groups, _ = group_by_source(pooled.keys())
        selected = select_files_for_split(groups, ["imgB"], "train")
        out_dir = tmp_path / "argus2"
        stats = materialize_split("train", selected, pooled, out_dir)
        assert stats.n_images == 2
        assert stats.n_missing_labels == 1
        # The image with no label still got copied; no .txt was fabricated.
        images = sorted(p.name for p in (out_dir / "train" / "images").iterdir())
        labels = sorted(p.stem for p in (out_dir / "train" / "labels").iterdir())
        assert len(images) == 2
        assert len(labels) == 1  # only the one that had a real label file

    def test_empty_label_file_is_kept_not_treated_as_missing(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw2"
        _write(raw / "train" / "images" / "empty_img.rf.eeee0000.jpg", "fakejpg")
        _write(raw / "train" / "labels" / "empty_img.rf.eeee0000.txt", "")  # legitimately "no objects"

        pooled, _ = pool_raw_images(raw)
        groups, _ = group_by_source(pooled.keys())
        selected = select_files_for_split(groups, ["empty_img"], "train")
        out_dir = tmp_path / "argus3"
        stats = materialize_split("train", selected, pooled, out_dir)

        assert stats.n_missing_labels == 0
        assert (out_dir / "train" / "labels" / "empty_img.rf.eeee0000.txt").is_file()
        for count in stats.instances_per_class.values():
            assert count == 0


# --------------------------------------------------------------------------
# normalize_label_text -- polygon/segment row -> detection row conversion
# (recovers images that Ultralytics would otherwise drop entirely for
# mixing detection and segment rows in one label file)
# --------------------------------------------------------------------------


class TestNormalizeLabelText:
    def test_polygon_row_converts_to_correct_bbox(self) -> None:
        # Rectangle drawn as a 4-point polygon: (0.2,0.2) (0.6,0.2) (0.6,0.7) (0.2,0.7).
        text = "1 0.2 0.2 0.6 0.2 0.6 0.7 0.2 0.7\n"
        normalized, stats = normalize_label_text(text)
        lines = normalized.splitlines()
        assert len(lines) == 1
        cls_id, cx, cy, w, h = lines[0].split()
        assert cls_id == "1"
        # hand-computed: cx=(0.2+0.6)/2=0.4, cy=(0.2+0.7)/2=0.45, w=0.6-0.2=0.4, h=0.7-0.2=0.5
        assert float(cx) == pytest.approx(0.4)
        assert float(cy) == pytest.approx(0.45)
        assert float(w) == pytest.approx(0.4)
        assert float(h) == pytest.approx(0.5)
        assert stats.polygon_rows_converted == 1
        assert stats.malformed_rows_skipped == 0
        assert stats.contains_polygon_row is True

    def test_mixed_detection_and_polygon_rows_both_survive(self) -> None:
        text = (
            "0 0.5 0.5 0.2 0.2\n"                    # plain detection row
            "1 0.2 0.2 0.6 0.2 0.6 0.7 0.2 0.7\n"     # polygon row (same rectangle as above)
        )
        normalized, stats = normalize_label_text(text)
        lines = normalized.splitlines()
        assert len(lines) == 2
        for line in lines:
            assert len(line.split()) == 5  # both rows are now valid 5-field detection rows
        assert lines[0].split()[0] == "0"
        assert lines[1].split()[0] == "1"
        assert stats.polygon_rows_converted == 1
        assert stats.malformed_rows_skipped == 0

    def test_polygon_points_are_clamped_to_unit_range(self) -> None:
        # 3-point polygon with one coordinate outside [0, 1] on each axis.
        text = "2 -0.2 0.5 0.3 1.3 0.6 0.9\n"
        normalized, stats = normalize_label_text(text)
        cls_id, cx, cy, w, h = normalized.splitlines()[0].split()
        # x's clamp to [0.0, 0.3, 0.6] -> min 0.0, max 0.6; y's clamp to [0.5, 1.0, 0.9] -> min 0.5, max 1.0
        assert float(cx) == pytest.approx(0.3)
        assert float(cy) == pytest.approx(0.75)
        assert float(w) == pytest.approx(0.6)
        assert float(h) == pytest.approx(0.5)
        assert stats.polygon_rows_converted == 1

    def test_degenerate_box_after_clamping_is_dropped(self) -> None:
        # All 3 x-coordinates are negative -> clamp to 0.0 -> zero width -> dropped.
        text = "3 -0.1 0.2 -0.05 0.5 -0.2 0.9\n"
        normalized, stats = normalize_label_text(text)
        assert normalized == ""
        assert stats.polygon_rows_converted == 0
        assert stats.malformed_rows_skipped == 0  # structurally valid row, just degenerate
        assert stats.contains_polygon_row is True  # it WAS a polygon-shaped row

    def test_degenerate_detection_row_is_dropped_too(self) -> None:
        text = "0 0.5 0.5 0.0 0.2\n"  # zero width
        normalized, _ = normalize_label_text(text)
        assert normalized == ""

    def test_even_field_count_row_is_malformed_skipped_not_crashed(self) -> None:
        # 8 fields (1 class + 7 coords) -- an incomplete trailing coordinate pair.
        text = "1 0.1 0.2 0.3 0.4 0.5 0.6 0.7\n"
        normalized, stats = normalize_label_text(text)
        assert normalized == ""
        assert stats.malformed_rows_skipped == 1
        assert stats.polygon_rows_converted == 0
        assert stats.contains_polygon_row is True  # still shaped like a mixed/segment row

    def test_too_few_fields_row_is_malformed_skipped(self) -> None:
        text = "0 0.5 0.5\n"  # only 3 fields, incomplete detection row
        normalized, stats = normalize_label_text(text)
        assert normalized == ""
        assert stats.malformed_rows_skipped == 1

    def test_non_numeric_field_is_malformed_skipped_not_crashed(self) -> None:
        text = "1 abc 0.2 0.3 0.4\n"
        normalized, stats = normalize_label_text(text)
        assert normalized == ""
        assert stats.malformed_rows_skipped == 1

    def test_empty_text_stays_empty(self) -> None:
        normalized, stats = normalize_label_text("")
        assert normalized == ""
        assert stats.polygon_rows_converted == 0
        assert stats.malformed_rows_skipped == 0


# --------------------------------------------------------------------------
# resolve_class_filter / apply_class_filter -- --classes / --single-class
# --------------------------------------------------------------------------


class TestResolveClassFilter:
    def test_keeps_only_requested_classes_and_remaps_contiguously(self) -> None:
        cf = resolve_class_filter(["warping", "spaghetti"], single_class=False)
        # Ordered by ORIGINAL class id (spaghetti=1, warping=3), not CLI order.
        assert cf.keep_ids == frozenset({1, 3})
        assert cf.id_remap == {1: 0, 3: 1}
        assert cf.names == ["spaghetti", "warping"]
        assert cf.single_class is False

    def test_output_order_independent_of_cli_argument_order(self) -> None:
        cf_a = resolve_class_filter(["warping", "spaghetti"], single_class=False)
        cf_b = resolve_class_filter(["spaghetti", "warping"], single_class=False)
        assert cf_a == cf_b

    def test_case_insensitive_matching(self) -> None:
        cf = resolve_class_filter(["SPAGHETTI"], single_class=False)
        assert cf.names == ["spaghetti"]

    def test_unknown_class_name_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_class_filter(["not_a_real_class"], single_class=False)

    def test_none_keeps_every_class_with_identity_remap(self) -> None:
        cf = resolve_class_filter(None, single_class=False)
        assert cf.names == list(CLASS_NAMES)
        assert cf.id_remap == {i: i for i in range(len(CLASS_NAMES))}

    def test_single_class_maps_kept_classes_to_zero(self) -> None:
        cf = resolve_class_filter(["spaghetti", "warping"], single_class=True)
        assert cf.id_remap == {1: 0, 3: 0}
        assert cf.names == ["spaghetti+warping"]
        assert cf.single_class is True

    def test_single_class_with_no_classes_filter_collapses_all_five(self) -> None:
        cf = resolve_class_filter(None, single_class=True)
        assert set(cf.id_remap.values()) == {0}
        assert len(cf.id_remap) == len(CLASS_NAMES)
        assert len(cf.names) == 1


class TestApplyClassFilter:
    def test_drops_unkept_classes_and_remaps_survivors(self) -> None:
        cf = resolve_class_filter(["spaghetti", "warping"], single_class=False)
        text = (
            "0 0.1 0.1 0.1 0.1\n"  # error extrusion -- dropped
            "1 0.2 0.2 0.1 0.1\n"  # spaghetti -- kept, remapped to 0
            "3 0.3 0.3 0.1 0.1\n"  # warping -- kept, remapped to 1
            "4 0.4 0.4 0.1 0.1\n"  # zits -- dropped
        )
        filtered = apply_class_filter(text, cf)
        lines = filtered.splitlines()
        assert len(lines) == 2
        assert lines[0].split()[0] == "0"  # was spaghetti (class 1)
        assert lines[1].split()[0] == "1"  # was warping (class 3)

    def test_single_class_collapses_kept_rows_to_id_zero(self) -> None:
        cf = resolve_class_filter(["spaghetti", "warping"], single_class=True)
        text = "1 0.2 0.2 0.1 0.1\n3 0.3 0.3 0.1 0.1\n"
        filtered = apply_class_filter(text, cf)
        assert filtered.splitlines()
        assert all(line.split()[0] == "0" for line in filtered.splitlines())

    def test_all_rows_filtered_out_yields_empty_text(self) -> None:
        cf = resolve_class_filter(["spaghetti"], single_class=False)
        text = "3 0.3 0.3 0.1 0.1\n"  # warping only -- not kept
        assert apply_class_filter(text, cf) == ""


class TestMaterializeSplitWithClassFilter:
    def test_images_with_all_labels_filtered_out_are_kept_as_backgrounds(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        # source has only a "warping" (class 3) label -- filtered away entirely.
        _write(raw / "train" / "images" / "onlyWarping.rf.aaaa0000.jpg", "fakejpg")
        _write(raw / "train" / "labels" / "onlyWarping.rf.aaaa0000.txt", "3 0.5 0.5 0.2 0.2\n")
        # source has a "spaghetti" (class 1) label -- survives the filter.
        _write(raw / "train" / "images" / "hasSpaghetti.rf.bbbb0000.jpg", "fakejpg")
        _write(raw / "train" / "labels" / "hasSpaghetti.rf.bbbb0000.txt", "1 0.5 0.5 0.2 0.2\n")

        pooled, _ = pool_raw_images(raw)
        groups, _ = group_by_source(pooled.keys())
        selected = select_files_for_split(groups, ["onlyWarping", "hasSpaghetti"], "train")

        cf = resolve_class_filter(["spaghetti"], single_class=False)
        out_dir = tmp_path / "argus_filtered"
        stats = materialize_split("train", selected, pooled, out_dir, cf)

        assert stats.n_images == 2
        assert stats.n_missing_labels == 0  # both images had real label files on disk

        # Both images are still copied -- background images are kept, not dropped.
        images = sorted(p.name for p in (out_dir / "train" / "images").iterdir())
        assert len(images) == 2

        # The warping-only image's label file exists but is now empty (background).
        warping_label = out_dir / "train" / "labels" / "onlyWarping.rf.aaaa0000.txt"
        assert warping_label.is_file()
        assert warping_label.read_text(encoding="utf-8").strip() == ""

        # The spaghetti image's label file survives with the remapped class id 0.
        spaghetti_label = out_dir / "train" / "labels" / "hasSpaghetti.rf.bbbb0000.txt"
        assert spaghetti_label.read_text(encoding="utf-8").split()[0] == "0"

        assert stats.instances_per_class == {"spaghetti": 1}
        assert stats.images_per_class == {"spaghetti": 1}


class TestClassFilterDoesNotAffectSplitAssignment:
    def test_split_assignment_identical_with_and_without_class_filter(
        self, synthetic_raw_dir: Path, tmp_path: Path
    ) -> None:
        out_no_filter = tmp_path / "argus_no_filter"
        out_with_filter = tmp_path / "argus_with_filter"

        main(["--raw", str(synthetic_raw_dir), "--out", str(out_no_filter), "--seed", "1337"])
        main(
            [
                "--raw", str(synthetic_raw_dir),
                "--out", str(out_with_filter),
                "--seed", "1337",
                "--classes", "spaghetti",
            ]
        )

        for split in ("train", "val", "test"):
            images_a = sorted(p.name for p in (out_no_filter / split / "images").iterdir())
            images_b = sorted(p.name for p in (out_with_filter / split / "images").iterdir())
            assert images_a == images_b


def test_class_names_order_matches_spec() -> None:
    assert CLASS_NAMES == ("error extrusion", "spaghetti", "stringing", "warping", "zits")
