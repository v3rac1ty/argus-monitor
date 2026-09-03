"""Unit tests for training/build_classification_dataset.py's pure dataset
selection/splitting/normalization logic.

These tests build small synthetic fixtures under `tmp_path` (in-memory numpy
images, hand-written YOLO label files, tiny candidate pools) and never touch
the real ~6GB dataset tree or the network -- the Hugging Face fetch/download
functions (`fetch_hf_parquet_urls`, `download_file`,
`download_hf_normal_parquets`) are never called; instead we build
`Candidate` objects directly (as `main()` would after fetching) so the
selection/split/materialize pipeline is exercised without any I/O beyond
`tmp_path`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.build_classification_dataset import (
    CLASS_NAMES,
    MIN_SESSIONS_FOR_EVAL,
    Candidate,
    argus_group_key,
    argus_spaghetti_candidates,
    assert_no_group_overlap,
    assign_splits_for_class,
    build_dataset,
    deterministic_sample,
    fdm_candidates,
    fdm_group_key,
    fdm_session_groups,
    has_class_id,
    parse_fdm_timestamp,
    resize_and_center_crop,
    sample_candidates,
    save_normalized_jpeg,
    scan_argus_spaghetti_images,
    select_fdm_defect_class,
    select_normal,
    select_spaghetti,
    split_groups,
)


# --------------------------------------------------------------------------
# fdm_group_key / argus_group_key -- stem grouping (leakage hazard #1 and #2)
# --------------------------------------------------------------------------


class TestFdmGroupKey:
    def test_strips_aug_suffix(self) -> None:
        assert fdm_group_key("Image_20231128195537006_aug") == "Image_20231128195537006"

    def test_strips_original_suffix(self) -> None:
        assert fdm_group_key("Image_20231128195537006_original") == "Image_20231128195537006"

    def test_plain_stem_unchanged(self) -> None:
        assert fdm_group_key("Image_20231128195537006") == "Image_20231128195537006"

    def test_groups_all_three_variants_together(self) -> None:
        stems = [
            "Image_X_aug",
            "Image_X_original",
            "Image_X",
        ]
        keys = {fdm_group_key(s) for s in stems}
        assert keys == {"Image_X"}

    def test_does_not_strip_aug_or_original_mid_stem(self) -> None:
        # Only a TRAILING "_aug"/"_original" is a variant marker.
        assert fdm_group_key("Image_augmented_2023") == "Image_augmented_2023"
        assert fdm_group_key("Image_original_2023") == "Image_original_2023"


class TestArgusGroupKey:
    def test_strips_rf_hash_suffix(self) -> None:
        assert (
            argus_group_key("00001_error_dataset_jpeg.rf.549c891f051b38ac8300be431e761f8c")
            == "00001_error_dataset_jpeg"
        )

    def test_groups_multiple_rf_variants_together(self) -> None:
        stems = [
            "00001_x.rf.aaaa1111",
            "00001_x.rf.bbbb2222",
            "00001_x.rf.cccc3333",
        ]
        keys = {argus_group_key(s) for s in stems}
        assert keys == {"00001_x"}

    def test_no_rf_marker_returns_stem_unchanged(self) -> None:
        assert argus_group_key("plain_photo_no_marker") == "plain_photo_no_marker"


# --------------------------------------------------------------------------
# parse_fdm_timestamp / fdm_session_groups -- leakage hazard #3: FDM frames
# are sequential time-lapse captures of a handful of real print jobs, not
# independent photos, so splitting must group by print SESSION.
# --------------------------------------------------------------------------


def _fdm_stem(dt: datetime, ms: int = 0, suffix: str = "") -> str:
    return f"Image_{dt.strftime('%Y%m%d%H%M%S')}{ms:03d}{suffix}"


class TestParseFdmTimestamp:
    def test_parses_plain_stem(self) -> None:
        dt = datetime(2023, 11, 28, 19, 53, 36)
        assert parse_fdm_timestamp(_fdm_stem(dt, ms=980)) == dt

    def test_aug_variant_parses_to_same_timestamp_as_source_frame(self) -> None:
        dt = datetime(2023, 11, 28, 19, 55, 37)
        plain = _fdm_stem(dt, ms=6)
        aug = _fdm_stem(dt, ms=6, suffix="_aug")
        assert parse_fdm_timestamp(aug) == parse_fdm_timestamp(plain) == dt

    def test_original_variant_parses_to_same_timestamp_as_source_frame(self) -> None:
        dt = datetime(2023, 11, 28, 19, 55, 37)
        plain = _fdm_stem(dt, ms=6)
        original = _fdm_stem(dt, ms=6, suffix="_original")
        assert parse_fdm_timestamp(original) == parse_fdm_timestamp(plain) == dt

    def test_milliseconds_are_ignored_for_ordering(self) -> None:
        dt = datetime(2023, 11, 28, 19, 53, 36)
        assert parse_fdm_timestamp(_fdm_stem(dt, ms=1)) == parse_fdm_timestamp(_fdm_stem(dt, ms=999))

    def test_non_image_prefix_returns_none(self) -> None:
        assert parse_fdm_timestamp("Photo_20231128195336980") is None

    def test_too_few_digits_returns_none(self) -> None:
        assert parse_fdm_timestamp("Image_2023112819") is None

    def test_invalid_calendar_date_returns_none(self) -> None:
        # Month 13 -- 17 digits, right shape, not a real date.
        assert parse_fdm_timestamp("Image_20231328195336980") is None

    def test_extra_digit_before_suffix_does_not_partial_match(self) -> None:
        # 18 digits, not 17 -- must not silently truncate to a "valid" 17.
        assert parse_fdm_timestamp("Image_202311281953369801_aug") is None


class TestFdmSessionGroups:
    def test_frames_30s_apart_form_one_session(self) -> None:
        t0 = datetime(2023, 11, 28, 19, 53, 0)
        stems = [_fdm_stem(t0), _fdm_stem(t0.replace(second=30)), _fdm_stem(t0.replace(minute=54))]
        groups = fdm_session_groups(stems, gap_s=600)
        assert len(set(groups.values())) == 1

    def test_gap_over_600s_starts_a_new_session(self) -> None:
        t0 = datetime(2023, 11, 28, 19, 53, 0)
        t1 = datetime(2023, 11, 28, 20, 5, 1)  # 721s later -- over the 600s default
        stems = [_fdm_stem(t0), _fdm_stem(t1)]
        groups = fdm_session_groups(stems, gap_s=600)
        assert len(set(groups.values())) == 2
        assert groups[stems[0]] != groups[stems[1]]

    def test_gap_exactly_at_threshold_stays_in_session(self) -> None:
        t0 = datetime(2023, 11, 28, 19, 53, 0)
        t1 = t0 + timedelta(seconds=600)  # exactly the threshold -- not OVER it
        stems = [_fdm_stem(t0), _fdm_stem(t1)]
        groups = fdm_session_groups(stems, gap_s=600)
        assert len(set(groups.values())) == 1

    def test_gap_threshold_is_configurable(self) -> None:
        t0 = datetime(2023, 11, 28, 19, 53, 0)
        t1 = t0 + timedelta(seconds=90)
        stems = [_fdm_stem(t0), _fdm_stem(t1)]
        # 90s gap: one session at the default 600s threshold ...
        assert len(set(fdm_session_groups(stems, gap_s=600).values())) == 1
        # ... but two sessions at a tighter 60s threshold.
        assert len(set(fdm_session_groups(stems, gap_s=60).values())) == 2

    def test_aug_and_original_variants_share_their_source_frames_session(self) -> None:
        t0 = datetime(2023, 11, 28, 19, 53, 0)
        t1 = t0 + timedelta(seconds=30)
        stems = [_fdm_stem(t0), _fdm_stem(t0, suffix="_aug"), _fdm_stem(t0, suffix="_original"), _fdm_stem(t1)]
        groups = fdm_session_groups(stems, gap_s=600)
        assert len(set(groups.values())) == 1  # all 4 in the one session

    def test_multiple_sessions_are_ordered_chronologically(self) -> None:
        t0 = datetime(2023, 11, 28, 19, 53, 0)
        session_a = [t0, t0 + timedelta(seconds=30)]
        session_b = [t0 + timedelta(seconds=900), t0 + timedelta(seconds=930)]
        session_c = [t0 + timedelta(seconds=1800)]
        stems = [_fdm_stem(t) for t in session_a + session_b + session_c]
        groups = fdm_session_groups(stems, gap_s=600)
        assert len(set(groups.values())) == 3
        assert groups[stems[0]] == groups[stems[1]]
        assert groups[stems[2]] == groups[stems[3]]
        assert len({groups[stems[0]], groups[stems[2]], groups[stems[4]]}) == 3

    def test_unparseable_stems_become_singleton_groups(self) -> None:
        t0 = datetime(2023, 11, 28, 19, 53, 0)
        good = _fdm_stem(t0)
        bad1 = "Corrupted_filename_1"
        bad2 = "Corrupted_filename_2"
        groups = fdm_session_groups([good, bad1, bad2], gap_s=600)
        assert groups[good] not in (groups[bad1], groups[bad2])
        assert groups[bad1] != groups[bad2]  # two different unparseable stems -> two different groups

    def test_unparseable_variants_of_the_same_bad_frame_still_group_together(self) -> None:
        bad = "Corrupted_filename"
        stems = [bad, f"{bad}_aug", f"{bad}_original"]
        groups = fdm_session_groups(stems, gap_s=600)
        assert len(set(groups.values())) == 1

    def test_deterministic_for_repeated_calls(self) -> None:
        t0 = datetime(2023, 11, 28, 19, 53, 0)
        stems = [_fdm_stem(t0.replace(minute=(53 + i) % 60)) for i in range(30)]
        a = fdm_session_groups(stems, gap_s=600)
        b = fdm_session_groups(stems, gap_s=600)
        assert a == b


# --------------------------------------------------------------------------
# has_class_id / scan_argus_spaghetti_images -- class-1 (spaghetti) selection
# --------------------------------------------------------------------------


class TestHasClassId:
    def test_single_matching_row(self) -> None:
        assert has_class_id("1 0.5 0.5 0.2 0.2\n", 1) is True

    def test_multi_class_label_file_with_match(self) -> None:
        text = "0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.1 0.1\n3 0.3 0.3 0.1 0.1\n"
        assert has_class_id(text, 1) is True

    def test_multi_class_label_file_without_match(self) -> None:
        text = "0 0.1 0.1 0.1 0.1\n3 0.3 0.3 0.1 0.1\n4 0.4 0.4 0.1 0.1\n"
        assert has_class_id(text, 1) is False

    def test_empty_label_file_is_false(self) -> None:
        assert has_class_id("", 1) is False
        assert has_class_id("   \n\n  ", 1) is False

    def test_malformed_leading_token_is_not_a_match_not_a_crash(self) -> None:
        assert has_class_id("not_a_number 0.5 0.5 0.2 0.2\n", 1) is False


def _write_label(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_fake_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-jpg-bytes")


class TestScanArgusSpaghettiImages:
    def test_picks_exactly_images_with_a_spaghetti_row(self, tmp_path: Path) -> None:
        root = tmp_path / "argus_v2"

        # has spaghetti (class 1)
        _write_fake_image(root / "train" / "images" / "hasSpaghetti.jpg")
        _write_label(root / "train" / "labels" / "hasSpaghetti.txt", "1 0.5 0.5 0.2 0.2\n")

        # multi-class row including spaghetti
        _write_fake_image(root / "train" / "images" / "multiClass.jpg")
        _write_label(root / "train" / "labels" / "multiClass.txt", "0 0.1 0.1 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")

        # no spaghetti row
        _write_fake_image(root / "val" / "images" / "noSpaghetti.jpg")
        _write_label(root / "val" / "labels" / "noSpaghetti.txt", "3 0.3 0.3 0.1 0.1\n")

        # empty label file -- legitimately "no objects", must be skipped
        _write_fake_image(root / "val" / "images" / "emptyLabel.jpg")
        _write_label(root / "val" / "labels" / "emptyLabel.txt", "")

        # image with no label file at all -- must be skipped, not crash
        _write_fake_image(root / "test" / "images" / "noLabelFile.jpg")

        result = scan_argus_spaghetti_images(root)
        names = sorted(p.name for p in result)
        assert names == ["hasSpaghetti.jpg", "multiClass.jpg"]

    def test_pools_across_splits(self, tmp_path: Path) -> None:
        root = tmp_path / "argus_v2"
        for split in ("train", "val", "test"):
            _write_fake_image(root / split / "images" / f"{split}_img.jpg")
            _write_label(root / split / "labels" / f"{split}_img.txt", "1 0.5 0.5 0.1 0.1\n")

        result = scan_argus_spaghetti_images(root)
        assert sorted(p.name for p in result) == ["test_img.jpg", "train_img.jpg", "val_img.jpg"]

    def test_missing_split_dir_is_skipped_not_an_error(self, tmp_path: Path) -> None:
        root = tmp_path / "argus_v2"
        _write_fake_image(root / "train" / "images" / "only.jpg")
        _write_label(root / "train" / "labels" / "only.txt", "1 0.5 0.5 0.1 0.1\n")
        # val/ and test/ don't exist at all.
        result = scan_argus_spaghetti_images(root)
        assert [p.name for p in result] == ["only.jpg"]


class TestArgusSpaghettiCandidates:
    def test_groups_rf_variants_and_tags_source(self, tmp_path: Path) -> None:
        root = tmp_path / "argus_v2"
        for variant in ("aaaa1111", "bbbb2222"):
            fn = f"00001_x.rf.{variant}"
            _write_fake_image(root / "train" / "images" / f"{fn}.jpg")
            _write_label(root / "train" / "labels" / f"{fn}.txt", "1 0.5 0.5 0.1 0.1\n")

        candidates = argus_spaghetti_candidates(root)
        assert len(candidates) == 2
        assert {c.group_key for c in candidates} == {"argus_v2:00001_x"}
        assert all(c.source == "argus_v2" for c in candidates)
        assert all(c.class_name == "spaghetti" for c in candidates)


def _write_fdm_image(root: Path, class_dir: str, dt: datetime, ms: int = 0, suffix: str = "") -> Path:
    p = root / class_dir / f"{_fdm_stem(dt, ms=ms, suffix=suffix)}.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake-jpg-bytes")  # never decoded -- fdm_candidates() only builds Candidates
    return p


class TestFdmCandidatesSessionGrouping:
    def test_group_key_is_session_frame_key_is_old_style(self, tmp_path: Path) -> None:
        root = tmp_path / "fdm_raw"
        t0 = datetime(2023, 11, 28, 19, 53, 0)
        t0_plus30 = t0 + timedelta(seconds=30)
        t0_plus60 = t0 + timedelta(seconds=60)
        t1 = t0 + timedelta(seconds=700)  # >600s later -- a second session

        _write_fdm_image(root, "Cracking", t0)
        _write_fdm_image(root, "Cracking", t0, suffix="_aug")
        _write_fdm_image(root, "Cracking", t0, suffix="_original")
        _write_fdm_image(root, "Cracking", t0_plus30)
        _write_fdm_image(root, "Cracking", t0_plus60)
        _write_fdm_image(root, "Cracking", t1)

        candidates = fdm_candidates(root, "Cracking", "cracking")
        assert len(candidates) == 6
        by_stem = {c.output_stem.removeprefix("fdm_cracking_"): c for c in candidates}

        t0_stem = _fdm_stem(t0)
        aug_stem = _fdm_stem(t0, suffix="_aug")
        original_stem = _fdm_stem(t0, suffix="_original")
        plus30_stem = _fdm_stem(t0_plus30)
        plus60_stem = _fdm_stem(t0_plus60)
        t1_stem = _fdm_stem(t1)

        # frame_key (hazard #1's old grouping) unifies only the _aug/_original
        # variants of the SAME source frame ...
        assert by_stem[t0_stem].frame_key == by_stem[aug_stem].frame_key == by_stem[original_stem].frame_key
        # ... and does NOT unify genuinely different frames.
        assert by_stem[t0_stem].frame_key != by_stem[plus30_stem].frame_key

        # group_key (the new session unit) unifies EVERY frame -- and their
        # variants -- of the same print session ...
        session0 = by_stem[t0_stem].group_key
        assert by_stem[aug_stem].group_key == session0
        assert by_stem[original_stem].group_key == session0
        assert by_stem[plus30_stem].group_key == session0
        assert by_stem[plus60_stem].group_key == session0
        # ... but starts a new session after the >600s gap.
        assert by_stem[t1_stem].group_key != session0

        assert all(c.source == "fdm" for c in candidates)
        assert all(c.class_name == "cracking" for c in candidates)

    def test_session_gap_s_is_threaded_through(self, tmp_path: Path) -> None:
        root = tmp_path / "fdm_raw"
        t0 = datetime(2023, 11, 28, 19, 53, 0)
        t1 = t0 + timedelta(seconds=90)
        _write_fdm_image(root, "Warping", t0)
        _write_fdm_image(root, "Warping", t1)

        # 90s gap: one session at the default 600s threshold ...
        loose = fdm_candidates(root, "Warping", "warping", session_gap_s=600)
        assert len({c.group_key for c in loose}) == 1
        # ... but two sessions at a tighter 60s threshold.
        tight = fdm_candidates(root, "Warping", "warping", session_gap_s=60)
        assert len({c.group_key for c in tight}) == 2


# --------------------------------------------------------------------------
# deterministic_sample / sample_candidates -- capping + determinism
# --------------------------------------------------------------------------


class TestDeterministicSample:
    def test_deterministic_for_fixed_seed(self) -> None:
        keys = [f"k{i:03d}" for i in range(100)]
        a = deterministic_sample(keys, k=20, seed=1337)
        b = deterministic_sample(keys, k=20, seed=1337)
        assert a == b
        assert len(a) == 20

    def test_different_seeds_can_differ(self) -> None:
        keys = [f"k{i:03d}" for i in range(100)]
        a = deterministic_sample(keys, k=20, seed=1337)
        b = deterministic_sample(keys, k=20, seed=42)
        assert a != b

    def test_input_order_does_not_matter(self) -> None:
        keys = [f"k{i:03d}" for i in range(50)]
        a = deterministic_sample(keys, k=10, seed=7)
        b = deterministic_sample(list(reversed(keys)), k=10, seed=7)
        assert a == b

    def test_k_greater_than_population_returns_everything(self) -> None:
        keys = ["a", "b", "c"]
        result = deterministic_sample(keys, k=100, seed=1)
        assert sorted(result) == sorted(keys)
        assert len(result) == 3

    def test_no_duplicates_in_sample(self) -> None:
        keys = [f"k{i:03d}" for i in range(200)]
        result = deterministic_sample(keys, k=50, seed=9)
        assert len(result) == len(set(result)) == 50


def _candidate(class_name: str, output_stem: str, group_key: str | None = None) -> Candidate:
    return Candidate(
        class_name=class_name,
        source="fdm",
        group_key=group_key or output_stem,
        output_stem=output_stem,
        loader=lambda: np.zeros((64, 64, 3), dtype=np.uint8),
    )


class TestSampleCandidates:
    def test_respects_k_and_is_deterministic(self) -> None:
        pool = [_candidate("cracking", f"c{i:03d}") for i in range(100)]
        a = sample_candidates(pool, k=30, seed=1337)
        b = sample_candidates(pool, k=30, seed=1337)
        assert len(a) == 30
        assert [c.output_stem for c in a] == [c.output_stem for c in b]

    def test_duplicate_output_stem_raises(self) -> None:
        pool = [_candidate("cracking", "dup"), _candidate("cracking", "dup")]
        with pytest.raises(ValueError):
            sample_candidates(pool, k=1, seed=1)


# --------------------------------------------------------------------------
# Per-class selection recipes: cap respected, deterministic
# --------------------------------------------------------------------------


class TestSelectFdmDefectClass:
    def test_uses_all_when_under_cap(self) -> None:
        pool = [_candidate("warping", f"w{i:03d}") for i in range(400)]
        result = select_fdm_defect_class(pool, cap=550, seed=1337)
        assert len(result) == 400

    def test_caps_when_over_cap(self) -> None:
        pool = [_candidate("warping", f"w{i:03d}") for i in range(600)]
        result = select_fdm_defect_class(pool, cap=550, seed=1337)
        assert len(result) == 550

    def test_deterministic_for_fixed_seed(self) -> None:
        pool = [_candidate("warping", f"w{i:03d}") for i in range(600)]
        a = select_fdm_defect_class(pool, cap=550, seed=1337)
        b = select_fdm_defect_class(pool, cap=550, seed=1337)
        assert [c.output_stem for c in a] == [c.output_stem for c in b]


class TestSelectNormal:
    def test_samples_down_to_target(self) -> None:
        pool = [_candidate("normal", f"n{i:04d}") for i in range(2000)]
        result = select_normal(pool, target=500, cap=550, seed=1337)
        assert len(result) == 500

    def test_smaller_pool_kept_in_full(self) -> None:
        pool = [_candidate("normal", f"n{i:04d}") for i in range(300)]
        result = select_normal(pool, target=500, cap=550, seed=1337)
        assert len(result) == 300

    def test_cap_below_target_wins(self) -> None:
        pool = [_candidate("normal", f"n{i:04d}") for i in range(2000)]
        result = select_normal(pool, target=500, cap=200, seed=1337)
        assert len(result) == 200


class TestSelectSpaghetti:
    def test_all_off_platform_plus_sampled_argus_reaches_target(self) -> None:
        off_platform = [_candidate("spaghetti", f"op{i:03d}") for i in range(91)]
        argus_pool = [_candidate("spaghetti", f"ax{i:04d}") for i in range(3579)]
        result = select_spaghetti(off_platform, argus_pool, target=500, cap=550, seed=1337)
        assert len(result) == 500
        stems = {c.output_stem for c in result}
        # every off_platform image is present
        assert all(f"op{i:03d}" in stems for i in range(91))
        # remaining ~409 come from argus
        argus_count = sum(1 for s in stems if s.startswith("ax"))
        assert argus_count == 500 - 91

    def test_small_argus_pool_yields_fewer_than_target(self) -> None:
        off_platform = [_candidate("spaghetti", f"op{i:03d}") for i in range(91)]
        argus_pool = [_candidate("spaghetti", f"ax{i:03d}") for i in range(10)]
        result = select_spaghetti(off_platform, argus_pool, target=500, cap=550, seed=1337)
        assert len(result) == 101  # 91 + all 10 available

    def test_cap_smaller_than_off_platform_still_trims_to_cap(self) -> None:
        off_platform = [_candidate("spaghetti", f"op{i:03d}") for i in range(91)]
        argus_pool = [_candidate("spaghetti", f"ax{i:03d}") for i in range(50)]
        result = select_spaghetti(off_platform, argus_pool, target=500, cap=40, seed=1337)
        assert len(result) == 40

    def test_deterministic_for_fixed_seed(self) -> None:
        off_platform = [_candidate("spaghetti", f"op{i:03d}") for i in range(91)]
        argus_pool = [_candidate("spaghetti", f"ax{i:04d}") for i in range(3579)]
        a = select_spaghetti(off_platform, argus_pool, target=500, cap=550, seed=1337)
        b = select_spaghetti(off_platform, argus_pool, target=500, cap=550, seed=1337)
        assert [c.output_stem for c in a] == [c.output_stem for c in b]


# --------------------------------------------------------------------------
# split_groups / assert_no_group_overlap / assign_splits_for_class
# --------------------------------------------------------------------------


class TestSplitGroups:
    def test_deterministic_for_fixed_seed(self) -> None:
        groups = [f"g{i:04d}" for i in range(300)]
        a = split_groups(groups, seed=1337)
        b = split_groups(groups, seed=1337)
        assert a == b

    def test_different_seeds_can_differ(self) -> None:
        groups = [f"g{i:04d}" for i in range(300)]
        a = split_groups(groups, seed=1337)
        b = split_groups(groups, seed=42)
        assert a != b

    def test_no_overlap_and_all_groups_preserved(self) -> None:
        groups = [f"g{i:04d}" for i in range(357)]  # not evenly divisible
        train, val, test = split_groups(groups, seed=99)
        assert set(train) & set(val) == set()
        assert set(train) & set(test) == set()
        assert set(val) & set(test) == set()
        assert set(train) | set(val) | set(test) == set(groups)

    def test_ratios_approximately_70_15_15(self) -> None:
        groups = [f"g{i:04d}" for i in range(1000)]
        train, val, test = split_groups(groups, seed=1337)
        assert len(train) + len(val) + len(test) == 1000
        assert 690 <= len(train) <= 710
        assert 140 <= len(val) <= 160
        assert 140 <= len(test) <= 160

    def test_rejects_bad_ratios(self) -> None:
        with pytest.raises(ValueError):
            split_groups(["a", "b"], seed=1, ratios=(0.5, 0.5, 0.5))


class TestAssertNoGroupOverlap:
    def test_passes_on_disjoint_splits(self) -> None:
        assert_no_group_overlap({"train": ["a", "b"], "val": ["c"], "test": ["d"]})  # no raise

    def test_fires_on_deliberately_overlapping_split(self) -> None:
        with pytest.raises(AssertionError):
            assert_no_group_overlap({"train": ["a", "b"], "val": ["b", "c"], "test": ["d"]})

    def test_fires_on_train_test_overlap(self) -> None:
        with pytest.raises(AssertionError):
            assert_no_group_overlap({"train": ["a"], "val": ["b"], "test": ["a"]})


class TestAssignSplitsForClass:
    def test_all_variants_of_a_group_land_in_the_same_split(self) -> None:
        # One group with 5 near-duplicate images -- must never be split.
        candidates = [_candidate("warping", f"w{i}", group_key="shared_group") for i in range(5)]
        # Pad with plenty of single-image groups so the 70/15/15 split has
        # room to place "shared_group" anywhere without forcing a 100% split.
        candidates += [_candidate("warping", f"solo{i:03d}") for i in range(95)]

        result = assign_splits_for_class(candidates, seed=1337)

        homes = set()
        for split_name, items in result.items():
            stems = {c.output_stem for c in items}
            if any(s.startswith("w") and s[1:].isdigit() for s in stems):
                homes.add(split_name)
        assert len(homes) == 1  # all 5 "w*" images landed in exactly one split

        total = sum(len(v) for v in result.values())
        assert total == 100

    def test_no_overlap_end_to_end(self) -> None:
        candidates = [_candidate("cracking", f"c{i:03d}") for i in range(200)]
        result = assign_splits_for_class(candidates, seed=1337)
        stems_by_split = {s: {c.output_stem for c in items} for s, items in result.items()}
        assert stems_by_split["train"] & stems_by_split["val"] == set()
        assert stems_by_split["train"] & stems_by_split["test"] == set()
        assert stems_by_split["val"] & stems_by_split["test"] == set()


# --------------------------------------------------------------------------
# Print-session splitting -- leakage hazard #3's consequence: sessions (not
# frames) are the split unit, and small classes need explicit evaluability
# handling -- see assign_splits_for_class / _ensure_min_one_per_split.
# --------------------------------------------------------------------------


class TestSessionSplitIntegrity:
    def test_all_frames_of_a_session_land_in_exactly_one_split(self) -> None:
        # 20 "print sessions", each contributing a different number of
        # near-identical time-lapse frames -- mirrors the real FDM shape
        # (one session = many frames sharing one group_key).
        candidates: list[Candidate] = []
        for session_idx in range(20):
            n_frames = 3 + (session_idx % 5)
            for frame_idx in range(n_frames):
                candidates.append(
                    _candidate(
                        "warping",
                        f"session{session_idx:02d}_frame{frame_idx:02d}",
                        group_key=f"fdm:Warping:session{session_idx:02d}",
                    )
                )

        result = assign_splits_for_class(candidates, seed=1337)

        session_to_splits: dict[str, set[str]] = {}
        for split_name, items in result.items():
            for c in items:
                session_to_splits.setdefault(c.group_key, set()).add(split_name)

        # The core property: every session's frames all landed together.
        assert all(len(splits) == 1 for splits in session_to_splits.values())
        assert sum(len(v) for v in result.values()) == len(candidates)
        assert len(session_to_splits) == 20


class TestSmallSessionCountEvaluability:
    def _session_candidates(self, n_sessions: int, frames_per_session: int = 4) -> list[Candidate]:
        out: list[Candidate] = []
        for session_idx in range(n_sessions):
            for frame_idx in range(frames_per_session):
                out.append(
                    _candidate(
                        "layer_shifting",
                        f"s{session_idx}f{frame_idx}",
                        group_key=f"fdm:Layer_shifting:session{session_idx}",
                    )
                )
        return out

    def test_three_sessions_gets_at_least_one_val_and_one_test_session(self) -> None:
        candidates = self._session_candidates(n_sessions=3)
        result = assign_splits_for_class(candidates, seed=1337)

        assert len({c.group_key for c in result["train"]}) >= 1
        assert len({c.group_key for c in result["val"]}) >= 1
        assert len({c.group_key for c in result["test"]}) >= 1
        # every frame of every session is still fully accounted for and
        # confined to a single split.
        assert sum(len(v) for v in result.values()) == len(candidates)
        session_to_splits: dict[str, set[str]] = {}
        for split_name, items in result.items():
            for c in items:
                session_to_splits.setdefault(c.group_key, set()).add(split_name)
        assert all(len(splits) == 1 for splits in session_to_splits.values())

    def test_five_sessions_also_guarantees_val_and_test(self) -> None:
        # A different small-but-above-threshold count: naive 70/15/15
        # rounding of 5 groups gives train=4, val=1, test=0 -- exactly the
        # "test rounds down to 0" case _ensure_min_one_per_split exists to
        # fix (distinct from n=3's "val rounds down to 0" case above).
        candidates = self._session_candidates(n_sessions=5)
        result = assign_splits_for_class(candidates, seed=1337)
        assert len({c.group_key for c in result["train"]}) >= 1
        assert len({c.group_key for c in result["val"]}) >= 1
        assert len({c.group_key for c in result["test"]}) >= 1

    def test_two_sessions_all_go_to_train_and_are_not_evaluable(self) -> None:
        candidates = self._session_candidates(n_sessions=2)
        result = assign_splits_for_class(candidates, seed=1337)

        assert result["val"] == []
        assert result["test"] == []
        assert len(result["train"]) == len(candidates)
        assert {c.group_key for c in result["train"]} == {c.group_key for c in candidates}

        # this is exactly how callers (materialize_class) derive "not
        # evaluable": val and test are both empty.
        evaluable = bool(result["val"]) and bool(result["test"])
        assert evaluable is False

    def test_one_session_all_go_to_train_and_are_not_evaluable(self) -> None:
        candidates = self._session_candidates(n_sessions=1)
        result = assign_splits_for_class(candidates, seed=1337)
        assert result["val"] == [] and result["test"] == []
        assert len(result["train"]) == len(candidates)

    def test_min_sessions_for_eval_constant_is_three(self) -> None:
        # Documents the exact threshold the two tests above straddle.
        assert MIN_SESSIONS_FOR_EVAL == 3


class TestAssignSplitsForClassDeterminism:
    def test_deterministic_with_min_session_guarantee_active(self) -> None:
        candidates: list[Candidate] = []
        for session_idx in range(5):
            for frame_idx in range(6):
                candidates.append(
                    _candidate(
                        "stringing",
                        f"s{session_idx}f{frame_idx}",
                        group_key=f"fdm:Stringing:session{session_idx}",
                    )
                )
        a = assign_splits_for_class(candidates, seed=1337)
        b = assign_splits_for_class(candidates, seed=1337)
        for split in ("train", "val", "test"):
            assert [c.output_stem for c in a[split]] == [c.output_stem for c in b[split]]

    def test_deterministic_for_the_not_evaluable_path(self) -> None:
        candidates = TestSmallSessionCountEvaluability()._session_candidates(n_sessions=2)
        a = assign_splits_for_class(candidates, seed=7)
        b = assign_splits_for_class(candidates, seed=7)
        assert [c.output_stem for c in a["train"]] == [c.output_stem for c in b["train"]]


# --------------------------------------------------------------------------
# resize_and_center_crop -- portrait, landscape, square -> exactly 512x512
# --------------------------------------------------------------------------


class TestResizeAndCenterCrop:
    def test_landscape_input(self) -> None:
        img = np.random.randint(0, 255, size=(400, 800, 3), dtype=np.uint8)  # h < w
        out = resize_and_center_crop(img, size=512)
        assert out.shape == (512, 512, 3)

    def test_portrait_input(self) -> None:
        img = np.random.randint(0, 255, size=(800, 400, 3), dtype=np.uint8)  # h > w
        out = resize_and_center_crop(img, size=512)
        assert out.shape == (512, 512, 3)

    def test_square_input(self) -> None:
        img = np.random.randint(0, 255, size=(300, 300, 3), dtype=np.uint8)
        out = resize_and_center_crop(img, size=512)
        assert out.shape == (512, 512, 3)

    def test_already_correct_size_is_a_no_op_crop(self) -> None:
        img = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)
        out = resize_and_center_crop(img, size=512)
        assert out.shape == (512, 512, 3)
        assert np.array_equal(out, img)

    def test_very_wide_landscape(self) -> None:
        img = np.random.randint(0, 255, size=(200, 2000, 3), dtype=np.uint8)
        out = resize_and_center_crop(img, size=512)
        assert out.shape == (512, 512, 3)

    def test_real_fdm_scale_landscape(self) -> None:
        # Mirrors the real FDM images' 2048x3072 (h < w) shape.
        img = np.random.randint(0, 255, size=(2048, 3072, 3), dtype=np.uint8)
        out = resize_and_center_crop(img, size=512)
        assert out.shape == (512, 512, 3)

    def test_crop_is_centered(self) -> None:
        # Landscape image with a distinctive vertical stripe in the exact
        # horizontal center; after resize+center-crop it should still be
        # centered in the output.
        h, w = 512, 1024
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, w // 2 - 2 : w // 2 + 2] = 255
        out = resize_and_center_crop(img, size=512)
        assert out.shape == (512, 512, 3)
        # The bright stripe should still be roughly centered horizontally.
        col_brightness = out[:, :, 0].mean(axis=0)
        brightest_col = int(np.argmax(col_brightness))
        assert 512 // 2 - 20 <= brightest_col <= 512 // 2 + 20


class TestSaveNormalizedJpeg:
    def test_writes_a_512x512_jpeg(self, tmp_path: Path) -> None:
        img = np.random.randint(0, 255, size=(400, 800, 3), dtype=np.uint8)
        dest = tmp_path / "out" / "img.jpg"
        save_normalized_jpeg(img, dest, quality=90)
        assert dest.is_file()
        loaded = cv2.imread(str(dest))
        assert loaded.shape == (512, 512, 3)


# --------------------------------------------------------------------------
# build_dataset -- end-to-end with synthetic in-memory candidate pools
# --------------------------------------------------------------------------


def _synthetic_pool(class_name: str, source: str, n: int, prefix: str, group_size: int = 1) -> list[Candidate]:
    """Build `n` synthetic candidates for `class_name`, optionally grouped
    `group_size`-at-a-time (to exercise multi-image groups)."""
    out: list[Candidate] = []
    for i in range(n):
        group_idx = i // group_size
        out.append(
            Candidate(
                class_name=class_name,
                source=source,
                group_key=f"{prefix}:group{group_idx:04d}",
                output_stem=f"{prefix}_{i:05d}",
                loader=lambda: np.random.randint(0, 255, size=(300, 500, 3), dtype=np.uint8),
            )
        )
    return out


class TestBuildDatasetEndToEnd:
    def test_full_pipeline_writes_512x512_images_and_report(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "argus_cls"

        fdm_defect_by_class = {
            "cracking": _synthetic_pool("cracking", "fdm", 40, "cracking"),
            "layer_shifting": _synthetic_pool("layer_shifting", "fdm", 40, "layer_shifting"),
            "stringing": _synthetic_pool("stringing", "fdm", 40, "stringing"),
            "warping": _synthetic_pool("warping", "fdm", 40, "warping"),
        }
        fdm_off_platform = _synthetic_pool("spaghetti", "fdm", 10, "off_platform")
        argus_pool = _synthetic_pool("spaghetti", "argus_v2", 60, "argus", group_size=3)
        hf_pool = _synthetic_pool("normal", "hf", 80, "hf")

        report = build_dataset(
            out_dir=out_dir,
            fdm_defect_by_class=fdm_defect_by_class,
            fdm_off_platform=fdm_off_platform,
            argus_spaghetti_pool=argus_pool,
            hf_normal_pool=hf_pool,
            seed=1337,
            per_class_cap=550,
            target_per_class=500,
        )

        # Every class directory got created for every split, and every
        # written JPEG is exactly 512x512.
        checked_any = False
        for class_name in CLASS_NAMES:
            for split in ("train", "val", "test"):
                class_dir = out_dir / split / class_name
                assert class_dir.is_dir()
            all_files = list((out_dir).rglob(f"*/{class_name}/*.jpg"))
            assert len(all_files) > 0
            for f in all_files[:3]:
                img = cv2.imread(str(f))
                assert img.shape == (512, 512, 3)
                checked_any = True
        assert checked_any

        # normal: all 80 kept (under target/cap)
        assert report["classes"]["normal"]["counts"]["total"] == 80
        # spaghetti: 10 off_platform + 60 argus = 70 (under target/cap)
        assert report["classes"]["spaghetti"]["counts"]["total"] == 70
        assert report["classes"]["spaghetti"]["source_breakdown"] == {"fdm": 10, "argus_v2": 60}
        # FDM defect classes: all 40 kept each
        for cname in ("cracking", "layer_shifting", "stringing", "warping"):
            assert report["classes"][cname]["counts"]["total"] == 40
            assert report["classes"][cname]["source_breakdown"] == {"fdm": 40}

        assert "PASS" in report["group_overlap_check"]
        assert "confound_warning" in report
        assert "normal" in report["confound_warning"].lower()

        report_json_path = out_dir / "split_report.json"
        # build_dataset itself doesn't write the report file (main() does);
        # confirm it's at least JSON-serializable end to end.
        import json

        json.dumps(report)  # must not raise

    def test_multi_image_groups_never_split_across_train_val_test(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "argus_cls2"
        fdm_defect_by_class = {
            "cracking": _synthetic_pool("cracking", "fdm", 30, "cracking"),
            "layer_shifting": _synthetic_pool("layer_shifting", "fdm", 30, "layer_shifting"),
            "stringing": _synthetic_pool("stringing", "fdm", 30, "stringing"),
            "warping": _synthetic_pool("warping", "fdm", 30, "warping"),
        }
        fdm_off_platform = _synthetic_pool("spaghetti", "fdm", 6, "off_platform")
        # argus pool grouped 4-at-a-time -- like real .rf. variants.
        argus_pool = _synthetic_pool("spaghetti", "argus_v2", 40, "argus", group_size=4)
        hf_pool = _synthetic_pool("normal", "hf", 50, "hf")

        report = build_dataset(
            out_dir=out_dir,
            fdm_defect_by_class=fdm_defect_by_class,
            fdm_off_platform=fdm_off_platform,
            argus_spaghetti_pool=argus_pool,
            hf_normal_pool=hf_pool,
            seed=7,
            per_class_cap=550,
            target_per_class=500,
        )
        # Re-derive group membership from what's actually on disk for the
        # spaghetti class and confirm no group's images landed in >1 split.
        group_to_splits: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            for f in (out_dir / split / "spaghetti").glob("argus_*.jpg"):
                # output_stem is "argus_{i:05d}"; group_size=4 -> group index i//4
                idx = int(f.stem.split("_")[1])
                group = f"argus:group{idx // 4:04d}"
                group_to_splits.setdefault(group, set()).add(split)
        assert all(len(v) == 1 for v in group_to_splits.values())
        assert report["classes"]["spaghetti"]["counts"]["total"] == 6 + 40


def _session_synthetic_pool(class_name: str, n_frames: int, frames_per_session: int, prefix: str) -> list[Candidate]:
    """Like `_synthetic_pool`, but with an explicit `frame_key` distinct
    from `group_key` -- mirrors real `fdm_candidates()` output where many
    frames (frame_key) collapse into fewer print sessions (group_key), so
    build_dataset's "n_frames vs n_sessions" reporting has something real
    to show."""
    out: list[Candidate] = []
    for i in range(n_frames):
        session_idx = i // frames_per_session
        out.append(
            Candidate(
                class_name=class_name,
                source="fdm",
                group_key=f"{prefix}:session{session_idx:04d}",
                frame_key=f"{prefix}:frame{i:04d}",
                output_stem=f"{prefix}_{i:05d}",
                loader=lambda: np.random.randint(0, 255, size=(300, 500, 3), dtype=np.uint8),
            )
        )
    return out


class TestBuildDatasetEvaluationValidityReporting:
    def test_thin_class_is_marked_not_evaluable_and_flagged(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "argus_cls_thin"

        # layer_shifting: 40 frames from only 2 "print sessions" (20 frames
        # each) -- below MIN_SESSIONS_FOR_EVAL, so ALL of it should land in
        # train and the class should be flagged NOT EVALUATED.
        fdm_defect_by_class = {
            "cracking": _synthetic_pool("cracking", "fdm", 40, "cracking"),
            "layer_shifting": _session_synthetic_pool("layer_shifting", n_frames=40, frames_per_session=20, prefix="ls"),
            "stringing": _synthetic_pool("stringing", "fdm", 40, "stringing"),
            "warping": _synthetic_pool("warping", "fdm", 40, "warping"),
        }
        fdm_off_platform = _synthetic_pool("spaghetti", "fdm", 10, "off_platform")
        argus_pool = _synthetic_pool("spaghetti", "argus_v2", 60, "argus", group_size=3)
        hf_pool = _synthetic_pool("normal", "hf", 80, "hf")

        report = build_dataset(
            out_dir=out_dir,
            fdm_defect_by_class=fdm_defect_by_class,
            fdm_off_platform=fdm_off_platform,
            argus_spaghetti_pool=argus_pool,
            hf_normal_pool=hf_pool,
            seed=1337,
            per_class_cap=550,
            target_per_class=500,
        )

        ls = report["classes"]["layer_shifting"]
        assert ls["evaluable"] is False
        assert ls["n_sessions"] == 2
        assert ls["n_frames"] == 40
        assert ls["counts"]["val"] == 0
        assert ls["counts"]["test"] == 0
        assert ls["counts"]["train"] == ls["counts"]["total"] == 40
        assert ls["sessions_per_split"] == {"train": 2, "val": 0, "test": 0}

        cracking = report["classes"]["cracking"]
        assert cracking["evaluable"] is True

        # the evaluation validity warning names the thin class as NOT
        # EVALUATED, and doesn't wrongly flag the healthy one.
        warning = report["evaluation_validity_warning"]
        assert "layer_shifting" in warning
        assert "NOT EVALUATED" in warning

        # no image was ever written into layer_shifting's val/test dirs.
        assert list((out_dir / "val" / "layer_shifting").glob("*.jpg")) == []
        assert list((out_dir / "test" / "layer_shifting").glob("*.jpg")) == []
        assert len(list((out_dir / "train" / "layer_shifting").glob("*.jpg"))) == 40

    def test_report_shows_frames_and_sessions_separately(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "argus_cls_frames_sessions"

        # cracking: 40 frames grouped 4-per-session -> 10 sessions. Every
        # frame still gets its own output file (the split UNIT changed,
        # not the number of images kept).
        fdm_defect_by_class = {
            "cracking": _session_synthetic_pool("cracking", n_frames=40, frames_per_session=4, prefix="crk"),
            "layer_shifting": _synthetic_pool("layer_shifting", "fdm", 40, "layer_shifting"),
            "stringing": _synthetic_pool("stringing", "fdm", 40, "stringing"),
            "warping": _synthetic_pool("warping", "fdm", 40, "warping"),
        }
        fdm_off_platform = _synthetic_pool("spaghetti", "fdm", 10, "off_platform")
        argus_pool = _synthetic_pool("spaghetti", "argus_v2", 60, "argus", group_size=3)
        hf_pool = _synthetic_pool("normal", "hf", 80, "hf")

        report = build_dataset(
            out_dir=out_dir,
            fdm_defect_by_class=fdm_defect_by_class,
            fdm_off_platform=fdm_off_platform,
            argus_spaghetti_pool=argus_pool,
            hf_normal_pool=hf_pool,
            seed=1337,
            per_class_cap=550,
            target_per_class=500,
        )

        cracking = report["classes"]["cracking"]
        assert cracking["n_frames"] == 40
        assert cracking["n_sessions"] == 10
        assert cracking["counts"]["total"] == 40  # all 40 images still written

        # a class with no session grouping (frame_key defaults to
        # group_key) reports the two numbers as equal.
        stringing = report["classes"]["stringing"]
        assert stringing["n_frames"] == stringing["n_sessions"] == 40

        assert report["total_frames"] >= report["total_sessions"]


def test_class_names_match_config_example_and_task_spec() -> None:
    assert CLASS_NAMES == ("normal", "spaghetti", "cracking", "layer_shifting", "stringing", "warping")
