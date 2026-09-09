"""Unit tests for training/ingest_roboflow_multi.py's class harmonization,
duplicate detection, redundant-source guard and group-aware splitting.

Every fixture is synthesized under `tmp_path` (tiny in-memory numpy images
written as lossless PNG/BMP, hand-written data.yaml + YOLO label files). The
real datasets are never touched and nothing here hits the network -- the
suite must run fully offline.

The four things these tests exist to protect:
  1. class harmonization -- source classes land in the right unified slot, and
     unknown/dropped ones are reported rather than silently renumbered;
  2. the split's zero-overlap guarantee -- no leakage group may ever appear in
     two splits, checked against what was actually written to disk;
  3. duplicate detection -- exact (re-encode-proof) and perceptual, including
     across dataset boundaries, which is the cross-upload re-upload check;
  4. the redundant-source guard -- an export whose source photos are already in
     another source must stop the merge before a single file is copied, because
     that is exactly what rf_defects/rf_failure did behind a big file count.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import training.ingest_roboflow_multi as ingest
from argus.types import Severity
from training.ingest_roboflow_multi import (
    CLASS_MAP,
    DEFAULT_PHASH_DISTANCE,
    DEFAULT_SESSION_CORROBORATION,
    DEFAULT_SPLIT_RATIOS,
    MIN_GROUPS_FOR_SPLIT,
    REDUNDANT_SOURCE_PCT,
    SEVERITY_BY_CLASS,
    SOURCE_SPEC_BY_KEY,
    UNIFIED_CLASSES,
    ImageRecord,
    SourceSpec,
    UnionFind,
    assert_no_group_overlap,
    assert_no_redundant_sources,
    build_class_remap,
    build_merged_dataset,
    build_stem_overlap_matrix,
    content_hash,
    dhash,
    filename_session_groups,
    find_redundant_sources,
    hamming,
    hash_records,
    is_known_class,
    main,
    map_source_class,
    normalize_class_name,
    output_filename,
    parse_filename_timestamp,
    parse_sequence_key,
    phash_neighbor_pairs,
    read_data_yaml_license,
    remap_label_text,
    scan_source,
    session_corroboration,
    session_is_believable,
    source_stem_sets,
    split_groups,
)

# --------------------------------------------------------------------------
# Synthetic fixture helpers
# --------------------------------------------------------------------------

#: The real class lists, in the real class-id order, straight from each export's
#: own data.yaml. Class 0 is 'error extrusion' for atco and 'Bed Adhesion' for
#: stereovision, so the default one-row label below exercises both.
ATCO_CLASSES = ["error extrusion", "spaghetti", "stringing", "warping", "zits"]
STEREO_CLASSES = ["Bed Adhesion", "Blob of Death", "Head", "Layer Separation", "Spaghetti", "Warping"]
#: A fourth-party naming variant kept as a fixture: it is the only source of
#: "Layer Split", which CLASS_MAP folds into layer_separation.
LAYER_SPLIT_CLASSES = ["Layer Split", "Spaghetti", "Stringing", "Warping"]


def _distinct_image(index: int, size: int = 64) -> np.ndarray:
    """A deterministic image whose dhash is far from every other index's.

    Values stay in 0..200 so a uniform +3 shift (used to build near-duplicates)
    never clips, which keeps the shifted copy's dhash bit-identical.
    """
    rng = np.random.default_rng(9_000 + index)
    return rng.integers(0, 201, (size, size, 3), dtype=np.uint8)


def _shifted(image: np.ndarray, delta: int = 3) -> np.ndarray:
    """Same scene, different pixels: a near-duplicate, not an exact duplicate."""
    return np.clip(image.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def _make_export(
    root: Path,
    class_names: list[str],
    images: dict[str, np.ndarray],
    labels: dict[str, str] | None = None,
    split_dir: str = "train",
    ext: str = ".png",
) -> Path:
    """Writes a minimal Roboflow-shaped YOLO export (data.yaml + <split>/images + labels)."""
    labels = labels or {}
    (root / split_dir / "images").mkdir(parents=True, exist_ok=True)
    (root / split_dir / "labels").mkdir(parents=True, exist_ok=True)
    names_literal = ", ".join(f"'{c}'" for c in class_names)
    (root / "data.yaml").write_text(
        f"train: train/images\nval: valid/images\nnc: {len(class_names)}\nnames: [{names_literal}]\n",
        encoding="utf-8",
    )
    for stem, arr in images.items():
        assert cv2.imwrite(str(root / split_dir / "images" / f"{stem}{ext}"), arr)
        (root / split_dir / "labels" / f"{stem}.txt").write_text(
            labels.get(stem, "0 0.5 0.5 0.2 0.2\n"), encoding="utf-8"
        )
    return root


def _spaced_export(
    root: Path,
    class_names: list[str],
    n: int,
    image_offset: int,
    rf_suffix: bool = False,
    label_text: str = "0 0.5 0.5 0.2 0.2\n",
    stem_prefix: str = "s",
) -> Path:
    """n independent images with sequence numbers spaced far enough apart
    (steps of 100 >> --seq-gap 3) that filename-session grouping keeps them
    separate, so a split has real groups to work with.

    stem_prefix must differ between two sources used in the same merge:
    identical stems across sources read as a redundant re-upload, which the
    pre-flight guard (correctly) refuses.
    """
    images: dict[str, np.ndarray] = {}
    for i in range(n):
        stem = f"{stem_prefix}{i * 100:04d}"
        if rf_suffix:
            stem = f"{stem}.rf.{i:032x}"
        images[stem] = _distinct_image(image_offset + i)
    return _make_export(root, class_names, images, {s: label_text for s in images})


def _scan_and_hash(root: Path, key: str, is_reference: bool = False):
    scan = scan_source(root, key, is_reference=is_reference)
    hash_records(scan.records, workers=1, progress_every=0)
    return scan


@pytest.fixture
def noncommercial_spec(monkeypatch: pytest.MonkeyPatch) -> SourceSpec:
    """Registers a synthetic CC BY-NC source ('ncsource') in the spec tables.

    Both real sources are permissive, so license gating has nothing to bite on
    in production; this keeps the capability under test without pretending a
    non-commercial source is part of the roster.
    """
    spec = SourceSpec(
        key="ncsource",
        origin="synthetic/noncommercial-fixture",
        license="CC BY-NC-SA 4.0",
        noncommercial=True,
        expected_classes=tuple(LAYER_SPLIT_CLASSES),
    )
    monkeypatch.setattr(ingest, "SOURCE_SPECS", ingest.SOURCE_SPECS + (spec,))
    monkeypatch.setattr(ingest, "SOURCE_SPEC_BY_KEY", {**ingest.SOURCE_SPEC_BY_KEY, "ncsource": spec})
    return spec


# --------------------------------------------------------------------------
# Class harmonization
# --------------------------------------------------------------------------


class TestNormalizeClassName:
    def test_lowercases_and_flattens_separators(self) -> None:
        assert normalize_class_name("Blob_of-Death") == "blob of death"
        assert normalize_class_name("  Layer   Separation  ") == "layer separation"
        assert normalize_class_name("Layer_Split") == "layer split"

    def test_idempotent(self) -> None:
        for name in ("spaghetti", "layer separation", "blob of death"):
            assert normalize_class_name(normalize_class_name(name)) == name


class TestMapSourceClass:
    @pytest.mark.parametrize(
        "source_name,unified",
        [
            ("spaghetti", "spaghetti"),
            ("Spaghetti", "spaghetti"),
            ("Layer Separation", "layer_separation"),
            ("Layer Split", "layer_separation"),
            ("Bed Adhesion", "bed_adhesion"),
            ("Blob of Death", "blob_of_death"),
            ("Warping", "warping"),
            ("warping", "warping"),
            ("stringing", "stringing"),
            ("Stringing", "stringing"),
            ("zits", "zits"),
            ("Head", "head"),
            ("error extrusion", "error_extrusion"),
            ("error_extrusion", "error_extrusion"),
            ("Error Extrusion", "error_extrusion"),
        ],
    )
    def test_every_documented_source_class_maps(self, source_name: str, unified: str) -> None:
        assert map_source_class(source_name) == unified
        assert unified in UNIFIED_CLASSES

    def test_every_class_of_every_declared_source_is_mapped(self) -> None:
        # A class the table doesn't know is silently dropped from the labels, so
        # the roster's own class lists must all resolve.
        for spec in SOURCE_SPEC_BY_KEY.values():
            for name in spec.expected_classes:
                assert map_source_class(name) is not None, f"{spec.key}: {name!r} is unmapped"

    def test_unknown_class_is_unmapped_not_dropped(self) -> None:
        assert map_source_class("Elephant") is None
        assert is_known_class("Elephant") is False
        assert is_known_class("Blob of Death") is True

    def test_layer_split_and_layer_separation_collapse_together(self) -> None:
        assert map_source_class("Layer Split") == map_source_class("Layer Separation")


class TestSeverity:
    def test_every_unified_class_has_a_severity(self) -> None:
        assert set(SEVERITY_BY_CLASS) == set(UNIFIED_CLASSES)

    def test_the_unified_space_is_nine_classes(self) -> None:
        # Every distinct source class is kept; nothing is merged away.
        assert len(UNIFIED_CLASSES) == 9
        assert len(set(UNIFIED_CLASSES)) == 9

    def test_catastrophic_and_cosmetic_partition(self) -> None:
        catastrophic = {c for c, s in SEVERITY_BY_CLASS.items() if s is Severity.CATASTROPHIC}
        cosmetic = {c for c, s in SEVERITY_BY_CLASS.items() if s is Severity.COSMETIC}
        assert catastrophic == {"spaghetti", "layer_separation", "bed_adhesion", "blob_of_death"}
        assert cosmetic == {"warping", "stringing", "zits", "head", "error_extrusion"}

    def test_head_is_a_non_actionable_distractor(self) -> None:
        # The printhead must never be able to trigger a pause/cancel.
        assert SEVERITY_BY_CLASS["head"] is Severity.COSMETIC

    def test_error_extrusion_is_recoverable_not_catastrophic(self) -> None:
        assert SEVERITY_BY_CLASS["error_extrusion"] is Severity.COSMETIC

    def test_every_class_map_target_is_a_unified_class(self) -> None:
        for target in CLASS_MAP.values():
            assert target is None or target in UNIFIED_CLASSES


class TestBuildClassRemap:
    def test_atco_classes(self) -> None:
        remap = build_class_remap(ATCO_CLASSES)
        assert remap.id_remap == {
            0: UNIFIED_CLASSES.index("error_extrusion"),
            1: UNIFIED_CLASSES.index("spaghetti"),
            2: UNIFIED_CLASSES.index("stringing"),
            3: UNIFIED_CLASSES.index("warping"),
            4: UNIFIED_CLASSES.index("zits"),
        }
        assert remap.unmapped == [] and remap.dropped == []

    def test_stereovision_classes(self) -> None:
        remap = build_class_remap(STEREO_CLASSES)
        assert remap.id_remap == {
            0: UNIFIED_CLASSES.index("bed_adhesion"),
            1: UNIFIED_CLASSES.index("blob_of_death"),
            2: UNIFIED_CLASSES.index("head"),
            3: UNIFIED_CLASSES.index("layer_separation"),
            4: UNIFIED_CLASSES.index("spaghetti"),
            5: UNIFIED_CLASSES.index("warping"),
        }
        assert remap.unmapped == [] and remap.dropped == []

    def test_layer_split_lands_in_layer_separation(self) -> None:
        remap = build_class_remap(LAYER_SPLIT_CLASSES)
        assert remap.id_remap[0] == UNIFIED_CLASSES.index("layer_separation")
        assert remap.mapped["Layer Split"] == "layer_separation"

    def test_unknown_class_reported_and_left_out_of_remap(self) -> None:
        remap = build_class_remap(["spaghetti", "Elephant", "zits"])
        assert remap.unmapped == ["Elephant"]
        assert 1 not in remap.id_remap
        assert remap.id_remap == {0: UNIFIED_CLASSES.index("spaghetti"), 2: UNIFIED_CLASSES.index("zits")}

    def test_class_mapped_to_none_is_dropped_and_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(CLASS_MAP, "elephant", None)
        remap = build_class_remap(["spaghetti", "Elephant"])
        assert remap.dropped == ["Elephant"]
        assert remap.unmapped == []
        assert 1 not in remap.id_remap

    def test_typo_in_class_map_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(CLASS_MAP, "elephant", "not_a_unified_class")
        with pytest.raises(ValueError, match="not in UNIFIED_CLASSES"):
            build_class_remap(["Elephant"])


class TestRemapLabelText:
    def test_rows_are_renumbered_into_unified_ids(self) -> None:
        cf = build_class_remap(STEREO_CLASSES).to_class_filter()
        text, stats = remap_label_text("0 0.5 0.5 0.2 0.2\n4 0.25 0.25 0.1 0.1\n", cf)
        bed_id = UNIFIED_CLASSES.index("bed_adhesion")
        spag_id = UNIFIED_CLASSES.index("spaghetti")
        assert [int(line.split()[0]) for line in text.strip().splitlines()] == [bed_id, spag_id]
        assert stats.rows_in == 2 and stats.rows_out == 2 and stats.rows_dropped_by_class == 0

    def test_atco_error_extrusion_survives_instead_of_being_dropped(self) -> None:
        # Before 'error extrusion' was in CLASS_MAP, every AtCo class-0 row was
        # silently discarded as "unmapped".
        cf = build_class_remap(ATCO_CLASSES).to_class_filter()
        text, stats = remap_label_text("0 0.5 0.5 0.2 0.2\n", cf)
        assert int(text.split()[0]) == UNIFIED_CLASSES.index("error_extrusion")
        assert stats.rows_out == 1 and stats.rows_dropped_by_class == 0

    def test_unmapped_class_rows_are_dropped_and_counted(self) -> None:
        cf = build_class_remap(["spaghetti", "Elephant"]).to_class_filter()
        text, stats = remap_label_text("0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n", cf)
        assert text.strip().splitlines() == ["0 0.500000 0.500000 0.200000 0.200000"]
        assert stats.rows_dropped_by_class == 1

    def test_polygon_rows_are_collapsed_to_bboxes(self) -> None:
        cf = build_class_remap(ATCO_CLASSES).to_class_filter()
        text, stats = remap_label_text("1 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n", cf)
        assert stats.polygon_rows_converted == 1
        cid, cx, cy, w, h = text.split()
        assert int(cid) == UNIFIED_CLASSES.index("spaghetti")
        assert float(w) == pytest.approx(0.2) and float(h) == pytest.approx(0.3)

    def test_empty_label_stays_empty(self) -> None:
        cf = build_class_remap(ATCO_CLASSES).to_class_filter()
        text, stats = remap_label_text("\n  \n", cf)
        assert text == "" and stats.rows_out == 0

    def test_out_of_range_class_id_is_dropped(self) -> None:
        cf = build_class_remap(ATCO_CLASSES).to_class_filter()
        text, stats = remap_label_text("9 0.5 0.5 0.2 0.2\n", cf)
        assert text == "" and stats.rows_dropped_by_class == 1


# --------------------------------------------------------------------------
# Duplicate detection: exact content hash
# --------------------------------------------------------------------------


class TestContentHash:
    def test_same_array_same_hash(self) -> None:
        a = _distinct_image(1)
        assert content_hash(a) == content_hash(a.copy())

    def test_different_arrays_different_hash(self) -> None:
        assert content_hash(_distinct_image(1)) != content_hash(_distinct_image(2))

    def test_reencoding_does_not_hide_a_duplicate(self, tmp_path: Path) -> None:
        # The whole point of hashing DECODED pixels: the same photo saved in two
        # lossless containers must still collide.
        arr = _distinct_image(7)
        png, bmp = tmp_path / "a.png", tmp_path / "a.bmp"
        assert cv2.imwrite(str(png), arr) and cv2.imwrite(str(bmp), arr)
        assert png.read_bytes() != bmp.read_bytes()
        assert content_hash(cv2.imread(str(png))) == content_hash(cv2.imread(str(bmp)))

    def test_shape_is_part_of_the_digest(self) -> None:
        flat = np.zeros((4, 6, 3), dtype=np.uint8)
        tall = np.zeros((6, 4, 3), dtype=np.uint8)
        assert content_hash(flat) != content_hash(tall)


# --------------------------------------------------------------------------
# Duplicate detection: perceptual hash
# --------------------------------------------------------------------------


class TestDhash:
    def test_is_64_bits(self) -> None:
        assert 0 <= dhash(_distinct_image(3)) < (1 << 64)

    def test_identical_images_hash_identically(self) -> None:
        arr = _distinct_image(4)
        assert dhash(arr) == dhash(arr.copy())

    def test_brightness_shift_is_a_near_duplicate(self) -> None:
        arr = _distinct_image(5)
        shifted = _shifted(arr)
        assert content_hash(arr) != content_hash(shifted)  # not an exact duplicate
        assert hamming(dhash(arr), dhash(shifted)) <= DEFAULT_PHASH_DISTANCE

    def test_mirrored_gradient_is_far_away(self) -> None:
        gradient = np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
        gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        assert hamming(dhash(gradient), dhash(gradient[:, ::-1])) > DEFAULT_PHASH_DISTANCE

    def test_independent_images_are_far_apart(self) -> None:
        hashes = [dhash(_distinct_image(i)) for i in range(12)]
        for i, a in enumerate(hashes):
            for b in hashes[i + 1 :]:
                assert hamming(a, b) > DEFAULT_PHASH_DISTANCE


class TestHamming:
    def test_known_distances(self) -> None:
        assert hamming(0b1010, 0b1010) == 0
        assert hamming(0b1010, 0b1011) == 1
        assert hamming(0, (1 << 64) - 1) == 64


class TestPhashNeighborPairs:
    def _clusters(self, hashes: dict[str, int], distance: int) -> list[set[str]]:
        uf = UnionFind(hashes)
        for a, b in phash_neighbor_pairs(hashes, distance):
            uf.union(a, b)
        return sorted((set(v) for v in uf.components().values()), key=lambda s: sorted(s)[0])

    def test_identical_hashes_link_even_at_distance_zero(self) -> None:
        clusters = self._clusters({"a": 0xFF, "b": 0xFF, "c": 0x0F}, 0)
        assert {"a", "b"} in clusters and {"c"} in clusters

    def test_links_within_threshold_and_not_beyond(self) -> None:
        hashes = {"a": 0, "b": 0b111, "c": (1 << 63) | 0xFFFF}
        clusters = self._clusters(hashes, 3)
        assert {"a", "b"} in clusters
        assert {"c"} in clusters

    @staticmethod
    def _spread_bits(n: int) -> int:
        """n set bits spread across the whole 64-bit word (one per band, worst
        case for pigeonhole banding)."""
        bits = 0
        for k in range(n):
            bits |= 1 << (k * 9 % 64)
        assert bin(bits).count("1") == n
        return bits

    @pytest.mark.parametrize("distance", [1, 3, 6])
    def test_finds_pairs_at_exactly_the_threshold(self, distance: int) -> None:
        # Pigeonhole banding must not miss a pair sitting right on the boundary
        # with its differing bits spread one per band.
        clusters = self._clusters({"a": 0, "b": self._spread_bits(distance)}, distance)
        assert {"a", "b"} in clusters, f"missed a pair at distance {distance}"

    @pytest.mark.parametrize("distance", [1, 3, 6])
    def test_does_not_link_one_bit_beyond_the_threshold(self, distance: int) -> None:
        clusters = self._clusters({"a": 0, "b": self._spread_bits(distance + 1)}, distance)
        assert {"a"} in clusters and {"b"} in clusters

    def test_transitive_chain_becomes_one_cluster(self) -> None:
        # Adjacent timelapse frames drift: 0 -> 1 -> 2 bits apart pairwise, all
        # one print job.
        hashes = {"f0": 0b0000, "f1": 0b0001, "f2": 0b0011, "f3": 0b0111}
        assert self._clusters(hashes, 1) == [{"f0", "f1", "f2", "f3"}]

    def test_deterministic(self) -> None:
        hashes = {f"k{i}": i * 7 for i in range(40)}
        assert phash_neighbor_pairs(hashes, 4) == phash_neighbor_pairs(dict(reversed(list(hashes.items()))), 4)

    def test_empty_input(self) -> None:
        assert phash_neighbor_pairs({}, 6) == []


class TestUnionFind:
    def test_singletons_by_default(self) -> None:
        uf = UnionFind(["a", "b", "c"])
        assert len(uf.components()) == 3

    def test_union_is_transitive(self) -> None:
        uf = UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")
        uf.union("b", "c")
        comps = {frozenset(v) for v in uf.components().values()}
        assert comps == {frozenset({"a", "b", "c"}), frozenset({"d"})}

    def test_long_chain_does_not_recurse(self) -> None:
        keys = [f"k{i:05d}" for i in range(5000)]
        uf = UnionFind(keys)
        for a, b in zip(keys, keys[1:]):
            uf.union(a, b)
        assert len(uf.components()) == 1


# --------------------------------------------------------------------------
# Session / timelapse recovery from filenames
# --------------------------------------------------------------------------


class TestParseFilenameTimestamp:
    def test_fdm_style_17_digits(self) -> None:
        assert parse_filename_timestamp("Image_20230417093015123") is not None
        assert parse_filename_timestamp("Image_20230417093015123").isoformat() == "2023-04-17T09:30:15"

    def test_compact_14_digits(self) -> None:
        assert parse_filename_timestamp("cam_20230417093015").isoformat() == "2023-04-17T09:30:15"

    def test_underscore_separated(self) -> None:
        assert parse_filename_timestamp("IMG_20230417_093015").isoformat() == "2023-04-17T09:30:15"

    def test_dashed(self) -> None:
        assert parse_filename_timestamp("snap-2023-04-17_09-30-15").isoformat() == "2023-04-17T09:30:15"

    def test_no_timestamp(self) -> None:
        assert parse_filename_timestamp("frame_0042") is None
        assert parse_filename_timestamp("plain-name") is None

    def test_invalid_date_digits(self) -> None:
        assert parse_filename_timestamp("x99999999999999") is None


class TestParseSequenceKey:
    def test_trailing_number(self) -> None:
        assert parse_sequence_key("frame_0042") == ("frame_|", 42)

    def test_binds_to_last_digit_run(self) -> None:
        family, num = parse_sequence_key("cam2_shot_0007")
        assert num == 7 and family.startswith("cam2_shot_")

    def test_suffix_is_kept_in_the_family(self) -> None:
        assert parse_sequence_key("img_0007_jpg") == ("img_|_jpg", 7)

    def test_no_digits(self) -> None:
        assert parse_sequence_key("photo") is None
        assert parse_sequence_key("a1") is None  # single digit is not a sequence


class TestFilenameSessionGroups:
    def test_timestamps_split_on_a_large_gap(self) -> None:
        stems = [
            "Image_20230417090000000",
            "Image_20230417090030000",  # +30s, same job
            "Image_20230417120000000",  # +3h, new job
        ]
        sessions = filename_session_groups(stems, gap_s=600)
        assert sessions[stems[0]] == sessions[stems[1]]
        assert sessions[stems[2]] != sessions[stems[0]]
        assert len(set(sessions.values())) == 2

    def test_consecutive_sequence_numbers_are_one_session(self) -> None:
        stems = [f"frame_{i:04d}" for i in range(6)]
        sessions = filename_session_groups(stems, seq_gap=3)
        assert len(set(sessions.values())) == 1

    def test_sequence_jump_starts_a_new_session(self) -> None:
        sessions = filename_session_groups(["frame_0001", "frame_0002", "frame_0900"], seq_gap=3)
        assert sessions["frame_0001"] == sessions["frame_0002"]
        assert sessions["frame_0900"] != sessions["frame_0001"]

    def test_different_families_never_share_a_session(self) -> None:
        sessions = filename_session_groups(["camA_0001", "camB_0001"], seq_gap=3)
        assert sessions["camA_0001"] != sessions["camB_0001"]

    def test_stems_with_no_signal_are_left_ungrouped(self) -> None:
        sessions = filename_session_groups(["opaque", "another"], seq_gap=3)
        assert sessions == {}

    def test_timestamp_wins_over_sequence(self) -> None:
        sessions = filename_session_groups(["IMG_20230417_093015"], seq_gap=3)
        assert sessions["IMG_20230417_093015"].startswith("ts")


# --------------------------------------------------------------------------
# Filename sessions must be corroborated before they are believed
# --------------------------------------------------------------------------


def _records(images: dict[str, np.ndarray], tmp_path: Path, key: str = "atco"):
    root = _make_export(tmp_path / f"src_{key}", ATCO_CLASSES, images)
    return _scan_and_hash(root, key).records


class TestSessionCorroboration:
    def test_a_real_capture_session_scores_high(self, tmp_path: Path) -> None:
        base = _distinct_image(3000)
        recs = _records({f"frame_{i:04d}": _shifted(base, i) for i in range(10)}, tmp_path)
        assert session_corroboration(recs) == pytest.approx(1.0)

    def test_a_numbered_collection_of_independent_photos_scores_zero(self, tmp_path: Path) -> None:
        # The AtCo shape: 0001_null_dataset ... 0585_null_dataset, consecutively
        # numbered but unrelated. Measured on the real export: 0.7%.
        recs = _records({f"{i:04d}_coll": _distinct_image(3100 + i) for i in range(10)}, tmp_path)
        assert session_corroboration(recs) == pytest.approx(0.0)

    def test_augmentation_copies_of_one_photo_cannot_inflate_the_score(self, tmp_path: Path) -> None:
        # One vote per SOURCE PHOTO, never per file: Roboflow copies of a single
        # photo are near-duplicates by construction and would score 100%.
        base = _distinct_image(3200)
        images = {}
        for photo in range(6):
            scene = _distinct_image(3300 + photo)
            for aug in range(4):
                images[f"{photo:04d}_coll.rf.{photo * 10 + aug:032x}"] = _shifted(scene, aug)
        assert session_corroboration(_records(images, tmp_path)) == pytest.approx(0.0)
        assert base is not None

    def test_empty_session_scores_zero(self) -> None:
        assert session_corroboration([]) == 0.0


class TestSessionIsBelievable:
    def test_corroborated_session_is_believed(self, tmp_path: Path) -> None:
        base = _distinct_image(3400)
        assert session_is_believable(_records({f"f_{i:04d}": _shifted(base, i) for i in range(8)}, tmp_path))

    def test_uncorroborated_session_is_rejected(self, tmp_path: Path) -> None:
        recs = _records({f"{i:04d}_coll": _distinct_image(3500 + i) for i in range(8)}, tmp_path)
        assert not session_is_believable(recs)

    def test_a_tiny_session_is_believed_without_a_vote(self, tmp_path: Path) -> None:
        recs = _records({f"{i:04d}_coll": _distinct_image(3600 + i) for i in range(2)}, tmp_path)
        assert session_is_believable(recs)

    def test_threshold_zero_believes_everything(self, tmp_path: Path) -> None:
        recs = _records({f"{i:04d}_coll": _distinct_image(3700 + i) for i in range(8)}, tmp_path)
        assert session_is_believable(recs, min_fraction=0.0)

    def test_default_threshold_separates_the_two_measured_cases(self) -> None:
        # Measured on the real roster: collections 0.0-0.7%, real sessions 61-64%.
        assert 0.01 < DEFAULT_SESSION_CORROBORATION < 0.6


class TestSessionCorroborationEndToEnd:
    def test_a_numbered_collection_does_not_collapse_into_one_group(self, tmp_path: Path) -> None:
        images = {f"{i:04d}_null_dataset": _distinct_image(4100 + i) for i in range(20)}
        root = _make_export(tmp_path / "coll", ATCO_CLASSES, images)
        report = build_merged_dataset(scans={"atco": _scan_and_hash(root, "atco")}, out_dir=tmp_path / "out")
        assert report["emitted_groups"] == 20
        assert report["sources"]["atco"]["filename_derived_sessions"] == 1
        assert report["sources"]["atco"]["filename_sessions_rejected_as_uncorroborated"] == 1
        assert report["sources"]["atco"]["filename_sessions_believed"] == 0

    def test_without_the_vote_the_same_collection_collapses(self, tmp_path: Path) -> None:
        # This is the pre-2026-09 behaviour that put 64% of all images in val.
        images = {f"{i:04d}_null_dataset": _distinct_image(4200 + i) for i in range(20)}
        root = _make_export(tmp_path / "coll", ATCO_CLASSES, images)
        with pytest.raises(ValueError, match="independent leakage groups"):
            build_merged_dataset(
                scans={"atco": _scan_and_hash(root, "atco")},
                out_dir=tmp_path / "out",
                session_corroboration_min=0.0,
            )

    def test_a_genuine_timelapse_is_still_kept_together(self, tmp_path: Path) -> None:
        # The guard must not cost us the Defect B protection it was built for.
        images: dict[str, np.ndarray] = {}
        for job in range(12):
            base = _distinct_image(4300 + job)
            for frame in range(5):
                images[f"job{job:02d}_f{frame:04d}"] = _shifted(base, frame)
        root = _make_export(tmp_path / "tl", ATCO_CLASSES, images)
        report = build_merged_dataset(scans={"atco": _scan_and_hash(root, "atco")}, out_dir=tmp_path / "out")
        assert report["emitted_groups"] == 12
        assert report["sources"]["atco"]["filename_sessions_believed"] == 12
        assert report["sources"]["atco"]["filename_sessions_rejected_as_uncorroborated"] == 0
        _assert_manifest_has_no_group_leakage(tmp_path / "out")


# --------------------------------------------------------------------------
# Group-aware splitting
# --------------------------------------------------------------------------


class TestSplitGroups:
    def test_default_ratios_are_60_25_15(self) -> None:
        # Val is deliberately fat: a 70/15/15 GROUP split measured out at
        # 89/4.9/6.1 by image count, and ~5% is too thin for threshold picking.
        assert DEFAULT_SPLIT_RATIOS == (0.60, 0.25, 0.15)
        assert sum(DEFAULT_SPLIT_RATIOS) == pytest.approx(1.0)

    def test_partitions_every_group_exactly_once(self) -> None:
        groups = [f"g{i:04d}" for i in range(100)]
        train, val, test = split_groups(groups, seed=1337)
        assert sorted(train + val + test) == sorted(groups)
        assert len(train) == 60 and len(val) == 25 and len(test) == 15

    def test_deterministic_for_a_seed_and_order_independent(self) -> None:
        groups = [f"g{i:04d}" for i in range(50)]
        assert split_groups(groups, seed=7) == split_groups(list(reversed(groups)), seed=7)

    def test_different_seeds_differ(self) -> None:
        groups = [f"g{i:04d}" for i in range(50)]
        assert split_groups(groups, seed=1) != split_groups(groups, seed=2)

    def test_bad_ratios_raise(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_groups(["a", "b"], seed=1, ratios=(0.5, 0.4, 0.4))


class TestAssertNoGroupOverlap:
    def test_passes_on_a_clean_split(self) -> None:
        assert_no_group_overlap({"train": ["a", "b"], "val": ["c"], "test": ["d"]})

    def test_raises_naming_both_splits(self) -> None:
        with pytest.raises(AssertionError, match="'train' and 'test'"):
            assert_no_group_overlap({"train": ["a"], "val": ["b"], "test": ["a"]})


# --------------------------------------------------------------------------
# Source roster
# --------------------------------------------------------------------------


class TestSourceRoster:
    def test_only_atco_and_stereovision_are_ingestable(self) -> None:
        # rf_defects and rf_failure are measured re-uploads of a 418-photo subset
        # of the AtCo set; they must not be reachable as CLI sources.
        assert sorted(SOURCE_SPEC_BY_KEY) == ["atco", "stereovision"]

    def test_neither_current_source_is_noncommercial(self) -> None:
        assert not any(s.noncommercial for s in SOURCE_SPEC_BY_KEY.values())

    def test_atco_points_at_the_yolo_export_not_a_reference_root(self) -> None:
        # datasets/raw ships a data.yaml, so it is an ingest source now.
        assert SOURCE_SPEC_BY_KEY["atco"].expected_classes == tuple(ATCO_CLASSES)
        assert SOURCE_SPEC_BY_KEY["stereovision"].expected_classes == tuple(STEREO_CLASSES)


# --------------------------------------------------------------------------
# Redundant-source guard (Defect D)
# --------------------------------------------------------------------------


class TestStemOverlapMatrix:
    def test_counts_shared_stems_with_both_directional_percentages(self) -> None:
        matrix = build_stem_overlap_matrix({"big": {"a", "b", "c", "d"}, "small": {"a", "b"}})
        cell = matrix["pairwise"]["big|small"]
        assert cell["n_shared_stems"] == 2
        assert cell["pct_of_big"] == pytest.approx(50.0)
        assert cell["pct_of_small"] == pytest.approx(100.0)

    def test_unique_stem_counts_are_reported_per_source(self) -> None:
        matrix = build_stem_overlap_matrix({"a": {"x", "y"}, "b": {"x"}})
        assert matrix["unique_source_stems"] == {"a": 2, "b": 1}

    def test_full_containment_is_listed_only_in_the_contained_direction(self) -> None:
        matrix = build_stem_overlap_matrix({"big": {"a", "b", "c", "d"}, "small": {"a", "b"}})
        entries = matrix["redundant_containments"]
        assert [e["source"] for e in entries] == ["small"]
        assert entries[0]["contained_in"] == "big"
        assert entries[0]["pct_of_source_already_present"] == pytest.approx(100.0)

    def test_disjoint_sources_produce_no_containments(self) -> None:
        matrix = build_stem_overlap_matrix({"a": {"x"}, "b": {"y"}})
        assert matrix["redundant_containments"] == []
        assert matrix["pairwise"]["a|b"]["n_shared_stems"] == 0

    def test_single_source_has_an_empty_pairwise_block(self) -> None:
        assert build_stem_overlap_matrix({"a": {"x"}})["pairwise"] == {}


class TestFindRedundantSources:
    def test_the_rf_failure_shape_is_caught(self) -> None:
        # 418 stems, all of them already in the AtCo set, hidden behind 8,853 files.
        stems = {str(i) for i in range(418)}
        found = find_redundant_sources(
            {"rf_failure": stems, "atco": stems | {str(i) for i in range(418, 1930)}},
            ["rf_failure", "atco"],
        )
        assert [c.source for c in found] == ["rf_failure"]
        assert found[0].contained_in == "atco"
        assert found[0].pct == pytest.approx(100.0)

    def test_the_real_atco_stereovision_pair_is_not_redundant(self) -> None:
        # Measured: 195 of stereovision's 2,909 photos are also in AtCo's 1,930.
        shared = {f"s{i}" for i in range(195)}
        stem_sets = {
            "atco": shared | {f"a{i}" for i in range(1735)},
            "stereovision": shared | {f"v{i}" for i in range(2714)},
        }
        assert find_redundant_sources(stem_sets, ["atco", "stereovision"]) == []

    def test_containment_in_a_reference_dataset_counts(self) -> None:
        stems = {str(i) for i in range(20)}
        found = find_redundant_sources({"newsrc": stems, "ondisk": stems}, ["newsrc"])
        assert [c.contained_in for c in found] == ["ondisk"]

    def test_containment_in_a_source_that_is_itself_excluded_does_not_count(self) -> None:
        # Being a copy of data that is not in the merge either is not redundancy.
        stems = {str(i) for i in range(20)}
        assert find_redundant_sources(
            {"kept": stems, "dropped": stems}, ["kept"], compare_against=["kept"]
        ) == []

    def test_just_below_the_threshold_passes(self) -> None:
        own = {str(i) for i in range(100)}
        other = {str(i) for i in range(89)}  # 89% < 90%
        assert find_redundant_sources({"a": own, "b": other}, ["a"]) == []

    def test_exactly_at_the_threshold_is_redundant(self) -> None:
        own = {str(i) for i in range(100)}
        other = {str(i) for i in range(int(REDUNDANT_SOURCE_PCT))}
        assert [c.source for c in find_redundant_sources({"a": own, "b": other}, ["a"])] == ["a"]

    def test_source_stem_sets_strips_the_roboflow_suffix(self, tmp_path: Path) -> None:
        root = _spaced_export(tmp_path / "sv", STEREO_CLASSES, n=3, image_offset=0, rf_suffix=True)
        sets = source_stem_sets({"stereovision": scan_source(root, "stereovision")})
        assert sets["stereovision"] == {"s0000", "s0100", "s0200"}


class TestAssertNoRedundantSources:
    def test_names_both_sources_and_the_percentage(self) -> None:
        stems = {str(i) for i in range(10)}
        found = find_redundant_sources({"dupe": stems, "orig": stems}, ["dupe"])
        with pytest.raises(ValueError) as exc:
            assert_no_redundant_sources(found)
        message = str(exc.value)
        assert "REDUNDANT SOURCE" in message
        assert "'dupe'" in message and "'orig'" in message and "100.0%" in message

    def test_allow_flag_lets_it_through(self) -> None:
        stems = {str(i) for i in range(10)}
        found = find_redundant_sources({"dupe": stems, "orig": stems}, ["dupe"])
        assert assert_no_redundant_sources(found, allow=True) is None

    def test_nothing_redundant_is_a_no_op(self) -> None:
        assert assert_no_redundant_sources([]) is None


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


class TestScanSource:
    def test_missing_root_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            scan_source(tmp_path / "nope", "atco")

    def test_missing_data_yaml_fails_loudly(self, tmp_path: Path) -> None:
        (tmp_path / "train" / "images").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="data.yaml"):
            scan_source(tmp_path, "atco")

    def test_data_yaml_without_names_fails_loudly(self, tmp_path: Path) -> None:
        (tmp_path / "train" / "images").mkdir(parents=True)
        (tmp_path / "data.yaml").write_text("nc: 2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no usable 'names'"):
            scan_source(tmp_path, "atco")

    def test_no_images_fails_loudly(self, tmp_path: Path) -> None:
        _make_export(tmp_path, ATCO_CLASSES, {})
        with pytest.raises(FileNotFoundError, match="No images found"):
            scan_source(tmp_path, "atco")

    def test_reads_classes_records_and_labels(self, tmp_path: Path) -> None:
        _spaced_export(tmp_path, ATCO_CLASSES, n=3, image_offset=0)
        scan = scan_source(tmp_path, "atco")
        assert scan.class_names == ATCO_CLASSES
        assert len(scan.records) == 3
        assert scan.n_label_files == 3 and scan.n_label_rows == 3
        assert scan.rf_fallback_count == 3  # no ".rf." suffix on these fixtures

    def test_roboflow_suffix_is_recognized(self, tmp_path: Path) -> None:
        _spaced_export(tmp_path, STEREO_CLASSES, n=3, image_offset=0, rf_suffix=True)
        scan = scan_source(tmp_path, "stereovision")
        assert scan.rf_fallback_count == 0
        assert all(r.rf_matched for r in scan.records)
        assert scan.records[0].source_stem == "s0000"

    def test_missing_label_file_is_counted(self, tmp_path: Path) -> None:
        _spaced_export(tmp_path, ATCO_CLASSES, n=2, image_offset=0)
        next((tmp_path / "train" / "labels").iterdir()).unlink()
        scan = scan_source(tmp_path, "atco")
        assert scan.n_missing_labels == 1

    def test_reference_scan_needs_no_data_yaml(self, tmp_path: Path) -> None:
        nested = tmp_path / "Warping"
        nested.mkdir(parents=True)
        assert cv2.imwrite(str(nested / "a.png"), _distinct_image(1))
        scan = scan_source(tmp_path, "fdm_raw", is_reference=True)
        assert len(scan.records) == 1 and scan.records[0].is_reference


class TestReadDataYamlLicense:
    def test_reads_the_roboflow_block(self, tmp_path: Path) -> None:
        (tmp_path / "data.yaml").write_text(
            "names: ['a']\nroboflow:\n  license: CC BY 4.0\n", encoding="utf-8"
        )
        assert read_data_yaml_license(tmp_path) == "CC BY 4.0"

    def test_missing_file_or_key_is_none_not_an_error(self, tmp_path: Path) -> None:
        assert read_data_yaml_license(tmp_path) is None
        (tmp_path / "data.yaml").write_text("names: ['a']\n", encoding="utf-8")
        assert read_data_yaml_license(tmp_path) is None


class TestOutputFilename:
    def test_source_tag_prefixes_the_name(self) -> None:
        rec = ImageRecord(source_key="atco", path=Path("a/b.png"), rel_path="train/images/b.png", is_reference=False)
        assert output_filename(rec, "both", set()) == "atco__b.png"
        assert output_filename(rec, "filename", set()) == "atco__b.png"
        assert output_filename(rec, "manifest", set()) == "b.png"

    def test_collisions_get_a_suffix(self) -> None:
        rec = ImageRecord(source_key="atco", path=Path("a/b.png"), rel_path="train/images/b.png", is_reference=False)
        used: set[str] = set()
        assert output_filename(rec, "both", used) == "atco__b.png"
        assert output_filename(rec, "both", used) == "atco__b_1.png"


# --------------------------------------------------------------------------
# End-to-end merge on synthetic exports
# --------------------------------------------------------------------------


def _manifest_rows(out_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]


def _assert_manifest_has_no_group_leakage(out_dir: Path) -> None:
    splits_by_group: dict[str, set[str]] = {}
    for row in _manifest_rows(out_dir):
        splits_by_group.setdefault(row["group"], set()).add(row["split"])
    leaked = {g: s for g, s in splits_by_group.items() if len(s) > 1}
    assert not leaked, f"leakage groups straddling splits: {leaked}"


def _two_source_scans(tmp_path: Path, n: int = 14):
    at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=n, image_offset=0, stem_prefix="a")
    sv = _spaced_export(tmp_path / "sv", STEREO_CLASSES, n=n, image_offset=100, rf_suffix=True, stem_prefix="v")
    return {"atco": _scan_and_hash(at, "atco"), "stereovision": _scan_and_hash(sv, "stereovision")}


class TestBuildMergedDataset:
    def test_writes_a_complete_leak_free_dataset(self, tmp_path: Path) -> None:
        scans = _two_source_scans(tmp_path)
        out = tmp_path / "out"
        report = build_merged_dataset(scans=scans, out_dir=out)

        for split in ("train", "val", "test"):
            assert (out / split / "images").is_dir() and (out / split / "labels").is_dir()
            assert report["splits"][split]["n_images"] > 0
        assert report["group_overlap_check"].startswith("PASS")
        _assert_manifest_has_no_group_leakage(out)

        n_written = sum(len(list((out / s / "images").iterdir())) for s in ("train", "val", "test"))
        assert n_written == report["emitted_images"] == 28

    def test_every_image_has_a_label_file(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        build_merged_dataset(scans=_two_source_scans(tmp_path), out_dir=out)
        for split in ("train", "val", "test"):
            images = sorted(p.stem for p in (out / split / "images").iterdir())
            labels = sorted(p.stem for p in (out / split / "labels").iterdir())
            assert images == labels

    def test_data_yaml_carries_the_unified_class_list(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        build_merged_dataset(scans=_two_source_scans(tmp_path), out_dir=out)
        text = (out / "data.yaml").read_text(encoding="utf-8")
        assert f"nc: {len(UNIFIED_CLASSES)}" in text
        for cname in UNIFIED_CLASSES:
            assert f"'{cname}'" in text
        assert "CC BY 4.0" in text  # license provenance is recorded

    def test_classes_are_harmonized_into_the_unified_space(self, tmp_path: Path) -> None:
        # atco class 0 == 'error extrusion'; stereovision class 0 == 'Bed Adhesion'.
        out = tmp_path / "out"
        report = build_merged_dataset(scans=_two_source_scans(tmp_path), out_dir=out)
        totals = {c: sum(report["splits"][s]["instances_per_class"][c] for s in ("train", "val", "test")) for c in UNIFIED_CLASSES}
        assert totals["error_extrusion"] == 14
        assert totals["bed_adhesion"] == 14
        assert totals["zits"] == 0

    def test_provenance_survives_in_filenames_and_manifest(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        build_merged_dataset(scans=_two_source_scans(tmp_path), out_dir=out)
        rows = _manifest_rows(out)
        assert len(rows) == 28
        assert {r["source"] for r in rows} == {"atco", "stereovision"}
        for row in rows:
            assert Path(row["output"]).name.startswith(row["source"] + "__")
            assert row["content_sha256"] and row["dhash"] and row["group"]

    def test_source_tag_manifest_only_drops_the_filename_prefix(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        build_merged_dataset(scans=_two_source_scans(tmp_path), out_dir=out, source_tag="manifest")
        rows = _manifest_rows(out)
        assert all(not Path(r["output"]).name.startswith(r["source"] + "__") for r in rows)

    def test_determinism(self, tmp_path: Path) -> None:
        scans = _two_source_scans(tmp_path)
        a = build_merged_dataset(scans=scans, out_dir=tmp_path / "a", seed=99)
        b = build_merged_dataset(scans=scans, out_dir=tmp_path / "b", seed=99)
        assert a["splits"] == b["splits"]
        assert _manifest_rows(tmp_path / "a") == _manifest_rows(tmp_path / "b")

    def test_dry_run_writes_nothing_but_still_counts_classes(self, tmp_path: Path) -> None:
        scans = _two_source_scans(tmp_path)
        report = build_merged_dataset(scans=scans, out_dir=tmp_path / "out", dry_run=True)
        assert not (tmp_path / "out").exists()
        assert report["emitted_images"] == 28
        # The whole point of --dry-run is inspecting balance BEFORE a multi-GB copy.
        wet = build_merged_dataset(scans=scans, out_dir=tmp_path / "wet")
        assert report["splits"] == wet["splits"]

    def test_reports_empty_and_unevaluable_classes(self, tmp_path: Path) -> None:
        # atco-only, every label class 0: nothing labels the other eight classes.
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=14, image_offset=0, stem_prefix="a")
        report = build_merged_dataset(scans={"atco": _scan_and_hash(at, "atco")}, out_dir=tmp_path / "out")
        empty = next(w for w in report["warnings"] if w.startswith("EMPTY CLASSES"))
        for cname in ("spaghetti", "layer_separation", "bed_adhesion", "blob_of_death", "head"):
            assert f"'{cname}'" in empty
        assert "'error_extrusion'" not in empty

    def test_too_few_groups_fails_loudly(self, tmp_path: Path) -> None:
        # Consecutive frame numbers + one repeated scene: everything collapses
        # into a single indivisible group, which must NOT be split.
        base = _distinct_image(500)
        images = {f"frame_{i:04d}": _shifted(base, i) for i in range(6)}
        root = _make_export(tmp_path / "tl", ATCO_CLASSES, images)
        scans = {"atco": _scan_and_hash(root, "atco")}
        with pytest.raises(ValueError, match="independent leakage groups"):
            build_merged_dataset(scans=scans, out_dir=tmp_path / "out")


class TestHonestSizeReporting:
    def test_three_numbers_are_reported_separately_per_source(self, tmp_path: Path) -> None:
        # 6 source photos x 3 Roboflow augmentation copies each = 18 files.
        images: dict[str, np.ndarray] = {}
        for photo in range(6):
            base = _distinct_image(2000 + photo)
            for aug in range(3):
                images[f"p{photo * 100:04d}.rf.{photo * 10 + aug:032x}"] = _shifted(base, aug)
        for i in range(12):  # enough independent groups to split
            images[f"solo{i}_x{i * 100:04d}"] = _distinct_image(2100 + i)
        root = _make_export(tmp_path / "at", ATCO_CLASSES, images)
        report = build_merged_dataset(scans={"atco": _scan_and_hash(root, "atco")}, out_dir=tmp_path / "out")

        block = report["sources"]["atco"]
        assert block["n_files"] == 30
        assert block["n_unique_source_stems"] == 18  # 6 augmented + 12 solo
        assert block["n_groups"] == 18
        assert block["files_per_unique_source_photo"] == pytest.approx(30 / 18, abs=0.01)

        size = report["dataset_size"]
        assert size["per_source"]["atco"] == {
            "role": "input",
            "included_in_output": True,
            "raw_files": 30,
            "unique_source_photos": 18,
            "independent_groups": 18,
        }
        assert size["included_total"]["raw_files"] == 30
        assert size["included_total"]["unique_source_photos"] == 18
        assert size["included_total"]["independent_groups"] == 18

    def test_overlap_matrix_is_always_in_the_report(self, tmp_path: Path) -> None:
        report = build_merged_dataset(scans=_two_source_scans(tmp_path), out_dir=tmp_path / "out")
        overlap = report["source_stem_overlap"]
        assert overlap["threshold_pct"] == REDUNDANT_SOURCE_PCT
        assert overlap["unique_source_stems"] == {"atco": 14, "stereovision": 14}
        assert overlap["pairwise"]["atco|stereovision"]["n_shared_stems"] == 0
        assert overlap["redundant_included_sources"] == []
        assert overlap["allow_redundant_sources"] is False


class TestRedundantSourceGuard:
    def _redundant_pair(self, tmp_path: Path):
        # Same stems in both exports: the second adds no new source photographs,
        # exactly the rf_failure-vs-datasets/raw shape.
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=14, image_offset=0, stem_prefix="a")
        sv = _spaced_export(tmp_path / "sv", STEREO_CLASSES, n=14, image_offset=100, stem_prefix="a")
        return {"atco": _scan_and_hash(at, "atco"), "stereovision": _scan_and_hash(sv, "stereovision")}

    def test_merge_is_refused_before_anything_is_copied(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        with pytest.raises(ValueError) as exc:
            build_merged_dataset(scans=self._redundant_pair(tmp_path), out_dir=out)
        message = str(exc.value)
        assert "REDUNDANT SOURCE" in message
        assert "'atco'" in message and "'stereovision'" in message and "100.0%" in message
        assert not out.exists(), "the guard must fire before a single file is written"

    def test_allow_flag_merges_it_and_warns(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        report = build_merged_dataset(
            scans=self._redundant_pair(tmp_path), out_dir=out, allow_redundant_sources=True
        )
        assert report["emitted_images"] == 28
        assert any(w.startswith("REDUNDANT SOURCE MERGED ANYWAY") for w in report["warnings"])
        overlap = report["source_stem_overlap"]
        assert overlap["allow_redundant_sources"] is True
        assert {c["source"] for c in overlap["redundant_included_sources"]} == {"atco", "stereovision"}
        _assert_manifest_has_no_group_leakage(out)

    def test_a_redundant_source_that_is_excluded_does_not_block_the_merge(self, tmp_path: Path) -> None:
        # Only sources headed for the output can be redundant.
        report = build_merged_dataset(
            scans=self._redundant_pair(tmp_path), out_dir=tmp_path / "out", include=["atco"]
        )
        assert report["license_gating"]["included_sources"] == ["atco"]
        assert report["source_stem_overlap"]["redundant_included_sources"] == []
        # ...but the overlap is still reported, unconditionally.
        assert report["source_stem_overlap"]["pairwise"]["atco|stereovision"]["n_shared_stems"] == 14

    def test_containment_in_a_reference_dataset_also_blocks(self, tmp_path: Path) -> None:
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=14, image_offset=0, stem_prefix="a")
        ref = tmp_path / "ondisk"
        ref.mkdir()
        for i in range(14):
            assert cv2.imwrite(str(ref / f"a{i * 100:04d}.png"), _distinct_image(500 + i))
        scans = {
            "atco": _scan_and_hash(at, "atco"),
            "ondisk": _scan_and_hash(ref, "ondisk", is_reference=True),
        }
        with pytest.raises(ValueError, match="REDUNDANT SOURCE"):
            build_merged_dataset(scans=scans, out_dir=tmp_path / "out")


class TestCrossDatasetDuplicates:
    def _sources_with_a_shared_image(self, tmp_path: Path, shared: np.ndarray, mutate=None):
        at_images = {f"a{i * 100:04d}": _distinct_image(i) for i in range(14)}
        sv_images = {f"v{i * 100:04d}": _distinct_image(100 + i) for i in range(14)}
        at_images["shared9900"] = shared
        sv_images["shared9900"] = shared if mutate is None else mutate(shared)
        at = _make_export(tmp_path / "at", ATCO_CLASSES, at_images)
        sv = _make_export(tmp_path / "sv", STEREO_CLASSES, sv_images)
        return {"atco": _scan_and_hash(at, "atco"), "stereovision": _scan_and_hash(sv, "stereovision")}

    def test_exact_cross_dataset_duplicate_is_detected_and_deduplicated(self, tmp_path: Path) -> None:
        scans = self._sources_with_a_shared_image(tmp_path, _distinct_image(777))
        report = build_merged_dataset(scans=scans, out_dir=tmp_path / "out")
        dupes = report["duplicates"]
        assert dupes["cross_dataset_exact_sets"] >= 1
        assert "atco|stereovision" in dupes["collision_matrix"]
        assert dupes["collision_matrix"]["atco|stereovision"]["exact"] == 2
        assert dupes["images_dropped_as_exact_duplicates"] == 1
        assert report["emitted_images"] == 29  # 30 scanned, 1 exact dupe collapsed
        assert any(w.startswith("CROSS-DATASET COLLISION") for w in report["warnings"])

    def test_near_duplicate_across_datasets_is_detected_and_grouped(self, tmp_path: Path) -> None:
        scans = self._sources_with_a_shared_image(tmp_path, _distinct_image(778), mutate=_shifted)
        report = build_merged_dataset(scans=scans, out_dir=tmp_path / "out")
        assert report["duplicates"]["cross_dataset_near_sets"] >= 1
        assert report["duplicates"]["cross_dataset_exact_sets"] == 0  # pixels differ
        # Both copies are kept (not pixel-identical) but must share a split.
        rows = _manifest_rows(tmp_path / "out")
        shared = [r for r in rows if Path(r["source_path"]).stem == "shared9900"]
        assert len(shared) == 2
        assert shared[0]["group"] == shared[1]["group"]
        assert shared[0]["split"] == shared[1]["split"]
        _assert_manifest_has_no_group_leakage(tmp_path / "out")

    def test_reference_dataset_collision_is_reported_without_emitting_it(self, tmp_path: Path) -> None:
        shared = _distinct_image(779)
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=14, image_offset=0, stem_prefix="a")
        assert cv2.imwrite(str(at / "train" / "images" / "extra9900.png"), shared)
        (at / "train" / "labels" / "extra9900.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        ref = tmp_path / "fdm_raw" / "Warping"
        ref.mkdir(parents=True)
        assert cv2.imwrite(str(ref / "Image_20230417093015123.png"), shared)

        scans = {
            "atco": _scan_and_hash(at, "atco"),
            "fdm_raw": _scan_and_hash(tmp_path / "fdm_raw", "fdm_raw", is_reference=True),
        }
        out = tmp_path / "out"
        report = build_merged_dataset(scans=scans, out_dir=out)
        assert "atco|fdm_raw" in report["duplicates"]["collision_matrix"]
        assert report["sources"]["fdm_raw"]["role"] == "reference"
        assert report["sources"]["fdm_raw"]["included_in_output"] is False
        assert {r["source"] for r in _manifest_rows(out)} == {"atco"}
        assert report["emitted_images"] == 15

    def test_drop_reference_duplicates_removes_the_overlap(self, tmp_path: Path) -> None:
        shared = _distinct_image(780)
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=14, image_offset=0, stem_prefix="a")
        assert cv2.imwrite(str(at / "train" / "images" / "extra9900.png"), shared)
        (at / "train" / "labels" / "extra9900.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        ref = tmp_path / "fdm_raw"
        ref.mkdir(parents=True)
        assert cv2.imwrite(str(ref / "dup.png"), shared)

        scans = {
            "atco": _scan_and_hash(at, "atco"),
            "fdm_raw": _scan_and_hash(ref, "fdm_raw", is_reference=True),
        }
        report = build_merged_dataset(scans=scans, out_dir=tmp_path / "out", drop_reference_duplicates=True)
        assert report["duplicates"]["images_dropped_as_reference_duplicates"] == 1
        assert report["emitted_images"] == 14


class TestLicenseGating:
    def _permissive_plus_nc(self, tmp_path: Path):
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=12, image_offset=0, stem_prefix="a")
        nc = _spaced_export(tmp_path / "nc", LAYER_SPLIT_CLASSES, n=12, image_offset=200, stem_prefix="n")
        return {"atco": _scan_and_hash(at, "atco"), "ncsource": _scan_and_hash(nc, "ncsource")}

    def test_no_license_warning_when_no_source_is_noncommercial(self, tmp_path: Path) -> None:
        # Both real sources are permissive, so the gating flag is a no-op today
        # and must not imply that anything was excluded.
        report = build_merged_dataset(scans=_two_source_scans(tmp_path), out_dir=tmp_path / "out")
        assert report["license_gating"]["excluded_sources"] == []
        assert not any(w.startswith("LICENSE GATING") for w in report["warnings"])
        assert not any("non-commercial sources" in w for w in report["warnings"])

    def test_no_license_warning_when_a_source_is_excluded_for_other_reasons(self, tmp_path: Path) -> None:
        report = build_merged_dataset(
            scans=_two_source_scans(tmp_path), out_dir=tmp_path / "out", include=["atco"]
        )
        assert report["license_gating"]["excluded_sources"] == ["stereovision"]
        assert not any(w.startswith("LICENSE GATING") for w in report["warnings"])

    def test_noncommercial_source_is_excluded_by_default(self, tmp_path: Path, noncommercial_spec) -> None:
        out = tmp_path / "out"
        report = build_merged_dataset(scans=self._permissive_plus_nc(tmp_path), out_dir=out)
        gating = report["license_gating"]
        assert gating["exclude_noncommercial"] is True
        assert gating["included_sources"] == ["atco"]
        assert gating["excluded_sources"] == ["ncsource"]
        assert gating["included_licenses"] == ["MIT"]
        assert {r["source"] for r in _manifest_rows(out)} == {"atco"}
        assert any(w.startswith("LICENSE GATING") for w in report["warnings"])

    def test_still_scanned_and_analyzed_when_excluded(self, tmp_path: Path, noncommercial_spec) -> None:
        report = build_merged_dataset(scans=self._permissive_plus_nc(tmp_path), out_dir=tmp_path / "out")
        assert report["sources"]["ncsource"]["n_files"] == 12
        assert report["sessions"]["ncsource"]["raw_frames"] == 12

    def test_included_with_the_flag_off(self, tmp_path: Path, noncommercial_spec) -> None:
        out = tmp_path / "out"
        report = build_merged_dataset(
            scans=self._permissive_plus_nc(tmp_path), out_dir=out, exclude_noncommercial=False
        )
        assert report["license_gating"]["included_sources"] == ["atco", "ncsource"]
        assert {r["source"] for r in _manifest_rows(out)} == {"atco", "ncsource"}
        assert any("non-commercial sources" in w for w in report["warnings"])

    def test_license_declared_in_the_export_is_reported_and_mismatches_warn(self, tmp_path: Path) -> None:
        scans = _two_source_scans(tmp_path)
        (tmp_path / "sv" / "data.yaml").write_text(
            (tmp_path / "sv" / "data.yaml").read_text(encoding="utf-8") + "roboflow:\n  license: MIT\n",
            encoding="utf-8",
        )
        scans["stereovision"] = _scan_and_hash(tmp_path / "sv", "stereovision")
        report = build_merged_dataset(scans=scans, out_dir=tmp_path / "out")
        assert report["sources"]["stereovision"]["license_declared_in_data_yaml"] == "MIT"
        assert any(w.startswith("LICENSE MISMATCH for 'stereovision'") for w in report["warnings"])


class TestSessionReporting:
    def test_timelapse_source_collapses_to_few_sessions_and_warns(self, tmp_path: Path) -> None:
        # 3 print jobs x 5 near-identical frames each, plus 12 independent
        # photos so the split still has enough groups.
        images: dict[str, np.ndarray] = {}
        for job in range(3):
            base = _distinct_image(600 + job)
            for frame in range(5):
                images[f"job{job}_frame_{frame:04d}"] = _shifted(base, frame)
        for i in range(12):
            images[f"solo{i}_x{i * 100:04d}"] = _distinct_image(700 + i)
        root = _make_export(tmp_path / "tl", ATCO_CLASSES, images)
        scans = {"atco": _scan_and_hash(root, "atco")}
        report = build_merged_dataset(scans=scans, out_dir=tmp_path / "out")

        block = report["sources"]["atco"]
        assert block["raw_frame_count"] == 27
        # 3 timelapse jobs + 12 independent photos == 15 real sessions.
        assert block["estimated_independent_sessions"] == 15
        assert block["frames_per_session"] == pytest.approx(27 / 15)
        _assert_manifest_has_no_group_leakage(tmp_path / "out")

    def test_a_heavier_timelapse_raises_the_warning(self, tmp_path: Path) -> None:
        images: dict[str, np.ndarray] = {}
        for job in range(12):
            base = _distinct_image(900 + job)
            for frame in range(6):  # 6 frames per job >> TIMELAPSE_FRAMES_PER_GROUP
                images[f"job{job:02d}_f{frame:04d}"] = _shifted(base, frame)
        root = _make_export(tmp_path / "tl", ATCO_CLASSES, images)
        report = build_merged_dataset(
            scans={"atco": _scan_and_hash(root, "atco")}, out_dir=tmp_path / "out"
        )
        assert any(w.startswith("TIMELAPSE SUSPECTED") for w in report["warnings"])
        assert report["sources"]["atco"]["estimated_independent_sessions"] == 12

    def test_lopsided_group_sizes_raise_the_split_imbalance_warning(self, tmp_path: Path) -> None:
        # One enormous print job plus many single photos: an honest GROUP split
        # still leaves val/test with very few images.
        images: dict[str, np.ndarray] = {}
        # Low dynamic range so 200 distinct brightness shifts never clip: every
        # frame has different pixels (not an exact duplicate) but the same dhash.
        big = _distinct_image(1200) // 4
        for frame in range(200):
            images[f"big_f{frame:04d}"] = _shifted(big, frame)
        for i in range(14):
            images[f"solo{i}_x{i * 100:04d}"] = _distinct_image(1300 + i)
        root = _make_export(tmp_path / "sk", ATCO_CLASSES, images)
        report = build_merged_dataset(
            scans={"atco": _scan_and_hash(root, "atco")}, out_dir=tmp_path / "out"
        )
        assert any(w.startswith("SPLIT IMBALANCE") for w in report["warnings"])
        # ...and the split is still leak-free, which is the point of tolerating it.
        _assert_manifest_has_no_group_leakage(tmp_path / "out")

    def test_frames_of_one_job_never_straddle_a_split(self, tmp_path: Path) -> None:
        images: dict[str, np.ndarray] = {}
        for job in range(14):
            base = _distinct_image(800 + job)
            for frame in range(4):
                images[f"job{job:02d}_f{frame:04d}"] = _shifted(base, frame)
        root = _make_export(tmp_path / "tl", ATCO_CLASSES, images)
        scans = {"atco": _scan_and_hash(root, "atco")}
        out = tmp_path / "out"
        report = build_merged_dataset(scans=scans, out_dir=out)
        assert report["emitted_groups"] == 14

        split_of_job: dict[str, str] = {}
        for row in _manifest_rows(out):
            job = Path(row["source_path"]).stem.split("_")[0]
            assert split_of_job.setdefault(job, row["split"]) == row["split"], f"{job} straddles splits"
        _assert_manifest_has_no_group_leakage(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _empty_reference(tmp_path: Path, name: str = "ref") -> Path:
    root = tmp_path / name
    (root / "x").mkdir(parents=True)
    assert cv2.imwrite(str(root / "x" / "refphoto.png"), _distinct_image(4000))
    return root


class TestMain:
    def test_missing_path_exits_with_a_clear_message(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--atco", str(tmp_path / "not_here"), "--out", str(tmp_path / "out")])
        assert "does not exist" in str(exc.value)

    def test_no_source_exits_with_a_clear_message(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["--out", str(tmp_path / "out")])
        assert "no source given" in str(exc.value)

    def test_the_dropped_flags_are_gone(self, tmp_path: Path) -> None:
        # rf_defects/rf_failure were 100% re-uploads; there is no flag for them,
        # and the old --printfail/--fdmnc4 sources no longer exist.
        for flag in ("--printfail", "--fdmnc4", "--rf-defects", "--rf-failure"):
            with pytest.raises(SystemExit):
                main([flag, str(tmp_path), "--out", str(tmp_path / "out")])

    def test_end_to_end_run(self, tmp_path: Path) -> None:
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=14, image_offset=0, stem_prefix="a")
        sv = _spaced_export(tmp_path / "sv", STEREO_CLASSES, n=14, image_offset=100, rf_suffix=True, stem_prefix="v")
        out = tmp_path / "out"

        main([
            "--atco", str(at),
            "--stereovision", str(sv),
            "--reference", str(_empty_reference(tmp_path)),
            "--out", str(out),
            "--workers", "1",
            "--force",
        ])

        report = json.loads((out / "split_report.json").read_text(encoding="utf-8"))
        assert report["group_overlap_check"].startswith("PASS")
        assert report["emitted_images"] == 28
        assert sorted(report["sources"]) == ["atco", "ref", "stereovision"]
        assert report["sources"]["ref"]["role"] == "reference"
        assert report["ratios"] == {"train": 0.60, "val": 0.25, "test": 0.15}
        assert (out / "data.yaml").is_file() and (out / "manifest.jsonl").is_file()
        _assert_manifest_has_no_group_leakage(out)

    def test_dry_run_writes_only_the_report(self, tmp_path: Path) -> None:
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=14, image_offset=0, stem_prefix="a")
        out = tmp_path / "out"
        main(["--atco", str(at), "--reference", str(_empty_reference(tmp_path)), "--out", str(out),
              "--workers", "1", "--dry-run"])
        assert (out / "split_report.json").is_file()
        assert not (out / "train").exists()
        assert not (out / "data.yaml").exists()

    def test_a_path_given_as_both_source_and_reference_is_scanned_once(self, tmp_path: Path) -> None:
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=14, image_offset=0, stem_prefix="a")
        out = tmp_path / "out"
        main(["--atco", str(at), "--reference", str(at), "--out", str(out), "--workers", "1", "--dry-run"])
        report = json.loads((out / "split_report.json").read_text(encoding="utf-8"))
        assert sorted(report["sources"]) == ["atco"]

    def test_redundant_source_exits_with_both_names_and_the_percentage(self, tmp_path: Path) -> None:
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=14, image_offset=0, stem_prefix="a")
        sv = _spaced_export(tmp_path / "sv", STEREO_CLASSES, n=14, image_offset=100, stem_prefix="a")
        out = tmp_path / "out"
        with pytest.raises(SystemExit) as exc:
            main(["--atco", str(at), "--stereovision", str(sv), "--out", str(out), "--workers", "1"])
        message = str(exc.value)
        assert "REDUNDANT SOURCE" in message and "100.0%" in message
        assert not (out / "train").exists()

    def test_allow_redundant_sources_flag_lets_the_run_through(self, tmp_path: Path) -> None:
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=14, image_offset=0, stem_prefix="a")
        sv = _spaced_export(tmp_path / "sv", STEREO_CLASSES, n=14, image_offset=100, stem_prefix="a")
        out = tmp_path / "out"
        main(["--atco", str(at), "--stereovision", str(sv), "--out", str(out), "--workers", "1",
              "--allow-redundant-sources"])
        report = json.loads((out / "split_report.json").read_text(encoding="utf-8"))
        assert report["emitted_images"] == 28
        assert any(w.startswith("REDUNDANT SOURCE MERGED ANYWAY") for w in report["warnings"])
        assert MIN_GROUPS_FOR_SPLIT <= report["emitted_groups"]

    def test_overlap_matrix_is_printed_and_persisted(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        at = _spaced_export(tmp_path / "at", ATCO_CLASSES, n=14, image_offset=0, stem_prefix="a")
        sv = _spaced_export(tmp_path / "sv", STEREO_CLASSES, n=14, image_offset=100, stem_prefix="v")
        out = tmp_path / "out"
        main(["--atco", str(at), "--stereovision", str(sv), "--out", str(out), "--workers", "1", "--dry-run"])
        printed = capsys.readouterr().out
        assert "SOURCE-PHOTO OVERLAP MATRIX" in printed
        assert "EFFECTIVE SIZE" in printed
        report = json.loads((out / "split_report.json").read_text(encoding="utf-8"))
        assert report["source_stem_overlap"]["pairwise"]["atco|stereovision"]["n_shared_stems"] == 0
