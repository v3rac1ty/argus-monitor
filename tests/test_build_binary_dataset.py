"""Unit tests for training/build_binary_dataset.py's pure dedup/clustering/
splitting logic.

These tests build small synthetic fixtures (hand-crafted `HfRecord`s and
in-memory numpy images) and never touch the real Hugging Face dataset or the
network -- `fetch_hf_parquet_urls`, `download_hf_split_parquets`,
`load_rows_from_parquet` and `gather_hf_records` are never called. The
split/cluster core (`plan_binary_split` and its helpers) takes plain
in-memory records and never calls a record's `loader`, which is exactly what
makes it testable without any I/O; several tests below assert that directly
by wiring a `loader` that raises if it's ever invoked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest

from training.build_binary_dataset import (
    CLASS_NAMES,
    DEFAULT_SPLIT_RATIOS,
    MIN_CLUSTERS_PER_SPLIT_CLASS,
    MIN_INDEPENDENT_SCENES_FOR_TRUST,
    SPLIT_NAMES,
    BinarySplitPlan,
    HfRecord,
    assert_no_cluster_overlap,
    assign_clusters_to_splits,
    build_dataset,
    build_evaluation_validity_warning,
    cap_clusters_per_class,
    class_name_for_label,
    cluster_by_phash,
    content_hash,
    dedupe_exact_duplicates,
    dhash,
    find_mixed_label_clusters,
    materialize_split,
    plan_binary_split,
    straddling_cluster_count,
    summarize_cluster_sizes,
)


def _no_call_loader(uid: str) -> Callable[[], np.ndarray]:
    """A loader that fails the test loudly if the pure split/cluster logic
    ever tries to decode an image -- it shouldn't need to."""

    def _load() -> np.ndarray:
        raise AssertionError(f"loader for {uid!r} should not be called by pure planning logic")

    return _load


def _rec(uid: str, orig_split: str, label: int, content_hash: str = "c", phash: int = 0, output_stem: str | None = None) -> HfRecord:
    return HfRecord(
        uid=uid,
        orig_split=orig_split,
        label=label,
        content_hash=content_hash,
        phash=phash,
        loader=_no_call_loader(uid),
        output_stem=output_stem or uid.replace(":", "_"),
    )


def _spread_hash(i: int) -> int:
    """A deterministic, well-dispersed 64-bit value -- used when a test needs
    many phash values that are very unlikely to accidentally cluster with
    each other (unlike small sequential ints, which are all close in Hamming
    distance). Offset by 1 so i=0 never collides with the phash=0 fixtures
    other tests deliberately use."""
    return ((i + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)


# --------------------------------------------------------------------------
# class_name_for_label -- the 0/1 -> normal/failure mapping
# --------------------------------------------------------------------------


class TestClassNameForLabel:
    def test_zero_is_normal(self) -> None:
        assert class_name_for_label(0) == "normal"

    def test_one_is_failure(self) -> None:
        assert class_name_for_label(1) == "failure"

    def test_unrecognized_label_raises(self) -> None:
        with pytest.raises(ValueError):
            class_name_for_label(2)

    def test_class_names_are_sorted_failure_first(self) -> None:
        # index 0 == failure, index 1 == normal -- downstream (training,
        # ONNX export, inference) all derive class order from this.
        assert CLASS_NAMES == ("failure", "normal")


# --------------------------------------------------------------------------
# dedupe_exact_duplicates
# --------------------------------------------------------------------------


class TestDedupeExactDuplicates:
    def test_no_duplicates_keeps_everything(self) -> None:
        records = [_rec(f"r{i}", "train", 0, content_hash=f"h{i}") for i in range(10)]
        kept, dropped = dedupe_exact_duplicates(records)
        assert dropped == 0
        assert len(kept) == 10

    def test_exact_duplicate_by_content_hash_is_dropped(self) -> None:
        a = _rec("a", "train", 0, content_hash="same", phash=0)
        b = _rec("b", "train", 0, content_hash="same", phash=999)  # phash irrelevant to exact-dup check
        kept, dropped = dedupe_exact_duplicates([a, b])
        assert dropped == 1
        assert len(kept) == 1

    def test_keeps_lexicographically_first_uid_regardless_of_input_order(self) -> None:
        a = _rec("a_first", "train", 0, content_hash="same")
        b = _rec("z_second", "train", 0, content_hash="same")
        kept_forward, _ = dedupe_exact_duplicates([a, b])
        kept_reversed, _ = dedupe_exact_duplicates([b, a])
        assert [r.uid for r in kept_forward] == ["a_first"]
        assert [r.uid for r in kept_reversed] == ["a_first"]

    def test_multiple_duplicate_groups_counted_independently(self) -> None:
        records = [
            _rec("a1", "train", 0, content_hash="groupA"),
            _rec("a2", "train", 0, content_hash="groupA"),
            _rec("a3", "train", 0, content_hash="groupA"),
            _rec("b1", "train", 1, content_hash="groupB"),
            _rec("b2", "train", 1, content_hash="groupB"),
        ]
        kept, dropped = dedupe_exact_duplicates(records)
        assert dropped == 3  # 2 extra from groupA, 1 extra from groupB
        assert len(kept) == 2


# --------------------------------------------------------------------------
# cluster_by_phash -- near-duplicate connected components
# --------------------------------------------------------------------------


class TestClusterByPhash:
    def test_close_hashes_form_one_cluster(self) -> None:
        a = _rec("a", "train", 0, content_hash="ca", phash=0b0000)
        b = _rec("b", "train", 0, content_hash="cb", phash=0b0001)  # hamming(a,b) == 1
        cluster_of = cluster_by_phash([a, b], phash_distance=2)
        assert cluster_of["a"] == cluster_of["b"]

    def test_far_hashes_form_separate_clusters(self) -> None:
        a = _rec("a", "train", 0, content_hash="ca", phash=0)
        c = _rec("c", "train", 0, content_hash="cc", phash=0b1111)  # hamming(a,c) == 4
        cluster_of = cluster_by_phash([a, c], phash_distance=2)
        assert cluster_of["a"] != cluster_of["c"]

    def test_transitive_chain_forms_one_cluster(self) -> None:
        # a-b within distance, b-c within distance, a-c NOT within distance
        # directly -- but a,b,c must still land in one cluster via the chain.
        a = _rec("a", "train", 0, content_hash="ca", phash=0b000000)
        b = _rec("b", "train", 0, content_hash="cb", phash=0b000011)  # hamming(a,b)=2
        c = _rec("c", "train", 0, content_hash="cc", phash=0b001111)  # hamming(b,c)=2, hamming(a,c)=4
        cluster_of = cluster_by_phash([a, b, c], phash_distance=2)
        assert cluster_of["a"] == cluster_of["b"] == cluster_of["c"]

    def test_singleton_with_no_neighbors_is_its_own_cluster(self) -> None:
        a = _rec("a", "train", 0, phash=0)
        b = _rec("b", "train", 0, phash=0xFFFFFFFF)
        cluster_of = cluster_by_phash([a, b], phash_distance=1)
        assert cluster_of["a"] != cluster_of["b"]

    def test_deterministic_regardless_of_input_order(self) -> None:
        a = _rec("a", "train", 0, phash=0)
        b = _rec("b", "train", 0, phash=1)
        forward = cluster_by_phash([a, b], phash_distance=2)
        backward = cluster_by_phash([b, a], phash_distance=2)
        assert forward["a"] == backward["a"]
        assert forward["b"] == backward["b"]
        assert forward["a"] == forward["b"]

    def test_cluster_id_is_min_uid_in_component(self) -> None:
        a = _rec("aaa", "train", 0, phash=0)
        z = _rec("zzz", "train", 0, phash=1)
        cluster_of = cluster_by_phash([z, a], phash_distance=2)
        assert cluster_of["aaa"] == "aaa"
        assert cluster_of["zzz"] == "aaa"


# --------------------------------------------------------------------------
# straddling_cluster_count -- the leakage-audit headline number
# --------------------------------------------------------------------------


class TestStraddlingClusterCount:
    def test_cluster_spanning_two_orig_splits_counts_as_straddling(self) -> None:
        a = _rec("a", "train", 0, phash=0)
        b = _rec("b", "test", 0, phash=0)  # identical hash -> same cluster, different orig_split
        cluster_of = cluster_by_phash([a, b], phash_distance=6)
        count, by_cluster = straddling_cluster_count([a, b], cluster_of)
        assert count == 1
        assert by_cluster[cluster_of["a"]] == {"train", "test"}

    def test_cluster_within_one_orig_split_does_not_straddle(self) -> None:
        a = _rec("a", "train", 0, phash=0)
        b = _rec("b", "train", 0, phash=0)
        cluster_of = cluster_by_phash([a, b], phash_distance=6)
        count, _ = straddling_cluster_count([a, b], cluster_of)
        assert count == 0

    def test_multiple_straddling_clusters_all_counted(self) -> None:
        a = _rec("a", "train", 0, content_hash="ca", phash=0)
        b = _rec("b", "validation", 0, content_hash="cb", phash=0)
        c = _rec("c", "train", 1, content_hash="cc", phash=0xFFFFFFFFFFFFFFFF)  # hamming(0, this) == 64, far from a/b
        d = _rec("d", "test", 1, content_hash="cd", phash=0xFFFFFFFFFFFFFFFF)
        cluster_of = cluster_by_phash([a, b, c, d], phash_distance=6)
        count, _ = straddling_cluster_count([a, b, c, d], cluster_of)
        assert count == 2


# --------------------------------------------------------------------------
# find_mixed_label_clusters -- a scene can't be both normal and failure
# --------------------------------------------------------------------------


class TestFindMixedLabelClusters:
    def test_cluster_with_both_labels_is_flagged(self) -> None:
        a = _rec("a", "train", 0, phash=0)
        b = _rec("b", "train", 1, phash=0)  # same hash -> same cluster, different label
        cluster_of = cluster_by_phash([a, b], phash_distance=6)
        mixed = find_mixed_label_clusters([a, b], cluster_of)
        assert len(mixed) == 1
        assert cluster_of["a"] in mixed
        assert mixed[cluster_of["a"]] == {0, 1}

    def test_single_label_cluster_is_not_flagged(self) -> None:
        a = _rec("a", "train", 0, phash=0)
        b = _rec("b", "train", 0, phash=0)
        cluster_of = cluster_by_phash([a, b], phash_distance=6)
        mixed = find_mixed_label_clusters([a, b], cluster_of)
        assert mixed == {}


# --------------------------------------------------------------------------
# summarize_cluster_sizes
# --------------------------------------------------------------------------


class TestSummarizeClusterSizes:
    def test_empty_returns_zeros(self) -> None:
        summary = summarize_cluster_sizes([])
        assert summary["count"] == 0
        assert summary["min"] == 0
        assert summary["max"] == 0

    def test_known_distribution(self) -> None:
        summary = summarize_cluster_sizes([1, 1, 2, 4, 10])
        assert summary["count"] == 5
        assert summary["min"] == 1
        assert summary["max"] == 10
        assert summary["median"] == 2
        assert summary["mean"] == pytest.approx(18 / 5)

    def test_even_count_median_is_averaged(self) -> None:
        summary = summarize_cluster_sizes([1, 2, 3, 4])
        assert summary["median"] == 2.5


# --------------------------------------------------------------------------
# assign_clusters_to_splits -- stratified, deterministic cluster-level split
# --------------------------------------------------------------------------


class TestAssignClustersToSplits:
    def test_deterministic_for_fixed_seed(self) -> None:
        clusters_by_label = {
            0: [(f"n{i}", 1) for i in range(50)],
            1: [(f"f{i}", 1) for i in range(50)],
        }
        a = assign_clusters_to_splits(clusters_by_label, seed=1337)
        b = assign_clusters_to_splits(clusters_by_label, seed=1337)
        assert a == b

    def test_different_seeds_can_differ(self) -> None:
        clusters_by_label = {
            0: [(f"n{i}", 1) for i in range(50)],
            1: [(f"f{i}", 1) for i in range(50)],
        }
        a = assign_clusters_to_splits(clusters_by_label, seed=1337)
        b = assign_clusters_to_splits(clusters_by_label, seed=42)
        assert a != b

    def test_every_cluster_gets_assigned_to_a_valid_split(self) -> None:
        clusters_by_label = {0: [(f"n{i}", 1) for i in range(30)]}
        result = assign_clusters_to_splits(clusters_by_label, seed=1)
        assert set(result.keys()) == {f"n{i}" for i in range(30)}
        assert set(result.values()) <= set(SPLIT_NAMES)

    def test_stratification_both_labels_present_in_every_split(self) -> None:
        # 60 clusters per label, single image each -- plenty of room for the
        # 70/15/15 target to place some of each label in every split.
        clusters_by_label = {
            0: [(f"n{i:03d}", 1) for i in range(60)],
            1: [(f"f{i:03d}", 1) for i in range(60)],
        }
        result = assign_clusters_to_splits(clusters_by_label, seed=1337)
        for split in SPLIT_NAMES:
            has_normal = any(cid.startswith("n") for cid, sp in result.items() if sp == split)
            has_failure = any(cid.startswith("f") for cid, sp in result.items() if sp == split)
            assert has_normal, f"split {split} has no normal cluster"
            assert has_failure, f"split {split} has no failure cluster"

    def test_ratios_approximately_match_target_by_image_count(self) -> None:
        clusters_by_label = {0: [(f"n{i:04d}", 1) for i in range(1000)]}
        result = assign_clusters_to_splits(clusters_by_label, seed=1337)
        counts = {s: sum(1 for sp in result.values() if sp == s) for s in SPLIT_NAMES}
        assert counts["train"] + counts["val"] + counts["test"] == 1000
        assert 650 <= counts["train"] <= 750
        assert 100 <= counts["val"] <= 200
        assert 100 <= counts["test"] <= 200

    def test_rejects_bad_ratios(self) -> None:
        with pytest.raises(ValueError):
            assign_clusters_to_splits({0: [("a", 1)]}, seed=1, ratios=(0.5, 0.5, 0.5))


# --------------------------------------------------------------------------
# cap_clusters_per_class
# --------------------------------------------------------------------------


class TestCapClustersPerClass:
    def test_caps_total_images_at_or_under_cap(self) -> None:
        cluster_label = {f"c{i}": 0 for i in range(20)}
        cluster_size = {f"c{i}": 5 for i in range(20)}  # 100 images total
        kept = cap_clusters_per_class(cluster_label, cluster_size, cap=37, seed=1)
        total = sum(cluster_size[cid] for cid in kept)
        assert total <= 37

    def test_keeps_whole_clusters_never_partial(self) -> None:
        cluster_label = {"a": 0, "b": 0, "c": 0}
        cluster_size = {"a": 10, "b": 10, "c": 10}
        kept = cap_clusters_per_class(cluster_label, cluster_size, cap=25, seed=1)
        # cap=25 can fit at most 2 whole 10-image clusters (20), never 2.5
        total = sum(cluster_size[cid] for cid in kept)
        assert total in (0, 10, 20)

    def test_under_cap_keeps_everything(self) -> None:
        cluster_label = {f"c{i}": 1 for i in range(5)}
        cluster_size = {f"c{i}": 2 for i in range(5)}
        kept = cap_clusters_per_class(cluster_label, cluster_size, cap=1000, seed=1)
        assert kept == set(cluster_label)

    def test_deterministic_for_fixed_seed(self) -> None:
        cluster_label = {f"c{i}": i % 2 for i in range(40)}
        cluster_size = {f"c{i}": 3 for i in range(40)}
        a = cap_clusters_per_class(cluster_label, cluster_size, cap=50, seed=7)
        b = cap_clusters_per_class(cluster_label, cluster_size, cap=50, seed=7)
        assert a == b

    def test_labels_capped_independently(self) -> None:
        cluster_label = {**{f"n{i}": 0 for i in range(10)}, **{f"f{i}": 1 for i in range(10)}}
        cluster_size = {cid: 1 for cid in cluster_label}
        kept = cap_clusters_per_class(cluster_label, cluster_size, cap=3, seed=1)
        n_normal = sum(1 for cid in kept if cluster_label[cid] == 0)
        n_failure = sum(1 for cid in kept if cluster_label[cid] == 1)
        assert n_normal <= 3
        assert n_failure <= 3


# --------------------------------------------------------------------------
# assert_no_cluster_overlap
# --------------------------------------------------------------------------


class TestAssertNoClusterOverlap:
    def test_passes_on_disjoint_splits(self) -> None:
        assert_no_cluster_overlap({"train": ["a", "b"], "val": ["c"], "test": ["d"]})  # no raise

    def test_fires_on_overlap(self) -> None:
        with pytest.raises(AssertionError):
            assert_no_cluster_overlap({"train": ["a"], "val": ["a"], "test": ["b"]})


# --------------------------------------------------------------------------
# plan_binary_split -- the pure end-to-end core, no network/disk required
# --------------------------------------------------------------------------


def _padding_records(n: int, start: int = 0) -> list[HfRecord]:
    """`n` well-separated, single-member, alternating-label clusters, so a
    real cluster-of-interest has room to land anywhere without being forced
    into a degenerate 100%-train split."""
    out = []
    for i in range(start, start + n):
        out.append(_rec(f"pad{i:04d}", "train", i % 2, content_hash=f"padhash{i}", phash=_spread_hash(i)))
    return out


class TestPlanBinarySplitNeverTouchesLoaders:
    def test_plan_never_calls_a_loader(self) -> None:
        records = _padding_records(60)
        # If plan_binary_split ever called .loader(), _no_call_loader would
        # raise AssertionError and this test would fail with that error.
        plan = plan_binary_split(records, seed=1337)
        assert isinstance(plan, BinarySplitPlan)


class TestPlanBinarySplitClusterIntegrity:
    def test_a_multi_member_cluster_never_straddles_a_split(self) -> None:
        # 5 near-duplicate records, chained within phash_distance=3, all one label.
        cluster_members = [_rec(f"m{i}", "train", 0, content_hash=f"cm{i}", phash=i) for i in range(5)]
        padding = _padding_records(60)
        plan = plan_binary_split(cluster_members + padding, seed=1337, phash_distance=3)

        homes = set()
        for split, recs in plan.records_by_split.items():
            if any(r.uid.startswith("m") for r in recs):
                homes.add(split)
        assert len(homes) == 1

    def test_no_overlap_end_to_end(self) -> None:
        records = _padding_records(120)
        plan = plan_binary_split(records, seed=1337)
        uids_by_split = {s: {r.uid for r in recs} for s, recs in plan.records_by_split.items()}
        assert uids_by_split["train"] & uids_by_split["val"] == set()
        assert uids_by_split["train"] & uids_by_split["test"] == set()
        assert uids_by_split["val"] & uids_by_split["test"] == set()


class TestPlanBinarySplitStratification:
    def test_both_classes_present_in_every_split(self) -> None:
        records = _padding_records(120)  # 60 normal, 60 failure, all singleton clusters
        plan = plan_binary_split(records, seed=1337)
        for split in SPLIT_NAMES:
            classes_present = {class_name_for_label(r.label) for r in plan.records_by_split[split]}
            assert "normal" in classes_present
            assert "failure" in classes_present


class TestPlanBinarySplitDeterminism:
    def test_deterministic_for_fixed_seed(self) -> None:
        records = _padding_records(100)
        p1 = plan_binary_split(records, seed=1337)
        p2 = plan_binary_split(records, seed=1337)
        stems1 = {s: sorted(r.output_stem for r in recs) for s, recs in p1.records_by_split.items()}
        stems2 = {s: sorted(r.output_stem for r in recs) for s, recs in p2.records_by_split.items()}
        assert stems1 == stems2
        assert p1.total_clusters == p2.total_clusters


class TestPlanBinarySplitExactDuplicateRemoval:
    def test_exact_duplicates_reflected_in_plan(self) -> None:
        a = _rec("a", "train", 0, content_hash="dup", phash=0)
        b = _rec("b", "train", 0, content_hash="dup", phash=999999)
        padding = _padding_records(60)
        plan = plan_binary_split([a, b] + padding, seed=1337)
        assert plan.exact_duplicates_dropped == 1
        all_uids = {r.uid for recs in plan.records_by_split.values() for r in recs}
        # only one of a/b should ever appear anywhere
        assert len(all_uids & {"a", "b"}) == 1


class TestPlanBinarySplitMixedLabelExclusion:
    def test_mixed_label_cluster_excluded_from_every_split(self) -> None:
        a = _rec("a", "train", 0, content_hash="ca", phash=0)
        b = _rec("b", "train", 1, content_hash="cb", phash=0)  # same phash -> same cluster, opposite label
        padding = _padding_records(60)
        plan = plan_binary_split([a, b] + padding, seed=1337)

        all_uids = {r.uid for recs in plan.records_by_split.values() for r in recs}
        assert "a" not in all_uids
        assert "b" not in all_uids
        assert plan.mixed_label_cluster_count == 1
        assert plan.mixed_label_images_excluded == 2

    def test_no_mixed_clusters_when_labels_are_clean(self) -> None:
        records = _padding_records(60)
        plan = plan_binary_split(records, seed=1337)
        assert plan.mixed_label_cluster_count == 0
        assert plan.mixed_label_images_excluded == 0


class TestPlanBinarySplitStraddleCounter:
    def test_straddle_count_reported_on_plan(self) -> None:
        a = _rec("a", "train", 0, content_hash="ca", phash=0)
        b = _rec("b", "test", 0, content_hash="cb", phash=0)  # same cluster, spans HF's own train/test
        padding = _padding_records(60)
        plan = plan_binary_split([a, b] + padding, seed=1337)
        assert plan.straddling_clusters >= 1

    def test_zero_straddle_when_every_cluster_is_single_split(self) -> None:
        records = _padding_records(60)  # all orig_split="train"
        plan = plan_binary_split(records, seed=1337)
        assert plan.straddling_clusters == 0


class TestPlanBinarySplitCap:
    def test_per_class_cap_reduces_total_images_kept(self) -> None:
        records = _padding_records(200)  # 100 normal, 100 failure singleton clusters
        uncapped = plan_binary_split(records, seed=1337)
        capped = plan_binary_split(records, seed=1337, per_class_cap=10)

        def total_images(plan: BinarySplitPlan) -> int:
            return sum(len(recs) for recs in plan.records_by_split.values())

        assert total_images(capped) < total_images(uncapped)
        assert capped.capped_out_cluster_count > 0

    def test_no_cap_means_capped_out_count_is_zero(self) -> None:
        records = _padding_records(40)
        plan = plan_binary_split(records, seed=1337, per_class_cap=None)
        assert plan.capped_out_cluster_count == 0


# --------------------------------------------------------------------------
# build_evaluation_validity_warning -- the critical honesty gate
# --------------------------------------------------------------------------


class TestBuildEvaluationValidityWarning:
    def test_below_total_scene_minimum_is_flagged(self) -> None:
        clusters_per_split_by_class = {s: {c: 10 for c in CLASS_NAMES} for s in SPLIT_NAMES}
        warning = build_evaluation_validity_warning(total_clusters=MIN_INDEPENDENT_SCENES_FOR_TRUST - 1, clusters_per_split_by_class=clusters_per_split_by_class)
        assert "EVALUATION VALIDITY WARNING" in warning

    def test_below_per_split_class_minimum_is_flagged(self) -> None:
        clusters_per_split_by_class = {s: {c: 10 for c in CLASS_NAMES} for s in SPLIT_NAMES}
        clusters_per_split_by_class["test"]["failure"] = MIN_CLUSTERS_PER_SPLIT_CLASS - 1
        warning = build_evaluation_validity_warning(total_clusters=1000, clusters_per_split_by_class=clusters_per_split_by_class)
        assert "EVALUATION VALIDITY WARNING" in warning
        assert "test" in warning
        assert "failure" in warning

    def test_meeting_both_thresholds_is_not_flagged(self) -> None:
        clusters_per_split_by_class = {s: {c: MIN_CLUSTERS_PER_SPLIT_CLASS + 5 for c in CLASS_NAMES} for s in SPLIT_NAMES}
        warning = build_evaluation_validity_warning(total_clusters=MIN_INDEPENDENT_SCENES_FOR_TRUST + 10, clusters_per_split_by_class=clusters_per_split_by_class)
        assert "EVALUATION VALIDITY WARNING" not in warning
        assert str(MIN_INDEPENDENT_SCENES_FOR_TRUST + 10) in warning


# --------------------------------------------------------------------------
# materialize_split / build_dataset -- end to end with real (synthetic) pixels
# --------------------------------------------------------------------------


def _real_image_record(uid: str, orig_split: str, label: int, seed: int, output_stem: str | None = None) -> HfRecord:
    """Builds an HfRecord backed by an actual decodable synthetic image, with
    content_hash/phash computed the same way gather_hf_records would from
    real data."""
    rng = np.random.RandomState(seed)
    image = rng.randint(0, 255, size=(300, 400, 3), dtype=np.uint8)
    return HfRecord(
        uid=uid,
        orig_split=orig_split,
        label=label,
        content_hash=content_hash(image),
        phash=dhash(image),
        loader=lambda: image,
        output_stem=output_stem or uid.replace(":", "_"),
    )


class TestMaterializeSplit:
    def test_writes_512x512_jpegs_into_class_dirs(self, tmp_path: Path) -> None:
        records = [_real_image_record(f"r{i}", "train", i % 2, seed=i) for i in range(10)]
        records_by_split = {"train": records, "val": [], "test": []}
        counts = materialize_split(records_by_split, tmp_path)

        total_written = sum(counts["train"].values())
        assert total_written == 10
        written_files = list(tmp_path.rglob("*.jpg"))
        assert len(written_files) == 10
        img = cv2.imread(str(written_files[0]))
        assert img.shape == (512, 512, 3)

    def test_writes_into_correct_class_subdirectory(self, tmp_path: Path) -> None:
        normal_rec = _real_image_record("n0", "train", 0, seed=1)
        failure_rec = _real_image_record("f0", "train", 1, seed=2)
        materialize_split({"train": [normal_rec, failure_rec], "val": [], "test": []}, tmp_path)
        assert (tmp_path / "train" / "normal" / f"{normal_rec.output_stem}.jpg").is_file()
        assert (tmp_path / "train" / "failure" / f"{failure_rec.output_stem}.jpg").is_file()


class TestBuildDatasetEndToEnd:
    def test_full_pipeline_writes_images_and_a_complete_report(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "argus_bin"
        records = [_real_image_record(f"hf_{i:04d}", "train" if i % 3 else "test", i % 2, seed=i) for i in range(90)]

        report = build_dataset(out_dir=out_dir, records=records, seed=1337)

        required_keys = {
            "seed",
            "phash_distance",
            "per_class_cap",
            "ratios",
            "output_size",
            "jpeg_quality",
            "class_names",
            "hf_dataset_id",
            "counts",
            "total_images_kept",
            "total_rows_pooled",
            "exact_duplicates_dropped",
            "mixed_label_clusters_excluded",
            "capped_out_cluster_count",
            "total_independent_scenes",
            "largest_cluster_size",
            "cluster_size_distribution",
            "clusters_per_split_by_class",
            "clusters_straddling_hf_original_split",
            "evaluation_validity_warning",
        }
        assert required_keys <= report.keys()

        # JSON-serializable end to end (this is what main() writes to disk).
        json.dumps(report)

        assert report["class_names"] == ["failure", "normal"]
        assert report["total_rows_pooled"] == 90
        assert report["ratios"] == {"train": DEFAULT_SPLIT_RATIOS[0], "val": DEFAULT_SPLIT_RATIOS[1], "test": DEFAULT_SPLIT_RATIOS[2]}

        # Files actually exist on disk for at least the train split.
        for cname in CLASS_NAMES:
            assert (out_dir / "train" / cname).is_dir()
        written = list(out_dir.rglob("*.jpg"))
        assert len(written) == report["total_images_kept"]
        assert len(written) > 0

    def test_evaluation_validity_warning_is_honest_about_a_thin_dataset(self, tmp_path: Path) -> None:
        # A handful of images/scenes -- well under MIN_INDEPENDENT_SCENES_FOR_TRUST.
        records = [_real_image_record(f"hf_{i:02d}", "train", i % 2, seed=i) for i in range(8)]
        report = build_dataset(out_dir=tmp_path / "argus_bin_thin", records=records, seed=1337)
        assert "EVALUATION VALIDITY WARNING" in report["evaluation_validity_warning"]
        # Also required to be surfaced honestly, not softened.
        assert str(report["total_independent_scenes"]) in report["evaluation_validity_warning"]
