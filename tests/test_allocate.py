"""Checks on the allocator's promises.

These are the properties an organiser would be upset to find violated: nobody
left out, nobody in two groups, groups roughly the size you asked for, and the
institution mixing rule actually enforced.
"""

from collections import Counter

import numpy as np
import pytest

from rigad.allocate import (
    allocate,
    coherence,
    default_quota,
    diversity,
    institution_span,
    same_institution_pairs,
)
from rigad.embed import normalize_rows

INSTITUTIONS = ["gu", "warwick", "essec"]


def make_cohort(n: int, *, n_themes: int = 3, dim: int = 32, seed: int = 0):
    """A synthetic cohort: people cluster on themes, spread across institutions."""
    rng = np.random.default_rng(seed)
    themes = normalize_rows(rng.normal(size=(n_themes, dim)))
    vectors = normalize_rows(
        np.array([themes[i % n_themes] + 0.35 * rng.normal(size=dim) for i in range(n)])
    )
    institutions = [INSTITUTIONS[i % len(INSTITUTIONS)] for i in range(n)]
    topic_sets = [{f"t{(i % n_themes) * 3 + j}" for j in range(2)} for i in range(n)]
    return vectors, topic_sets, institutions


@pytest.mark.parametrize("n", [12, 47, 60, 61, 116])
def test_everyone_allocated_exactly_once(n):
    vectors, topics, institutions = make_cohort(n)
    groups = allocate(vectors, topics, institutions, group_size=4, seed=0)

    allocated = sorted(i for g in groups for i in g.members)
    assert allocated == list(range(n)), "every person must appear in exactly one group"


@pytest.mark.parametrize("n", [12, 47, 60, 61, 116])
def test_group_sizes_within_one_of_target(n):
    vectors, topics, institutions = make_cohort(n)
    group_size = 4
    groups = allocate(vectors, topics, institutions, group_size=group_size, seed=0)

    for group in groups:
        assert group_size - 1 <= len(group) <= group_size + 1


@pytest.mark.parametrize("group_size", [3, 4, 5, 6])
def test_an_explicit_quota_is_respected(group_size):
    """A hard quota is opt-in now, but must still be honoured when asked for."""
    vectors, topics, institutions = make_cohort(90)
    quota = default_quota(group_size, len(set(institutions)))
    groups = allocate(vectors, topics, institutions, group_size=group_size,
                      quota=quota, seed=0)

    for group in groups:
        counts = Counter(institutions[i] for i in group.members)
        assert max(counts.values()) <= quota


def test_quota_of_one_forces_all_distinct_institutions():
    vectors, topics, institutions = make_cohort(60)
    groups = allocate(vectors, topics, institutions, group_size=3, quota=1, seed=0)

    for group in groups:
        homes = [institutions[i] for i in group.members]
        assert len(set(homes)) == len(homes)


def test_allocation_is_deterministic_for_a_seed():
    vectors, topics, institutions = make_cohort(60)
    first = allocate(vectors, topics, institutions, group_size=4, seed=7)
    second = allocate(vectors, topics, institutions, group_size=4, seed=7)

    assert [g.members for g in first] == [g.members for g in second]


def test_different_seeds_can_give_different_allocations():
    vectors, topics, institutions = make_cohort(60)
    first = allocate(vectors, topics, institutions, group_size=4, seed=1)
    second = allocate(vectors, topics, institutions, group_size=4, seed=2)

    assert [g.members for g in first] != [g.members for g in second]


def test_beats_random_on_coherence():
    """The whole point is to do better than shuffling people into groups."""
    vectors, topics, institutions = make_cohort(90)
    similarity = vectors @ vectors.T

    groups = allocate(vectors, topics, institutions, group_size=4, beta=0.0, seed=0)
    ours = np.mean([coherence(g.members, similarity) for g in groups])

    rng = np.random.default_rng(0)
    random_scores = []
    for _ in range(20):
        order = rng.permutation(len(institutions))
        chunks = [order[i : i + 4].tolist() for i in range(0, len(order), 4)]
        random_scores.append(np.mean([coherence(c, similarity) for c in chunks if len(c) > 1]))

    assert ours > np.mean(random_scores)


def test_empty_cohort_returns_no_groups():
    empty = np.zeros((0, 8))
    assert allocate(empty, [], [], group_size=4) == []


def test_mismatched_input_lengths_are_rejected():
    vectors, topics, institutions = make_cohort(10)
    with pytest.raises(ValueError):
        allocate(vectors, topics[:5], institutions, group_size=4)


# --- metric behaviour ------------------------------------------------------


def test_coherence_of_identical_vectors_is_one():
    vectors = normalize_rows(np.ones((3, 8)))
    assert coherence([0, 1, 2], vectors @ vectors.T) == pytest.approx(1.0)


def test_coherence_of_single_member_is_zero():
    vectors = normalize_rows(np.ones((3, 8)))
    assert coherence([0], vectors @ vectors.T) == 0.0


def test_diversity_is_one_when_no_topics_overlap():
    assert diversity([0, 1], [{"a"}, {"b"}]) == pytest.approx(1.0)


def test_diversity_falls_when_topics_repeat():
    assert diversity([0, 1], [{"a"}, {"a"}]) == pytest.approx(0.5)


def test_diversity_handles_people_without_topics():
    assert diversity([0, 1], [set(), set()]) == 0.0


def test_institution_span_counts_distinct_homes():
    assert institution_span([0, 1, 2], ["gu", "gu", "warwick"]) == 2


def test_default_quota_forces_mixing():
    # Four people, three institutions: at most two may share a home.
    assert default_quota(4, 3) == 2
    # Three people, three institutions: all must differ.
    assert default_quota(3, 3) == 1
    # More people than institutions still allows a group to form.
    assert default_quota(9, 3) == 3


# --- unbalanced institutions -----------------------------------------------
# The synthetic cohorts above spread people evenly across institutions — the
# case where a strict quota is always satisfiable. Real cohorts are lopsided,
# so these tests use uneven institution sizes on purpose.

UNEVEN = {"monash": 168, "asu": 137, "gu": 93, "warwick": 85, "essec": 31}


def make_uneven_cohort(sizes: dict[str, int], *, dim: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    institutions = [key for key, count in sizes.items() for _ in range(count)]
    n = len(institutions)
    vectors = normalize_rows(rng.normal(size=(n, dim)))
    topic_sets = [{f"t{i % 40}", f"u{i % 17}"} for i in range(n)]
    return vectors, topic_sets, institutions


def test_unbalanced_institutions_still_fill_groups():
    """Uneven institution sizes must never produce undersized groups.

    Five institutions with groups of four imply a strict quota of one member
    per institution, which a lopsided cohort cannot supply — every group must
    still be filled to within one of the target size.
    """
    vectors, topics, institutions = make_uneven_cohort(UNEVEN)
    group_size = 4
    groups = allocate(vectors, topics, institutions, group_size=group_size, seed=0)

    for group in groups:
        assert group_size - 1 <= len(group) <= group_size + 1

    allocated = sorted(i for g in groups for i in g.members)
    assert allocated == list(range(len(institutions)))


def test_soft_penalty_still_mixes_institutions_without_a_quota():
    """Mixing should fall out of the tiebreaker, with no hard rule imposed."""
    vectors, topics, institutions = make_uneven_cohort(UNEVEN)
    groups = allocate(vectors, topics, institutions, group_size=4, seed=0)

    same = np.mean([same_institution_pairs(g.members, institutions) for g in groups])
    assert same < 0.25, "most pairs should come from different institutions"


def test_institution_penalty_can_be_switched_off():
    vectors, topics, institutions = make_uneven_cohort(UNEVEN)
    mixed = allocate(vectors, topics, institutions, group_size=4, seed=0)
    ignored = allocate(vectors, topics, institutions, group_size=4,
                       institution_penalty=0.0, seed=0)

    with_penalty = np.mean([same_institution_pairs(g.members, institutions) for g in mixed])
    without = np.mean([same_institution_pairs(g.members, institutions) for g in ignored])
    assert with_penalty <= without


def test_theme_first_grouping_beats_diversity_first_on_coherence():
    """The default should build themed groups, not deliberately mixed ones."""
    vectors, topics, institutions = make_cohort(64)
    similarity = vectors @ vectors.T

    themed = allocate(vectors, topics, institutions, group_size=4, seed=0)
    diverse = allocate(vectors, topics, institutions, group_size=4, beta=1.0, seed=0)

    assert np.mean([coherence(g.members, similarity) for g in themed]) > np.mean(
        [coherence(g.members, similarity) for g in diverse]
    )


def test_unbalanced_groups_still_cross_institutions():
    vectors, topics, institutions = make_uneven_cohort(UNEVEN)
    groups = allocate(vectors, topics, institutions, group_size=4, seed=0)
    spans = [institution_span(g.members, institutions) for g in groups]
    assert np.mean(spans) > 2.5


def test_one_institution_swamping_the_rest_is_handled():
    """An extreme case: no quota can spread this evenly, so it must not hang."""
    vectors, topics, institutions = make_uneven_cohort({"big": 200, "tiny": 3})
    groups = allocate(vectors, topics, institutions, group_size=4, seed=0)
    allocated = sorted(i for g in groups for i in g.members)
    assert allocated == list(range(len(institutions)))
