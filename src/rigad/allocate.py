"""Form cross-institutional research groups.

The problem RIGAD solves is not "who is most similar to whom". Ranking people
by similarity produces groups of near-duplicates drawn from the same
institution, which is the opposite of what an alliance wants.

**A group is built around a shared research theme.** That is the point of
putting people in a room together, and it is the term that dominates:

    maximise  sum over groups of
        [ alpha * coherence(G) - penalty * same_institution_pairs(G)
          + beta * diversity(G) ]
    subject to group sizes within group_size +/- 1

``coherence`` — mean pairwise cosine similarity — carries the theme, and is
the only term that matters much. Institutional mixing is a **tiebreaker**: a
small penalty for pairs from the same institution, enough to prefer a mixed
group when two options are thematically equivalent, never enough to break up
a good theme. There is no hard quota by default.

``diversity`` rewards members having *different* OpenAlex topic labels, and
defaults to **off**. It is available for the case where an organiser
deliberately wants dissimilar people in a room, but it works directly against
thematic grouping — a group whose members share no topic at all scores
perfectly on it.

The search is a greedy construction followed by local search over swaps. That
is deliberate: it is fast, deterministic given a seed, and every step can be
explained to a sceptical colleague, which matters more here than squeezing out
the last fraction of a percent that an ILP solver might find.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class Group:
    """One formed group, as indices into the researcher list."""

    members: list[int]

    def __len__(self) -> int:
        return len(self.members)


def coherence(members: list[int], similarity: np.ndarray) -> float:
    """Mean pairwise cosine similarity within a group.

    A single-member group has no pairs; we call that 0.0 rather than 1.0 so
    that the objective never rewards leaving someone on their own.
    """
    if len(members) < 2:
        return 0.0
    idx = np.array(members)
    block = similarity[np.ix_(idx, idx)]
    n = len(members)
    # Sum of the off-diagonal entries, divided by the number of ordered pairs.
    return float((block.sum() - np.trace(block)) / (n * (n - 1)))


def diversity(members: list[int], topic_sets: list[set[str]]) -> float:
    """Share of group topics that are distinct.

    1.0 means every member works on entirely different topics; 1/n means they
    all work on exactly the same ones. Returns 0.0 when nobody has topics.
    """
    if not members:
        return 0.0
    total = sum(len(topic_sets[i]) for i in members)
    if total == 0:
        return 0.0
    union = set().union(*(topic_sets[i] for i in members))
    return len(union) / total


def institution_span(members: list[int], institutions: list[str]) -> int:
    """How many distinct institutions a group draws on."""
    return len({institutions[i] for i in members})


def same_institution_pairs(members: list[int], institutions: list[str]) -> float:
    """Share of pairs in a group that come from the same institution.

    0.0 when everyone is from somewhere different, 1.0 when all are from the
    same place. Used as a mild penalty so that mixing breaks ties without
    overriding the theme.
    """
    n = len(members)
    if n < 2:
        return 0.0
    same = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if institutions[members[i]] == institutions[members[j]]
    )
    return same / (n * (n - 1) / 2)


def group_theme(members: list[int], topic_sets: list[set[str]]) -> str:
    """The topic that best describes a group, for showing to an organiser.

    The most widely shared topic if there is one; otherwise the topic of the
    member closest to the group's centre is not available here, so we fall
    back to naming the most common label. An organiser wants "digital health",
    not four indices.
    """
    counts = Counter(topic for i in members for topic in topic_sets[i])
    if not counts:
        return ""
    topic, shared_by = counts.most_common(1)[0]
    return topic if shared_by > 1 else f"{topic} (and {len(counts) - 1} other topics)"


def default_quota(group_size: int, n_institutions: int) -> int:
    """The strictest per-institution quota a group size allows, if you want one.

    Only relevant when passing ``quota`` explicitly to ``allocate``. The
    default path uses a soft penalty instead: a hard quota can be quietly
    unsatisfiable — five institutions and groups of four imply a quota of 1,
    which unevenly sized institutions cannot supply — and the shortfall would
    surface as undersized groups rather than as an error.
    """
    return max(1, math.ceil(group_size / n_institutions))


class _Allocation:
    """Working state for the search. Holds groups and scores them."""

    def __init__(
        self,
        similarity: np.ndarray,
        topic_sets: list[set[str]],
        institutions: list[str],
        *,
        alpha: float,
        beta: float,
        penalty: float,
        quota: int | None,
    ) -> None:
        self.similarity = similarity
        self.topic_sets = topic_sets
        self.institutions = institutions
        self.alpha = alpha
        self.beta = beta
        self.penalty = penalty
        self.quota = quota
        self.groups: list[list[int]] = []

    def group_score(self, members: list[int]) -> float:
        score = self.alpha * coherence(members, self.similarity)
        score -= self.penalty * same_institution_pairs(members, self.institutions)
        if self.beta:
            score += self.beta * diversity(members, self.topic_sets)
        return score

    def total_score(self) -> float:
        return sum(self.group_score(g) for g in self.groups)

    def can_accept(self, group: list[int], person: int, max_size: int) -> bool:
        """Whether adding ``person`` keeps the group legal."""
        if len(group) >= max_size:
            return False
        if self.quota is None:
            return True  # institution handled as a soft penalty, not a rule
        home = self.institutions[person]
        return sum(1 for i in group if self.institutions[i] == home) < self.quota


def allocate(
    vectors: np.ndarray,
    topic_sets: list[set[str]],
    institutions: list[str],
    *,
    group_size: int = 4,
    alpha: float = 1.0,
    beta: float = 0.0,
    institution_penalty: float = 0.15,
    quota: int | None = None,
    seed: int = 0,
    max_passes: int = 40,
) -> list[Group]:
    """Allocate everyone into cross-institutional groups.

    ``vectors`` must be L2-normalised so that a dot product is cosine
    similarity. ``topic_sets`` and ``institutions`` are parallel to it.

    Groups form around shared research themes. ``institution_penalty`` nudges
    them to mix institutions where that costs nothing thematically; pass
    ``quota`` to make it a hard limit instead, at the risk of undersized
    groups when institutions are unevenly sized.

    Returns groups sized ``group_size`` give or take one. The result is
    deterministic for a given ``seed``.
    """
    n = len(institutions)
    if n == 0:
        return []
    if vectors.shape[0] != n or len(topic_sets) != n:
        raise ValueError("vectors, topic_sets and institutions must be the same length")

    similarity = vectors @ vectors.T
    state = _Allocation(
        similarity, topic_sets, institutions,
        alpha=alpha, beta=beta, penalty=institution_penalty, quota=quota,
    )

    n_groups = max(1, round(n / group_size))
    # Allow one over the nominal size so that everyone can be placed even when
    # n does not divide evenly.
    max_size = group_size + 1

    _construct(state, n_groups, max_size, seed=seed)
    _improve(state, max_size, max_passes=max_passes, seed=seed)

    return [Group(sorted(g)) for g in state.groups if g]


def _construct(state: _Allocation, n_groups: int, max_size: int, *, seed: int) -> None:
    """Build a first allocation: spread-out seeds, then greedy assignment.

    Seeds are chosen farthest-point style so that groups start on distinct
    themes rather than all converging on whatever the corpus talks about most.
    """
    rng = random.Random(seed)
    n = len(state.institutions)
    similarity = state.similarity

    # First seed is fixed by the RNG; each subsequent seed is whoever is least
    # similar to the seeds picked so far.
    seeds = [rng.randrange(n)]
    while len(seeds) < min(n_groups, n):
        worst_similarity = similarity[:, seeds].max(axis=1)
        worst_similarity[seeds] = np.inf  # never pick the same person twice
        seeds.append(int(worst_similarity.argmin()))

    state.groups = [[s] for s in seeds]
    placed = set(seeds)

    # Assign the rest in random order, each to whichever legal group it helps
    # most. Random order keeps one arbitrary sequence from dominating; the seed
    # keeps it reproducible.
    remaining = [i for i in range(n) if i not in placed]
    rng.shuffle(remaining)

    nominal_size = max_size - 1
    for person in remaining:
        # Fill groups up to the nominal size before letting any group run
        # over, so sizes stay even. Only if every under-size group is blocked
        # by the institution quota do we allow an over-size group.
        best_group = _best_group_for(state, person, limit=nominal_size)
        if best_group is None:
            best_group = _best_group_for(state, person, limit=max_size)

        if best_group is not None:
            best_group.append(person)
        else:
            # Every group is full or blocked by the institution quota. Start a
            # new group rather than drop the person: leaving someone
            # unallocated is never an acceptable answer for a real cohort.
            state.groups.append([person])


def _best_group_for(state: _Allocation, person: int, *, limit: int) -> list[int] | None:
    """The legal group under ``limit`` members that gains most from ``person``."""
    best_group, best_gain = None, -np.inf
    for group in state.groups:
        if not state.can_accept(group, person, limit):
            continue
        gain = state.group_score(group + [person]) - state.group_score(group)
        if gain > best_gain:
            best_group, best_gain = group, gain
    return best_group


def _improve(state: _Allocation, max_size: int, *, max_passes: int, seed: int) -> None:
    """Local search: swap pairs between groups while that improves the score.

    Only swaps are tried, never moves, so group sizes stay as constructed.
    """
    rng = random.Random(seed + 1)

    for _ in range(max_passes):
        improved = False
        order = list(range(len(state.groups)))
        rng.shuffle(order)

        for gi in order:
            for gj in order:
                if gi >= gj:
                    continue
                group_a, group_b = state.groups[gi], state.groups[gj]
                before = state.group_score(group_a) + state.group_score(group_b)

                for pos_a, person_a in enumerate(group_a):
                    for pos_b, person_b in enumerate(group_b):
                        swapped_a = group_a[:pos_a] + [person_b] + group_a[pos_a + 1 :]
                        swapped_b = group_b[:pos_b] + [person_a] + group_b[pos_b + 1 :]
                        if not _legal(state, swapped_a) or not _legal(state, swapped_b):
                            continue
                        after = state.group_score(swapped_a) + state.group_score(swapped_b)
                        if after > before + 1e-12:
                            state.groups[gi], state.groups[gj] = swapped_a, swapped_b
                            improved = True
                            break
                    else:
                        continue
                    break

        if not improved:
            break  # local optimum


def _legal(state: _Allocation, group: list[int]) -> bool:
    """Whether a group respects the per-institution quota, if there is one."""
    if state.quota is None:
        return True
    counts: dict[str, int] = {}
    for i in group:
        home = state.institutions[i]
        counts[home] = counts.get(home, 0) + 1
        if counts[home] > state.quota:
            return False
    return True


# --- Reporting -------------------------------------------------------------


def describe(
    groups: list[Group],
    similarity: np.ndarray,
    topic_sets: list[set[str]],
    institutions: list[str],
) -> dict[str, float]:
    """Summarise an allocation, for comparing strategies against each other."""
    if not groups:
        return {}
    sizes = [len(g) for g in groups]
    return {
        "n_groups": len(groups),
        "mean_size": float(np.mean(sizes)),
        "coherence": float(np.mean([coherence(g.members, similarity) for g in groups])),
        "diversity": float(np.mean([diversity(g.members, topic_sets) for g in groups])),
        "institution_span": float(
            np.mean([institution_span(g.members, institutions) for g in groups])
        ),
        # The share of groups that are not single-institution: the headline
        # number for an alliance whose stated purpose is breaking silos.
        "pct_cross_institutional": 100.0
        * float(np.mean([institution_span(g.members, institutions) > 1 for g in groups])),
    }
