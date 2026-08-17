"""Compare RIGAD's allocation against baselines, and measure what the
cross-institutional constraint costs.

Four strategies, same cohort, same group size:

``random``          shuffle people into groups. The floor.
``similarity``      agglomerative clustering on the embeddings. What a naive
                    "match similar people" tool produces — good themes, but
                    uneven group sizes and no attention to who meets whom.
``rigad``           theme-first: coherence, with a small penalty for pairs
                    from the same institution. The default.
``rigad-diverse``   the same optimiser with the diversity term switched on,
                    which deliberately builds groups whose members share no
                    topics — a comparison point for organisers considering
                    that option.

Evaluated at the sizes a real event has. RIGAD groups the people at a
workshop, a PhD course or a summer school — twenty to sixty of them, not the
five hundred in the corpus. The corpus is reference data; a cohort is what
gets grouped, and the constraint behaves quite differently at the two scales.
So each strategy is run on repeated random cohorts drawn from the corpus at
several realistic sizes, and the results averaged.

Then a sweep over beta, which traces the trade-off curve: how much topical
coherence you give up to buy topic diversity within a group.

Writes metrics to docs/metrics.json for the notebooks and the report to read.

    python scripts/evaluate.py [corpus.json] [metrics-out.json]

Defaults to the institutional sweep. Pass ``roster_profiles.json`` to evaluate
on the actual departmental cohort instead.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from rigad.allocate import (
    allocate,
    coherence,
    diversity,
    institution_span,
    same_institution_pairs,
)
from rigad.config import FIGURES_DIR, SAMPLE_DIR
from rigad.corpus import load_profiles
from rigad.embed import embed_texts

GROUP_SIZE = 4
SEED = 0
# Enough repeats to give the random baseline a stable mean without making the
# script slow; the spread across seeds is small at this cohort size.
N_RANDOM_REPEATS = 20

# The sizes real events actually are: a doctoral course, a workshop, a summer
# school. Nothing here is a five-hundred-person cohort.
COHORT_SIZES = (20, 32, 48, 64)

# Independent cohorts drawn per size, so a result is not an artefact of one
# lucky sample.
N_COHORTS = 12

def get_embeddings(texts: list[str], cache: Path) -> tuple[np.ndarray, str]:
    """Embed, caching to disk — the neural model takes ~40s on this corpus."""
    if cache.exists():
        cached = np.load(cache, allow_pickle=False)
        if cached["vectors"].shape[0] == len(texts):
            return cached["vectors"], str(cached["backend"])

    vectors, backend = embed_texts(texts)
    np.savez_compressed(cache, vectors=vectors, backend=backend)
    return vectors, backend


# --- Baselines -------------------------------------------------------------


def random_groups(n: int, group_size: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return [order[i : i + group_size].tolist() for i in range(0, n, group_size)]


def similarity_groups(vectors: np.ndarray, group_size: int) -> list[list[int]]:
    """Unconstrained agglomerative clustering — the naive 'match similar people'.

    Cluster sizes are uneven by nature, which is itself part of the finding:
    an organiser needs groups of roughly equal size, and plain clustering does
    not give them that.
    """
    from sklearn.cluster import AgglomerativeClustering

    n_clusters = max(1, round(len(vectors) / group_size))
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine",
                                     linkage="average").fit_predict(vectors)
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return list(groups.values())


def score(groups: list[list[int]], similarity, topic_sets, institutions) -> dict:
    """Metrics for a list of raw index groups (baselines return these)."""
    usable = [g for g in groups if len(g) > 1]
    if not usable:
        return {}
    sizes = [len(g) for g in groups]
    return {
        "n_groups": len(groups),
        "mean_size": float(np.mean(sizes)),
        "size_spread": float(np.max(sizes) - np.min(sizes)),
        "coherence": float(np.mean([coherence(g, similarity) for g in usable])),
        "same_institution_pairs": float(
            np.mean([same_institution_pairs(g, institutions) for g in usable])
        ),
        "diversity": float(np.mean([diversity(g, topic_sets) for g in usable])),
        "institution_span": float(np.mean([institution_span(g, institutions) for g in usable])),
        "pct_cross_institutional": 100.0
        * float(np.mean([institution_span(g, institutions) > 1 for g in usable])),
    }


def evaluate_cohort(indices, vectors, topic_sets, institutions) -> dict[str, dict]:
    """Run every strategy on one sampled cohort."""
    sub_vectors = vectors[indices]
    sub_topics = [topic_sets[i] for i in indices]
    sub_institutions = [institutions[i] for i in indices]
    similarity = sub_vectors @ sub_vectors.T
    n = len(indices)

    out: dict[str, dict] = {}

    runs = [
        score(random_groups(n, GROUP_SIZE, seed), similarity, sub_topics, sub_institutions)
        for seed in range(5)
    ]
    out["random"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}

    out["similarity"] = score(
        similarity_groups(sub_vectors, GROUP_SIZE), similarity, sub_topics, sub_institutions
    )

    for name, beta in [("rigad", 0.0), ("rigad-diverse", 1.0)]:
        groups = allocate(sub_vectors, sub_topics, sub_institutions,
                          group_size=GROUP_SIZE, alpha=1.0, beta=beta, seed=SEED)
        out[name] = score([g.members for g in groups], similarity, sub_topics, sub_institutions)

    # Ablation: same optimiser with the institution tiebreaker switched off,
    # to show how much of the mixing it is responsible for.
    ignoring = allocate(sub_vectors, sub_topics, sub_institutions,
                        group_size=GROUP_SIZE, institution_penalty=0.0, seed=SEED)
    out["rigad-ignoring-institution"] = score(
        [g.members for g in ignoring], similarity, sub_topics, sub_institutions
    )
    return out


def sample_cohort(rng, institutions: list[str], size: int) -> list[int]:
    """Draw a plausible cohort: a spread of institutions, not one department.

    Sampling uniformly from the corpus would over-represent whichever
    institution contributed most people. A real workshop invites across the
    alliance, so we take a roughly even share from each.
    """
    by_institution: dict[str, list[int]] = {}
    for i, inst in enumerate(institutions):
        by_institution.setdefault(inst, []).append(i)

    per = max(1, size // len(by_institution))
    chosen: list[int] = []
    for members in by_institution.values():
        take = min(per, len(members))
        chosen += list(rng.choice(members, size=take, replace=False))

    # Top up to the requested size from whoever is left.
    remaining = [i for i in range(len(institutions)) if i not in set(chosen)]
    if len(chosen) < size and remaining:
        extra = min(size - len(chosen), len(remaining))
        chosen += list(rng.choice(remaining, size=extra, replace=False))
    return [int(i) for i in chosen[:size]]


def main(corpus: str | None = None, metrics_out: str | None = None) -> None:
    corpus_path = Path(corpus) if corpus else SAMPLE_DIR / "profiles.json"
    people = load_profiles(corpus_path)
    print(f"corpus: {corpus_path.name}")
    texts = [p.profile_text for p in people]
    topic_sets = [set(p.topics) for p in people]
    institutions = [p.institution_key for p in people]

    print(f"cohort: {len(people)} researchers, {len(set(institutions))} institutions")

    t0 = time.time()
    vectors, backend = get_embeddings(
        texts, SAMPLE_DIR / f"embeddings_{corpus_path.stem}.npz"
    )
    print(f"embeddings: {backend}, {vectors.shape} ({time.time() - t0:.0f}s)")

    topic_sets_all = topic_sets
    rng = np.random.default_rng(SEED)

    # --- realistic cohorts --------------------------------------------------
    by_size: dict[int, dict[str, dict]] = {}
    for size in COHORT_SIZES:
        if size > len(people):
            continue
        runs: list[dict[str, dict]] = []
        for _ in range(N_COHORTS):
            indices = sample_cohort(rng, institutions, size)
            runs.append(evaluate_cohort(indices, vectors, topic_sets_all, institutions))

        averaged = {
            strategy: {
                metric: float(np.mean([r[strategy][metric] for r in runs]))
                for metric in runs[0][strategy]
            }
            for strategy in runs[0]
        }
        by_size[size] = averaged

        print(f"\ncohort of {size}  (mean of {N_COHORTS} draws)")
        print(f"  {'strategy':<28}{'coherence':>10}{'same-inst pairs':>17}{'span':>7}{'size':>7}")
        for name, m in averaged.items():
            print(f"  {name:<28}{m['coherence']:>10.3f}{m['same_institution_pairs']:>17.2f}"
                  f"{m['institution_span']:>7.2f}{m['mean_size']:>7.2f}")

    # Headline numbers come from the mid-size cohort — a typical workshop.
    headline_size = 32 if 32 in by_size else max(by_size)
    results = by_size[headline_size]

    # --- trade-off sweep, on one representative cohort ----------------------
    indices = sample_cohort(np.random.default_rng(SEED), institutions, headline_size)
    sub_vectors = vectors[indices]
    sub_topics = [topic_sets_all[i] for i in indices]
    sub_institutions = [institutions[i] for i in indices]
    sub_similarity = sub_vectors @ sub_vectors.T

    sweep = []
    print(f"\nbeta sweep on a cohort of {headline_size}")
    for beta in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        groups = allocate(sub_vectors, sub_topics, sub_institutions,
                          group_size=GROUP_SIZE, alpha=1.0, beta=beta, seed=SEED)
        m = score([g.members for g in groups], sub_similarity, sub_topics, sub_institutions)
        sweep.append({"beta": beta, **m})
        print(f"  beta={beta:<5} coherence={m['coherence']:.3f} diversity={m['diversity']:.3f}")

    FIGURES_DIR.parent.mkdir(parents=True, exist_ok=True)
    out = Path(metrics_out) if metrics_out else FIGURES_DIR.parent / "metrics.json"
    out.write_text(json.dumps(
        {"corpus": corpus_path.name, "corpus_size": len(people),
         "headline_cohort_size": headline_size, "n_cohorts_per_size": N_COHORTS,
         "group_size": GROUP_SIZE, "backend": backend,
         "strategies": results, "by_cohort_size": by_size,
         "beta_sweep": sweep}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
