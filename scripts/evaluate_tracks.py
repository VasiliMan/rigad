"""Evaluate draft-to-track matching without any ground-truth track labels.

Nobody publishes a dataset of papers labelled with the ICIS track they were
submitted to, so we cannot measure accuracy directly. We can measure something
almost as useful, and arguably harder to game:

**Consistency.** OpenAlex assigns every paper its own topic labels, entirely
independently of this tool. Two papers that OpenAlex says are about the same
topic ought to be routed to the same track. So for each OpenAlex topic with
enough papers, we ask what share of them land on the same top track, and
compare that against what you would get by assigning tracks at random.

**Is the margin worth reporting?** The tool tells authors how far ahead the
winning track was. That is only worth showing if a wide margin really does
mean a more reliable answer — so we split the drafts at the median margin and
check whether the confident half is measurably more consistent than the
uncertain half. If it is not, the number should not be on screen.

**Do the institutions look different, in the way they should?** Departments
specialise, so papers will not spread evenly across tracks and a flat
distribution would be the suspicious result, not the good one. If the matcher
is reading real topical structure rather than noise, the three institutions
should show *distinguishable and plausible* track profiles. Nothing in the
pipeline sees an institution label, so this is an independent check — and a
stronger one than consistency, because the expected shape is known in advance.

    python scripts/evaluate_tracks.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from rigad.config import FIGURES_DIR, REPO_ROOT, SAMPLE_DIR
from rigad.corpus import load_profiles
from rigad.tracks import ambiguity, build_index, load_tracks

# An OpenAlex topic needs at least this many papers before its consistency
# score means anything.
MIN_PAPERS_PER_TOPIC = 5

# Chi-square needs reasonable expected cell counts; pool out rarer tracks.
MIN_PAPERS_PER_TRACK_FOR_TEST = 15
N_RANDOM_REPEATS = 50


def institution_profiles(
    institutions: list[str], top_track: list[str], track_names: list[str]
) -> dict:
    """Test whether the institutions' track distributions genuinely differ.

    A chi-square test of independence on the institution x track table, with
    Cramer's V as the effect size — the p-value alone only says the profiles
    are not identical, which at this sample size is easy. V says how far from
    identical, and a moderate value is the expected answer: three information
    systems departments should overlap substantially while still having their
    own emphases.

    Sparse tracks are pooled out before testing, since chi-square is unreliable
    when expected cell counts are tiny.
    """
    from scipy.stats import chi2_contingency

    order = sorted(set(institutions))
    table = np.array(
        [[sum(1 for i, t in zip(institutions, top_track) if i == inst and t == name)
          for name in track_names]
         for inst in order]
    )
    keep = table.sum(axis=0) >= MIN_PAPERS_PER_TRACK_FOR_TEST
    tested = table[:, keep]
    kept_names = [n for n, k in zip(track_names, keep) if k]

    chi2, p_value, dof, _ = chi2_contingency(tested)
    total = tested.sum()
    cramers_v = float(np.sqrt(chi2 / (total * (min(tested.shape) - 1))))

    corpus_share = tested.sum(axis=0) / total
    over: dict[str, dict] = {}
    for row, inst in enumerate(order):
        share = tested[row] / tested[row].sum()
        # Ignore tracks too rare within an institution for the ratio to mean
        # anything — a single paper can otherwise show a huge "lift".
        lift = np.where(share > 0.04, share / corpus_share, 0.0)
        j = int(np.argmax(lift))
        over[inst] = {
            "track": kept_names[j],
            "share": float(share[j]),
            "corpus_share": float(corpus_share[j]),
            "lift": float(lift[j]),
        }

    return {
        "chi2": float(chi2), "dof": int(dof), "p_value": float(p_value),
        "cramers_v": cramers_v, "n": int(total),
        "n_tracks_tested": int(keep.sum()), "over_represented": over,
    }


def consistency(assignments: dict[str, list[str]]) -> float:
    """Mean share of a topic's papers that share the same top track."""
    scores = [
        Counter(tracks).most_common(1)[0][1] / len(tracks)
        for tracks in assignments.values()
        if len(tracks) >= MIN_PAPERS_PER_TOPIC
    ]
    return float(np.mean(scores)) if scores else 0.0


def main() -> None:
    tracks, conference = load_tracks(REPO_ROOT / "data" / "tracks" / "icis2026.json")
    index = build_index(tracks)
    print(f"{conference}: {len(tracks)} tracks, {len(index.facet_texts)} facets "
          f"({index.backend})")

    people = load_profiles(SAMPLE_DIR / "profiles.json")
    # Keep each paper's institution so we can compare departmental profiles.
    tagged: dict[str, tuple] = {}
    for person in people:
        for work in person.works:
            if work.topics:
                tagged.setdefault(work.id, (person.institution_short, work))
    institutions_of = [inst for inst, _ in tagged.values()]
    works = [w for _, w in tagged.values()]
    print(f"drafts: {len(works)} distinct publications with topic labels")

    vectors = index.embed_drafts([w.text for w in works])

    top_track: list[str] = []
    margins: list[float] = []
    for vector in vectors:
        matches = index.match(vector, top_k=3)
        top_track.append(matches[0].track.name)
        margins.append(ambiguity(matches))

    margins_array = np.array(margins)

    # --- Consistency against OpenAlex topics --------------------------------
    by_topic: dict[str, list[str]] = defaultdict(list)
    for work, track in zip(works, top_track):
        for topic in work.topics:
            by_topic[topic].append(track)

    scored_topics = {t: v for t, v in by_topic.items() if len(v) >= MIN_PAPERS_PER_TOPIC}
    observed = consistency(by_topic)

    # Random baseline: shuffle the track assignments, keep the topic structure.
    rng = np.random.default_rng(0)
    random_scores = []
    for _ in range(N_RANDOM_REPEATS):
        shuffled = list(top_track)
        rng.shuffle(shuffled)
        by_topic_random: dict[str, list[str]] = defaultdict(list)
        for work, track in zip(works, shuffled):
            for topic in work.topics:
                by_topic_random[topic].append(track)
        random_scores.append(consistency(by_topic_random))
    baseline = float(np.mean(random_scores))

    print(f"\ntopics with >={MIN_PAPERS_PER_TOPIC} papers: {len(scored_topics)}")
    print(f"topic consistency (RIGAD):  {observed:.3f}")
    print(f"topic consistency (random): {baseline:.3f}")
    print(f"lift: {observed / baseline:.2f}x")

    # --- Does the margin predict reliability? -------------------------------
    median_margin = float(np.median(margins_array))
    confident = margins_array >= median_margin

    def subset_consistency(mask: np.ndarray) -> float:
        subset: dict[str, list[str]] = defaultdict(list)
        for work, track, keep in zip(works, top_track, mask):
            if keep:
                for topic in work.topics:
                    subset[topic].append(track)
        return consistency(subset)

    confident_score = subset_consistency(confident)
    uncertain_score = subset_consistency(~confident)
    print(f"\nmedian margin: {median_margin:.3f}")
    print(f"consistency, confident half: {confident_score:.3f}")
    print(f"consistency, uncertain half: {uncertain_score:.3f}")

    # --- How are drafts spread across tracks? -------------------------------
    # Not expected to be flat: three specialised IS departments should cluster.
    # What matters is that no track is unreachable, not that usage is even.
    usage = Counter(top_track)
    print(f"\ntracks receiving at least one draft: {len(usage)} of {len(tracks)}")
    for name, count in usage.most_common(6):
        print(f"  {count:5d}  {name[:52]}")

    # --- Do the institutions differ, plausibly? -----------------------------
    profile = institution_profiles(institutions_of, top_track, [t.name for t in tracks])
    print(f"\ninstitution x track: chi2={profile['chi2']:.1f} "
          f"dof={profile['dof']} p={profile['p_value']:.1e} "
          f"Cramer V={profile['cramers_v']:.3f}")
    for inst, row in profile["over_represented"].items():
        print(f"  {inst:12s} {row['track'][:42]:44s} "
              f"{100 * row['share']:4.1f}% vs {100 * row['corpus_share']:4.1f}% "
              f"({row['lift']:.2f}x)")

    payload = {
        "conference": conference,
        "n_tracks": len(tracks),
        "n_facets": len(index.facet_texts),
        "n_drafts": len(works),
        "backend": index.backend,
        "topic_consistency": observed,
        "topic_consistency_random": baseline,
        "lift": observed / baseline if baseline else None,
        "n_topics_scored": len(scored_topics),
        "median_margin": median_margin,
        "consistency_confident_half": confident_score,
        "consistency_uncertain_half": uncertain_score,
        "tracks_used": len(usage),
        "track_usage": usage.most_common(),
        "institution_profiles": profile,
        "margins": margins,
    }
    FIGURES_DIR.parent.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR.parent / "track_metrics.json"
    out.write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
