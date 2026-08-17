"""The one thing RIGAD does: point it at a folder of drafts, get advice back.

    from rigad import analyse
    result = analyse("~/my-drafts")
    result.show()

For each draft it suggests conference tracks and EUTOPIA mentors. Given enough
drafts it also proposes working groups. Nothing else needs configuring.
"""

from __future__ import annotations

import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .allocate import Group, allocate
from .config import REPO_ROOT
from .discover import Mentor, MentorMatch, build_index as build_mentor_index, load_mentors
from .documents import Draft, load_drafts
from .tracks import Track, TrackMatch, ambiguity, build_index as build_track_index, load_tracks

DEFAULT_TRACKS = REPO_ROOT / "data" / "tracks" / "icis2026.json"
DEFAULT_MENTORS = REPO_ROOT / "data" / "mentors" / "eutopia_mentors.json"

# Below this many drafts, grouping is not meaningful — you cannot form a
# balanced set of working groups out of a handful of people.
MIN_DRAFTS_FOR_GROUPING = 5
DEFAULT_GROUP_SIZE = 4

# Confidence bands for a track recommendation, taken from the margin
# distribution measured over 3,622 real papers (see docs/track_metrics.json).
# "High" is the top quartile of margins; "moderate" is above the median.
# These are not round numbers because they are measured, not chosen — and the
# evaluation showed wide-margin drafts really are matched more consistently
# (0.458) than narrow-margin ones (0.351), which is what makes the label
# worth printing at all.
#
# Re-derive them with scripts/evaluate_tracks.py if the corpus or the track
# set changes.
MARGIN_HIGH = 0.038
MARGIN_MODERATE = 0.020


def confidence(margin: float) -> str:
    """Turn the gap between the top two matches into a word."""
    if margin >= MARGIN_HIGH:
        return "high"
    if margin >= MARGIN_MODERATE:
        return "moderate"
    return "low"


@dataclass
class DraftResult:
    """Everything RIGAD has to say about one draft."""

    draft: Draft
    tracks: list[TrackMatch]
    mentors: list[MentorMatch]
    # Named on every draft, not only in the run header: a reader looking at one
    # draft's recommendations should not have to scroll up to learn whose
    # tracks these are. It also stops "Suggested tracks" reading as though the
    # tool invented the categories itself.
    conference: str = ""

    @property
    def margin(self) -> float:
        return ambiguity(self.tracks)

    @property
    def confidence(self) -> str:
        return confidence(self.margin)

    def show(self, width: int = 78) -> None:
        """Print the recommendations for this draft.

        Long text is wrapped rather than clipped: a track's closest topic is
        the whole reason the match is understandable, and cutting it mid-word
        throws that away.
        """
        print(f"\n{'─' * width}")
        for line in textwrap.wrap(self.draft.title, width):
            print(line)
        print(f"  {self.draft.path.name}"
              + (f"  ·  {self.draft.institution}" if self.draft.institution else ""))

        label = f"Suggested {self.conference} tracks" if self.conference else "Suggested tracks"
        print(f"\n  {label}")
        for i, m in enumerate(self.tracks):
            marker = "→" if i == 0 else " "
            print(f"   {marker} {m.track.name:<48} {m.score:.3f}")
            for line in textwrap.wrap(m.matched_topic, width - 14):
                print(f"       {line}")

        note = {
            "high": "the leading track is clearly ahead",
            "moderate": "the leading track is somewhat ahead",
            "low": "the top tracks are close — choose by the audience you want",
        }[self.confidence]
        print(f"\n   confidence: {self.confidence.upper()} "
              f"(margin {self.margin:.3f}) — {note}")

        if self.mentors:
            print("\n  Possible mentors at other EUTOPIA institutions")
            for m in self.mentors:
                print(f"     {m.mentor.name[:26]:28s} {m.mentor.institution:10s} "
                      f"{m.score:.3f}  {m.mentor.basis[:38]}")
                if m.shared_topics:
                    print(f"       shared interests: {', '.join(m.shared_topics[:3])}")


@dataclass
class Analysis:
    """The result of analysing a folder."""

    results: list[DraftResult]
    groups: list[Group] = field(default_factory=list)
    skipped: list[tuple[Path, str]] = field(default_factory=list)
    conference: str = ""

    def __len__(self) -> int:
        return len(self.results)

    def show(self, *, max_drafts: int | None = None) -> None:
        """Print the whole analysis."""
        print(f"RIGAD — {len(self.results)} draft(s)"
              + (f", tracks from {self.conference}" if self.conference else ""))

        for result in self.results[:max_drafts]:
            result.show()

        if self.groups:
            self.show_groups()
        elif len(self.results) < MIN_DRAFTS_FOR_GROUPING:
            print(f"\n{'─' * 72}")
            print(f"Grouping needs at least {MIN_DRAFTS_FOR_GROUPING} drafts "
                  f"(this folder has {len(self.results)}).")

    def show_groups(self) -> None:
        print(f"\n{'═' * 72}")
        print(f"Suggested working groups ({len(self.groups)})")
        for n, group in enumerate(self.groups, 1):
            print(f"\n  Group {n}")
            # Name the group by the track its drafts most often matched. The
            # track name is short and already meaningful to an organiser,
            # unlike the raw facet text behind a match.
            tracks = Counter(self.results[i].tracks[0].track.name for i in group.members)
            theme, shared = tracks.most_common(1)[0]
            qualifier = "" if shared > 1 else " (mixed)"
            print(f"    theme: {theme}{qualifier}")
            for i in group.members:
                result = self.results[i]
                where = f"  ·  {result.draft.institution}" if result.draft.institution else ""
                title = textwrap.shorten(result.draft.title, 56, placeholder="…")
                print(f"     {title}{where}")
                print(f"       {result.tracks[0].track.name}")


def analyse(
    folder: str | Path,
    *,
    tracks_file: str | Path = DEFAULT_TRACKS,
    mentors_file: str | Path | None = DEFAULT_MENTORS,
    top_tracks: int = 3,
    top_mentors: int = 3,
    group_size: int = DEFAULT_GROUP_SIZE,
    backend: str = "auto",
    verbose: bool = True,
) -> Analysis:
    """Analyse every draft in ``folder``.

    Suggests tracks and mentors for each draft, and — once there are at least
    ``MIN_DRAFTS_FOR_GROUPING`` of them — proposes working groups. When drafts
    sit in institution subfolders, groups are required to cross institutions.
    """
    drafts, skipped = load_drafts(folder, verbose=verbose)
    if not drafts:
        return Analysis(results=[], skipped=skipped)

    tracks, conference = load_tracks(tracks_file)
    track_index = build_track_index(tracks, backend=backend)
    draft_texts = [d.text for d in drafts]
    track_vectors = track_index.embed_drafts(draft_texts)

    mentor_index = None
    mentor_vectors = None
    if mentors_file and Path(mentors_file).exists():
        mentor_index = build_mentor_index(load_mentors(mentors_file), backend=backend)
        mentor_vectors = mentor_index.embed_drafts(draft_texts)

    results: list[DraftResult] = []
    for i, draft in enumerate(drafts):
        matches = track_index.match(track_vectors[i], top_k=top_tracks)
        # Topics the draft is associated with, borrowed from its best-matching
        # tracks — the only topic vocabulary available for an unpublished draft.
        draft_topics = {m.matched_topic for m in matches}

        mentors: list[MentorMatch] = []
        if mentor_index is not None:
            mentors = mentor_index.match(
                mentor_vectors[i],
                top_k=top_mentors,
                exclude_institution=draft.institution,
                draft_topics=draft_topics,
            )

        results.append(
            DraftResult(draft=draft, tracks=matches, mentors=mentors, conference=conference)
        )

    groups: list[Group] = []
    if len(drafts) >= MIN_DRAFTS_FOR_GROUPING:
        institutions = [d.institution or "unknown" for d in drafts]
        # Each draft's topic vocabulary comes from the tracks it matched, so
        # the diversity term still has something to work with.
        topic_sets = [{m.matched_topic for m in r.tracks} for r in results]
        # With no institution information every draft is "unknown", so the
        # quota would make grouping impossible; lift it in that case.
        quota = None if len(set(institutions)) > 1 else len(drafts)
        groups = allocate(
            track_vectors, topic_sets, institutions,
            group_size=group_size, quota=quota, seed=0,
        )

    return Analysis(results=results, groups=groups, skipped=skipped, conference=conference)
