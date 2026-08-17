"""Match a draft paper to the conference tracks it should be submitted to.

Choosing a track is a real and recurring problem. A large conference has
twenty-odd tracks with overlapping scope; the descriptions are long; and a
doctoral student submitting for the first time has no feel for where their
work will get read sympathetically. Submitting to the wrong track means
reviewers with the wrong expertise, and there is no second chance that year.

The matching works on **facets** rather than whole tracks. A track description
like "Social media" runs to five thousand words and covers a dozen distinct
concerns; a single vector for all of it is a blur, and a draft about one of
those concerns matches it only weakly. So each track is split into facets — its
prose description, plus every "topic of interest" it lists — and a draft's
score for a track is its best score against any one facet.

In practice the description usually carries the match, because it is twenty
times longer than a topic phrase and long text scores higher against a long
draft. So alongside the ranking each match also reports the track's *closest
listed topic*: "of the topics this track invites, yours sits nearest to data
governance and stewardship" is far more use to an author than a number, even
though it is not what decided the ranking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embed import embed_texts

# Facets shorter than this are things like "Other." — too generic to match on
# and they attract unrelated drafts.
MIN_FACET_CHARS = 25


@dataclass
class Facet:
    """One separately-matchable piece of a track.

    ``is_topic`` is carried alongside the text rather than encoded into it:
    a marker placed inside the string would be embedded with everything else
    and shift the vector. Metadata about a facet must never end up in the
    text being embedded.
    """

    text: str
    is_topic: bool


@dataclass
class Track:
    """One conference track, and the text describing what belongs in it."""

    id: str
    name: str
    description: str
    topics: list[str]

    def facets(self) -> list[Facet]:
        """The separately-matchable pieces of this track.

        The description is included whole. Each topic of interest is prefixed
        with the track name so a bare phrase like "Trust" still carries some
        context into the embedding.
        """
        pieces: list[Facet] = []
        if len(self.description) >= MIN_FACET_CHARS:
            pieces.append(Facet(f"{self.name}. {self.description}", is_topic=False))
        for topic in self.topics:
            if len(topic) >= MIN_FACET_CHARS:
                pieces.append(Facet(f"{self.name}: {topic}", is_topic=True))
        return pieces or [Facet(self.name, is_topic=False)]


@dataclass
class TrackMatch:
    """A suggested track, with the evidence for suggesting it."""

    track: Track
    score: float
    matched_facet: str
    matched_on_topic: bool = False
    closest_topic: str = ""

    @property
    def matched_topic(self) -> str:
        """What matched, phrased for a reader.

        A named topic of interest is strong, specific evidence. Matching the
        overall description instead means the draft fits the track's general
        scope without hitting any of its listed topics — weaker evidence, and
        the wording says so rather than dumping the description back.
        """
        if self.matched_on_topic:
            _, _, tail = self.matched_facet.partition(": ")
            return tail or self.matched_facet
        if self.closest_topic:
            return f"overall scope; closest listed topic: {self.closest_topic}"
        return "the track's overall scope"


@dataclass
class TrackIndex:
    """Tracks plus their embedded facets, ready to match drafts against.

    ``mean`` is the facet centroid, kept so drafts can be centred the same way
    the facets were — see ``rigad.embed.center``. Without it the top three
    tracks for any draft sit within about 0.006 of each other and the ranking
    is not meaningful; with it that gap is roughly five times wider.
    """

    tracks: list[Track]
    facet_vectors: np.ndarray
    facet_texts: list[str]
    facet_track: list[int]  # index into `tracks` for each facet row
    backend: str
    facet_is_topic: list[bool] = field(default_factory=list)
    embedder: object | None = None
    mean: np.ndarray | None = None

    def embed_drafts(self, texts: list[str]) -> np.ndarray:
        """Embed draft texts into the same space as this index.

        Uses the embedder fitted on the track facets, so drafts and facets are
        genuinely comparable — re-fitting per call would put them in unrelated
        spaces and the scores would be meaningless.
        """
        from .embed import normalize_rows

        vectors = self.embedder.transform(texts, kind="query")
        if self.mean is None:
            return vectors
        return normalize_rows(vectors - self.mean)

    def match(self, draft_vector: np.ndarray, top_k: int = 3) -> list[TrackMatch]:
        """Rank tracks for one already-embedded draft.

        A track's score is its best facet's score. In practice that is nearly
        always the track description: descriptions run to a couple of thousand
        characters against about a hundred for a topic of interest, and long
        text scores systematically higher against a long draft (measured on
        this corpus, descriptions average 0.23 and topics -0.03).

        Ranking deliberately uses that plain maximum rather than standardising
        the two kinds to compete on equal terms: per-kind scores are noisy for
        tracks that list only a handful of topics, and the ranking degrades.
        The nearest listed topic is instead reported alongside, as supporting
        detail rather than as the deciding factor.
        """
        facet_scores = self.facet_vectors @ draft_vector

        # Best facet per track — a track is as relevant as its most relevant part.
        best_score = np.full(len(self.tracks), -np.inf)
        best_facet = np.zeros(len(self.tracks), dtype=int)
        for facet_index, track_index in enumerate(self.facet_track):
            if facet_scores[facet_index] > best_score[track_index]:
                best_score[track_index] = facet_scores[facet_index]
                best_facet[track_index] = facet_index

        order = np.argsort(best_score)[::-1][:top_k]
        return [
            TrackMatch(
                track=self.tracks[i],
                score=float(best_score[i]),
                matched_facet=self.facet_texts[best_facet[i]],
                matched_on_topic=(
                    self.facet_is_topic[best_facet[i]] if self.facet_is_topic else False
                ),
                closest_topic=self._closest_topic(i, facet_scores),
            )
            for i in order
        ]

    def _closest_topic(self, track_index: int, facet_scores: np.ndarray) -> str:
        """The track's own listed topic that best fits this draft, if any.

        Reported as supporting detail — "of the topics this track lists, yours
        is nearest to this one" — which is far more use to an author than the
        track's general blurb, even though it is not what decided the ranking.
        """
        if not self.facet_is_topic:
            return ""
        best, best_score = "", -np.inf
        for i, owner in enumerate(self.facet_track):
            if owner != track_index or not self.facet_is_topic[i]:
                continue
            if facet_scores[i] > best_score:
                best_score = facet_scores[i]
                _, _, tail = self.facet_texts[i].partition(": ")
                best = tail
        return best


def load_tracks(path: Path) -> tuple[list[Track], str]:
    """Read a track corpus file. Returns the tracks and the conference name."""
    payload = json.loads(Path(path).read_text())
    tracks = [
        Track(
            id=t["id"],
            name=t["name"],
            description=t.get("description", ""),
            topics=t.get("topics", []),
        )
        for t in payload["tracks"]
    ]
    return tracks, payload.get("conference", "")


def build_index(
    tracks: list[Track], *, backend: str = "auto", centre: bool = True
) -> TrackIndex:
    """Embed every facet of every track.

    Facets are embedded as passages (the side being searched) and centred on
    their own centroid; drafts are then embedded as queries and centred on the
    same point via ``TrackIndex.embed_drafts``.
    """
    from .embed import center as _center
    from .embed import fit_embedder

    facet_texts: list[str] = []
    facet_track: list[int] = []
    facet_is_topic: list[bool] = []
    for track_index, track in enumerate(tracks):
        for facet in track.facets():
            facet_texts.append(facet.text)
            facet_track.append(track_index)
            facet_is_topic.append(facet.is_topic)

    embedder = fit_embedder(facet_texts, backend=backend)
    vectors = embedder.transform(facet_texts, kind="passage")

    mean = None
    if centre:
        vectors, mean = _center(vectors)

    return TrackIndex(
        tracks=tracks,
        facet_vectors=vectors,
        facet_texts=facet_texts,
        facet_track=facet_track,
        facet_is_topic=facet_is_topic,
        backend=embedder.backend,
        embedder=embedder,
        mean=mean,
    )


def ambiguity(matches: list[TrackMatch]) -> float:
    """How close the runner-up is to the winner.

    Near zero means the top two tracks fit about equally well — which is worth
    telling the author, because it usually means the draft genuinely straddles
    two communities and the choice should be made on which audience they want,
    not on which number is larger.
    """
    if len(matches) < 2:
        return 1.0
    return matches[0].score - matches[1].score
