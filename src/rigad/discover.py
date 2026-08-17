"""Suggest a mentor for a draft, from the EUTOPIA researcher pool.

Given a draft paper, find the researchers across the alliance whose recent work
is closest to it. The pool ships with the repository (`data/mentors/`) so this
needs no API access at run time.

The structure deliberately mirrors ``rigad.tracks``: a pool is embedded once as
passages, drafts are embedded as queries into the same centred space, and a
match reports the topics the two have in common as its evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Role titles suggesting someone senior enough to mentor. Matched
# case-insensitively as substrings.
SENIOR_ROLE_MARKERS = ("professor", "reader", "senior lecturer", "docent", "director")

from .config import MIN_WORKS_MENTOR_NOT_DEPARTMENTAL


@dataclass
class Mentor:
    """A researcher who could be suggested as a mentor.

    ``in_is_department`` distinguishes the two ways someone enters the pool.
    Departmental staff come from a published staff directory. The others were
    found because they publish information-systems research from elsewhere in
    the same university — a medical school's health-informatics group, say.

    Both are worth suggesting: cross-departmental collaborators are, if
    anything, harder to discover than cross-institutional ones. But they are
    not the same kind of suggestion, and a student deserves to know which they
    are looking at, so the distinction is always shown rather than hidden.
    """

    name: str
    institution: str
    profile: str
    role: str = ""
    topics: list[str] = field(default_factory=list)
    n_works: int = 0
    openalex_id: str = ""
    in_is_department: bool = True

    @property
    def basis(self) -> str:
        """Why this person is in the pool, phrased for a reader."""
        if self.in_is_department:
            return self.role or "IS department"
        return "publishes IS research (not IS dept)"

    @property
    def is_senior(self) -> bool:
        """Whether the role title suggests seniority.

        Directories are not consistent across institutions, so this is a
        heuristic. Someone found through their publications has no job title
        at all, and treating a missing title as "senior" would silently exempt
        every one of them from the filter — so for them the evidence is their
        publication record instead.
        """
        if not self.in_is_department:
            return self.n_works >= MIN_WORKS_MENTOR_NOT_DEPARTMENTAL
        if not self.role:
            return True
        return any(marker in self.role.lower() for marker in SENIOR_ROLE_MARKERS)


@dataclass
class MentorMatch:
    """A suggested mentor and the evidence for suggesting them."""

    mentor: Mentor
    score: float
    shared_topics: list[str]


@dataclass
class MentorIndex:
    """An embedded mentor pool, ready to match drafts against."""

    mentors: list[Mentor]
    vectors: np.ndarray
    backend: str
    embedder: object | None = None
    mean: np.ndarray | None = None

    def embed_drafts(self, texts: list[str]) -> np.ndarray:
        """Embed drafts into the same space as the pool."""
        from .embed import normalize_rows

        vectors = self.embedder.transform(texts, kind="query")
        if self.mean is None:
            return vectors
        return normalize_rows(vectors - self.mean)

    def match(
        self,
        draft_vector: np.ndarray,
        *,
        top_k: int = 3,
        senior_only: bool = True,
        departmental_only: bool = False,
        exclude_institution: str | None = None,
        draft_topics: set[str] | None = None,
    ) -> list[MentorMatch]:
        """Rank mentors for one embedded draft.

        ``exclude_institution`` drops the author's own institution — a
        researcher already knows who works down the corridor, and the point of
        an alliance is the people they would otherwise never meet.

        ``departmental_only`` restricts suggestions to confirmed IS department
        staff, excluding researchers found through their publications
        elsewhere in the same universities.
        """
        scores = self.vectors @ draft_vector
        draft_topics = draft_topics or set()

        candidates: list[MentorMatch] = []
        for i, mentor in enumerate(self.mentors):
            if departmental_only and not mentor.in_is_department:
                continue
            if senior_only and not mentor.is_senior:
                continue
            if exclude_institution and mentor.institution.lower() == exclude_institution.lower():
                continue
            candidates.append(
                MentorMatch(
                    mentor=mentor,
                    score=float(scores[i]),
                    shared_topics=sorted(draft_topics & set(mentor.topics)),
                )
            )

        candidates.sort(key=lambda m: m.score, reverse=True)
        return candidates[:top_k]


def load_mentors(path: str | Path) -> list[Mentor]:
    """Read a mentor pool file."""
    payload = json.loads(Path(path).read_text())
    return [
        Mentor(
            name=m["name"],
            institution=m.get("institution", ""),
            profile=m.get("profile", ""),
            role=m.get("role", ""),
            topics=m.get("topics", []),
            n_works=m.get("n_works", 0),
            openalex_id=m.get("openalex_id", ""),
            in_is_department=m.get("in_is_department", True),
        )
        for m in payload["mentors"]
    ]


def build_index(
    mentors: list[Mentor], *, backend: str = "auto", centre: bool = True
) -> MentorIndex:
    """Embed a mentor pool once, for matching many drafts against."""
    from .embed import center, fit_embedder

    texts = [f"{m.name}. {m.profile}" for m in mentors]
    embedder = fit_embedder(texts, backend=backend)
    vectors = embedder.transform(texts, kind="passage")

    mean = None
    if centre:
        vectors, mean = center(vectors)

    return MentorIndex(
        mentors=mentors,
        vectors=vectors,
        backend=embedder.backend,
        embedder=embedder,
        mean=mean,
    )
