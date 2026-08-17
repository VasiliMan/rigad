"""Checks on track matching.

These run on the TF-IDF backend so the suite stays fast and needs no model
download. Behaviour that depends on the neural model is asserted only where it
holds for both backends.
"""

import json

import numpy as np
import pytest

from rigad.tracks import Track, TrackIndex, ambiguity, build_index, load_tracks

TRACKS = [
    Track(
        id="security",
        name="Cybersecurity and Privacy",
        description=(
            "Research on security breaches, encryption, threat detection, privacy "
            "regulation, and how organisations protect information assets from attack."
        ),
        topics=["Data breach notification and response", "Privacy-preserving computation"],
    ),
    Track(
        id="health",
        name="Healthcare and Wellbeing",
        description=(
            "Research on clinical information systems, electronic health records, "
            "telemedicine, patient outcomes, and digital health interventions."
        ),
        topics=["Electronic health record adoption in hospitals", "Telemedicine and remote care"],
    ),
    Track(
        id="learning",
        name="Digital Learning and Pedagogy",
        description=(
            "Research on online education, learning analytics, educational technology, "
            "student engagement in digital classrooms, and curriculum design."
        ),
        topics=["Learning analytics and student performance", "Online course design"],
    ),
]


@pytest.fixture(scope="module")
def index() -> TrackIndex:
    return build_index(TRACKS, backend="tfidf")


def test_every_track_contributes_facets(index):
    assert set(index.facet_track) == set(range(len(TRACKS)))
    assert len(index.facet_texts) == len(index.facet_vectors)


def test_facets_split_description_and_topics():
    facets = TRACKS[0].facets()
    # One for the description, one per topic of interest.
    assert len(facets) == 1 + len(TRACKS[0].topics)
    assert any("Data breach" in f.text for f in facets)
    # Exactly one description facet; the rest are topics.
    assert sum(not f.is_topic for f in facets) == 1
    # The topic marker must never appear in the text that gets embedded.
    assert all("topic of interest" not in f.text.lower() for f in facets)


def test_track_with_no_usable_text_still_yields_a_facet():
    bare = Track(id="x", name="Mystery Track", description="", topics=["Other."])
    facets = bare.facets()
    assert [f.text for f in facets] == ["Mystery Track"]
    assert not facets[0].is_topic


def test_match_returns_requested_number_of_tracks(index):
    vector = index.embed_drafts(["A study of hospital record systems."])[0]
    assert len(index.match(vector, top_k=2)) == 2


def test_match_ranks_by_descending_score(index):
    vector = index.embed_drafts(["Encryption and data breaches in firms."])[0]
    scores = [m.score for m in index.match(vector, top_k=3)]
    assert scores == sorted(scores, reverse=True)


def test_obviously_clinical_draft_matches_the_health_track(index):
    draft = (
        "We study the adoption of electronic health records across hospitals and "
        "the effect of telemedicine on patient outcomes in clinical care."
    )
    vector = index.embed_drafts([draft])[0]
    assert index.match(vector, top_k=1)[0].track.id == "health"


def test_obviously_security_draft_matches_the_security_track(index):
    draft = (
        "This paper analyses data breach notification behaviour and encryption "
        "practices used to protect information assets from attackers."
    )
    vector = index.embed_drafts([draft])[0]
    assert index.match(vector, top_k=1)[0].track.id == "security"


def test_matched_facet_is_reported_and_belongs_to_the_winning_track(index):
    vector = index.embed_drafts(["Learning analytics in online courses."])[0]
    best = index.match(vector, top_k=1)[0]
    assert best.matched_facet in index.facet_texts
    assert best.matched_facet.startswith(best.track.name)


def test_matched_topic_reports_a_named_topic_of_interest():
    from rigad.tracks import TrackMatch

    match = TrackMatch(
        track=TRACKS[0], score=0.5,
        matched_facet="Cybersecurity and Privacy: Privacy-preserving computation",
        matched_on_topic=True,
    )
    assert match.matched_topic == "Privacy-preserving computation"


def test_closest_topic_is_reported_when_the_description_won(index):
    """The description usually wins, so the nearest topic is the useful detail."""
    vector = index.embed_drafts(
        ["A broad essay about protecting organisational information assets."]
    )[0]
    best = index.match(vector, top_k=1)[0]
    assert best.closest_topic
    assert best.closest_topic in " ".join(best.track.topics)


def test_matched_topic_says_so_when_only_the_overall_scope_matched():
    """A description hit is weaker evidence, and must not echo the description."""
    from rigad.tracks import TrackMatch

    match = TrackMatch(track=TRACKS[0], score=0.5,
                       matched_facet=f"{TRACKS[0].name}. {TRACKS[0].description}")
    assert not match.matched_on_topic
    assert "overall scope" in match.matched_topic
    assert TRACKS[0].description not in match.matched_topic


def test_ambiguity_is_the_gap_between_first_and_second(index):
    vector = index.embed_drafts(["Encryption and data breaches."])[0]
    matches = index.match(vector, top_k=3)
    assert ambiguity(matches) == pytest.approx(matches[0].score - matches[1].score)


def test_ambiguity_of_a_single_match_is_maximal():
    from rigad.tracks import TrackMatch

    assert ambiguity([TrackMatch(track=TRACKS[0], score=0.4, matched_facet="x")]) == 1.0


def test_centring_widens_the_gap_between_top_tracks():
    """Centring is what makes the ranking decisive — assert it actually does."""
    drafts = [
        "Electronic health records and telemedicine in hospitals.",
        "Encryption, breaches and privacy regulation in firms.",
        "Online course design and learning analytics for students.",
    ]

    gaps = {}
    for centre in (False, True):
        idx = build_index(TRACKS, backend="tfidf", centre=centre)
        vectors = idx.embed_drafts(drafts)
        gaps[centre] = np.mean([ambiguity(idx.match(v, top_k=3)) for v in vectors])

    assert gaps[True] > gaps[False]


def test_load_tracks_reads_a_corpus_file(tmp_path):
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps({
        "conference": "Test Conf 2026",
        "tracks": [{"id": "a", "name": "Track A", "description": "About A.", "topics": ["A1"]}],
    }))
    tracks, conference = load_tracks(path)
    assert conference == "Test Conf 2026"
    assert tracks[0].name == "Track A"
    assert tracks[0].topics == ["A1"]


def test_shipped_icis_corpus_is_well_formed():
    from rigad.config import REPO_ROOT

    tracks, conference = load_tracks(REPO_ROOT / "data" / "tracks" / "icis2026.json")
    assert "ICIS" in conference
    assert len(tracks) >= 20
    # Each track must be distinguishable: unique name and its own description.
    assert len({t.name for t in tracks}) == len(tracks)
    assert len({t.description for t in tracks}) == len(tracks)
    assert all(len(t.description) > 100 for t in tracks)


# --- degenerate documents ---------------------------------------------------
# The tool reads whatever a user's folder contains. Nothing in here should be
# able to put NaN or inf into a similarity computation, because the user sees
# that as "divide by zero encountered in matmul" and it explains nothing.


def test_a_draft_sharing_no_vocabulary_scores_without_warnings(index):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        vector = index.embed_drafts(["zzzz qqqq xxxx"])[0]
        matches = index.match(vector, top_k=3)

    assert len(matches) == 3
    assert all(np.isfinite(m.score) for m in matches)


def test_normalise_survives_nan_and_infinity():
    from rigad.embed import normalize_rows

    rows = np.array([[np.nan, 1.0], [np.inf, 0.0], [0.0, 0.0]], dtype=np.float32)
    out = normalize_rows(rows)
    assert np.isfinite(out).all()
