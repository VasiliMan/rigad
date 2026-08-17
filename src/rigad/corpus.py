"""Build researcher profiles for a known roster of people.

RIGAD starts from a list of participants — a department roster, a summer-school
sign-up sheet, a doctoral cohort — and needs a research profile for each. That
is what this module does: for every person on the roster it finds their
publications in OpenAlex and assembles them into one profile.

Resolving a name to the right researcher is the hard part, because names are
ambiguous and because the obvious method is too expensive. OpenAlex meters
requests against a daily credit budget in which a ``search`` filter costs ten
credits against one for a plain filter — so looking up a few hundred people by
name search exhausts a whole day's budget before finishing. Nothing here uses
search. Instead:

1. Page through everyone whose *current* affiliation is the institution, using
   a plain filter. Two hundred people per credit.
2. Match the roster against that list locally, on accent-folded first-and-last
   name forms. Free, and the institution does the disambiguating: there are
   many David Smiths, but few at Warwick.
3. Fetch each matched author's works by author ID — again a plain filter. This
   recovers work published under previous affiliations, so someone who moved
   recently still gets a complete profile.

A person who cannot be matched is recorded as unresolved rather than guessed
at, so a bad match never silently pollutes the corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .config import (
    CACHE_DIR,
    IS_SUBFIELDS,
    MIN_WORKS_ROSTER_MEMBER,
    MIN_WORKS_TO_LIST_AUTHOR,
    MIN_WORKS_VIA_PUBLICATIONS,
    OPENALEX_BASE,
    OPENALEX_MAILTO,
    Institution,
)

# Only consider reasonably recent work: a profile should reflect what someone
# is doing now, not what they did fifteen years ago.
MIN_YEAR = 2015

# Cap per researcher so that a handful of very prolific people cannot dominate
# the corpus. Most recent first.
MAX_WORKS_PER_PERSON = 40

@dataclass
class Work:
    """One publication, reduced to the fields RIGAD actually uses."""

    id: str
    title: str
    abstract: str
    year: int
    topics: list[str]
    cited_by_count: int = 0

    @property
    def text(self) -> str:
        """Title plus abstract — what gets embedded."""
        return f"{self.title}. {self.abstract}".strip()


@dataclass
class Researcher:
    """A researcher together with the publications we found for them.

    ``roster_confirmed`` records whether this person also appears on the
    published staff list of one of the partner IS departments. The corpus
    contains both: confirmed department members, and other researchers at the
    same universities who publish in the same subfields. Keeping the wider set
    makes mentor matching and reading lists more useful, while the flag lets
    the allocator work on a cohort of people who genuinely belong to the
    departments the pilot is about.
    """

    name: str
    institution_key: str
    institution_short: str
    role: str = ""
    openalex_id: str = ""
    roster_confirmed: bool = False
    works: list[Work] = field(default_factory=list)

    @property
    def n_works(self) -> int:
        return len(self.works)

    @property
    def topics(self) -> list[str]:
        """Distinct OpenAlex topic labels across this researcher's works.

        OpenAlex assigns these independently of anything RIGAD does, which is
        what makes them usable as ground truth when checking whether an
        allocation is thematically sensible.
        """
        seen: dict[str, None] = {}
        for work in self.works:
            for topic in work.topics:
                seen.setdefault(topic, None)
        return list(seen)

    @property
    def profile_text(self) -> str:
        """Text representing this researcher when embedding.

        Newest work first, so that if the text is truncated downstream it is
        current interests that survive.
        """
        ordered = sorted(self.works, key=lambda w: w.year, reverse=True)
        return "\n\n".join(w.text for w in ordered)


# --- OpenAlex access -------------------------------------------------------


# OpenAlex documents a 10-requests/second ceiling, but in practice it starts
# returning 429 well below that if you sustain a burst. Building the whole
# corpus is only a few hundred calls and every response is cached, so one
# request per second costs a few minutes once and nothing thereafter.
_MIN_SECONDS_BETWEEN_REQUESTS = 1.0
_last_request_at = 0.0

# OpenAlex meters requests against a daily credit budget (1000 credits / $0.10
# at the time of writing) and reports the state on every response. Two things
# matter when planning a run:
#
#   * a plain filter costs 1 credit; a `search` filter costs 10. Resolving a
#     few hundred people by name search alone exceeds a whole day's budget,
#     which is why nothing in this module uses search.
#   * exceeding the budget returns 429 until midnight UTC, so a long job must
#     stop while it still has room rather than discovering the wall.
#
# We stop with a clear error at this floor instead of burning the remainder.
CREDIT_FLOOR = 40

# Updated from response headers as requests are made.
credits_remaining: int | None = None


class BudgetExhausted(RuntimeError):
    """Raised when the OpenAlex credit budget is spent, or nearly so."""


def _note_budget(response: httpx.Response) -> None:
    """Record remaining credits, and refuse to spend the last of them."""
    global credits_remaining
    raw = response.headers.get("x-ratelimit-remaining")
    if raw is None:
        return
    try:
        credits_remaining = int(raw)
    except ValueError:
        return
    if credits_remaining < CREDIT_FLOOR:
        raise BudgetExhausted(
            f"OpenAlex credits nearly exhausted ({credits_remaining} left). "
            "Cached progress is saved; re-run after the daily reset (midnight UTC)."
        )


def _throttle() -> None:
    """Space out requests so we stay inside OpenAlex's rate limit."""
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    _last_request_at = time.monotonic()


def _get(client: httpx.Client, url: str, use_cache: bool = True) -> dict:
    """GET with a transparent on-disk cache and retries on transient failures."""
    path = CACHE_DIR / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text())

    last_error: Exception | None = None
    for attempt in range(5):
        _throttle()
        try:
            response = client.get(url, timeout=60.0)
            # Being rate-limited is not a failure, just a request to slow
            # down, so back off harder and for longer than for other errors.
            if response.status_code == 429:
                # Distinguish "slow down" from "out of budget until tomorrow":
                # retrying the latter for an hour accomplishes nothing.
                if "budget" in response.text.lower():
                    raise BudgetExhausted(
                        "OpenAlex daily budget exhausted. Cached progress is saved; "
                        "re-run after the reset (midnight UTC)."
                    )
                time.sleep(float(response.headers.get("retry-after", 5 * (attempt + 1))))
                continue
            _note_budget(response)
            response.raise_for_status()
            payload = response.json()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
            return payload
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(2**attempt)  # 1s, 2s, 4s, 8s, 16s
    raise RuntimeError(f"OpenAlex request failed after 5 attempts: {url}") from last_error


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Rebuild abstract text from OpenAlex's inverted index.

    OpenAlex stores abstracts as ``{word: [positions]}`` rather than plain
    text. Inverting that is lossy at the punctuation level but fine for
    embedding.
    """
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, spots in inverted_index.items() for pos in spots]
    positions.sort()
    return " ".join(word for _, word in positions)


def _parse_work(raw: dict) -> Work | None:
    """Convert an OpenAlex record into a ``Work``, or None if unusable."""
    abstract = reconstruct_abstract(raw.get("abstract_inverted_index"))
    title = raw.get("title") or ""
    # Very short abstracts are usually publisher boilerplate ("No abstract
    # available", copyright lines) and add noise rather than signal.
    if not title or len(abstract) < 200:
        return None
    return Work(
        id=raw["id"],
        title=title,
        abstract=abstract,
        year=raw.get("publication_year") or 0,
        topics=[t["display_name"] for t in raw.get("topics", []) if t.get("display_name")],
        cited_by_count=raw.get("cited_by_count", 0),
    )


def name_keys(name: str) -> set[str]:
    """Comparable forms of a personal name: 'first|last' and 'f|last'.

    Accents are stripped and case folded so that "Ivana Ljubić" and "Ivana
    Ljubic" match — a difference in how two sources store a name, not a
    difference in who the person is.
    """
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    tokens = re.sub(r"[^a-z ]", " ", folded.lower()).split()
    if len(tokens) < 2:
        return set()
    first, last = tokens[0], tokens[-1]
    return {f"{first}|{last}", f"{first[0]}|{last}"}


def fetch_institution_authors(
    client: httpx.Client,
    institution: Institution,
    *,
    min_works: int = MIN_WORKS_TO_LIST_AUTHOR,
    use_cache: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """List everyone whose current affiliation is this institution.

    This is the search-free way to resolve names to OpenAlex author IDs. A
    ``search`` filter costs ten credits per request, so looking up a few
    hundred people by name would exhaust a day's budget on its own; paging
    through the institution's author list costs one credit per 200 people and
    lets the name matching happen locally, for free.

    ``last_known_institutions`` rather than ``affiliations`` because a staff
    directory lists people who are *there now*, and the affiliations filter
    additionally returns everyone who ever passed through.
    """
    filters = (
        f"last_known_institutions.ror:https://ror.org/{institution.ror},"
        f"works_count:>{min_works}"
    )
    authors: list[dict] = []
    cursor = "*"

    while cursor:
        url = (
            f"{OPENALEX_BASE}/authors?filter={filters}"
            f"&per-page=200&cursor={cursor}&mailto={OPENALEX_MAILTO}"
        )
        payload = _get(client, url, use_cache=use_cache)
        results = payload.get("results", [])
        for author in results:
            if author.get("id") and author.get("display_name"):
                authors.append(
                    {
                        "id": author["id"],
                        "name": author["display_name"],
                        "works_count": author.get("works_count", 0),
                    }
                )
        cursor = payload.get("meta", {}).get("next_cursor")
        if not results:
            break

    if verbose:
        print(f"  {institution.short:12s} {len(authors):6d} authors listed")
    return authors


def build_pool_from_works(
    client: httpx.Client,
    institution: Institution,
    *,
    subfields: dict[str, str] | None = None,
    from_date: str = "2020-01-01",
    min_works: int = MIN_WORKS_VIA_PUBLICATIONS,
    max_pages: int = 40,
    use_cache: bool = True,
    verbose: bool = True,
) -> list[Researcher]:
    """Derive an institution's active researchers from its publication output.

    The second way to build a pool, for institutions where no staff directory
    is available — several publish theirs only as JavaScript, or refuse
    automated requests outright.

    Rather than listing people and then finding their papers, this lists the
    institution's papers *in the relevant subfields* and finds the people. The
    result is subtly different and worth stating plainly: it is "researchers
    at this institution active in information systems and management", not
    "members of its IS department". Someone in another department publishing
    in these subfields is included; a department member who has not published
    recently is not.

    It is also far cheaper for a large university — a few thousand works is a
    dozen credits, against hundreds to page through every affiliated author.
    """
    subfields = subfields or IS_SUBFIELDS
    filters = ",".join([
        f"authorships.institutions.ror:https://ror.org/{institution.ror}",
        f"topics.subfield.id:{'|'.join(subfields)}",
        f"from_publication_date:{from_date}",
        "has_abstract:true",
        "type:article",
    ])

    # author id -> (display name, works)
    by_author: dict[str, tuple[str, list[Work]]] = {}
    cursor = "*"
    pages = 0

    while cursor and pages < max_pages:
        url = (
            f"{OPENALEX_BASE}/works?filter={filters}"
            f"&per-page=200&cursor={cursor}&mailto={OPENALEX_MAILTO}"
        )
        payload = _get(client, url, use_cache=use_cache)
        results = payload.get("results", [])
        if not results:
            break

        for raw in results:
            work = _parse_work(raw)
            if work is None:
                continue
            for authorship in raw.get("authorships", []):
                author = authorship.get("author") or {}
                # Only credit the author for institutions we asked about —
                # a paper's co-authors elsewhere are not this institution's staff.
                here = any(
                    (inst.get("ror") or "").endswith(institution.ror)
                    for inst in authorship.get("institutions", [])
                )
                if here and author.get("id") and author.get("display_name"):
                    name, works = by_author.setdefault(
                        author["id"], (author["display_name"], [])
                    )
                    works.append(work)

        cursor = payload.get("meta", {}).get("next_cursor")
        pages += 1

    researchers = [
        Researcher(
            name=name,
            institution_key=institution.key,
            institution_short=institution.short,
            openalex_id=author_id,
            roster_confirmed=False,
            works=sorted(works, key=lambda w: w.year, reverse=True)[:MAX_WORKS_PER_PERSON],
        )
        for author_id, (name, works) in by_author.items()
        if len(works) >= min_works
    ]
    researchers.sort(key=lambda r: r.n_works, reverse=True)

    if verbose:
        print(f"  {institution.short:12s} {pages:3d} pages -> "
              f"{len(researchers):4d} researchers with >={min_works} works")
    return researchers


def match_roster(roster: list[dict], authors_by_institution: dict[str, list[dict]]) -> dict:
    """Map roster entries to OpenAlex author IDs by name, within institution.

    Where several OpenAlex authors share a name form at the same institution,
    the most prolific is taken — the alternative is to guess, and the roster
    lists research staff rather than one-paper visitors.
    """
    resolved: dict[str, dict] = {}

    for key, authors in authors_by_institution.items():
        index: dict[str, list[dict]] = {}
        for author in authors:
            for form in name_keys(author["name"]):
                index.setdefault(form, []).append(author)

        for person in roster:
            if person["institution_key"] != key or person["name"] in resolved:
                continue
            candidates: list[dict] = []
            for form in name_keys(person["name"]):
                candidates.extend(index.get(form, []))
            if candidates:
                best = max(candidates, key=lambda a: a["works_count"])
                resolved[person["name"]] = {**person, **best}

    return resolved


def fetch_author_works(
    client: httpx.Client, author_id: str, *, use_cache: bool = True
) -> list[Work]:
    """Fetch an author's recent works, newest first."""
    filters = f"author.id:{author_id},from_publication_date:{MIN_YEAR}-01-01,has_abstract:true"
    url = (
        f"{OPENALEX_BASE}/works?filter={filters}"
        f"&sort=publication_date:desc&per-page={MAX_WORKS_PER_PERSON}"
        f"&mailto={OPENALEX_MAILTO}"
    )
    payload = _get(client, url, use_cache=use_cache)
    works = [_parse_work(raw) for raw in payload.get("results", [])]
    return [w for w in works if w is not None]


def build_roster_profiles(
    roster: list[dict],
    institutions: tuple[Institution, ...],
    *,
    min_works: int = MIN_WORKS_ROSTER_MEMBER,
    use_cache: bool = True,
    verbose: bool = True,
) -> tuple[list[Researcher], list[dict]]:
    """Resolve every roster entry to an OpenAlex profile.

    Returns ``(researchers, unresolved)`` so callers can see and report who was
    dropped — coverage is a result worth knowing, not something to hide.
    """
    by_key = {inst.key: inst for inst in institutions}
    researchers: list[Researcher] = []
    unresolved: list[dict] = []

    with httpx.Client(headers={"User-Agent": f"RIGAD ({OPENALEX_MAILTO})"}) as client:
        if verbose:
            print("listing institution authors (1 credit per 200 people)")
        authors_by_institution = {
            inst.key: fetch_institution_authors(
                client, inst, use_cache=use_cache, verbose=verbose
            )
            for inst in institutions
        }

        resolved = match_roster(roster, authors_by_institution)
        if verbose:
            print(f"\nname-matched {len(resolved)}/{len(roster)} roster members")
            print("fetching their publications (1 credit each)")

        for person in roster:
            match = resolved.get(person["name"])
            if match is None:
                unresolved.append(
                    {**person, "reason": "no OpenAlex author of that name at this institution"}
                )
                continue

            institution = by_key[person["institution_key"]]
            # One person failing to fetch must not lose the other hundred-odd
            # profiles. Responses are cached, so re-running picks up where a
            # failed run left off and only retries what is missing. A spent
            # budget is different: nothing will succeed until the reset, so it
            # propagates rather than being swallowed per person.
            try:
                works = fetch_author_works(client, match["id"], use_cache=use_cache)
            except BudgetExhausted:
                raise
            except RuntimeError as exc:
                unresolved.append({**person, "reason": f"API error: {exc}"[:120]})
                continue

            if len(works) < min_works:
                unresolved.append({**person, "reason": f"only {len(works)} usable works"})
                continue

            researchers.append(
                Researcher(
                    name=person["name"],
                    institution_key=person["institution_key"],
                    institution_short=institution.short,
                    role=person.get("role", ""),
                    openalex_id=match["id"],
                    roster_confirmed=True,
                    works=works,
                )
            )

    if verbose:
        print(
            f"\nresolved {len(researchers)}/{len(roster)} roster members"
            f" ({credits_remaining} credits left)"
        )
    return researchers, unresolved


# --- Persistence -----------------------------------------------------------


def save_profiles(researchers: list[Researcher], path: Path) -> None:
    """Write profiles to JSON so notebooks can run without network access."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": r.name,
            "institution_key": r.institution_key,
            "institution_short": r.institution_short,
            "role": r.role,
            "openalex_id": r.openalex_id,
            "roster_confirmed": r.roster_confirmed,
            "works": [vars(w) for w in r.works],
        }
        for r in researchers
    ]
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))


# Fields of ``Work`` that we persist; anything else in a stored record is from
# an older snapshot format and is ignored on load.
_WORK_FIELDS = {"id", "title", "abstract", "year", "topics", "cited_by_count"}


def load_profiles(path: Path) -> list[Researcher]:
    """Read profiles written by ``save_profiles``."""
    payload = json.loads(path.read_text())
    return [
        Researcher(
            name=item["name"],
            institution_key=item["institution_key"],
            institution_short=item["institution_short"],
            role=item.get("role", ""),
            openalex_id=item.get("openalex_id", ""),
            roster_confirmed=item.get("roster_confirmed", False),
            works=[Work(**{k: v for k, v in w.items() if k in _WORK_FIELDS})
                   for w in item["works"]],
        )
        for item in payload
    ]
