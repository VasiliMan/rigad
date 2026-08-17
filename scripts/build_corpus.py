"""Build the RIGAD corpus for all partner institutions.

The corpus is a **union of two definitions**, because neither alone captures
what "information systems research at this university" means:

    corpus(X) =  papers in strict IS subfields authored at X
              ∪  everything published by members of X's IS department

The first half catches IS work done elsewhere in the university — the medical
school's health-informatics paper, the library's digital-scholarship work.
The second half catches the breadth of what IS academics actually do: an IS
professor publishing in a healthcare or education venue is doing IS work even
when OpenAlex files the paper under Health Informatics. Taken together they
describe the field as a community *and* as a topic, which is what the field
actually is.

Two mechanisms from ``rigad.corpus`` supply the halves — roster resolution for
the department members, and a subfield sweep for the topical papers. This
script holds the policy: which institutions, which subfields, how the halves
are merged. Neither mechanism uses OpenAlex ``search`` filters, which cost ten
credits against one.

Safe to re-run: every response is cached, so a second run only fetches what
the first missed.

    python scripts/build_corpus.py
"""

from __future__ import annotations

import json
from collections import Counter

import httpx

from rigad import corpus as corpus_module
from rigad.config import IS_SUBFIELDS, OPENALEX_MAILTO, PILOT_INSTITUTIONS, SAMPLE_DIR
from rigad.corpus import (
    Researcher,
    build_pool_from_works,
    build_roster_profiles,
    save_profiles,
)


def merge(roster_people: list[Researcher], works_people: list[Researcher]) -> list[Researcher]:
    """Union the two halves, preferring the roster record for the same person.

    A roster record is richer: it carries the job title, the confirmation that
    this person really is departmental staff, and their complete publication
    list rather than only their in-subfield papers.
    """
    merged: dict[str, Researcher] = {}
    for person in works_people:
        if person.openalex_id:
            merged[person.openalex_id] = person
    for person in roster_people:
        if person.openalex_id:
            merged[person.openalex_id] = person  # roster wins
    return sorted(merged.values(), key=lambda r: r.n_works, reverse=True)


def main() -> None:
    roster = json.loads((SAMPLE_DIR / "roster.json").read_text())
    print(f"roster: {len(roster)} people across "
          f"{len({p['institution_key'] for p in roster})} institutions\n")

    print("half 1 — department members, with everything they publish")
    roster_people, unresolved = build_roster_profiles(roster, PILOT_INSTITUTIONS)

    print(f"\nhalf 2 — strict IS papers at each institution "
          f"({', '.join(IS_SUBFIELDS.values())})")
    works_people: list[Researcher] = []
    with httpx.Client(headers={"User-Agent": f"RIGAD ({OPENALEX_MAILTO})"}) as client:
        for institution in PILOT_INSTITUTIONS:
            works_people += build_pool_from_works(
                client, institution, subfields=IS_SUBFIELDS
            )

    people = merge(roster_people, works_people)

    save_profiles(people, SAMPLE_DIR / "profiles.json")
    (SAMPLE_DIR / "unresolved.json").write_text(
        json.dumps(unresolved, indent=1, ensure_ascii=False)
    )

    confirmed = [p for p in people if p.roster_confirmed]
    print(f"\ncorpus: {len(people)} researchers "
          f"({len(confirmed)} confirmed department staff, "
          f"{len(people) - len(confirmed)} found via IS publications)")
    print(f"by institution: {dict(Counter(p.institution_short for p in people))}")
    print(f"distinct topics: {len({t for p in people for t in p.topics})}")
    print(f"credits left: {corpus_module.credits_remaining}")

    print(f"\nroster members not resolved: {len(unresolved)}")
    for reason, count in Counter(u["reason"].split(":")[0] for u in unresolved).most_common():
        print(f"  {count:3d}  {reason}")


if __name__ == "__main__":
    main()
