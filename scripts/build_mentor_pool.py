"""Turn the corpus into the mentor pool the tool ships with.

The corpus holds full publication records and is large; the mentor pool is a
trimmed version small enough to commit and quick to embed at run time.

Each entry records **how the person got there** — from a department's staff
directory, or because they publish information-systems research from elsewhere
in the same university. Both are worth suggesting, but they are different
kinds of suggestion and the tool shows which is which.

    python scripts/build_mentor_pool.py
"""

from __future__ import annotations

import json
from collections import Counter

from rigad.config import MIN_WORKS_MENTOR_NOT_DEPARTMENTAL, SAMPLE_DIR
from rigad.corpus import load_profiles

# Enough recent work to characterise someone's interests to a reader.
WORKS_IN_PROFILE = 8
PROFILE_CHARS = 4000


def main() -> None:
    people = load_profiles(SAMPLE_DIR / "profiles.json")

    pool = []
    for person in people:
        if not person.roster_confirmed and person.n_works < MIN_WORKS_MENTOR_NOT_DEPARTMENTAL:
            continue
        recent = sorted(person.works, key=lambda w: w.year, reverse=True)[:WORKS_IN_PROFILE]
        pool.append({
            "name": person.name,
            "institution": person.institution_short,
            "role": person.role,
            "in_is_department": person.roster_confirmed,
            "openalex_id": person.openalex_id,
            "n_works": person.n_works,
            "topics": person.topics[:12],
            "profile": " ".join(w.text for w in recent)[:PROFILE_CHARS],
        })

    out = SAMPLE_DIR.parent / "mentors" / "eutopia_mentors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "source": (
            "Union of two definitions: members of each partner IS department "
            "(from published staff directories), and researchers elsewhere in "
            "the same universities who publish in OpenAlex's information-systems "
            "subfields. Publication records from OpenAlex (CC0)."
        ),
        "institutions": sorted({p["institution"] for p in pool}),
        "mentors": pool,
    }, indent=1, ensure_ascii=False))

    departmental = sum(1 for p in pool if p["in_is_department"])
    print(f"{len(pool)} mentors -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {departmental} from IS department staff directories")
    print(f"  {len(pool) - departmental} found via IS publications elsewhere")
    print(f"  by institution: {dict(Counter(p['institution'] for p in pool))}")


if __name__ == "__main__":
    main()
