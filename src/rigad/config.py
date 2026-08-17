"""Institutions, topic scope and thresholds for the RIGAD pilot.

Five institutions, chosen because the "Sustainable Digital Transformation"
partnership already works with each: the host University of Gothenburg, plus
Warwick Business School (seed funding 2024), ESSEC (2025), and the EUTOPIA
global partners Arizona State and Monash.

They are not all measured the same way, and the difference matters when
reading results. Gothenburg, Warwick, ESSEC and ASU publish staff directories
we could capture, so their department members are known by name. Monash's
directory lists only eight people, so its corpus entry comes almost entirely
from publication output. Any per-institution comparison should be read with
that asymmetry in mind.

``EUTOPIA_PARTNERS`` carries the wider alliance so extending the pipeline is a
configuration change rather than a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SAMPLE_DIR = DATA_DIR / "sample"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"

# Contact address for the OpenAlex "polite pool", which gets faster and more
# reliable service than anonymous requests. See https://docs.openalex.org/
OPENALEX_MAILTO = "vasili.mankevich@ait.gu.se"
OPENALEX_BASE = "https://api.openalex.org"


@dataclass(frozen=True)
class Institution:
    """An institution in the corpus.

    ``ror`` is the Research Organization Registry ID, which OpenAlex uses as a
    stable filter key. ``short`` is what appears in figures and group listings.
    """

    key: str
    short: str
    display_name: str
    ror: str
    country: str
    note: str = ""

    @property
    def ror_url(self) -> str:
        return f"https://ror.org/{self.ror}"


# --- Pilot scope -----------------------------------------------------------
# Warwick Business School and the Department of Applied IT have no separate
# OpenAlex entities: OpenAlex indexes affiliations at the parent-university
# level. We therefore filter the parent institution and narrow to the
# management/IS subfields below, which is what actually isolates the business
# school's output. ESSEC is a standalone institution and needs no such
# narrowing, but the same topic filter is applied for comparability.
PILOT_INSTITUTIONS: tuple[Institution, ...] = (
    Institution(
        key="gu",
        short="Gothenburg",
        display_name="University of Gothenburg",
        ror="01tm6cn81",
        country="SE",
        note="Host institution; Department of Applied IT.",
    ),
    Institution(
        key="warwick",
        short="Warwick",
        display_name="University of Warwick",
        ror="01a77tt86",
        country="GB",
        note="Warwick Business School; EUTOPIA associate partner. Seed funding 2024.",
    ),
    Institution(
        key="essec",
        short="ESSEC",
        display_name="ESSEC Business School",
        ror="02dga6j42",
        country="FR",
        note="Partner of CY Cergy Paris Universite. Seed funding 2025.",
    ),
    Institution(
        key="asu",
        short="ASU",
        display_name="Arizona State University",
        ror="03efmqc40",
        country="US",
        note=(
            "W. P. Carey Department of Information Systems; EUTOPIA global partner. "
            "Members derived from publication output rather than a staff "
            "directory — see build_pool_from_works."
        ),
    ),
    Institution(
        key="monash",
        short="Monash",
        display_name="Monash University",
        ror="02bfwt286",
        country="AU",
        note=(
            "Department of Human-Centred Computing / Information Systems; EUTOPIA "
            "global partner. Members derived from publication output — the staff "
            "directory refuses automated requests."
        ),
    ),
)

# --- Full alliance ---------------------------------------------------------
# Nine full partners plus Warwick as associate partner. Present so that
# widening the corpus is a configuration change, not a code change.
EUTOPIA_PARTNERS: tuple[Institution, ...] = (
    Institution("vub", "Brussels", "Vrije Universiteit Brussel", "006e5kg04", "BE"),
    Institution("cy", "Cergy", "CY Cergy Paris Universite", "043htjv09", "FR"),
    Institution("ljubljana", "Ljubljana", "University of Ljubljana", "05njb9z20", "SI"),
    Institution("upf", "Pompeu Fabra", "Universitat Pompeu Fabra", "04n0g0b29", "ES"),
    Institution("gu", "Gothenburg", "University of Gothenburg", "01tm6cn81", "SE"),
    Institution("cafoscari", "Ca' Foscari", "Ca' Foscari University of Venice", "04yzxz566", "IT"),
    Institution("nova", "NOVA Lisbon", "Universidade Nova de Lisboa", "02xankh89", "PT"),
    Institution("tud", "Dresden", "Technische Universitat Dresden", "042aqky30", "DE"),
    Institution("ubb", "Babes-Bolyai", "Babes-Bolyai University", "02rmd1t30", "RO"),
    Institution("warwick", "Warwick", "University of Warwick", "01a77tt86", "GB",
                "Associate partner."),
)

# --- Topic scope -----------------------------------------------------------
# OpenAlex subfield IDs covering information systems and management research,
# matching the "Information Systems" EUTOPIA Connected Community this project
# sits within. Narrowing to these keeps the corpus coherent and tractable:
# an unfiltered institutional sweep returns tens of thousands of works
# dominated by medicine and the natural sciences.
IS_SUBFIELDS: dict[str, str] = {
    "1710": "Information Systems",
    "1404": "Management Information Systems",
    "1802": "Information Systems and Management",
}

# A wider net that also takes in general management. Deliberately NOT the
# default. When a corpus is built from a roster, the subfield filter only
# helps locate people and breadth is harmless; when it is built from
# publication output, the filter is the *only* constraint — measured on ASU
# and Monash, these three extra subfields alone admitted 41% of the papers,
# none with any information-systems connection.
MANAGEMENT_SUBFIELDS: dict[str, str] = {
    **IS_SUBFIELDS,
    "1408": "Strategy and Management",
    "1407": "Organizational Behavior and Human Resource Management",
    "1403": "Business and International Management",
}

# Corpus window. Recent enough to reflect current interests, wide enough that
# researchers accumulate several works.
FROM_PUBLICATION_DATE = "2020-01-01"

# --- How much published work is enough? ------------------------------------
# Four different questions get four different answers. They live together here
# so the thresholds can be compared at a glance and there is one place to
# change them.

# Listing an institution's authors: skip people with almost no output, purely
# to keep the listing small. Cheap and generous.
MIN_WORKS_TO_LIST_AUTHOR = 2

# Building a department member's profile from the roster: two papers is thin
# but usable, and staff we can name are worth keeping even when quiet.
MIN_WORKS_ROSTER_MEMBER = 2

# Entering the corpus through publications alone, with no roster to vouch for
# you: a little more evidence required, since nothing else confirms the person
# belongs in an IS pool.
MIN_WORKS_VIA_PUBLICATIONS = 3

# Being offered to a student as a mentor without appearing on any IS staff
# directory: the highest bar of the four. One IS-adjacent paper does not make
# someone an IS mentor.
MIN_WORKS_MENTOR_NOT_DEPARTMENTAL = 5


def institution_by_ror(institutions: tuple[Institution, ...]) -> dict[str, Institution]:
    """Index institutions by bare ROR ID for fast lookup during profile building."""
    return {inst.ror: inst for inst in institutions}
