"""Extract ICIS 2026 track descriptions into the track-corpus format.

The conference publishes tracks as one long HTML page. Each track appears as:

    <track name>
    Track Chairs:
    ...names and bios...
    Track Description:
    ...prose, then "Topics of interest include..." and a bulleted list...
    Associate Editors:
    ...names...

We keep the name, the prose description, and the topic list. Chair and editor
names are dropped: they say nothing about what the track is *about*, and
leaving them in would let a draft match a track because it shares an author's
institution.

    python scripts/parse_icis_tracks.py <saved-page.html> data/tracks/icis2026.json
"""

from __future__ import annotations

import html as html_lib
import json
import re
import sys
from pathlib import Path

CHAIRS = "Track Chairs:"
EDITORS = "Associate Editors:"
TOPICS_MARKER = "Topics of interest"

# The page is not consistent about the colon — the theme track's heading is
# "Track Description" while every other track has "Track Description:".
# Matching only the colon form silently attributes the theme track the *next*
# track's text, so match either.
DESCRIPTION_HEADING = re.compile(r"^Track Description:?$")

# Lines that are people, not content: "Name, Institution, Country" or a bare
# email address. Used to stop a description bleeding into an editor list.
PERSON_LINE = re.compile(r"^[^,]+,\s+[^,]+,\s+[A-Za-z ]+$")
EMAIL_LINE = re.compile(r"^\S+@\S+\.\S+$")


def visible_lines(markup: str) -> list[str]:
    markup = re.sub(r"<script.*?</script>", " ", markup, flags=re.S)
    markup = re.sub(r"<style.*?</style>", " ", markup, flags=re.S)
    text = html_lib.unescape(re.sub(r"<[^>]+>", "\n", markup))
    return [line.strip() for line in text.split("\n") if line.strip()]


def parse(lines: list[str]) -> list[dict]:
    tracks: list[dict] = []

    chair_positions = [i for i, line in enumerate(lines) if line == CHAIRS]
    for order, chair_at in enumerate(chair_positions):
        name = lines[chair_at - 1] if chair_at else ""
        # The page's own navigation repeats the heading; skip those.
        if not name or name.lower().startswith("track description"):
            continue

        # Where does this track's own section end? At the next track's chairs.
        section_end = (
            chair_positions[order + 1] - 1
            if order + 1 < len(chair_positions)
            else len(lines)
        )

        description_at = next(
            (i for i in range(chair_at, section_end)
             if DESCRIPTION_HEADING.match(lines[i])),
            None,
        )
        if description_at is None:
            continue

        # Description runs to the associate-editor list, or to the section end.
        end = section_end
        for i in range(description_at + 1, section_end):
            if lines[i] == EDITORS:
                end = i
                break

        body = lines[description_at + 1 : end]

        prose: list[str] = []
        topics: list[str] = []
        in_topics = False
        for line in body:
            if TOPICS_MARKER.lower() in line.lower():
                in_topics = True
                continue
            if EMAIL_LINE.match(line) or PERSON_LINE.match(line):
                continue
            (topics if in_topics else prose).append(line)

        tracks.append(
            {
                "id": re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-"),
                "name": name,
                "order": order,
                "description": " ".join(prose).strip(),
                "topics": topics,
            }
        )

    return tracks


def main(source: str, destination: str) -> None:
    lines = visible_lines(Path(source).read_text(encoding="utf-8", errors="ignore"))
    tracks = parse(lines)

    # Drop anything that came out empty — a heading matched but no real content.
    tracks = [t for t in tracks if len(t["description"]) > 120]

    out = Path(destination)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"conference": "ICIS 2026",
         "source_url": "https://icis2026.aisconferences.org/submissions/track-descriptions/",
         "tracks": tracks}, indent=1, ensure_ascii=False))

    print(f"{len(tracks)} tracks -> {out}")
    for track in tracks:
        print(f"  {track['name'][:52]:54s} {len(track['description']):5d} chars, "
              f"{len(track['topics']):3d} topics")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
