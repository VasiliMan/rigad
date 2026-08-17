"""Read draft papers out of a folder.

The entry point to RIGAD is a folder of drafts — whatever the organiser or the
author already has, in the formats academics actually use: PDF, Word, plain
text, Markdown.

Institutions are read from subfolder names, if there are any:

    drafts/                          drafts/
      gothenburg/alice.pdf             alice.pdf
      warwick/bob.docx                 bob.docx
      essec/chen.pdf                   chen.pdf

    -> institutions known,           -> no institutions; grouping still works,
       groups can be required           but cannot require cross-institutional
       to cross them                    mixing

Nothing needs configuring either way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}

# Files that live in a drafts folder to explain it, not to be analysed.
# Without this a folder's own README is read as somebody's paper.
NON_DRAFT_STEMS = {"readme", "read me", "notes", "index", "license", "licence"}

# Below this a file is almost certainly a cover sheet, a form, or a scanned
# page image that yielded no text — not something we can match on.
MIN_USABLE_CHARS = 400

# Only the front of a paper is used. The opening pages carry the title,
# abstract and introduction, which is what characterises the work; later pages
# are increasingly related work and appendices, which pull a draft towards
# whatever it cites rather than what it argues.
MAX_CHARS = 12_000


@dataclass
class Draft:
    """One draft paper read from disk."""

    path: Path
    text: str
    institution: str | None = None

    @property
    def name(self) -> str:
        """A human label — the filename without extension, tidied up."""
        return self.path.stem.replace("_", " ").replace("-", " ").strip()

    @property
    def title(self) -> str:
        """Best guess at the paper's title: the first substantial line.

        A heuristic, and it is allowed to be wrong — it is used for display,
        never for matching.
        """
        for line in self.text.split("\n"):
            line = line.strip()
            if 15 <= len(line) <= 200 and not line.lower().startswith(("abstract", "http")):
                return line
        return self.name


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
        if sum(len(p) for p in parts) > MAX_CHARS:
            break
    return "\n".join(parts)


def read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


READERS = {".pdf": read_pdf, ".docx": read_docx, ".txt": read_text, ".md": read_text}


def tidy(raw: str) -> str:
    """Clean up extracted text enough to embed.

    PDF extraction leaves hyphenated line breaks and hard-wrapped lines that
    split words and sentences; both confuse the embedding model.
    """
    raw = raw.replace("\x00", " ")
    raw = re.sub(r"-\n(\w)", r"\1", raw)      # re-join words split across lines
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def load_drafts(
    folder: str | Path, *, verbose: bool = True
) -> tuple[list[Draft], list[tuple[Path, str]]]:
    """Read every supported document under ``folder``.

    Returns the drafts and a list of ``(path, reason)`` for anything skipped,
    so an organiser can see that a file was ignored instead of quietly getting
    results for fewer people than they have.
    """
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"not a folder: {root}")

    drafts: list[Draft] = []
    skipped: list[tuple[Path, str]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith((".", "~$")):
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped.append((path, f"unsupported format ({path.suffix or 'no extension'})"))
            continue
        if path.stem.lower() in NON_DRAFT_STEMS:
            skipped.append((path, "looks like folder documentation, not a draft"))
            continue

        try:
            text = tidy(READERS[path.suffix.lower()](path))
        except Exception as exc:  # noqa: BLE001 — any reader failure is the same to us
            skipped.append((path, f"could not read: {type(exc).__name__}"))
            continue

        if len(text) < MIN_USABLE_CHARS:
            skipped.append((path, f"too little text ({len(text)} chars — scanned image?)"))
            continue

        # A subfolder directly under the root names the institution.
        relative = path.relative_to(root)
        institution = relative.parts[0] if len(relative.parts) > 1 else None

        drafts.append(Draft(path=path, text=text[:MAX_CHARS], institution=institution))

    if verbose:
        print(f"read {len(drafts)} drafts from {root}")
        if skipped:
            print(f"skipped {len(skipped)}:")
            for path, reason in skipped[:10]:
                print(f"   {path.name[:44]:46s} {reason}")
        institutions = {d.institution for d in drafts if d.institution}
        if institutions:
            print(f"institutions from subfolders: {', '.join(sorted(institutions))}")

    return drafts, skipped
