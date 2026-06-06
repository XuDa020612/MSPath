from __future__ import annotations

import re


def _clean(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def to_structured_report(raw_report: str) -> str:
    """Normalize raw report text into a free-form summary (one or more paragraphs)."""
    text = _clean(str(raw_report))
    if not text:
        return "Pathology summary: no description available."

    # Collapse excessive blank lines but keep paragraph breaks.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return "Pathology summary: no description available."
    return "\n\n".join(paragraphs)
