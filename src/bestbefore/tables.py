"""Markdown tables for the summaries in results/ and on screen."""
from __future__ import annotations


def markdown_table(rows: list[dict[str, str]], left: tuple[str, ...] = ()) -> str:
    """One row per dict, columns in the first dict's order. Columns named in
    `left` hold text and are left-aligned; the rest hold numbers."""
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" if c in left else "---:" for c in cols) + "|"]
    lines += ["| " + " | ".join(r[c] for c in cols) + " |" for r in rows]
    return "\n".join(lines)
