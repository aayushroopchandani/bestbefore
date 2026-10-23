"""The generator's shares and ranges, fixed by rule before any photo is made
(docs/m2-synthetic-generator.md, "The spec"). Never tuned on the store
photos, which nothing here was designed from.

S4 writes the parts it uses: the text and the print styles. S5 adds
geometry, size, composing and camera effects, the GenSpec holding them all,
and its hash.
"""
from __future__ import annotations

from dataclasses import dataclass

Shares = tuple[tuple[str, float], ...]     # (name, share) pairs that sum to 1
Range = tuple[float, float]                # drawn uniformly


@dataclass(frozen=True)
class TextSpec:
    """What is printed ("Text")."""
    years: tuple[int, int] = (2018, 2035)       # every date falls in these years
    shelf_months: tuple[int, int] = (6, 36)     # two dates: expiry = mfg + this many months
    kinds: Shares = (("single", 0.35), ("pair", 0.35), ("panel", 0.20),
                     ("range", 0.05), ("relative", 0.05))
    month_word: float = 0.45     # of dates outside ranges; the rest are numbers
    full_name: float = 0.50      # of month words; the rest are abbreviations
    day_numbers: float = 0.70    # numeric dates that print a day
    day_words: float = 0.40      # month-word dates that print a day
    ordinal: float = 0.50        # of month-word dates with a day: 19th
    case: Shares = (("upper", 0.50), ("title", 0.45), ("lower", 0.05))   # month words
    single_word: float = 0.70    # a single date with its expiry word
    mfg_first: float = 0.80      # an mfg line printed above the expiry's
    panel_lines: tuple[int, int] = (1, 3)    # a panel's extra lines: codes and distractors
    # Left open by the plan, fixed in S4:
    panel_codes: float = 0.50    # of a panel's extra lines, codes; the rest distractors
    range_word: float = 0.70     # a range with an expiry word, as a single date
    relative_months: tuple[int, ...] = (6, 9, 12, 18, 24, 36)   # Best Before N months from Mfg.


TEXT = TextSpec()
