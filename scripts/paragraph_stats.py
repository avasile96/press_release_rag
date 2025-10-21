"""Compute paragraph word-count statistics for a text file.

Outputs:
- number of paragraphs
- average words per paragraph (mean)
- 80th percentile words per paragraph
- min and max words per paragraph

Usage:
    python3 scripts/paragraph_stats.py data/raw/press_releases.txt

If no path is provided, it defaults to data/raw/press_releases.txt in the repo.
"""

import sys
from pathlib import Path
import math
import re
from typing import List


def split_paragraphs(text: str) -> List[str]:
    # Split on one or more blank lines (handles windows/mac line endings)
    parts = re.split(r"\r?\n\s*\r?\n+", text.strip())
    # filter out empty
    return [p.strip() for p in parts if p.strip()]


def count_words(s: str) -> int:
    # Split on whitespace; this is a reasonable approximation of words
    return len(s.split())


def percentile_sorted(values: List[int], p: float) -> int:
    # values must be non-empty and sorted ascending
    n = len(values)
    if n == 0:
        return 0
    # use the "nearest-rank" method (Ceil) common for percentiles
    rank = math.ceil(p * n)
    idx = max(0, min(n - 1, rank - 1))
    return values[idx]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw/press_releases.txt")
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    text = path.read_text(encoding="utf8")
    paragraphs = split_paragraphs(text)
    counts = [count_words(p) for p in paragraphs]

    if not counts:
        print("No paragraphs found.")
        return

    total = len(counts)
    mean = sum(counts) / total
    sorted_counts = sorted(counts)
    p80 = percentile_sorted(sorted_counts, 0.8)
    mn = sorted_counts[0]
    mx = sorted_counts[-1]

    print(f"Paragraphs: {total}")
    print(f"Average words per paragraph: {mean:.2f}")
    print(f"80th percentile words per paragraph: {p80}")
    print(f"Min words per paragraph: {mn}")
    print(f"Max words per paragraph: {mx}")


if __name__ == "__main__":
    main()
