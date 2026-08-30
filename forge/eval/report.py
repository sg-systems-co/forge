"""Markdown / JSON reporting for milestone tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def markdown_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return "_(no rows)_"
    cols = columns or list(rows[0].keys())

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    widths = {c: max(len(c), *(len(fmt(r.get(c, ""))) for r in rows)) for c in cols}
    head = "| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |"
    sep = "|" + "|".join("-" * (widths[c] + 2) for c in cols) + "|"
    body = [
        "| " + " | ".join(fmt(r.get(c, "")).ljust(widths[c]) for c in cols) + " |" for r in rows
    ]
    return "\n".join([head, sep, *body])


def write_report(path: str | Path, title: str, rows: list[dict[str, Any]], notes: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{markdown_table(rows)}\n\n{notes}\n")
    path.with_suffix(".json").write_text(json.dumps(rows, indent=2))
