#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "microcli-toolkit",
# ]
# ///

"""note - Create structured notes with YAML frontmatter.

Usage:
    uv run note.py create [args...]

Commands:
    create    Create a new note
"""
from pathlib import Path
import sys

import microcli as m
from datetime import datetime
import re

NOTES_DIR = Path(".knowledge/notes")


def slugify(title: str) -> str:
    """Convert title to URL-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def format_note(title: str, slug: str, tags: list, content: str) -> str:
    """Format note with YAML frontmatter."""
    date = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "---",
        f"title: {title}",
        f"slug: {slug}",
        f"date: {date}",
        f"tags: [{', '.join(tags)}]" if tags else "tags: []",
        "---",
        "",
        content,
    ]

    return "\n".join(lines)


@m.command
def create(
    title: str,
    slug: str = "",
    tags: str = "",
    save: bool = False,
):
    """Create a structured note from stdin content."""
    content = sys.stdin.read().strip()

    if not content:
        m.fail("No content provided via stdin")

    # Generate slug if not provided
    if not slug:
        slug = slugify(title)

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    # Format the note
    formatted = format_note(title, slug, tag_list, content)

    if not save:
        # Draft mode - show how to save
        m.info("Note is in draft mode. To save, create the exact same note with --save:")
        m.info(f"  {create.explain(title=title, slug=slug, tags=tags, save=True)}")
    else:
        NOTES_DIR.mkdir(parents=True, exist_ok=True)
        filepath = NOTES_DIR / f"{slug}.md"

        if filepath.exists():
            m.fail(f"File already exists: {filepath}")

        filepath.write_text(formatted)
        m.ok(f"Saved to: {filepath}")


if __name__ == "__main__":
    m.main()
