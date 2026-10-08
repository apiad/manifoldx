"""What an export left out or changed, by name, so nothing is dropped silently."""

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Entry:
    kind: str  # system, handler, gui, compute, task, material, texture, light
    name: str
    effect: str  # "dropped" or "approximated"
    detail: str
    where: str | None = None  # file:line for Python code
    count: int = 1


class ExportReport:
    def __init__(self):
        self.entries: list[Entry] = []

    def add(self, kind, name, effect, detail, where=None, count=1):
        for e in self.entries:
            if (e.kind, e.name, e.effect, e.detail, e.where) == (kind, name, effect, detail, where):
                e.count += count
                return
        self.entries.append(Entry(kind, name, effect, detail, where, count))

    def extend(self, other):
        for e in other.entries:
            self.add(e.kind, e.name, e.effect, e.detail, e.where, e.count)

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def format(self):
        if not self.entries:
            return "glTF export: everything was exported."
        lines = [f"glTF export: {len(self.entries)} things not exported or approximated:"]
        for e in self.entries:
            what = e.name + (f" x{e.count}" if e.count > 1 else "")
            where = f" ({e.where})" if e.where else ""
            lines.append(f"  {e.kind:<8} {e.effect:<12} {what}{where}: {e.detail}")
        return "\n".join(lines)

    def print(self, file=None):
        print(self.format(), file=file or sys.stderr)

    def to_dict(self):
        return {"complete": not self.entries, "entries": [asdict(e) for e in self.entries]}

    def write_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


class ExportIncomplete(RuntimeError):
    """Raised by a strict export when the report is not empty."""

    def __init__(self, report):
        super().__init__(report.format())
        self.report = report
