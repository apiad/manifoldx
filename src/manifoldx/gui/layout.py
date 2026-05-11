"""Stack + flex layout algorithm for the GUI layer.

Operates on plain spec dicts so it stays decoupled from widget classes.
A spec is::

    {
        "direction": "v" | "h",
        "padding":   (top, right, bottom, left),
        "gap":       int,
        "width":     int | None,        # explicit size; overrides intrinsic
        "height":    int | None,
        "flex":      int | None,        # share of free space (containers only)
        "intrinsic": (w, h),            # default size if no explicit/flex
        "children":  [spec, ...],
    }

`compute_layout(root, viewport)` returns ``{id(spec): LayoutBox}`` for every
spec in the tree, keyed by Python object id (callers retain the spec
references themselves).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LayoutBox:
    x: float
    y: float
    w: float
    h: float


def compute_layout(
    root: dict[str, Any],
    viewport: LayoutBox,
) -> dict[int, LayoutBox]:
    """Compute boxes for every node in the tree rooted at `root`."""
    out: dict[int, LayoutBox] = {}
    _layout_node(root, viewport, out)
    return out


def _layout_node(
    node: dict[str, Any],
    box: LayoutBox,
    out: dict[int, LayoutBox],
) -> None:
    out[id(node)] = box
    children = node.get("children") or []
    if not children:
        return

    top, right, bottom, left = node["padding"]
    inner = LayoutBox(
        box.x + left,
        box.y + top,
        max(0.0, box.w - left - right),
        max(0.0, box.h - top - bottom),
    )
    direction = node["direction"]
    gap = node["gap"]
    main_axis = "h" if direction == "v" else "w"
    main_size = getattr(inner, main_axis)

    total_gap = gap * max(0, len(children) - 1)
    free = main_size - total_gap

    # First pass: subtract fixed + intrinsic-without-flex from `free`.
    main_sizes: list[float] = []
    total_flex = 0
    for child in children:
        flex = child.get("flex")
        explicit = child.get("height") if direction == "v" else child.get("width")
        if flex is not None:
            total_flex += flex
            main_sizes.append(0.0)  # filled in second pass.
            continue
        if explicit is not None:
            size = float(explicit)
        else:
            iw, ih = child["intrinsic"]
            size = float(ih if direction == "v" else iw)
        main_sizes.append(size)
        free -= size

    # Second pass: distribute remaining `free` among flex children.
    if total_flex > 0 and free > 0:
        for i, child in enumerate(children):
            flex = child.get("flex")
            if flex is None:
                continue
            main_sizes[i] = free * (flex / total_flex)

    # Place children sequentially along the main axis.
    cursor = 0.0
    for child, size in zip(children, main_sizes):
        if direction == "v":
            child_box = LayoutBox(inner.x, inner.y + cursor, inner.w, size)
        else:
            child_box = LayoutBox(inner.x + cursor, inner.y, size, inner.h)
        _layout_node(child, child_box, out)
        cursor += size + gap
