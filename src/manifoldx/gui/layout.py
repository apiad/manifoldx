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
    cross_axis = "w" if direction == "v" else "h"
    main_size = getattr(inner, main_axis)
    cross_size = getattr(inner, cross_axis)

    total_gap = gap * max(0, len(children) - 1)
    free = main_size - total_gap

    # First pass: subtract fixed from `free`.
    # Children without explicit size or flex: use intrinsic if non-default, else fill.
    main_sizes: list[float] = []
    total_flex = 0
    num_fill_children = 0
    for child in children:
        flex = child.get("flex")
        explicit = child.get("height") if direction == "v" else child.get("width")
        if flex is not None:
            total_flex += flex
            main_sizes.append(0.0)  # filled in second pass.
            continue
        if explicit is not None:
            size = float(explicit)
            main_sizes.append(size)
            free -= size
        else:
            # No explicit size and no flex: check intrinsic
            iw, ih = child["intrinsic"]
            intrinsic_main = ih if direction == "v" else iw
            # Use intrinsic only if it's non-zero and not the default (10, 10)
            if intrinsic_main > 0 and (iw, ih) != (10, 10):
                size = float(intrinsic_main)
                main_sizes.append(size)
                free -= size
            else:
                # Zero or default intrinsic: will fill remaining space.
                num_fill_children += 1
                main_sizes.append(0.0)  # filled in second pass.

    # Second pass: distribute remaining `free` among flex children and fill children.
    total_shares = total_flex + num_fill_children
    if total_shares > 0 and free > 0:
        share_size = free / total_shares
        for i, child in enumerate(children):
            if main_sizes[i] == 0.0:  # flex or fill child
                flex = child.get("flex")
                explicit = child.get("height") if direction == "v" else child.get("width")
                if flex is not None or explicit is None:
                    # This is either a flex child (flex is not None) or a fill child (explicit is None)
                    if flex is not None:
                        main_sizes[i] = share_size * flex
                    else:
                        main_sizes[i] = share_size

    # Place children sequentially along the main axis.
    # If child has explicit cross-axis size, use it; otherwise use available space.
    cursor = 0.0
    for child, size in zip(children, main_sizes):
        if direction == "v":
            # vertical layout: main axis is h, cross axis is w
            cross_explicit = child.get("width")
            child_cross = float(cross_explicit) if cross_explicit is not None else cross_size
            child_box = LayoutBox(inner.x, inner.y + cursor, child_cross, size)
        else:
            # horizontal layout: main axis is w, cross axis is h
            cross_explicit = child.get("height")
            child_cross = float(cross_explicit) if cross_explicit is not None else cross_size
            child_box = LayoutBox(inner.x + cursor, inner.y, size, child_cross)
        _layout_node(child, child_box, out)
        cursor += size + gap
