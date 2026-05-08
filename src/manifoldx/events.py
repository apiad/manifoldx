"""Event-driven system: bus, handlers, read-only views, frame waiters."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable

_RO_MUTATION_MSG = (
    "Event handlers cannot mutate ECS data directly. Use "
    "engine.commands.append(...), engine.spawn(...), or engine.destroy(...)."
)


class _ReadOnlyAccessor:
    """Wraps a ComponentAccessor to forbid attribute writes."""

    def __init__(self, accessor):
        object.__setattr__(self, "_accessor", accessor)

    def __getattr__(self, name):
        return getattr(self._accessor, name)

    def __setattr__(self, name, value):
        raise RuntimeError(_RO_MUTATION_MSG)


class ReadOnlyView:
    """Wraps a ComponentView; reads pass through, writes raise.

    Caveat: numpy arrays returned by reads are not frozen — in-place
    mutations like `view[C].field[i] = x` cannot be intercepted at the
    Python level and remain undefined behavior in event handlers.
    """

    def __init__(self, view):
        object.__setattr__(self, "_view", view)

    def __getitem__(self, component):
        accessor = self._view[component]
        return _ReadOnlyAccessor(accessor)

    def __setitem__(self, component, value):
        raise RuntimeError(_RO_MUTATION_MSG)

    def __setattr__(self, name, value):
        raise RuntimeError(_RO_MUTATION_MSG)

    def __len__(self):
        return len(self._view)

    def __iter__(self):
        return iter(self._view)

    def get_component_data(self, component_name):
        return self._view.get_component_data(component_name)
