"""System registration and execution."""

import typing
from typing import Any, Callable


# =============================================================================
# System Registry
# =============================================================================


class System:
    """A registered system function."""

    def __init__(self, func: Callable, component_names: list):
        self.func = func
        self.component_names = component_names
        self.removed = False

    def run(self, engine, dt: float):
        """Execute the system with component view."""
        view = engine.store.get_component_view(self.component_names, engine)
        self.func(view, dt)


class Query:
    """
    Type annotation for systems. Specifies required components.

    Usage:
        @engine.system
        def my_system(query: Query[Transform, Mesh], dt: float):
            ...
    """

    def __init__(self, components: tuple):
        self.components = components

    @classmethod
    def __class_getitem__(cls, components: tuple):
        return cls(components)


class SystemRegistry:
    """Registry of all systems."""

    def __init__(self):
        self._systems: list[System] = []

    def register(self, func: Callable, component_names: list = None) -> Callable:
        """Register a system function with optional component names."""
        if component_names is None:
            component_names = []

        system = System(func, component_names)
        self._systems.append(system)
        return func

    def remove(self, func: Callable) -> None:
        """Unregister every registration of `func`; ValueError if it has none.

        Takes effect at once: a system removed mid-frame (even by itself, or by
        an earlier system) does not run again, this frame included.
        """
        matches = [s for s in self._systems if s.func is func]
        if not matches:
            raise ValueError(f"{func!r} is not a registered system")
        for system in matches:
            system.removed = True
        self._systems = [s for s in self._systems if s.func is not func]

    def run_all(self, engine, dt: float):
        """Execute all systems in order.

        Iterates a snapshot, so systems may be added or removed while it runs;
        systems added mid-frame start on the next frame.
        """
        for system in list(self._systems):
            if not system.removed:
                system.run(engine, dt)


__all__ = [
    "System",
    "Query",
    "SystemRegistry",
]
