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
        
    def run(self, engine, dt: float):
        """Execute the system with component view."""
        # Get view for the query components
        view = engine.store.get_component_view(self.component_names)
        # Call the system function
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
        
    def run_all(self, engine, dt: float):
        """Execute all systems in order."""
        for system in self._systems:
            system.run(engine, dt)


__all__ = [
    'System',
    'Query',
    'SystemRegistry',
]
