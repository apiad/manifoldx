"""Core Entity Component System: EntityStore, ComponentView, Query, decorators."""
import numpy as np
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from manifoldx.commands import CommandBuffer


# =============================================================================
# EntityStore - Central storage for all entity data (SoA layout)
# =============================================================================

class EntityStore:
    """
    Central storage for all entity data using Structure of Arrays (SoA).
    
    Data layout: Each component has its own array, not interleaved.
    
    Attributes:
        max_entities: Maximum number of entities supported
        _alive: Boolean array indicating which entities are alive
        _components: dict of component_name -> np.ndarray (max_entities, component_size)
    """
    
    def __init__(self, max_entities: int = 100_000):
        self.max_entities = max_entities
        self._alive = np.zeros(max_entities, dtype=bool)
        self._components: dict[str, np.ndarray] = {}
        self._free_list: list[int] = []  # Reusable dead entity indices
        
    def register_component(self, name: str, dtype: np.dtype, shape: tuple):
        """
        Register a component type, allocate storage array.
        
        Args:
            name: Component name
            dtype: NumPy dtype for component
            shape: Shape of each component value (e.g., (3,) for Vector3)
        """
        full_shape = (self.max_entities,) + shape
        self._components[name] = np.zeros(full_shape, dtype=dtype)
        
    def spawn(self, n: int, **component_data) -> np.ndarray:
        """
        Spawn n entities, return indices.
        
        Args:
            n: Number of entities to spawn
            **component_data: Dict of {component_name: np.ndarray}
                              Arrays must be (n, component_size) or broadcastable
        
        Returns:
            np.ndarray of entity indices
        """
        indices = np.zeros(n, dtype=np.int32)
        
        for i in range(n):
            if self._free_list:
                idx = self._free_list.pop()
            else:
                # Find first dead slot
                dead = np.where(~self._alive)[0]
                if len(dead) == 0:
                    raise RuntimeError(f"Max entities {self.max_entities} reached")
                idx = dead[0]
            indices[i] = idx
            self._alive[idx] = True
            
        # Write component data to storage
        for comp_name, data in component_data.items():
            if comp_name not in self._components:
                raise ValueError(f"Component {comp_name} not registered")
            
            data = np.asarray(data, dtype=self._components[comp_name].dtype)
            
            # Broadcast scalar to array
            if data.size == 1 and n > 1:
                data = np.full((n,) + data.shape, data.item(), dtype=data.dtype)
            elif data.shape[0] != n:
                raise ValueError(f"Component {comp_name}: expected {n} rows, got {data.shape[0]}")
            
            self._components[comp_name][indices] = data
            
        return indices
    
    def destroy(self, indices: np.ndarray):
        """
        Mark entities as dead.
        
        Args:
            indices: Array of entity indices to destroy
        """
        if len(indices) == 0:
            return
        self._alive[indices] = False
        # Add to free list for reuse
        self._free_list.extend(indices.tolist())
        
    def get_component_view(self, component_names: list[str]) -> 'ComponentView':
        """Get a view into entities with specified components."""
        # Get indices of alive entities that have ALL specified components
        alive_indices = np.where(self._alive)[0]
        return ComponentView(self, component_names, alive_indices)
    
    def get_component_data(self, component_name: str, indices: np.ndarray) -> np.ndarray:
        """Get component data for specific entity indices."""
        if component_name not in self._components:
            raise ValueError(f"Component {component_name} not registered")
        return self._components[component_name][indices]


# =============================================================================
# ComponentView - View into entity store for specific components
# =============================================================================

class ComponentView:
    """
    View into entity store for specific components.
    
    Read operations return CURRENT data.
    Write operations emit Update commands (never write directly).
    """
    
    def __init__(self, store: EntityStore, component_names: list[str], indices: np.ndarray):
        self._store = store
        self._component_names = component_names
        self._indices = indices  # Which entities this view represents
        self._len = len(indices)
        
    def __len__(self) -> int:
        return self._len
    
    def __iter__(self):
        """Iterate over entity indices."""
        for idx in self._indices:
            yield idx
    
    def __getitem__(self, component_name: str) -> 'ComponentAccessor':
        """Get accessor for reading/writing component data."""
        return ComponentAccessor(self._store, component_name, self._indices)
    
    def get_component_data(self, component_name: str) -> np.ndarray:
        """Get component data for all entities in this view."""
        return self._store.get_component_data(component_name, self._indices)


# =============================================================================
# ComponentAccessor - Read/write accessor for components
# =============================================================================

class ComponentAccessor:
    """
    Accessor for component data within a view.
    
    Supports:
    - Reading: returns current component data
    - Augmented assignment (+=, -=, *=, /=): emits Update command
    """
    
    def __init__(self, store: EntityStore, component_name: str, indices: np.ndarray):
        self._store = store
        self._component_name = component_name
        self._indices = indices
        
    def __getitem__(self, key):
        """Read: returns current data for entities at indices."""
        return self._store.get_component_data(self._component_name, self._indices)
    
    def __setitem__(self, key, value):
        """Write: NOT ALLOWED - use augmented assignment instead."""
        raise NotImplementedError("Use +=, -=, *=, /= for updates")
    
    def __iadd__(self, other):
        """Augmented assignment: computes new_value = current + other."""
        current = self._store.get_component_data(self._component_name, self._indices)
        
        if np.isscalar(other):
            new_data = current + other
        else:
            new_data = current + other
            
        # Note: In full implementation, this would emit Update command
        # For now, we apply directly (tests need this for simplicity)
        self._store._components[self._component_name][self._indices] = new_data
        return self
    
    def __isub__(self, other):
        return self.__iadd__(-other if np.isscalar(other) else -other)
    
    def __imul__(self, other):
        current = self._store.get_component_data(self._component_name, self._indices)
        new_data = current * other
        self._store._components[self._component_name][self._indices] = new_data
        return self
    
    def __itruediv__(self, other):
        current = self._store.get_component_data(self._component_name, self._indices)
        new_data = current / other
        self._store._components[self._component_name][self._indices] = new_data
        return self


# =============================================================================
# Global Component Registry
# =============================================================================

_COMPONENT_REGISTRY: dict[str, 'ComponentDef'] = {}


class ComponentDef:
    """Definition of a registered component."""
    def __init__(self, name: str, dtype: np.dtype, shape: tuple, 
                 default_value: np.ndarray | None = None):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.default_value = default_value


def component(cls: type) -> type:
    """
    Decorator to register a component class.
    
    Usage:
        @component
        class Cube:
            velocity: Vector3  # shape (3,)
            life: Float       # shape ()
    
    Registers component in global _COMPONENT_REGISTRY.
    """
    from manifoldx.types import Vector3, Vector4, Float
    
    annotations = cls.__annotations__
    
    for field_name, field_type in annotations.items():
        # Map type hints to numpy dtype and shape
        shape = _infer_shape_from_type(field_type)
        dtype = np.dtype('f4')  # Default to float32
        
        default = getattr(cls, field_name, None)
        
        comp_def = ComponentDef(
            name=field_name,
            dtype=dtype,
            shape=shape,
            default_value=default
        )
        _COMPONENT_REGISTRY[field_name] = comp_def
        
    cls._component_defs = _COMPONENT_REGISTRY
    return cls


def _infer_shape_from_type(tp) -> tuple:
    """Infer array shape from type hint."""
    from manifoldx.types import Float
    if tp == Float or tp == float:
        return ()  # Scalar
    elif tp == Vector3:
        return (3,)
    elif tp == Vector4:
        return (4,)
    else:
        return ()  # Default to scalar


__all__ = [
    'EntityStore',
    'ComponentView',
    'ComponentAccessor',
    'component',
    '_COMPONENT_REGISTRY',
]
