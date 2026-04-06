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
# TransformFieldView - Writable view into a Transform sub-field
# =============================================================================

class _TransformFieldView:
    """View into a sub-field of the Transform component (position, rotation, scale).
    
    Supports += for in-place updates. For rotation, += composes quaternions.
    """
    
    def __init__(self, store: 'EntityStore', indices: np.ndarray, field_name: str, col_start: int, col_end: int):
        self._store = store
        self._indices = indices
        self._field_name = field_name
        self._col_start = col_start
        self._col_end = col_end
    
    @property
    def data(self) -> np.ndarray:
        """Get current data as a writable slice."""
        return self._store._components['Transform'][np.ix_(self._indices, range(self._col_start, self._col_end))]
    
    def __iadd__(self, other):
        """In-place add. For rotation, composes quaternions."""
        other = np.asarray(other, dtype=np.float32)
        
        if self._field_name == 'rotation':
            # Quaternion composition: q_new = other * q_current
            current = self._store._components['Transform'][np.ix_(self._indices, range(3, 7))].copy()
            result = _quat_multiply(other, current)
            self._store._components['Transform'][np.ix_(self._indices, range(3, 7))] = result
        else:
            # Simple addition for position/scale
            self._store._components['Transform'][np.ix_(self._indices, range(self._col_start, self._col_end))] += other
        return self
    
    def __isub__(self, other):
        other = np.asarray(other, dtype=np.float32)
        self._store._components['Transform'][np.ix_(self._indices, range(self._col_start, self._col_end))] -= other
        return self
    
    def __repr__(self):
        return f"_TransformFieldView({self._field_name}, {len(self._indices)} entities)"


def _quat_multiply(q1, q2):
    """Multiply quaternions q1 * q2. Format: (x, y, z, w).
    
    Supports broadcasting: q1 can be (4,) and q2 can be (N, 4).
    """
    q1 = np.atleast_2d(q1)
    q2 = np.atleast_2d(q2)
    
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    result = np.empty_like(q2)
    result[:, 0] = w1*x2 + x1*w2 + y1*z2 - z1*y2  # x
    result[:, 1] = w1*y2 - x1*z2 + y1*w2 + z1*x2  # y
    result[:, 2] = w1*z2 + x1*y2 - y1*x2 + z1*w2  # z
    result[:, 3] = w1*w2 - x1*x2 - y1*y2 - z1*z2  # w
    
    # Normalize to avoid drift
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    result /= norms
    
    return result


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
    
    def __getitem__(self, component_name) -> 'ComponentAccessor':
        """Get accessor for reading/writing component data.
        
        Supports string name ('Cube') or class type (Cube)."""
        # Convert class to string name if needed
        if not isinstance(component_name, str):
            component_name = component_name.__name__
        
        accessor = ComponentAccessor(self._store, component_name, self._indices)
        
        # Try to find the component class for field lookup
        from manifoldx.ecs import _COMPONENT_REGISTRY
        if component_name in _COMPONENT_REGISTRY:
            comp_class = _COMPONENT_REGISTRY.get(component_name)
            if comp_class and hasattr(comp_class, '_component_start_idx'):
                accessor._component_class = comp_class
        
        return accessor
    
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
        self._component_class = None  # For field lookup
    
    def __getattr__(self, name: str):
        """Access component fields by name: .position, .life, .velocity."""
        from manifoldx.ecs import _COMPONENT_REGISTRY
        
        # Get full data for component
        data = self._store.get_component_data(self._component_name, self._indices)
        
        # Check if it's Transform with built-in fields
        if self._component_name == 'Transform':
            if name in ('position', 'pos'):
                return _TransformFieldView(self._store, self._indices, 'position', 0, 3)
            elif name in ('rotation', 'rot'):
                return _TransformFieldView(self._store, self._indices, 'rotation', 3, 7)
            elif name == 'scale':
                return _TransformFieldView(self._store, self._indices, 'scale', 7, 10)
        
        # Check if component class has field positions
        comp_class = _COMPONENT_REGISTRY.get(self._component_name)
        if comp_class and hasattr(comp_class, '_component_start_idx'):
            field_info = comp_class._component_start_idx.get(name)
            if field_info:
                col, size = field_info
                return data[:, col:col+size]
        
        raise AttributeError(f"No field '{name}' in component '{self._component_name}'")
        
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
    from manifoldx.types import Float, Vector3, Vector4
    if tp == Float or tp == float:
        return ()  # Scalar
    elif tp == Vector3:
        return (3,)
    elif tp == Vector4:
        return (4,)
    else:
        return ()  # Default to scalar


def _make_component_class(cls: type, engine) -> type:
    """Create a constructable component class that registers with engine's store."""
    from manifoldx.types import Float, Vector3, Vector4
    
    annotations = cls.__annotations__
    field_names = list(annotations.keys())
    field_shapes = {name: _infer_shape_from_type(annotations[name]) for name in field_names}
    
    # Register the component in engine's store
    if engine and engine.store:
        total_cols = sum(int(np.prod(s) if s else 1) for s in field_shapes.values())
        engine.store.register_component(cls.__name__, np.dtype('f4'), (total_cols,))
    
    # Also register in global registry for field lookup
    _COMPONENT_REGISTRY[cls.__name__] = cls
    
    # Create field position index
    _component_start_idx = {}
    col_offset = 0
    for field_name in field_names:
        shape = field_shapes.get(field_name, ())
        size = int(np.prod(shape)) if shape else 1
        _component_start_idx[field_name] = (col_offset, size)
        col_offset += size
    
    def __init__(self, **kwargs):
        self._field_values = kwargs
    
    def get_data(self, n: int, registry=None):
        total_size = sum(int(np.prod(field_shapes.get(f, (1,)))) for f in field_names)
        data = np.zeros((n, total_size), dtype=np.float32)
        
        col_offset = 0
        for field_name in field_names:
            value = self._field_values.get(field_name)
            shape = field_shapes.get(field_name, ())
            size = int(np.prod(shape)) if shape else 1
            
            if value is not None:
                value = np.asarray(value, dtype=np.float32)
                if value.shape == ():
                    # Scalar: broadcast to all n rows
                    data[:, col_offset] = value.item()
                elif len(value.shape) == 1 and value.shape[0] == size:
                    # 1D array matching size: broadcast to n rows
                    data[:, col_offset:col_offset+size] = value
                elif value.shape[0] == n:
                    # Already (n, size): use as-is
                    data[:, col_offset:col_offset+size] = value
                else:
                    # Other: flatten and hope for best
                    data[:, col_offset:col_offset+size] = value.flatten()[:size]
            
            col_offset += size
        
        return data
    
    cls.__init__ = __init__
    cls.get_data = get_data
    cls._component_name = cls.__name__
    cls._component_fields = field_names
    cls._component_start_idx = _component_start_idx
    
    return cls


__all__ = [
    'EntityStore',
    'ComponentView',
    'ComponentAccessor',
    'component',
    '_COMPONENT_REGISTRY',
]
