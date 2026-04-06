# Implement Full Operator Suite for _FieldView

## Goal
Enable `_FieldView` to be used seamlessly in expressions like:
- `query[Cube].angular * dt` - already works ✓
- `query[Transform].position += query[Cube].velocity * dt` - already works ✓
- `query[Transform].position + query[Other].offset` - needs `__add__` with field_view
- `query[FieldA].value * query[FieldB].value` - needs `__mul__` with field_view
- `np.log(query[Cube].life + 1.0)` - needs `__add__`, `__radd__`
- `query[Transform].scale = np.log(query[Cube].life + 1.0)` - needs `__radd__`

## Current State

**Implemented:**
- `__iadd__`, `__isub__`, `__imul__`, `__itruediv__` (in-place: +=, -=, *=, /=)
- `__mul__` (with scalars) ✓
- `__rmul__` (scalar * field_view) ✓
- `__le__`, `__lt__`, `__ge__`, `__gt__` (comparisons)

**Missing (blocking expressions):**

| Operator | Method | Purpose |
|----------|--------|---------|
| `+` | `__add__` | `field_view + ndarray` or `field_view + field_view` → ndarray |
| `-` | `__sub__` | `field_view - ndarray` → ndarray |
| `/` | `__truediv__` | `field_view / scalar` → ndarray |
| `scalar + x` | `__radd__` | `scalar + field_view` → ndarray |
| `scalar - x` | `__rsub__` | `scalar - field_view` → ndarray |
| `scalar / x` | `__rtruediv__` | `scalar / field_view` → ndarray |
| `array - x` | `__rsub__` | `array - field_view` → ndarray |

## Implementation

Add to `_FieldView` class in `src/manifoldx/ecs.py`:

```python
def __add__(self, other):
    """field_view + other (scalar, array, or field_view)."""
    if isinstance(other, _FieldView):
        other = other._get_data()
    return self._get_data() + np.asarray(other, dtype=np.float32)

def __sub__(self, other):
    """field_view - other (scalar, array, or field_view)."""
    if isinstance(other, _FieldView):
        other = other._get_data()
    return self._get_data() - np.asarray(other, dtype=np.float32)

def __truediv__(self, other):
    """field_view / other"""
    return self._get_data() / np.asarray(other, dtype=np.float32)

def __radd__(self, other):
    """other + field_view (scalar, array, or field_view)"""
    if isinstance(other, _FieldView):
        other = other._get_data()
    return np.asarray(other, dtype=np.float32) + self._get_data()

def __rsub__(self, other):
    """other - field_view (scalar or array)"""
    return np.asarray(other, dtype=np.float32) - self._get_data()

def __rtruediv__(self, other):
    """other / field_view (scalar)"""
    return np.asarray(other, dtype=np.float32) / self._get_data()

# Update __mul__ to handle field_view too
def __mul__(self, other):
    """field_view * other (scalar, array, or field_view)."""
    if isinstance(other, _FieldView):
        other = other._get_data()
    return self._get_data() * np.asarray(other, dtype=np.float32)

# Add __rmul__ for field_view * scalar (already have scalar * field_view)
def __rmul__(self, other):
    """other * field_view (scalar, array, or field_view)."""
    return self.__mul__(other)
```