# Lazy import to avoid wgpu dependency for types-only imports
def __getattr__(name):
    if name == 'Engine':
        from manifoldx.engine import Engine
        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def hello() -> str:
    return "Hello from manifoldx!"
