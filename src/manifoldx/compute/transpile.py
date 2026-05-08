"""Phase-2 Python → WGSL transpiler for Compute kernels.

See `.knowledge/analysis/2026-05-08-compute-phase-2-shader-compiler-design.md`.
"""
from __future__ import annotations


class ComputeShaderCompileError(Exception):
    """Raised when a Compute kernel cannot be transpiled to valid WGSL.

    Carries structured fields for the IDE/REPL to surface clearly:
    file, line, column, error category, the offending source line, and
    a human-readable message.
    """

    def __init__(
        self,
        *,
        category: str,
        message: str,
        filename: str,
        line: int,
        col: int,
        source_line: str | None = None,
    ):
        self.category = category
        self.message = message
        self.filename = filename
        self.line = line
        self.col = col
        self.source_line = source_line
        super().__init__(self._render())

    def _render(self) -> str:
        head = f"{self.filename}:{self.line}:{self.col}: {self.category}: {self.message}"
        if self.source_line is None:
            return head
        caret = " " * self.col + "^"
        return f"{head}\n  {self.source_line}\n  {caret}"
