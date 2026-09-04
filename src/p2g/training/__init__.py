"""Public Pixel4DGS training components.

Submodules are intentionally imported explicitly by their consumers.  Keeping
this package initializer free of eager imports lets artifact inspection and
CPU-only tooling run without loading Torch or a ROCm runtime.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
