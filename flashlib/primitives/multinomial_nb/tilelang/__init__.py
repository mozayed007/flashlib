"""Multinomial NB TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.primitives.multinomial_nb.tilelang.nb import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_multinomial_nb,
)

__all__ = [
    "tilelang_multinomial_nb",
]
