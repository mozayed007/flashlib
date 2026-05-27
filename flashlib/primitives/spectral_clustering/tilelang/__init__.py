"""Spectral Clustering TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.primitives.spectral_clustering.tilelang.spectral import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_spectral_clustering,
)

__all__ = [
    "tilelang_spectral_clustering",
]
