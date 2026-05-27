"""gram_gemm TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.linalg.gram_gemm.tilelang.gram_gemm import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_gram_gemm,
)

__all__ = [
    "tilelang_gram_gemm",
]
