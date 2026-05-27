"""cov_gemm TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.linalg.cov_gemm.tilelang.cov_gemm import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_cov_gemm,
)

__all__ = [
    "tilelang_cov_gemm",
]
