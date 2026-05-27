"""ab_gemm(A, B) -> A.T @ B for tall-skinny inputs sharing N dim."""
from flashlib.linalg.ab_gemm.triton import ab_gemm
from flashlib.linalg.ab_gemm import cost

# TODO: Phase 4 — add TileLang import
#   from flashlib.linalg.ab_gemm.tilelang import tilelang_ab_gemm

__all__ = ["ab_gemm", "cost"]
