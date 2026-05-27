"""gram_gemm(X) -> X @ X.T, for D >> N (dual of cov_gemm)."""
from flashlib.linalg.gram_gemm.triton import gram_gemm
from flashlib.linalg.gram_gemm import cost

# TODO: Phase 4 — add TileLang import
#   from flashlib.linalg.gram_gemm.tilelang import tilelang_gram_gemm

__all__ = ["gram_gemm", "cost"]
