"""TileLang gram_gemm: X @ X^T for wide X (D >> N).

Phase: 4
Reference: flashlib/linalg/gram_gemm/triton/gram_gemm.py
Guide: docs/phase-4-gemm.md
"""
from __future__ import annotations

from typing import Optional

import torch

# ---------------------------------------------------------------------------
# TileLang availability guard
# ---------------------------------------------------------------------------

_TILELANG_AVAILABLE = False
_TILELANG_IMPORT_ERROR: Optional[Exception] = None


def _try_init_tilelang() -> bool:
    global _TILELANG_AVAILABLE, _TILELANG_IMPORT_ERROR
    if _TILELANG_AVAILABLE:
        return True
    if _TILELANG_IMPORT_ERROR is not None:
        return False
    try:
        import tilelang  # noqa: F401
        _TILELANG_AVAILABLE = True
        return True
    except Exception as e:  # pragma: no cover
        _TILELANG_IMPORT_ERROR = e
        return False


# ---------------------------------------------------------------------------
# Triton fallback (always available)
# ---------------------------------------------------------------------------
from flashlib.linalg.gram_gemm.triton import gram_gemm as triton_gram_gemm  # noqa: E402


# ---------------------------------------------------------------------------
# TileLang gram_gemm
# ---------------------------------------------------------------------------

def tilelang_gram_gemm(X: torch.Tensor) -> torch.Tensor:
    """X @ X^T using TileLang GEMM with symmetric optimization.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    X : (N, D) float32 CUDA tensor, D >> N.

    Returns
    -------
    (N, N) float32 Gram matrix.
    """
    if not _try_init_tilelang():
        return triton_gram_gemm(X)

    import tilelang
    from tilelang import T

    N, D = X.shape

    # TODO: Phase 4 — implement TileLang gram_gemm kernel
    #
    # Design:
    #   - Output is (N, N) — may be large for big N
    #   - Stream D in panels of BLOCK_D columns
    #   - Symmetric optimization: compute upper triangle only, mirror
    #   - Each block handles a (BLOCK_NI, BLOCK_NJ) tile of the output
    #
    # Kernel structure:
    #   @tilelang.jit(out_idx=[-1])
    #   def _build_gram_gemm(N, D, BLOCK_N, BLOCK_D):
    #       @T.prim_func
    #       def kernel(X, Out):
    #           pid_i = T.get_block_idx(0)
    #           pid_j = T.get_block_idx(1)
    #           T.clear(acc_frag)
    #           for d_start in T.serial(0, D, BLOCK_D):
    #               T.copy(X[ni:ni+BLOCK_N, d_start:d_start+BLOCK_D], xi_shared)
    #               T.copy(X[nj:nj+BLOCK_N, d_start:d_start+BLOCK_D], xj_shared)
    #               T.gemm(xi_shared, xj_shared, acc_frag, transpose_A=True)
    #           T.copy(acc_frag, Out[ni:ni+BLOCK_N, nj:nj+BLOCK_N])
    #       return kernel
    #
    # Post-processing:
    #   result = torch.triu(result) + torch.triu(result, diagonal=1).T
    #
    # Key gotchas:
    #   - T.gemm with transpose_A=True: xi_shared^T @ xj_shared -> (BLOCK_N, BLOCK_N)
    #   - Accumulates across D chunks — T.clear() before D-loop
    #   - Two shared buffers: xi_shared and xj_shared (same shape, different row ranges)
    raise NotImplementedError(
        "TileLang gram_gemm not yet implemented — see docs/phase-4-gemm.md"
    )
