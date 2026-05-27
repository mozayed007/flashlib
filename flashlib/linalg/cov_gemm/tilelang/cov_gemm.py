"""TileLang cov_gemm: X^T @ X for tall-skinny X (N >> D).

Phase: 3-4
Reference: flashlib/linalg/cov_gemm/triton/cov_gemm.py
Guide: docs/phase-3-pca.md, docs/phase-4-gemm.md
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
from flashlib.linalg.cov_gemm.triton import cov_gemm as triton_cov_gemm  # noqa: E402


# ---------------------------------------------------------------------------
# TileLang cov_gemm
# ---------------------------------------------------------------------------

def tilelang_cov_gemm(X: torch.Tensor) -> torch.Tensor:
    """X^T @ X using TileLang tall-skinny GEMM with symmetric optimization.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    X : (N, D) float32 CUDA tensor, N >> D.

    Returns
    -------
    (D, D) float32 covariance matrix.
    """
    if not _try_init_tilelang():
        return triton_cov_gemm(X)

    import tilelang
    from tilelang import T

    N, D = X.shape

    # TODO: Phase 3-4 — implement TileLang cov_gemm kernel
    #
    # Design:
    #   - Output is (D, D) — small, fits in L2 cache
    #   - Stream X in panels of BLOCK_N rows
    #   - Symmetric optimization: compute upper triangle only, mirror
    #   - Each block handles a (BLOCK_DI, BLOCK_DJ) tile of the output
    #
    # Kernel structure:
    #   @tilelang.jit(out_idx=[-1])
    #   def _build_cov_gemm(N, D, BLOCK_N, BLOCK_D):
    #       @T.prim_func
    #       def kernel(X, Out):
    #           pid_i = T.get_block_idx(0)
    #           pid_j = T.get_block_idx(1)
    #           # Skip lower triangle
    #           T.clear(acc_frag)
    #           for n_start in T.serial(0, N, BLOCK_N):
    #               T.copy(X[n_start:, di_start:di_start+BLOCK_D], xi_shared)
    #               T.copy(X[n_start:, dj_start:dj_start+BLOCK_D], xj_shared)
    #               T.gemm(xi_shared, xj_shared, acc_frag, transpose_A=True)
    #           T.copy(acc_frag, Out[di:di+BLOCK_D, dj:dj+BLOCK_D])
    #       return kernel
    #
    # Post-processing:
    #   result = torch.triu(result) + torch.triu(result, diagonal=1).T
    #
    # Key gotchas:
    #   - T.gemm with transpose_A=True computes X^T @ X_chunk
    #   - T.gemm accumulates — T.clear(acc_frag) before the N-loop
    #   - Need TWO shared buffers for xi and xj (different column ranges)
    #   - Symmetric: only compute where dj_start + BLOCK_D > di_start
    raise NotImplementedError(
        "TileLang cov_gemm not yet implemented — see docs/phase-4-gemm.md"
    )
