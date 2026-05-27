"""TileLang ab_gemm: A^T @ B for tall-skinny inputs sharing N dim.

Phase: 4
Reference: flashlib/linalg/ab_gemm/triton/ab_gemm.py
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
from flashlib.linalg.ab_gemm.triton import ab_gemm as triton_ab_gemm  # noqa: E402


# ---------------------------------------------------------------------------
# TileLang ab_gemm
# ---------------------------------------------------------------------------

def tilelang_ab_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A^T @ B using TileLang tall-skinny GEMM.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    A : (N, D1) float32 CUDA tensor
    B : (N, D2) float32 CUDA tensor

    Returns
    -------
    (D1, D2) float32 = A^T @ B.
    """
    if not _try_init_tilelang():
        return triton_ab_gemm(A, B)

    import tilelang
    from tilelang import T

    N, D1 = A.shape
    D2 = B.shape[1]

    # TODO: Phase 4 — implement TileLang ab_gemm kernel
    #
    # Design:
    #   - Output is (D1, D2) — similar to cov_gemm but not symmetric
    #   - Stream N in panels of BLOCK_N rows
    #   - Each block handles a (BLOCK_D1, BLOCK_D2) tile of the output
    #
    # Kernel structure:
    #   @tilelang.jit(out_idx=[-1])
    #   def _build_ab_gemm(N, D1, D2, BLOCK_N, BLOCK_D1, BLOCK_D2):
    #       @T.prim_func
    #       def kernel(A, B, Out):
    #           pid_i = T.get_block_idx(0)  # along D1
    #           pid_j = T.get_block_idx(1)  # along D2
    #           T.clear(acc_frag)
    #           for n_start in T.serial(0, N, BLOCK_N):
    #               T.copy(A[n_start:, d1_start:d1_start+BLOCK_D1], a_shared)
    #               T.copy(B[n_start:, d2_start:d2_start+BLOCK_D2], b_shared)
    #               T.gemm(a_shared, b_shared, acc_frag, transpose_A=True)
    #           T.copy(acc_frag, Out[d1:d1+BLOCK_D1, d2:d2+BLOCK_D2])
    #       return kernel
    #
    # Key gotchas:
    #   - NOT symmetric (unlike cov_gemm/gram_gemm) — no mirroring needed
    #   - T.gemm with transpose_A=True: a_shared^T @ b_shared -> (BLOCK_D1, BLOCK_D2)
    #   - T.clear(acc_frag) before the N-loop
    #   - Two shared buffers for A and B columns
    raise NotImplementedError(
        "TileLang ab_gemm not yet implemented — see docs/phase-4-gemm.md"
    )
