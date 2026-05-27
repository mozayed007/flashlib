# TileLang API Reference

## Overview

TileLang is a Python DSL for writing GPU kernels. It compiles Python-embedded kernel definitions to TIR (TVM Intermediate Representation) and then to native GPU code. Key advantages over raw Triton:

- Explicit shared memory management via `T.alloc_shared`
- Register fragment allocation via `T.alloc_fragment`
- Built-in `T.gemm` for matrix multiply (maps to cuBLAS / WGMMA)
- `@tilelang.jit` for automatic compilation and caching

## JIT Compilation

```python
import tilelang
from tilelang import T

@tilelang.jit(
    out_idx=[-1],           # which outputs to return (-1 = last)
    pass_configs={"tl.disable_tma": True},  # optional
)
def my_kernel(N, D, K, BLOCK_N, BLOCK_K):
    # Shape params (N, D, K) are symbolic at compile time
    # Block params (BLOCK_N, BLOCK_K) are compile-time constants

    @T.prim_func
    def kernel(
        X: T.Tensor[[N, D], "float16"],
        C: T.Tensor[[K, D], "float16"],
        Out: T.Tensor[[N], "int32"],
    ):
        # Kernel body here
        ...

    return kernel
```

**Critical:** The JIT caches by (shape, block) signature. First call per unique signature is slow (30-120s). Subsequent calls with the same args are instant.

## Memory Allocation

```python
# Shared memory (visible to all threads in a block)
X_shared = T.alloc_shared([BLOCK_N, D], dtype="float16")
C_shared = T.alloc_shared([BLOCK_K, D], dtype="float16")

# Register fragment (private to each thread)
cross_frag = T.alloc_fragment([BLOCK_N, BLOCK_K], dtype="float32")
best_dist = T.alloc_fragment([BLOCK_N], dtype="float32")
```

Shared memory is limited (~48KB per SM on A100, ~228KB on H100 with opt-in). Plan tile sizes accordingly.

## Copy Operations

```python
# Global memory -> Shared memory
T.copy(X_global_slice, X_shared)

# Shared memory -> Register fragment
T.copy(X_shared_slice, X_frag)

# Register fragment -> Global memory
T.copy(Out_frag, Out_global_slice)
```

TileLang `T.copy` handles boundary masking automatically when the source/destination slices go out of bounds.

## GEMM

```python
# C = A @ B^T (accumulate into C)
T.gemm(A_shared, B_shared, C_frag, transpose_B=True)

# C += A @ B (T.gemm accumulates — adds to existing values!)
T.gemm(A_shared, B_shared, C_frag)
```

**Critical:** `T.gemm` accumulates (adds to `C`). You must `T.clear(C_frag)` before the first GEMM if you want `C = A @ B` instead of `C += A @ B`.

## Loops

```python
# Serial loop (for epilogue / reduction)
T.serial(k_start, k_end):
    body

# Parallel loop (thread-level parallelism)
T.Parallel(i_start, i_end):
    body

# Grid-level loop (block mapping)
# Handled by the @T.prim_func structure — each block gets a program_id
```

## Clear / Fill

```python
T.clear(frag)                                    # frag[:] = 0
T.fill(frag, T.infinity("float32"))              # frag[:] = +inf
T.fill(frag, value)                              # frag[:] = value
```

## Reductions

TileLang reduction is less mature than Triton's `tl.min`/`tl.argmin`. Use serial epilogues:

```python
# Argmin over K dimension (serial)
T.serial(k, K):
    if dist[local_i, k] < best_dist[local_i]:
        best_dist[local_i] = dist[local_i, k]
        best_idx[local_i] = k
```

## Program IDs and Block Mapping

```python
@T.prim_func
def kernel(...):
    # Get block indices
    pid_n = T.get_block_idx(0)  # blockIdx.x
    pid_b = T.get_block_idx(1)  # blockIdx.y
```

## Type Annotations

```python
# Tensor parameters
X: T.Tensor[[N, D], "float16"]       # 2D tensor, N x D, float16
C: T.Tensor[[K, D], "float16"]       # 2D tensor, K x D, float16
Out: T.Tensor[[N], "int32"]          # 1D tensor, N, int32

# Scalar constants
N: T.int32                           # runtime shape
BLOCK_N: T.int32 = 64                # compile-time constant
```

## Autotuning

```python
import tilelang.autotune

@tilelang.autotune(
    configs=[
        {"block_M": 64, "block_N": 64, "block_K": 32, "num_stages": 2, "threads": 128},
        {"block_M": 128, "block_N": 64, "block_K": 32, "num_stages": 3, "threads": 256},
    ],
    keys=["M", "N", "K"],
)
@tilelang.jit(...)
def kernel(M, N, K, block_M, block_N, block_K, num_stages, threads):
    ...
```

## dtype String Conversion

```python
# From torch dtype to TileLang string
dtype_str = str(x.dtype).split(".")[-1]  # torch.float16 -> "float16"

# TileLang also accepts torch dtypes directly
X: T.Tensor[[N, D], torch.float16]
```

## Common Patterns

### Euclidean Distance Assign (KMeans)

```python
# Load X tile to shared
T.copy(X_slice, X_shared)
# Load C tile to shared
T.copy(C_slice, C_shared)
# Clear accumulator
T.clear(cross_frag)
# Compute cross term: X @ C^T
T.gemm(X_shared, C_shared, cross_frag, transpose_B=True)
# Epilogue: dist = c_sq - 2 * cross, argmin
T.serial(k, BLOCK_K):
    dist = c_sq[k] - 2.0 * cross_frag[local_i, k]
    if dist < best_dist[local_i]:
        best_dist[local_i] = dist
        best_idx[local_i] = k_start + k
```

### Tall-Skinny GEMM (cov_gemm: X^T @ X)

```python
# Each block computes a (BLOCK_DI, BLOCK_DJ) tile of the output
# Stream X in panels of BLOCK_N rows
T.clear(acc_frag)
for n_start in T.serial(0, N, BLOCK_N):
    T.copy(X[n_start:n_start+BLOCK_N, di_start:di_start+BLOCK_DI], xi_shared)
    T.copy(X[n_start:n_start+BLOCK_N, dj_start:dj_start+BLOCK_DJ], xj_shared)
    T.gemm(xi_shared, xj_shared, acc_frag, transpose_A=True)
```
