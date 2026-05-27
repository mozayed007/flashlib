# TileLang API Reference

## Overview

TileLang is a Python DSL for writing GPU kernels. It compiles Python-embedded kernel definitions to TIR (TVM Intermediate Representation) and then to native GPU code. Key advantages over raw Triton:

- Explicit shared memory management via `T.alloc_shared`
- Register fragment allocation via `T.alloc_fragment`
- Built-in `T.gemm` for matrix multiply (maps to cuBLAS / WGMMA)
- `@tilelang.jit` for automatic compilation and caching

## JIT Compilation

TileLang has two JIT API styles (both compile to the same TIR):

**Modern style (recommended — from official README):**

```python
import tilelang
from tilelang import T

@tilelang.jit
def my_kernel(A, B, block_M=64, block_N=64, block_K=64, dtype=T.float16):
    M, N, K = T.const('M, N, K')
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty([M, N], dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])
    return C
```

**Older style (also works — used in some examples):**

```python
@tilelang.jit(out_idx=[-1])
def my_kernel(N, D, K, BLOCK_N, BLOCK_K):
    @T.prim_func
    def kernel(
        X: T.Tensor[[N, D], "float16"],
        C: T.Tensor[[K, D], "float16"],
        Out: T.Tensor[[N], "int32"],
    ):
        pid_n = T.get_block_binding(0)
        ...
    return kernel
```

Both styles work. The kernel examples in the phase guides use the older style for simplicity. Study the official TileLang README (https://github.com/tile-ai/tilelang) for the modern pattern.

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
for k in T.serial(K):
    body

# Serial loop with explicit start/end
for k in T.serial(start, end):
    body

# Pipelined loop (with software pipelining stages)
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    body

# Parallel loop (thread-level parallelism, block sizes not ranges)
for i, j in T.Parallel(block_M, block_N):
    body

# Grid-level loop (block mapping)
# Handled by T.Kernel context manager — see Program IDs section
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
for k in T.serial(K):
    if dist[local_i, k] < best_dist[local_i]:
        best_dist[local_i] = dist[local_i, k]
        best_idx[local_i] = k
```

## Program IDs and Block Mapping

The idiomatic way to define block context is with `T.Kernel`:

```python
# Modern pattern (recommended — from official README)
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    # bx = blockIdx.x, by = blockIdx.y
    n_start = by * block_M
    m_start = bx * block_N
```

Alternative using `T.get_block_binding`:

```python
# Low-level alternative
pid_n = T.get_block_binding(0)  # blockIdx.x
pid_b = T.get_block_binding(1)  # blockIdx.y
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
for k in T.serial(BLOCK_K):
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
