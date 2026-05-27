# Debugging TileLang Kernels

## Common Errors

### 1. `tilelang` Not Installed

```
ImportError: tilelang is not installed
```

**Fix:** `pip install tilelang>=0.1.6`

Check with: `python -c "from flashlib._tilelang import tilelang_available; print(tilelang_available())"`

### 2. Shared Memory Overflow

```
RuntimeError: SMEM allocation failed
```

**Cause:** Your tile sizes are too large for the GPU's shared memory.

**Fix:** Reduce BLOCK_N, BLOCK_K, or D. Check:
```python
# Rule of thumb: BLOCK_N * D * dtype_bytes + BLOCK_K * D * dtype_bytes < 48KB (A100) or 228KB (H100)
# float16 = 2 bytes, float32 = 4 bytes
smem_needed = (BLOCK_N + BLOCK_K) * D * 2  # for float16
```

### 3. dtype Mismatch

```
TypeError: expected float16, got float32
```

**Cause:** TileLang kernels are dtype-strict. The kernel signature declares exact dtypes.

**Fix:** Cast inputs before passing to the kernel:
```python
x = x.to(torch.float16)
```

Or use `"float32"` in the kernel signature if you want fp32.

### 4. Shape Mismatch

```
AssertionError: Expected shape (N, D), got (N, D+1)
```

**Cause:** Input tensor shape doesn't match kernel declaration.

**Fix:** Ensure inputs are contiguous and correctly shaped. Pad if needed (like KNN does for D < 16).

### 5. JIT Compilation Timeout

```
TimeoutError: TileLang JIT compilation exceeded timeout
```

**Cause:** First compilation of a new signature takes 30-120s.

**Fix:** This is expected. Increase timeout or wait. Subsequent calls with the same args are instant.

### 6. `T.gemm` Accumulation Bug

```
# Wrong: C contains garbage from previous iteration
T.gemm(A, B, C)  # C += A @ B, but C was uninitialized!

# Correct:
T.clear(C)        # C = 0
T.gemm(A, B, C)  # C = A @ B
```

**Symptom:** Output values are 2x or 3x expected, or randomly large.

**Fix:** Always `T.clear()` or `T.fill()` before the first `T.gemm`.

### 7. Wrong `transpose_B` in `T.gemm`

```python
# For cross term <x, c>: x is (BLOCK_N, D), c is (BLOCK_K, D)
# We want: cross = x @ c^T, shape (BLOCK_N, BLOCK_K)
T.gemm(X_shared, C_shared, cross_frag, transpose_B=True)  # correct

# If you forget transpose_B:
T.gemm(X_shared, C_shared, cross_frag)  # x @ c, shape (BLOCK_N, D) — WRONG
```

**Symptom:** Shape errors or garbage output.

### 8. Argmin Off-by-One

```python
# Wrong: using local index instead of global
best_idx[local_i] = k  # local index within BLOCK_K

# Correct: add k_start offset
best_idx[local_i] = k_start + k  # global centroid index
```

**Symptom:** Assignments are wrong but distances look correct.

### 9. Boundary Masking Missing

```python
# Wrong: accessing out-of-bounds centroids
T.copy(C_slice, C_shared)  # may read garbage for k >= K

# Correct: TileLang handles masking in T.copy, but verify your slicing
```

### 10. Triton Fallback Not Working

```python
# Wrong: crash when tilelang unavailable
import tilelang  # ImportError!

# Correct: use the guard pattern
if not _try_init_tilelang():
    return triton_<op>(...)
```

## Debugging Techniques

### Add Print Statements

TileLang supports `T.print` for debugging inside kernels:

```python
T.print("x_shared[0, 0] =", X_shared[0, 0])
```

### Test with Small Shapes

Start with tiny shapes to verify correctness before scaling up:

```python
B, N, D, K = 1, 64, 16, 8  # tiny
```

### Compare Against Torch Reference

```python
# Torch reference (always correct)
ref = torch.matmul(x.float(), c.float().transpose(-1, -2))

# TileLang kernel
out = tilelang_kernel(x, c)

# Check
torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-1)
```

### Use the Interactive Shell

```bash
modal run scripts/modal/shell.py
```

Then debug interactively:
```python
import torch
from flashlib.primitives.kmeans.tilelang.assign import tilelang_assign_euclid
x = torch.randn(1, 64, 16, device='cuda', dtype=torch.float32)
c = torch.randn(1, 8, 16, device='cuda', dtype=torch.float32)
result = tilelang_assign_euclid(x, c)
print(result)
```

### Check TileLang Compilation Output

TileLang prints compilation info when `verbose=True`:

```python
@tilelang.jit(verbose=True)
def kernel(...):
    ...
```

### Profile with CUDA Events

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
result = tilelang_kernel(x, c)
end.record()

torch.cuda.synchronize()
print(f"Time: {start.elapsed_time(end):.2f} ms")
```

## Error Checklist

When a kernel fails:

1. [ ] Is tilelang installed? (`python -c "import tilelang"`)
2. [ ] Are input dtypes correct? (match kernel signature)
3. [ ] Are input shapes correct? (check N, D, K)
4. [ ] Are inputs contiguous? (`x = x.contiguous()`)
5. [ ] Did you `T.clear()` before `T.gemm`?
6. [ ] Is `transpose_B` set correctly?
7. [ ] Are boundary masks correct?
8. [ ] Does the TileLang kernel compile? (run with `verbose=True`)
9. [ ] Does it work with tiny shapes? (N=64, D=16, K=8)
10. [ ] Is the epilogue correct? (argmin offset, etc.)
