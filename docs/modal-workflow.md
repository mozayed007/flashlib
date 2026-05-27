# Modal H100 Workflow

## Setup

### 1. Install Modal CLI

```bash
pip install modal
modal setup  # authenticate with your Modal account
```

### 2. Verify Credits

Check your budget at https://modal.com/settings/usage

H100 costs $3.95/hr. Budget: 72 hours ($284.52).

## Running Tests

### Run a specific test

```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_kmeans
```

### Run all TileLang tests

```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py
```

### Run with verbose output

```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_kmeans --tb long
```

## Interactive Shell

```bash
modal run scripts/modal/shell.py
```

Inside the shell:

```bash
# Check tilelang is available
python -c "from flashlib._tilelang import tilelang_available; print(tilelang_available())"

# Run diagnostics
python -c "import flashlib; flashlib.diagnose()"

# Run a single test
python -m pytest tests/test_tilelang_parity.py -k test_kmeans -v

# Debug a kernel interactively
python -c "
import torch
from flashlib.primitives.kmeans.tilelang.assign import tilelang_assign_euclid
x = torch.randn(1, 64, 16, device='cuda', dtype=torch.float32)
c = torch.randn(1, 8, 16, device='cuda', dtype=torch.float32)
print(tilelang_assign_euclid(x, c))
"
```

## Benchmarks

### Run a benchmark

```bash
modal run scripts/modal/bench_primitive.py --primitive kmeans
```

### Specify backend

```bash
modal run scripts/modal/bench_primitive.py --primitive kmeans --backend tilelang
modal run scripts/modal/bench_primitive.py --primitive kmeans --backend triton
```

### Custom shapes

```bash
modal run scripts/modal/bench_primitive.py --primitive kmeans --sizes "4096,128,64"
```

## How Modal Syncs Your Code

Modal auto-syncs the local working tree to the container when you run `modal run`. This means:

- Local edits are immediately testable — no git push needed
- The container sees exactly what's on your disk
- New files you create locally are available in the container

## Budget Management

| Phase | GPU Hrs | Cost |
|-------|---------|------|
| 0 | 0.9 | $3.56 |
| 1 | 3.0 | $11.85 |
| 2 | 4.0 | $15.80 |
| 3 | 2.6 | $10.27 |
| 4 | 3.5 | $13.83 |
| 5 | 9.0 | $35.55 |
| 6 | 1.4 | $5.53 |
| 7 | 4.6 | $18.17 |
| 8 | 6.4 | $25.28 |
| Buffer | 8.0 | $31.60 |
| **Total** | **43.4** | **$171.44** |

### Tips to Save Budget

- Use `timeout=300` for quick tests (default is 1800s)
- Kill idle shells — they still count against your hours
- Run tests with `-k` to only run the test you're working on
- Use `--x` (stop on first failure) to avoid wasting time on cascading failures
- Compile TileLang kernels locally (CPU) when possible, only use H100 for execution

## Modal Script Reference

| Script | Purpose |
|--------|---------|
| `scripts/modal/setup_env.py` | Image definition (don't run directly) |
| `scripts/modal/run_test.py` | Run pytest on H100 |
| `scripts/modal/bench_primitive.py` | Run benchmarks on H100 |
| `scripts/modal/shell.py` | Interactive bash shell on H100 |
