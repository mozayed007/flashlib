# Phase 0: Infrastructure Setup

## What You're Building

Verification that all infrastructure files work correctly. No GPU needed.

## Files Created (by Kilo)

| File | Purpose |
|------|---------|
| `flashlib/_tilelang.py` | TileLang availability check (mirrors cutedsl pattern) |
| `scripts/modal/setup_env.py` | Modal H100 image definition |
| `scripts/modal/run_test.py` | Run pytest on Modal H100 |
| `scripts/modal/bench_primitive.py` | Run benchmarks on Modal H100 |
| `scripts/modal/shell.py` | Interactive H100 shell |
| `tests/test_tilelang_parity.py` | Skeleton parity tests with skip markers |
| `benchmarks/tilelang/__init__.py` | Empty init for benchmark package |
| Updated `pyproject.toml` | Added `tilelang = ["tilelang>=0.1.6"]` optional dep |
| Updated `flashlib/diagnose.py` | Added tilelang version to diagnostics |

## Step-by-Step

### 1. Verify `flashlib/_tilelang.py`

```bash
python -c "from flashlib._tilelang import _try_init_tilelang; print(_try_init_tilelang())"
```

Expected output: `False` (tilelang not installed locally) or `True` (if installed).

### 2. Verify `pyproject.toml`

```bash
python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb')); print(d['project']['optional-dependencies']['tilelang'])"
```

Expected: `['tilelang>=0.1.6']`

### 3. Verify `diagnose.py`

```bash
python -c "import flashlib; flashlib.diagnose()"
```

Expected: Shows `tilelang: NOT INSTALLED (ModuleNotFoundError)` or a version number.

### 4. Verify test skeleton skips cleanly

```bash
python -m pytest tests/test_tilelang_parity.py -v
```

Expected: All tests skipped with "tilelang not installed" message.

### 5. Verify Modal scripts exist

```bash
ls scripts/modal/
```

Expected: `setup_env.py  run_test.py  bench_primitive.py  shell.py`

### 6. Install tilelang (optional, for local testing)

```bash
pip install tilelang>=0.1.6
```

Then re-run step 1 — should return `True`.

## Troubleshooting

**`_try_init_tilelang()` returns `False` even after installing tilelang:**
- Check: `pip show tilelang`
- Try: `python -c "import tilelang; print(tilelang.__version__)"`

**Modal scripts fail to import:**
- Ensure you're in the repo root: `cd /path/to/flashlib`
- Ensure `scripts/modal/` has an `__init__.py` (it doesn't need one for `modal run`)

## Checkpoint

- [ ] `python -c "from flashlib._tilelang import _try_init_tilelang; print(_try_init_tilelang())"` runs without error
- [ ] `python -c "import flashlib; flashlib.diagnose()"` shows tilelang line
- [ ] `python -m pytest tests/test_tilelang_parity.py -v` skips all tests
- [ ] `ls scripts/modal/` shows 4 files
- [ ] `ls benchmarks/tilelang/__init__.py` exists
