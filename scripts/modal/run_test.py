"""Run pytest on Modal H100.

Usage::

    modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py
    modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_kmeans
    modal run scripts/modal/run_test.py --path tests/ -x --tb=short

Auto-syncs the local working tree to the container so edits are
immediately testable without a git push.
"""
from __future__ import annotations

import subprocess

import modal

from scripts.modal.setup_env import app, image


@app.function(
    image=image,
    gpu="H100",
    timeout=1800,
    concurrency_limit=1,
)
def run_pytest(path: str = "tests/", k: str = "", x: bool = True,
               tb: str = "short", verbose: bool = True):
    """Execute pytest inside the H100 container."""
    cmd = ["python", "-m", "pytest", path]
    if k:
        cmd += ["-k", k]
    if x:
        cmd.append("-x")
    cmd += [f"--tb={tb}"]
    if verbose:
        cmd.append("-v")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


@app.local_entrypoint()
def main(path: str = "tests/", k: str = "", x: bool = True,
         tb: str = "short", verbose: bool = True):
    """Run tests on Modal H100.

    Args:
        path: Test file or directory (default: tests/).
        k: Pytest -k expression to filter tests.
        x: Stop on first failure (default: True).
        tb: Traceback style (default: short).
        verbose: Verbose output (default: True).
    """
    exit_code = run_pytest.remote(path=path, k=k, x=x, tb=tb, verbose=verbose)
    if exit_code != 0:
        print(f"Tests failed (exit code {exit_code})")
        raise SystemExit(exit_code)
    print("All tests passed.")
