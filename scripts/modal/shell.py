"""Interactive H100 shell for debugging TileLang kernels.

Usage::

    modal run scripts/modal/shell.py

Drops you into a bash shell on an H100 with flashlib installed.
Local working tree is synced — edit locally, test on GPU immediately.

Inside the shell:
    python -c "from flashlib._tilelang import tilelang_available; print(tilelang_available())"
    python -m pytest tests/test_tilelang_parity.py -k test_kmeans -v
    python -c "import flashlib; flashlib.diagnose()"
"""
from __future__ import annotations

import subprocess

import modal

from scripts.modal.setup_env import app, image


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,
    concurrency_limit=1,
)
def shell():
    """Launch an interactive bash shell on H100."""
    subprocess.run(["bash", "-l"], check=False)


@app.local_entrypoint()
def main():
    """Open an interactive shell on Modal H100."""
    shell.remote()
