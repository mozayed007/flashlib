"""Modal H100 image definition for flashlib TileLang development.

Usage::

    import modal
    from scripts.modal.setup_env import image, app

The image installs:
    - PyTorch 2.5 + CUDA 12.4
    - Triton 3.6
    - tilelang (from PyPI or source)
    - flashlib in editable mode (auto-syncs local changes)
    - pytest + scikit-learn for parity tests
"""
from __future__ import annotations

import modal

app = modal.App("flashlib-tilelang")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "wget", "cmake", "ninja-build")
    .pip_install(
        "torch>=2.5",
        "triton>=3.6",
        "numpy",
        "numba",
        "tqdm",
        "nvidia-cutlass-dsl",
    )
    .pip_install("tilelang>=0.1.6")
    .pip_install("pytest>=7", "scikit-learn>=1.5,<1.8")
    .add_local_dir(".", remote_path="/root/flashlib")
    .workdir("/root/flashlib")
    .env({"FLASHLIB_TILELANG_DEV": "1"})
)
