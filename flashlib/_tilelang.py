"""TileLang availability check.

Mirrors the CuteDSL pattern in every ``cutedsl/__init__.py``: a module-level
``_try_init_tilelang()`` guard that caches the import result so repeated
calls are free.

Usage::

    from flashlib._tilelang import tilelang_available, tilelang

    if tilelang_available():
        import tilelang as tl
        ...
"""
from __future__ import annotations

import importlib
from typing import Any, Optional

__all__ = ["tilelang_available", "tilelang", "_try_init_tilelang"]

_TILELANG_AVAILABLE = False
_TILELANG_IMPORT_ERROR: Optional[Exception] = None
_tilelang: Any = None


def _try_init_tilelang() -> bool:
    """Attempt to import ``tilelang``. Returns True on success.

    The result is cached — subsequent calls return immediately.
    """
    global _TILELANG_AVAILABLE, _TILELANG_IMPORT_ERROR, _tilelang
    if _TILELANG_AVAILABLE:
        return True
    if _TILELANG_IMPORT_ERROR is not None:
        return False
    try:
        _tilelang = importlib.import_module("tilelang")
        _TILELANG_AVAILABLE = True
        return True
    except Exception as e:  # pragma: no cover - depends on local install
        _TILELANG_IMPORT_ERROR = e
        return False


def tilelang_available() -> bool:
    """Return True if ``tilelang`` is importable."""
    return _try_init_tilelang()


def tilelang():
    """Return the ``tilelang`` module, or raise if unavailable.

    Raises
    ------
    ImportError
        If ``tilelang`` is not installed.
    """
    if not _try_init_tilelang():
        raise ImportError(
            "tilelang is not installed. Install it with: "
            "pip install tilelang>=0.1.6"
        ) from _TILELANG_IMPORT_ERROR
    return _tilelang
