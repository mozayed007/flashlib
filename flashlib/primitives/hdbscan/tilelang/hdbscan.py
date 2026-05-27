"""TileLang stub for HDBSCAN — delegates to Triton.

Phase: 5
This primitive's algorithm (MRD + Boruvka MST + tree condensation) is too
complex for a simple TileLang port. The TileLang backend delegates to
the Triton implementation.
"""
from flashlib.primitives.hdbscan.triton import (  # noqa: E402
    flash_hdbscan as triton_hdbscan,
)


def tilelang_hdbscan(*args, **kwargs):
    """TileLang stub — delegates to Triton."""
    return triton_hdbscan(*args, **kwargs)
