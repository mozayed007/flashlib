"""TileLang stub for UMAP — delegates to Triton.

Phase: 5
This primitive's algorithm (KNN graph + fuzzy simplicial set + SGD layout)
is too complex for a simple TileLang port. The TileLang backend delegates
to the Triton implementation.
"""
from flashlib.primitives.umap.triton.flash_umap import (  # noqa: E402
    flash_umap as triton_flash_umap,
)


def tilelang_flash_umap(*args, **kwargs):
    """TileLang stub — delegates to Triton."""
    return triton_flash_umap(*args, **kwargs)
