"""TileLang stub for t-SNE — delegates to Triton.

Phase: 5
This primitive's algorithm (P-matrix + gradient SGD) is too complex for
a simple TileLang port. The TileLang backend delegates to the Triton
implementation.
"""
from flashlib.primitives.tsne.triton.train import (  # noqa: E402
    triton_tsne as triton_tsne_impl,
)


def tilelang_tsne(*args, **kwargs):
    """TileLang stub — delegates to Triton."""
    return triton_tsne_impl(*args, **kwargs)
