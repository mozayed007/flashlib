"""TileLang stub for Random Forest — delegates to Triton.

Phase: 5
RandomForest has no impl.py dispatcher — it's a direct
FlashRandomForestClassifier class. The TileLang backend wraps it
as a module-level function that delegates to the Triton class.
"""


def tilelang_random_forest(*args, backend=None, **kwargs):
    """TileLang stub — delegates to the Triton FlashRandomForestClassifier.

    Parameters
    ----------
    *args, **kwargs : forwarded to FlashRandomForestClassifier
    backend : ignored (always uses Triton)
    """
    from flashlib.primitives.random_forest.impl import FlashRandomForestClassifier
    return FlashRandomForestClassifier(*args, **kwargs)
