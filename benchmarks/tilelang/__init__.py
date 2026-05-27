"""TileLang benchmarks — compare TileLang vs Triton vs CuteDSL.

Each ``bench_<primitive>.py`` script runs a timed comparison for a
specific primitive.  Use the Modal runner for H100 benchmarks::

    modal run scripts/modal/bench_primitive.py --primitive kmeans
"""
