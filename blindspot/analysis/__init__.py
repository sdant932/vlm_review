"""Runs into numbers: aggregation, per-dataset analyses, ablation scoring.

Everything here reads `results/*.jsonl` and writes JSON. No HTML — keeping
rendering out means the numbers stay independently checkable and a report
can be rebuilt without re-scoring thousands of rows.
"""
