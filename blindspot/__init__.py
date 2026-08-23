"""Haiku 4.5 perception blind-spot eval harness.

Four subpackages, in the order the pipeline uses them:

    core      dataset adapters, prompt construction, the API runner, scorers
    judging   LLM-judge grading and ground-truth adjudication
    analysis  scoring runs into numbers (aggregation, per-dataset analyses)
    reporting numbers into figures, tables and HTML

`core` is the only one the others depend on. Nothing in `core` imports
upward, so the harness can be used for a new dataset without pulling in
any of the reporting code.
"""

__all__ = ["core", "judging", "analysis", "reporting"]
