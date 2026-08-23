"""LLM-judge grading and ground-truth adjudication.

Separate from `core.scoring` because these paths cost money and are
non-deterministic: they call a model rather than compare strings.
"""
