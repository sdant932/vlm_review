"""Runnable entry points, grouped by pipeline stage.

    download/  fetch the scraped benchmarks into data/
    generate/  build and verify the synthetic dataset in data/svg_localization
    run/       call the API and write results/*.jsonl
    analyze/   turn results into diagnostic pages

A package rather than a loose directory so a few scripts can share code
(`scripts.run.official_eval` is imported by two analyses). Run them as
files -- `python scripts/generate/gen_svg_localization.py` -- not as
modules; the editable install puts `blindspot` on the path either way.
"""
