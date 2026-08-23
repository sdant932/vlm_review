"""Numbers into figures, tables and pages.

The live report pipeline is four modules, run in this order:

    report_examples  ->  report_tables  ->  report_index  ->  report_paste

The rest are earlier, superseded renderers, kept for provenance because the
report still traces some figures back to them. See docs/REPO_MAP.md.
"""
