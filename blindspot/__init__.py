"""Haiku 4.5 perception blind-spot eval harness.

One flat package of seventeen modules. There are no subpackages: an earlier
layout split this four ways by role, and `legacy/` holds the pre-consolidation
copies for reference.

Foundations -- imported by everything else, importing nothing else here:

    core                dataset adapters, prompt construction, the API runner, scorers
    charxiv             CharXiv's own constants/templates, vendored verbatim
    flow                the generic stage/step launcher every pipeline is built on
    tools               repository-inspection helpers (verify-install, etc.)

Acquiring and building data:

    download            fetch a published benchmark into data/<name>/
    generate            build the SVG localization dataset and its documentation
    generate_finetune   the resolution ladder, exact-ink boxes and SFT records

Spending money:

    run_api             the experiment arms the main runner does not cover
    judge               LLM-judge grading and ground-truth adjudication

Turning runs into numbers:

    eval                aggregation and per-dataset analyses
    diagnose            focused diagnostics, one question and one page each

Turning numbers into pages:

    report              the main report build
    report_pages        standalone evidence pages, outside the report chain
    report_finetune     the Part 3 finetuning artefacts
    report_worked       worked examples of what the reward actually does
    render_markdown     markdown to a self-contained HTML page

Driving all of the above:

    pipelines           every pipeline, declared as ordered lists of steps

`core` is the only module the others depend on, and nothing in `core` imports
upward, so the harness can be pointed at a new dataset without pulling in any
of the reporting code.
"""

__all__ = [
    "charxiv",
    "core",
    "diagnose",
    "download",
    "eval",
    "flow",
    "generate",
    "generate_finetune",
    "judge",
    "pipelines",
    "render_markdown",
    "report",
    "report_finetune",
    "report_pages",
    "report_worked",
    "run_api",
    "tools",
]
