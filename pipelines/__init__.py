"""The three efforts, each a launch script and nothing else.

A pipeline aggregates: it declares which generic scripts under `scripts/` to
call, in what order, with what arguments. It contains no logic of its own, so
moving work between pipelines never means moving code.

    literature_eval           evaluate on the published benchmarks we download
    synth_localization_eval   generate our dataset and evaluate on it
    finetune_data             build the SFT / GRPO training data
"""
