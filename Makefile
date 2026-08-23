# Thin wrappers over the commands in docs/PIPELINE.md. Nothing here is magic --
# every target is one line you could type yourself.

PY ?= python
SPEND ?= 0.10          # USD hard stop for the smoke-test eval
COUNT ?= 200           # scenes for a full dataset regeneration

.PHONY: help setup test verify dataset dataset-verify eval-smoke download clean-pyc

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

setup:            ## install deps, create runtime dirs, verify the install
	./setup.sh

test:             ## run the unit tests (offline, no API key needed)
	$(PY) -m pytest

verify:           ## compile + import every module, check each CLI, load the dataset
	$(PY) -m compileall -q blindspot scripts
	$(PY) scripts/verify_install.py
	$(PY) -m pytest

dataset:          ## regenerate data/svg_localization from scratch (deterministic)
	$(PY) scripts/generate/gen_svg_localization.py --count $(COUNT) --complexity 4 --seed 17
	$(PY) scripts/generate/gen_svg_derived.py

dataset-verify:   ## render a visual audit of the dataset's ground truth
	$(PY) scripts/generate/verify_svg_localization.py --open

eval-smoke:       ## 20 localization questions against Haiku 4.5 (~$0.10; override with SPEND=)
	$(PY) -m blindspot.core.runner --datasets svg_localization --limit 20 --max-spend $(SPEND)

download:         ## fetch the scraped benchmarks into data/ (large, slow)
	$(PY) scripts/download/download_datasets.py

clean-pyc:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
