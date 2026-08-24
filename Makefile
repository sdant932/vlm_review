# Thin wrappers over the commands in docs/PIPELINE.md. Nothing here is magic --
# every target is one line you could type yourself.

PY ?= python
# Trailing comments would land inside the value (make keeps the whitespace
# before the `#`), so these sit on their own lines.
# SPEND: USD hard stop for the smoke-test eval.
SPEND ?= 0.10
# COUNT: scenes for a full dataset regeneration.
COUNT ?= 200

.PHONY: help setup test verify dataset dataset-verify eval-smoke download clean-pyc

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

setup:            ## install deps, create runtime dirs, verify the install
	./setup.sh

test:             ## run the unit tests (offline, no API key needed)
	$(PY) -m pytest

verify:           ## compile + import every module, check each CLI, load the dataset
	$(PY) -m compileall -q blindspot
	$(PY) -m blindspot.tools verify-install
	$(PY) -m pytest

# OUT is mandatory and has no default. The generator has drifted from the
# committed data/svg_localization, and results/*.jsonl is keyed by uid, so
# regenerating in place would bind existing answers to different questions.
# blindspot.pipelines refuses that; this target must not be the way around it.
dataset:          ## build a NEW dataset: make dataset OUT=/tmp/svgloc_new
ifndef OUT
	@echo "OUT is required: make dataset OUT=/tmp/svgloc_new"
	@echo "Refusing to default to data/svg_localization -- see docs/runme/SYNTHETIC.md section 0."
	@exit 2
endif
	$(PY) -m blindspot.generate scenes --count $(COUNT) --complexity 4 --seed 17 --out $(OUT)
	$(PY) -m blindspot.generate questions --data $(OUT)

dataset-verify:   ## render a visual audit of the dataset's ground truth
	$(PY) -m blindspot.generate audit --open

eval-smoke:       ## 20 localization questions against Haiku 4.5 (~$0.10; override with SPEND=)
	$(PY) -m blindspot.core --datasets svg_localization --limit 20 --max-spend $(SPEND)

download:         ## fetch the scraped benchmarks into data/ (large, slow)
	$(PY) -m blindspot.download hf

clean-pyc:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
