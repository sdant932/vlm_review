# Raw results inventory

The raw API responses — one JSONL row per question, holding the prompt, the model's
answer, the thinking token counts, the as-sent image resolution and the score — are
**not** in this repository. They are ~64MB across 42 files.

This file inventories them so a claim can be traced to a specific run, and so a
regenerated set can be checked against the original.

Regenerate with the commands in [PIPELINE.md](PIPELINE.md). The runs are keyed by uid and
resumable, but they are **not** bit-reproducible: `temperature` is unavailable in
anthropic 1.0.0 and thinking pins it to 1, so a re-run will differ. The measured
item-level disagreement rate between two identical runs is 10.1% (2,121 items, 214
disagreed). Checksums below identify the original files; they are not a reproducibility
target.

## Naming

```
<dataset>__<model>_<thinking>_<image>_r<run>.jsonl     main runs
<dataset>__<tag>.jsonl                                 judge / audit / ablation passes
```

So `charxiv__haiku-4-5_think2000_native_r0.jsonl` is CharXiv on Haiku 4.5, thinking at
2,000 tokens, images sent at native resolution, run 0. A `.todo.json` or `.missing.json`
sidecar records what a run had left to do when it stopped.

## Files

| File | Rows | Size | sha256 (first 16) |
|---|---:|---:|---|
| `ai2d__haiku-4-5_think2000_native_r0.jsonl` | 3,088 | 5.9 MB | `a67c6bfe999f30ba` |
| `charxiv__equiv.jsonl` | 474 | 0.2 MB | `071fc2cfcd1fa911` |
| `charxiv__gtaudit.jsonl` | 202 | 0.2 MB | `538ed77aee2e8f52` |
| `charxiv__haiku-4-5_think2000_native_r0.jsonl` | 7,121 | 11.1 MB | `2ee838d4a58731f6` |
| `charxiv__haiku-4-5_think2000_native_r0.judged.jsonl` | 4,288 | 0.8 MB | `144a5cb787c3d580` |
| `control_blind.jsonl` | 2,000 | 4.2 MB | `cbb98bf0ee206cfb` |
| `control_grid4.jsonl` | 350 | 0.9 MB | `3cdd3210b0f39555` |
| `control_onepage0.jsonl` | 567 | 1.3 MB | `b8914c40515658d9` |
| `infographicvqa__equiv.jsonl` | 388 | 0.2 MB | `ac1dc47e5035a04c` |
| `infographicvqa__gtaudit.jsonl` | 413 | 0.4 MB | `1b0472770d202f0d` |
| `infographicvqa__haiku-4-5_think2000_native_r0.jsonl` | 2,803 | 4.3 MB | `e10cb72750b58e52` |
| `screenspot__haiku-4-5_official_r0.jsonl` | 200 | 0.1 MB | `3943cd8702d2a37d` |
| `screenspot__haiku-4-5_think2000_native_r0.jsonl` | 200 | 0.4 MB | `9996eb93e2d39808` |
| `screenspot__sonnet-5_think2000_edge1568_r0.jsonl` | 5 | 0.0 MB | `f29dace9594473ff` |
| `screenspot__sonnet-5_think2000_native_r0.jsonl` | 5 | 0.0 MB | `9420411a0c2c6a3e` |
| `screenspot__think2000_edge1568_r0.jsonl` | 200 | 0.4 MB | `78d0de88af12a3bd` |
| `screenspot_pro__gtaudit.jsonl` | 36 | 0.0 MB | `46881e8a0be61ed3` |
| `screenspot_pro__gtaudit.labelled.jsonl` | 171 | 0.2 MB | `54caabe589461a18` |
| `screenspot_pro__haiku-4-5_official_r0.jsonl` | 1,581 | 1.1 MB | `f59c1dcf76c4ba67` |
| `screenspot_pro__haiku-4-5_think2000_native_r0.jsonl` | 1,756 | 4.7 MB | `fe2e66f150b36557` |
| `screenspot_pro__sonnet-5_think2000_edge1568_r0.jsonl` | 5 | 0.0 MB | `59cb5ab8e61e9625` |
| `screenspot_pro__sonnet-5_think2000_native_r0.jsonl` | 5 | 0.0 MB | `694ae39f7ca6d860` |
| `screenspot_pro__think2000_edge1568_r0.jsonl` | 200 | 0.5 MB | `3df0aba4a5fcd377` |
| `screenspot_pro__tiled3x3_r0.jsonl` | 50 | 0.0 MB | `dbfc9d0fbec506dd` |
| `slidevqa__haiku-4-5_think2000_native_r0.jsonl` | 1,003 | 1.5 MB | `ada2709d69f9e2b4` |
| `slidevqa_allpages__haiku-4-5_think2000_native_r0.jsonl` | 494 | 0.9 MB | `f153c3361d16ca38` |
| `svg_counting__blind_haiku-4-5_think2000_native_r0.jsonl` | 476 | 1.0 MB | `bf7a73af9e8b40d1` |
| `svg_counting__haiku-4-5_think2000_native_r0.jsonl` | 476 | 0.7 MB | `246713990ce2cbca` |
| `svg_localization__haiku-4-5_think2000_native_r0.jsonl` | 4,723 | 11.6 MB | `83d2a3f7b8c7e797` |
| `svg_localization__probe_haiku-4-5_think2000_native_r0.jsonl` | 108 | 0.3 MB | `142b998120a3596a` |
| `svg_localization__probe_sonnet-5_think2000_edge1568_r0.jsonl` | 108 | 0.1 MB | `7f35aa8dc1e41714` |
| `svg_localization__probe_sonnet-5_think2000_native_r0.jsonl` | 108 | 0.1 MB | `e74980184c19c30c` |
| `svg_word_mc__blind_haiku-4-5_think2000_native_r0.jsonl` | 736 | 1.7 MB | `ad4ada156fc27d19` |
| `svg_word_mc__haiku-4-5_think2000_native_r0.jsonl` | 736 | 1.1 MB | `311a72d9a602e36d` |
| `svgloc_abl_bbox__haiku-4-5_think2000.jsonl` | 300 | 0.9 MB | `72ef759a44e13916` |
| `svgloc_abl_careful__haiku-4-5_think2000.jsonl` | 300 | 1.0 MB | `b5a86065f6f52f8b` |
| `svgloc_abl_cell_then_point__haiku-4-5_think2000.jsonl` | 300 | 1.1 MB | `d7452641c886dc1d` |
| `svgloc_abl_crop__haiku-4-5_think2000.jsonl` | 300 | 0.8 MB | `9acffe7cf4bd416e` |
| `svgloc_abl_describe__haiku-4-5_think2000.jsonl` | 300 | 1.0 MB | `6dd06b17e14de030` |
| `svgloc_abl_landmark__haiku-4-5_think2000.jsonl` | 300 | 1.1 MB | `fbbe8575f5e2857e` |
| `svgloc_abl_quadrant_mc__haiku-4-5_think2000.jsonl` | 300 | 0.7 MB | `af4434019c1f40db` |
| `svgloc_abl_repeat__haiku-4-5_think2000.jsonl` | 300 | 0.8 MB | `28a7f1fef9c8f0d3` |

**42 files, 37,476 rows, 64 MB total.**
