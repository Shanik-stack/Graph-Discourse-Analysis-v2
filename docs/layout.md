# Project layout

The runners live at the repository root. Their implementation is in [`src/debate_graph/`](../src/debate_graph/).

```text
run_pipeline.py      graph export and analysis entry point
benchmark_models.py  model comparison and ablation entry point
src/debate_graph/    importable Python package
docs/                guides and debate index
research/            reports, presentation sources, and assets
data/                input datasets
models/              local trained models and feature caches (ignored)
outputs/             regenerated results (mostly ignored)
examples/            curated, portable visualizations for GitHub
legacy/              historical experiments pending complete parity review
```

Start with the [main README](../README.md), then the [pipeline guide](pipeline.md). See the [legacy audit](../legacy/README.md) before removing historical code.
