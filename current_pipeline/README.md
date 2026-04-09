# Full Pipeline

This folder is the "one place to run everything" view of the cleaned project.

It combines the end-to-end stance workflow:

1. Download or reuse debate graph data
2. Build stance-training pairs
3. Load base stance/NLI models
4. Train or load the meta stance detector
5. Predict graph edges and write `nodes.tsv` / `edges.tsv`
6. Run graph algorithms
7. Generate graph visualizations

## Entry point

Run from `clean_project_v2/`:

```bash
python current_pipeline/run_pipeline.py
```

Optional flags:

```bash
python current_pipeline/run_pipeline.py --force-retrain-meta
python current_pipeline/run_pipeline.py --max-pairs 2000 --max-edges 2000 --max-examples 1000
python current_pipeline/run_pipeline.py --no-generic-visualization
python current_pipeline/run_pipeline.py --skip-training
python current_pipeline/run_pipeline.py --skip-visuals
python current_pipeline/run_pipeline.py --small
```

## Speed modes

- `--skip-training`
  - reuses `meta_lr.joblib` and skips the feature-building and meta-training pass
  - biggest savings when training features would otherwise be recomputed
- `--skip-visuals`
  - skips all visualization folders and only keeps graph export plus algorithm outputs
- `--small`
  - caps the run to a much smaller sample for quick debugging
  - currently limits debates, pairs, edges, and examples to a small preset

## Outputs

Everything is written under:

`outputs/full_pipeline/`

Important subfolders:

- `syncialo_snapshot/`
- `models/`
- `graphs/`
- `analysis/`
- `viz_basic/`
- `viz_generic/`
- `viz_algo/`

The run summary is stored at:

`outputs/full_pipeline/run_summary.json`
