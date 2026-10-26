# Pipeline guide

The root-level commands run the end-to-end stance and argument-graph workflow.

It combines the end-to-end stance workflow:

1. Download or reuse debate graph data
2. Build stance-training pairs
3. Load base stance/NLI models
4. Train or load the selected graph export model
5. Predict graph edges and write `nodes.tsv` / `edges.tsv`
6. Run graph algorithms
7. Generate graph visualizations

## Entry point

Run from the repository root:

```bash
python run_pipeline.py
```

Optional flags:

```bash
python run_pipeline.py --model meta
python run_pipeline.py --model graph-aware
python run_pipeline.py --force-retrain-meta
python run_pipeline.py --max-pairs 2000 --max-edges 2000 --max-examples 1000
python run_pipeline.py --no-generic-visualization
python run_pipeline.py --skip-training
python run_pipeline.py --skip-visuals
python run_pipeline.py --small
python run_pipeline.py --use-existing-graphs
python run_pipeline.py --analysis-only
python run_pipeline.py --debates node_link_data-debate-eval-0001,node_link_data-debate-eval-0004
python run_pipeline.py --debate-numbers 1,4
python run_pipeline.py --debate-range 1 4
```

Useful combinations:

```bash
python run_pipeline.py --use-existing-graphs --skip-training
python run_pipeline.py --analysis-only --skip-training
python run_pipeline.py --use-existing-graphs --skip-training --debates node_link_data-debate-eval-0007
python run_pipeline.py --analysis-only --skip-training --debate-numbers 7
python run_pipeline.py --model graph-aware --small
```

## Benchmark entry point

To benchmark explicit train/test debate splits and compare models:

```bash
python benchmark_models.py --model compare --small
python benchmark_models.py --dataset createdebate --model compare --small
python benchmark_models.py --model context-aware --small
python benchmark_models.py --dataset createdebate --model context-aware --small
python benchmark_models.py --model ablation --small
python benchmark_models.py --model context-ablation --small
python benchmark_models.py --model context-feature-ablation --small
python benchmark_models.py --model context-feature-only --small
python benchmark_models.py --model context-all-ablations --small
python benchmark_models.py --model graph-feature-ablation --small
python benchmark_models.py --model graph-feature-only --small
python benchmark_models.py --model ablation --small --test-only
python benchmark_models.py --model graph-feature-ablation --small --test-only
python benchmark_models.py --model graph-feature-only --small --test-only
```

Model choices:

- `deberta`
- `stance`
- `meta`
- `graph-aware`
- `context-aware`
- `compare`
- `ablation`
- `graph-feature-ablation`
- `graph-feature-only`
- `context-ablation`
- `context-feature-ablation`
- `context-feature-only`
- `context-all-ablations`

`ablation` runs a feature-group study over:

- `nli`
- `stance`
- `graph`
- `nli+stance`
- `nli+graph`
- `stance+graph`
- `nli+stance+graph`

It writes:

- a JSON summary under `outputs/benchmark_reports/`
- a TSV ranking table under `outputs/benchmark_reports/`

`graph-feature-ablation` runs a separate leave-one-feature-out study inside the 7 graph features. It:

- trains the full graph-aware reference model
- then trains 7 separate models, each dropping exactly one graph feature:
  - `cosine_similarity`
  - `normalized_source_degree`
  - `normalized_target_degree`
  - `target_in_source`
  - `source_in_target`
  - `neighbor_jaccard_overlap`
  - `token_jaccard_overlap`

It stores results separately from the earlier feature-group ablation:

- `*_graph_feature_ablation_summary.json`
- `*_graph_feature_ablation_table.tsv`

`graph-feature-only` runs the exact one-feature-at-a-time experiment. It:

- trains 7 separate models
- keeps the standard 6 textual features in every run
- adds exactly one graph feature in each run
- therefore tests `nli + stance + 1 graph feature`

The 7 one-feature runs are:

- `cosine_similarity`
- `normalized_source_degree`
- `normalized_target_degree`
- `target_in_source`
- `source_in_target`
- `neighbor_jaccard_overlap`
- `token_jaccard_overlap`

It stores results separately from both earlier ablations:

- `*_graph_feature_only_summary.json`
- `*_graph_feature_only_table.tsv`

`context-aware` is a new model separate from the existing `graph-aware` model. It keeps:

- the 6 textual features from NLI + stance

and adds a separate set of ancestor/tree-context features inspired by GraphNLI, including:

- normalized source depth
- normalized target depth
- normalized depth gap
- normalized ancestor counts
- similarity of each claim to its own ancestor context
- cross-context similarity between a claim and the other claim's ancestor context
- similarity between the two ancestor-context embeddings

This lets you benchmark a discourse-context model without replacing the older graph-aware model.

`context-ablation` runs the feature-group ablation for the new context-aware setup:

- `nli`
- `stance`
- `context`
- `nli+stance`
- `nli+context`
- `stance+context`
- `nli+stance+context`

It stores results separately:

- `*_context_ablation_summary.json`
- `*_context_ablation_table.tsv`

`context-feature-ablation` runs a leave-one-context-feature-out study on the full context-aware model.

The 10 context features are:

- `normalized_source_depth`
- `normalized_target_depth`
- `normalized_depth_gap`
- `normalized_source_ancestor_count`
- `normalized_target_ancestor_count`
- `source_to_own_ancestor_context`
- `target_to_own_ancestor_context`
- `source_to_target_ancestor_context`
- `target_to_source_ancestor_context`
- `ancestor_context_to_ancestor_context`

It stores results separately:

- `*_context_feature_ablation_summary.json`
- `*_context_feature_ablation_table.tsv`

`context-feature-only` runs the one-feature-at-a-time version for the context-aware model:

- keeps the same 6 textual features
- adds exactly one of the 10 context features

It stores results separately:

- `*_context_feature_only_summary.json`
- `*_context_feature_only_table.tsv`

`context-all-ablations` runs all three context-aware ablation families in one command:

- `context-ablation`
- `context-feature-ablation`
- `context-feature-only`

Useful benchmark flags:

- `--force-retrain`
  - retrains the selected benchmark classifier(s) even if saved models already exist
- `--test-only`
  - does not train benchmark classifiers
  - reuses already saved benchmark model files and only evaluates them
  - fails clearly if a required saved model is missing
- `--dataset syncialo|createdebate`
  - chooses whether benchmarking runs on the existing Syncialo debate graphs or the CreateDebate parent-child argument dataset
  - for `createdebate`, parent-child pairs are labeled as `support` when the child and parent share the same side and `attack` otherwise; neutral examples are still sampled within each debate

## Speed modes

- `--skip-training`
  - reuses the saved trained model for the selected export mode
  - for `--model meta`, that is the cached logistic regression meta model
  - for `--model graph-aware`, that is the cached graph-aware classifier
- `--skip-visuals`
  - skips all visualization folders and only keeps graph export plus algorithm outputs
- `--small`
  - caps the run to a much smaller sample for quick debugging
  - currently limits debates, pairs, edges, and examples to a small preset
- `--use-existing-graphs`
  - skips graph export entirely and reuses the existing `nodes.tsv` and `edges.tsv`
  - useful when you only want graph algorithms or visualizations
- `--analysis-only`
  - alias for `--use-existing-graphs`
  - useful when you want a clearer "just analyze existing graph files" command

## Debate selection

You can restrict both export and downstream analysis to:

- specific debate ids with `--debates`
- use full JSON file stems such as `node_link_data-debate-eval-0002`
- stable debate numbers with `--debate-numbers`
- an inclusive stable debate number range with `--debate-range START END`

The stable debate numbering is documented in:

- `docs/debate-index.md`

When `--use-existing-graphs` is enabled, these filters are applied to the already exported TSV files before analysis and visualization.

## Outputs

Everything is written under:

`outputs/pipeline/`

Important subfolders:

- `data/syncialo_snapshot/`
- `models/pipeline/`
- `outputs/pipeline/graphs/`
- `outputs/pipeline/analysis/`
- `outputs/pipeline/viz_basic/`
- `outputs/pipeline/viz_generic/`
- `outputs/pipeline/viz_algo/`

The run summary is stored at:

`outputs/pipeline/run_summary.json`

Benchmark reports are written under:

`outputs/benchmark_reports/`
