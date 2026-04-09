# Full Pipeline Walkthrough

This is the single-file explanation of the end-to-end workflow used in the cleaned project.

Primary runner:

- [run_pipeline.py](C:\All Codes\BTP\clean_project_v2\current_pipeline\run_pipeline.py)

Core orchestration code:

- [pipeline.py](C:\All Codes\BTP\clean_project_v2\src\btp_clean\full_pipeline\pipeline.py)

## Big picture

The pipeline goes through these stages:

1. Download or reuse debate graph data
2. Convert debates into stance-learning examples
3. Load the base NLI and stance models
4. Build cached features for a meta-classifier
5. Train or load the meta stance detector
6. Predict support/attack/neutral edges for debate graphs
7. Save graph files (`nodes.tsv`, `edges.tsv`)
8. Run graph algorithms on the predicted graph
9. Generate graph visualizations
10. Save a final run summary

## Stage-by-stage flow

### 1. Configuration

The run starts from `FullPipelineConfig` in
[pipeline.py](C:\All Codes\BTP\clean_project_v2\src\btp_clean\full_pipeline\pipeline.py).

This config controls:

- which dataset split is used
- how many debates/pairs/edges are sampled
- whether the meta model should be retrained
- which Hugging Face models are loaded
- where outputs are written

Important output root:

- `outputs/full_pipeline/`

Downloaded source snapshot:

- `data/syncialo_snapshot/`

## 2. Dataset snapshot

Function:

- `snapshot_download_syncialo(...)`

Source file:

- [mata_benchmark_syncialo.py](C:\All Codes\BTP\clean_project_v2\src\btp_clean\stance_pipeline\mata_benchmark_syncialo.py)

What happens:

- The Syncialo debate dataset snapshot is downloaded or reused.
- Debate JSON files are stored under:
  - `data/syncialo_snapshot/`

## 3. Build training pairs

Functions:

- `load_debate_json(...)`
- `parse_nodes_and_links(...)`
- `build_dataset(...)`

What happens:

- Each debate JSON is parsed into:
  - claim nodes
  - labeled edges
- Gold edges are mapped into:
  - `support`
  - `attack`
- Non-edges are sampled as:
  - `neutral`

The output of this stage is a list of training examples like:

- `(debate_id, premise, hypothesis, gold_label)`

## 4. Load base models

Classes:

- `HFNLIModel`
- `HFStanceModel`

Source file:

- [meta_nli.py](C:\All Codes\BTP\clean_project_v2\src\btp_clean\stance_pipeline\meta_nli.py)

What happens:

- The NLI model produces logits for:
  - contradiction
  - neutral
  - entailment
- The stance model produces logits for:
  - against
  - none
  - favor

These are the base learned signals used by the final classifier.

## 5. Build cached features

Function:

- `featurize_all(...)`

What happens:

- For each training pair:
  - NLI logits are computed
  - stance logits are computed
- They are concatenated into one 6-dimensional feature vector

Feature layout:

- `[nli_contradiction, nli_neutral, nli_entailment, stance_against, stance_none, stance_favor]`

Saved artifact:

- `outputs/full_pipeline/models/meta_features.npz`

## 6. Train or load the meta stance detector

Function:

- `train_eval_on_cached_logits(...)`

What happens:

- A logistic regression classifier is trained on the cached features.
- It predicts:
  - `support`
  - `attack`
  - `neutral`

Saved artifact:

- `outputs/full_pipeline/models/meta_lr.joblib`

This is the main stance detector used for graph-edge prediction in the full pipeline.

## 7. Rebuild graph edge rows

Function:

- `build_debate_edge_dataset(...)`

What happens:

- The debates are converted back into graph-oriented rows with IDs:
  - `debate_id`
  - `src_id`
  - `tgt_id`
  - `premise`
  - `hypothesis`
  - `gold_label`

This stage keeps the node IDs so the predictions can be written back into graph form.

## 8. Predict graph edges and export graph files

Function:

- `run_meta_with_graph_logs(...)`

What happens:

- The base models compute logits for each edge candidate.
- The trained meta model predicts the final label.
- Graph files are written:
  - `nodes.tsv`
  - `edges.tsv`

Saved artifacts:

- [nodes.tsv](C:\All Codes\BTP\clean_project_v2\outputs\full_pipeline\graphs\nodes.tsv)
- [edges.tsv](C:\All Codes\BTP\clean_project_v2\outputs\full_pipeline\graphs\edges.tsv)

These two files are the bridge between stance detection and graph analysis.

## 9. Basic graph visualization

Functions:

- `build_graph(...)`
- `visualize(...)`

Source file:

- [graph_builder.py](C:\All Codes\BTP\clean_project_v2\src\btp_clean\stance_pipeline\graph_builder.py)

What happens:

- The graph is filtered by confidence and label type.
- Separate HTML files are generated for:
  - combined graph
  - attack-only graph
  - support-only graph

Output folder:

- `outputs/full_pipeline/viz_basic/`

## 10. Generic graph export

Function:

- `run_visualization(...)`

Source file:

- [visualize.py](C:\All Codes\BTP\clean_project_v2\src\btp_clean\stance_pipeline\visualize.py)

What happens:

- Builds one graph per debate
- Exports:
  - HTML
  - PNG
  - GEXF
  - summary CSV

Output folder:

- `outputs/full_pipeline/viz_generic/`

## 11. Run graph algorithms

Functions:

- `compute_support_indegree(...)`
- `compute_pagerank_support(...)`
- `mincut_vulnerability(...)`
- `cluster_support_graph(...)`
- `cluster_themes(...)`

Source file:

- [graph_algorithms.py](C:\All Codes\BTP\clean_project_v2\src\btp_clean\stance_pipeline\graph_algorithms.py)

What happens:

- The predicted graph is converted into a support subgraph
- The following are computed:
  - local support strength via in-degree
  - global influence via PageRank
  - bottlenecks via min-cut/articulation points
  - communities via support-graph clustering
  - cluster keywords via TF-IDF

Output folder:

- `outputs/full_pipeline/analysis/`

## 12. Algorithm-focused visualization

Functions:

- `viz_support_indegree(...)`
- `viz_pagerank_support_and_attack(...)`
- `viz_mincut_all_clusters_one_html(...)`
- `viz_clusters(...)`

Source file:

- [algo_visualizer.py](C:\All Codes\BTP\clean_project_v2\src\btp_clean\stance_pipeline\algo_visualizer.py)

What happens:

- Produces interactive HTML outputs that show algorithm results directly on the graph:
  - support in-degree
  - support PageRank
  - attack PageRank
  - net influence
  - min-cut vulnerability by cluster
  - cluster visualization with themes

Output folder:

- `outputs/full_pipeline/viz_algo/`

## 13. Final run summary

Function:

- `run_full_pipeline(...)`

What happens:

- Collects the output paths and summary stats from each stage
- Saves one final summary JSON

Saved file:

- [run_summary.json](C:\All Codes\BTP\clean_project_v2\outputs\full_pipeline\run_summary.json)

## Call order inside the orchestrator

At a high level, `run_full_pipeline(...)` does this:

1. `_ensure_dirs(cfg)`
2. `_run_training_and_graph_export(cfg)`
3. `_load_graph_frames(cfg)`
4. `_run_basic_graph_visualizations(cfg, nodes_df, edges_df)`
5. `_run_generic_visualization(cfg)` if enabled
6. `_run_graph_algorithms(cfg, nodes_df, edges_df, debate_id)`
7. `_run_algorithm_visualizations(cfg, nodes_df, edges_df, debate_id)`
8. save `run_summary.json`

## Most important idea

The real backbone of the full system is:

`debate dataset -> training pairs -> base model logits -> meta stance classifier -> predicted graph edges -> graph algorithms -> graph visualizations`

That is the end-to-end pipeline of this codebase.
