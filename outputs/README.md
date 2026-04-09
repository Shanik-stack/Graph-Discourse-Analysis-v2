# Outputs

This folder is reserved for generated artifacts in the v2 layout.

Suggested organization:

- `full_pipeline/`
  - end-to-end run outputs
- `graphs/`
  - exported `nodes.tsv`, `edges.tsv`, and graph files
- `analysis/`
  - PageRank, support in-degree, clustering, min-cut summaries
- `visualizations/`
  - HTML, PNG, GEXF, and algorithm-oriented graph views

Note:

In v2, `btp_clean.paths` has been remapped so active pipeline outputs now point at `outputs/`.
