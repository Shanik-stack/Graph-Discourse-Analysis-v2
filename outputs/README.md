# Generated outputs

`run_pipeline.py` writes into `outputs/pipeline/`:

| Path | Contents |
| --- | --- |
| `graphs/nodes.tsv`, `graphs/edges.tsv` | Graph input and predicted labels; included to reproduce the README demo |
| `analysis/` | Support in-degree, PageRank, minimum cut, clusters, and themes |
| `viz_basic/` | Combined, attack, and support HTML graphs |
| `viz_algo/` | Algorithm-focused interactive HTML graphs |
| `viz_generic/` | Generic HTML, PNG, GEXF, and summary CSV |
| `run_summary.json` | Configuration and output paths for the latest run |

Benchmark runs write reports to `outputs/benchmark_reports/`. Regenerated files are ignored by Git except the two graph TSVs used by the README demo. Portable examples live in [`examples/debate-0002/`](../examples/debate-0002/).
