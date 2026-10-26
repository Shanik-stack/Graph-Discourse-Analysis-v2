from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from debate_graph.pipeline import FullPipelineConfig, run_full_pipeline
from debate_graph.paths import DATA_DIR


def _sorted_debate_files(corpus_id: str = "synthetic_corpus-001", split: str = "eval") -> list[Path]:
    local_dir = DATA_DIR / "syncialo_snapshot" / "data" / corpus_id / split
    files = local_dir.rglob("*.json")
    return sorted(files, key=lambda p: (p.parent.name, p.name))


def _parse_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(int(chunk))
    return out


def _resolve_selected_debates(
    *,
    debate_names: str,
    debate_numbers: str,
    debate_range: list[int] | None,
    corpus_id: str = "synthetic_corpus-001",
    split: str = "eval",
) -> list[str] | None:
    selected: list[str] = []

    if debate_names:
        for item in debate_names.split(","):
            item = item.strip()
            if item and item not in selected:
                selected.append(item)

    numbered_requests: list[int] = []
    if debate_numbers:
        numbered_requests.extend(_parse_int_csv(debate_numbers))
    if debate_range:
        start, end = debate_range
        lo, hi = sorted((start, end))
        numbered_requests.extend(range(lo, hi + 1))

    if numbered_requests:
        files = _sorted_debate_files(corpus_id=corpus_id, split=split)
        for idx in numbered_requests:
            if idx < 1 or idx > len(files):
                raise ValueError(f"Debate number {idx} is out of range. Valid debate numbers are 1 to {len(files)}.")
            debate_id = files[idx - 1].stem
            if debate_id not in selected:
                selected.append(debate_id)

    return selected or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full stance-to-graph pipeline.")
    parser.add_argument("--model", choices=["meta", "graph-aware"], default="meta", help="Model used for graph export.")
    parser.add_argument("--force-retrain-meta", action="store_true", help="Retrain the selected graph export model even if a saved one exists.")
    parser.add_argument("--max-pairs", type=int, default=5000, help="Maximum pair count for meta training data.")
    parser.add_argument("--max-edges", type=int, default=5000, help="Maximum edge rows used for graph export.")
    parser.add_argument("--max-examples", type=int, default=2000, help="Maximum examples evaluated when writing graph outputs.")
    parser.add_argument("--max-debates", type=int, default=30, help="Maximum debates sampled from the dataset snapshot.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--skip-training", action="store_true", help="Reuse a saved model and skip feature building and model training.")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip all visualization outputs and only write graph/analysis artifacts.")
    parser.add_argument("--small", action="store_true", help="Run a smaller quick-check pipeline with reduced debates, pairs, edges, and examples.")
    parser.add_argument("--use-existing-graphs", action="store_true", help="Skip graph export and reuse existing nodes.tsv and edges.tsv for analysis/visualization.")
    parser.add_argument("--analysis-only", action="store_true", help="Alias for --use-existing-graphs. Reuse existing graph TSVs and only run downstream analysis/visualization.")
    parser.add_argument("--debates", type=str, default="", help="Comma-separated debate ids to analyze or export.")
    parser.add_argument("--debate-numbers", type=str, default="", help="Comma-separated stable debate numbers from SYNCIALO_DEBATE_INDEX.md.")
    parser.add_argument("--debate-range", nargs=2, type=int, metavar=("START", "END"), help="Inclusive stable debate number range from SYNCIALO_DEBATE_INDEX.md.")
    parser.add_argument("--no-generic-visualization", action="store_true", help="Skip the generic HTML/PNG/GEXF visualization pass.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_existing_graphs = args.use_existing_graphs or args.analysis_only
    selected_debate_ids = _resolve_selected_debates(
        debate_names=args.debates,
        debate_numbers=args.debate_numbers,
        debate_range=args.debate_range,
    )

    cfg = FullPipelineConfig(
        model_choice=args.model,
        force_retrain_meta=args.force_retrain_meta,
        max_pairs=args.max_pairs,
        max_edges=args.max_edges,
        max_examples=args.max_examples,
        max_debates=args.max_debates,
        seed=args.seed,
        skip_training=args.skip_training,
        skip_visuals=args.skip_visuals,
        small_mode=args.small,
        use_existing_graphs=use_existing_graphs,
        selected_debate_ids=selected_debate_ids,
        debate_index_start=None,
        debate_index_end=None,
        run_generic_visualization=(not args.no_generic_visualization) and (not args.skip_visuals),
    )
    summary = run_full_pipeline(cfg)
    print("Full pipeline completed.")
    print(f"Summary: {cfg.summary_path}")
    print(f"Working debate: {summary['graph_algorithms'].get('debate_id', 'unknown')}")

if __name__ == "__main__":
    main()
