from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from btp_clean.full_pipeline import FullPipelineConfig, run_full_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full stance-to-graph pipeline.")
    parser.add_argument("--force-retrain-meta", action="store_true", help="Retrain the meta classifier even if a saved one exists.")
    parser.add_argument("--max-pairs", type=int, default=5000, help="Maximum pair count for meta training data.")
    parser.add_argument("--max-edges", type=int, default=5000, help="Maximum edge rows used for graph export.")
    parser.add_argument("--max-examples", type=int, default=2000, help="Maximum examples evaluated when writing graph outputs.")
    parser.add_argument("--max-debates", type=int, default=30, help="Maximum debates sampled from the dataset snapshot.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--skip-training", action="store_true", help="Reuse a saved meta model and skip feature building and meta training.")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip all visualization outputs and only write graph/analysis artifacts.")
    parser.add_argument("--small", action="store_true", help="Run a smaller quick-check pipeline with reduced debates, pairs, edges, and examples.")
    parser.add_argument("--no-generic-visualization", action="store_true", help="Skip the generic HTML/PNG/GEXF visualization pass.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FullPipelineConfig(
        force_retrain_meta=args.force_retrain_meta,
        max_pairs=args.max_pairs,
        max_edges=args.max_edges,
        max_examples=args.max_examples,
        max_debates=args.max_debates,
        seed=args.seed,
        skip_training=args.skip_training,
        skip_visuals=args.skip_visuals,
        small_mode=args.small,
        run_generic_visualization=(not args.no_generic_visualization) and (not args.skip_visuals),
    )
    summary = run_full_pipeline(cfg)
    print("Full pipeline completed.")
    print(f"Summary: {cfg.summary_path}")
    print(f"Working debate: {summary['graph_algorithms'].get('debate_id', 'unknown')}")


if __name__ == "__main__":
    main()
