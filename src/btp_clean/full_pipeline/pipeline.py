from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict
import joblib

import pandas as pd

from btp_clean.paths import ARTIFACTS_DIR, DATA_DIR
from btp_clean.stance_pipeline import algo_visualizer as av
from btp_clean.stance_pipeline import graph_algorithms as ga
from btp_clean.stance_pipeline import graph_builder as gb
from btp_clean.stance_pipeline import mata_benchmark_syncialo as meta_bench
from btp_clean.stance_pipeline import visualize as generic_viz
from btp_clean.stance_pipeline.meta_nli import HFNLIModel, HFStanceModel


@dataclass
class FullPipelineConfig:
    corpus_id: str = "synthetic_corpus-001"
    split: str = "eval"
    neutrals_per_pos: float = 1.0
    max_pairs: int = 5000
    max_edges: int = 5000
    max_debates: int = 30
    seed: int = 0
    test_size: float = 0.2
    max_examples: int = 2000
    force_retrain_meta: bool = False
    skip_training: bool = False
    skip_visuals: bool = False
    small_mode: bool = False
    reuse_graph_features: bool = True

    nli_model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    stance_model_name: str = "krishnagarg09/stance-detection-semeval2016"
    nli_max_length: int = 256
    stance_max_length: int = 128
    stance_device: str = "cpu"

    graph_min_score: float = 0.55
    generic_viz_min_score: float = 0.60
    generic_viz_top_k_out: int = 5
    run_generic_visualization: bool = True

    @property
    def root_dir(self) -> Path:
        return ARTIFACTS_DIR / "full_pipeline"

    @property
    def snapshot_dir(self) -> Path:
        return DATA_DIR / "syncialo_snapshot"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models"

    @property
    def graphs_dir(self) -> Path:
        return self.root_dir / "graphs"

    @property
    def analysis_dir(self) -> Path:
        return self.root_dir / "analysis"

    @property
    def basic_viz_dir(self) -> Path:
        return self.root_dir / "viz_basic"

    @property
    def generic_viz_dir(self) -> Path:
        return self.root_dir / "viz_generic"

    @property
    def algo_viz_dir(self) -> Path:
        return self.root_dir / "viz_algo"

    @property
    def summary_path(self) -> Path:
        return self.root_dir / "run_summary.json"


def _apply_small_mode(cfg: FullPipelineConfig) -> FullPipelineConfig:
    if not cfg.small_mode:
        return cfg

    cfg.max_pairs = min(cfg.max_pairs, 300)
    cfg.max_edges = min(cfg.max_edges, 300)
    cfg.max_debates = min(cfg.max_debates, 3)
    cfg.max_examples = min(cfg.max_examples, 200)
    return cfg


def _ensure_dirs(cfg: FullPipelineConfig) -> None:
    for path in [
        cfg.root_dir,
        cfg.snapshot_dir,
        cfg.models_dir,
        cfg.graphs_dir,
        cfg.analysis_dir,
        cfg.basic_viz_dir,
        cfg.generic_viz_dir,
        cfg.algo_viz_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def _load_graph_frames(cfg: FullPipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_path = cfg.graphs_dir / "nodes.tsv"
    edges_path = cfg.graphs_dir / "edges.tsv"

    nodes = pd.read_csv(nodes_path, sep="\t")
    edges = pd.read_csv(edges_path, sep="\t")

    nodes["debate_id"] = nodes["debate_id"].astype(str)
    nodes["claim_id"] = nodes["claim_id"].astype(str)
    nodes["text"] = nodes["text"].astype(str)

    edges["debate_id"] = edges["debate_id"].astype(str)
    edges["src_id"] = edges["src_id"].astype(str)
    edges["tgt_id"] = edges["tgt_id"].astype(str)
    edges["pred_label"] = edges["pred_label"].astype(str)
    edges["pred_score"] = pd.to_numeric(edges["pred_score"], errors="coerce").fillna(0.0)

    return nodes, edges


def _run_training_and_graph_export(cfg: FullPipelineConfig) -> Dict[str, Any]:
    local_dir = meta_bench.snapshot_download_syncialo(
        corpus_id=cfg.corpus_id,
        split=cfg.split,
        local_dir=str(cfg.snapshot_dir),
    )

    pairs = meta_bench.build_dataset(
        local_dir=local_dir,
        corpus_id=cfg.corpus_id,
        split=cfg.split,
        neutrals_per_pos=cfg.neutrals_per_pos,
        max_pairs=cfg.max_pairs,
        max_debates=cfg.max_debates,
        seed=cfg.seed,
    )

    nli = HFNLIModel(
        cfg.nli_model_name,
        max_length=cfg.nli_max_length,
    )
    stance = HFStanceModel(
        cfg.stance_model_name,
        max_length=cfg.stance_max_length,
        device=cfg.stance_device,
        force_eager_attention=True,
    )

    features_path = cfg.models_dir / "meta_features.npz"
    lr_path = cfg.models_dir / "meta_lr.joblib"
    graph_features_path = cfg.models_dir / "graph_eval_features.npz"
    graph_features_cache_preexisting = graph_features_path.exists()

    training_summary: Dict[str, Any]
    if cfg.skip_training:
        if not lr_path.exists():
            raise FileNotFoundError(
                f"--skip-training was requested, but no saved meta model was found at {lr_path}"
            )
        payload = joblib.load(lr_path)
        clf = payload["clf"]
        training_summary = {
            "pairs": None,
            "feature_shape": None,
            "group_count": None,
            "meta_model_loaded_or_trained": True,
            "meta_classes": int(len(getattr(clf, "classes_", []))),
            "features_path": str(features_path),
            "model_path": str(lr_path),
            "training_skipped": True,
            "features_recomputed": False,
        }
    else:
        X, y, groups = meta_bench.featurize_all(
            pairs,
            nli=nli,
            stance=stance,
            cache_path=str(features_path),
            verbose_every=200,
        )

        clf = meta_bench.train_eval_on_cached_logits(
            X,
            y,
            groups,
            test_size=cfg.test_size,
            seed=cfg.seed,
            lr_path=str(lr_path),
            force_retrain=cfg.force_retrain_meta,
        )
        training_summary = {
            "pairs": len(pairs),
            "feature_shape": list(X.shape),
            "group_count": int(len(set(groups.tolist()))),
            "meta_model_loaded_or_trained": True,
            "meta_classes": int(len(getattr(clf, "classes_", []))),
            "features_path": str(features_path),
            "model_path": str(lr_path),
            "training_skipped": False,
            "features_recomputed": True,
        }

    rows, claims_by_debate = meta_bench.build_debate_edge_dataset(
        local_dir=local_dir,
        corpus_id=cfg.corpus_id,
        split=cfg.split,
        neutrals_per_pos=cfg.neutrals_per_pos,
        max_edges=cfg.max_edges,
        max_debates=cfg.max_debates,
        seed=cfg.seed,
    )

    meta_bench.run_meta_with_graph_logs(
        rows=rows,
        claims_by_debate=claims_by_debate,
        nli=nli,
        stance=stance,
        lr_path=str(lr_path),
        out_dir=str(cfg.graphs_dir),
        max_examples=cfg.max_examples,
        graph_features_path=(str(graph_features_path) if cfg.reuse_graph_features else None),
    )

    training_summary.update({
        "edge_rows": len(rows),
        "graphs_dir": str(cfg.graphs_dir),
        "graph_features_path": str(graph_features_path),
        "graph_features_cache_enabled": bool(cfg.reuse_graph_features),
        "graph_features_cache_preexisting": bool(graph_features_cache_preexisting),
    })
    return training_summary


def _run_basic_graph_visualizations(cfg: FullPipelineConfig, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Dict[str, Any]:
    debate_id = gb.pick_single_debate(edges_df)

    combined = gb.build_graph(nodes_df, edges_df, debate_id, keep_labels={"support", "attack"})
    attack = gb.build_graph(nodes_df, edges_df, debate_id, keep_labels={"attack"})
    support = gb.build_graph(nodes_df, edges_df, debate_id, keep_labels={"support"})

    combined_path = cfg.basic_viz_dir / f"{debate_id}_combined.html"
    attack_path = cfg.basic_viz_dir / f"{debate_id}_attack_only.html"
    support_path = cfg.basic_viz_dir / f"{debate_id}_support_only.html"

    gb.visualize(combined, str(combined_path), "COMBINED (support+attack)")
    gb.visualize(attack, str(attack_path), "ATTACK ONLY")
    gb.visualize(support, str(support_path), "SUPPORT ONLY")

    return {
        "debate_id": debate_id,
        "combined_html": str(combined_path),
        "attack_html": str(attack_path),
        "support_html": str(support_path),
        "nodes": int(combined.number_of_nodes()),
        "edges": int(combined.number_of_edges()),
    }


def _run_generic_visualization(cfg: FullPipelineConfig) -> Dict[str, Any]:
    nodes_path = cfg.graphs_dir / "nodes.tsv"
    edges_path = cfg.graphs_dir / "edges.tsv"

    generic_viz.run_visualization(
        str(nodes_path),
        str(edges_path),
        out_dir=str(cfg.generic_viz_dir),
        min_score=cfg.generic_viz_min_score,
        drop_neutral=True,
        keep_top_k_out=cfg.generic_viz_top_k_out,
    )

    return {
        "out_dir": str(cfg.generic_viz_dir),
        "summary_csv": str(cfg.generic_viz_dir / "summary.csv"),
    }


def _run_graph_algorithms(cfg: FullPipelineConfig, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, debate_id: str) -> Dict[str, Any]:
    filtered_edges = edges_df[edges_df["pred_score"] >= float(cfg.graph_min_score)]
    filtered_edges = filtered_edges[filtered_edges["pred_label"].isin(["support", "attack"])]

    ndf, edf = ga.restrict_to_debate(nodes_df, filtered_edges, debate_id)
    G = ga.build_directed_graph(ndf, edf)
    S = ga.support_subgraph(G)

    support_indegree = ga.compute_support_indegree(S)
    pagerank_support = ga.compute_pagerank_support(S)
    vulnerability = ga.mincut_vulnerability(S)
    clusters = ga.cluster_support_graph(S)
    themes = ga.cluster_themes(S, clusters)

    support_indegree_path = cfg.analysis_dir / "support_indegree.csv"
    pagerank_path = cfg.analysis_dir / "pagerank_support.csv"
    vulnerability_path = cfg.analysis_dir / "vulnerability_mincut.json"
    clusters_path = cfg.analysis_dir / "clusters.json"
    themes_path = cfg.analysis_dir / "cluster_themes.json"

    support_indegree.to_csv(support_indegree_path, index=False)
    pagerank_support.to_csv(pagerank_path, index=False)
    vulnerability_path.write_text(json.dumps({"debate_id": debate_id, **vulnerability}, indent=2), encoding="utf-8")
    clusters_path.write_text(json.dumps({"debate_id": debate_id, "clusters": clusters}, indent=2), encoding="utf-8")
    themes_path.write_text(json.dumps({"debate_id": debate_id, **themes}, indent=2), encoding="utf-8")

    return {
        "debate_id": debate_id,
        "out_dir": str(cfg.analysis_dir),
        "support_indegree_csv": str(support_indegree_path),
        "pagerank_support_csv": str(pagerank_path),
        "vulnerability_json": str(vulnerability_path),
        "clusters_json": str(clusters_path),
        "cluster_themes_json": str(themes_path),
        "support_graph_nodes": int(S.number_of_nodes()),
        "support_graph_edges": int(S.number_of_edges()),
    }


def _run_algorithm_visualizations(cfg: FullPipelineConfig, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, debate_id: str) -> Dict[str, Any]:
    filtered_edges = edges_df[edges_df["pred_score"] >= float(cfg.graph_min_score)]
    filtered_edges = filtered_edges[filtered_edges["pred_label"].isin(["support", "attack"])]

    G = av.build_combined_graph(nodes_df, filtered_edges, debate_id)
    S = av.support_subgraph(G)

    support_indegree_path = cfg.algo_viz_dir / f"{debate_id}_support_indegree.html"
    clusters_path = cfg.algo_viz_dir / f"{debate_id}_clusters.html"

    av.viz_support_indegree(S, str(support_indegree_path))
    av.viz_pagerank_support_and_attack(G, str(cfg.algo_viz_dir))
    av.viz_mincut_all_clusters_one_html(S, str(cfg.algo_viz_dir))
    av.viz_clusters(S, str(clusters_path))

    return {
        "out_dir": str(cfg.algo_viz_dir),
        "support_indegree_html": str(support_indegree_path),
        "clusters_html": str(clusters_path),
    }


def run_full_pipeline(cfg: FullPipelineConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or FullPipelineConfig()
    cfg = _apply_small_mode(cfg)
    _ensure_dirs(cfg)

    training_summary = _run_training_and_graph_export(cfg)
    nodes_df, edges_df = _load_graph_frames(cfg)

    debate_id = gb.pick_single_debate(edges_df)

    basic_viz_summary: Dict[str, Any] | None = None
    generic_viz_summary: Dict[str, Any] | None = None
    algo_viz_summary: Dict[str, Any] | None = None
    if not cfg.skip_visuals:
        basic_viz_summary = _run_basic_graph_visualizations(cfg, nodes_df, edges_df)
        debate_id = basic_viz_summary["debate_id"]
        if cfg.run_generic_visualization:
            generic_viz_summary = _run_generic_visualization(cfg)
        algo_viz_summary = _run_algorithm_visualizations(cfg, nodes_df, edges_df, debate_id)

    analysis_summary = _run_graph_algorithms(cfg, nodes_df, edges_df, debate_id)

    summary = {
        "config": asdict(cfg),
        "paths": {
            "root_dir": str(cfg.root_dir),
            "snapshot_dir": str(cfg.snapshot_dir),
            "models_dir": str(cfg.models_dir),
            "graphs_dir": str(cfg.graphs_dir),
            "analysis_dir": str(cfg.analysis_dir),
            "basic_viz_dir": str(cfg.basic_viz_dir),
            "generic_viz_dir": str(cfg.generic_viz_dir),
            "algo_viz_dir": str(cfg.algo_viz_dir),
        },
        "training": training_summary,
        "basic_visualization": basic_viz_summary,
        "generic_visualization": generic_viz_summary,
        "graph_algorithms": analysis_summary,
        "algorithm_visualization": algo_viz_summary,
        "modes": {
            "skip_training": cfg.skip_training,
            "skip_visuals": cfg.skip_visuals,
            "small_mode": cfg.small_mode,
        },
    }

    cfg.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
