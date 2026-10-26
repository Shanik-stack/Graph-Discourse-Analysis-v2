from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import joblib

import numpy as np
import pandas as pd

from debate_graph.paths import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR
from debate_graph.stance import algo_visualizer as av
from debate_graph.stance import graph_algorithms as ga
from debate_graph.stance import graph_builder as gb
from debate_graph.stance import visualize as generic_viz


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
    use_existing_graphs: bool = False
    model_choice: str = "meta"
    selected_debate_ids: Optional[List[str]] = None
    debate_index_start: Optional[int] = None
    debate_index_end: Optional[int] = None

    nli_model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    stance_model_name: str = "krishnagarg09/stance-detection-semeval2016"
    nli_max_length: int = 256
    stance_max_length: int = 128
    stance_device: str = "cpu"
    similarity_top_k: int = 8
    similarity_threshold: float = 0.15

    graph_min_score: float = 0.55
    generic_viz_min_score: float = 0.60
    generic_viz_top_k_out: int = 5
    run_generic_visualization: bool = True

    @property
    def root_dir(self) -> Path:
        return ARTIFACTS_DIR / "pipeline"

    @property
    def snapshot_dir(self) -> Path:
        return DATA_DIR / "syncialo_snapshot"

    @property
    def models_dir(self) -> Path:
        return MODELS_DIR / "pipeline"

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


def _cache_key(cfg: FullPipelineConfig) -> str:
    payload = {
        "corpus_id": cfg.corpus_id,
        "split": cfg.split,
        "neutrals_per_pos": cfg.neutrals_per_pos,
        "max_pairs": cfg.max_pairs,
        "max_edges": cfg.max_edges,
        "max_debates": cfg.max_debates,
        "seed": cfg.seed,
        "test_size": cfg.test_size,
        "max_examples": cfg.max_examples,
        "nli_model_name": cfg.nli_model_name,
        "stance_model_name": cfg.stance_model_name,
        "nli_max_length": cfg.nli_max_length,
        "stance_max_length": cfg.stance_max_length,
        "stance_device": cfg.stance_device,
        "small_mode": cfg.small_mode,
        "model_choice": cfg.model_choice,
        "selected_debate_ids": cfg.selected_debate_ids,
        "debate_index_start": cfg.debate_index_start,
        "debate_index_end": cfg.debate_index_end,
        "similarity_top_k": cfg.similarity_top_k,
        "similarity_threshold": cfg.similarity_threshold,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def _cache_paths(cfg: FullPipelineConfig) -> Dict[str, Path]:
    key = _cache_key(cfg)
    return {
        "features": cfg.models_dir / f"meta_features_{key}.npz",
        "model": cfg.models_dir / f"meta_lr_{key}.joblib",
        "graph_features": cfg.models_dir / f"graph_eval_features_{key}.npz",
        "graph_model": cfg.models_dir / f"graph_aware_model_{key}.joblib",
        "manifest": cfg.models_dir / f"cache_manifest_{key}.json",
    }


def _write_cache_manifest(cfg: FullPipelineConfig, cache_paths: Dict[str, Path]) -> None:
    manifest = {
        "cache_key": _cache_key(cfg),
        "config": asdict(cfg),
        "paths": {name: str(path) for name, path in cache_paths.items()},
    }
    cache_paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _maybe_import_benchmark_graph_model(cfg: FullPipelineConfig, target_path: Path) -> Optional[Path]:
    if target_path.exists():
        return target_path

    benchmark_dir = MODELS_DIR / "syncialo_benchmarks"
    if not benchmark_dir.exists():
        return None

    preferred_name = f"{cfg.corpus_id}_{cfg.split}_{cfg.max_debates}_seed{cfg.seed}_graph_aware_rf.joblib"
    candidates = list(benchmark_dir.glob("*_graph_aware_rf.joblib"))
    preferred = benchmark_dir / preferred_name

    source_path: Optional[Path] = None
    if preferred.exists():
        source_path = preferred
    elif candidates:
        source_path = max(candidates, key=lambda p: p.stat().st_mtime)

    if source_path is None:
        return None

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return source_path


def _load_graph_frames(cfg: FullPipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes_path = cfg.graphs_dir / "nodes.tsv"
    edges_path = cfg.graphs_dir / "edges.tsv"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"Graph files were not found in {cfg.graphs_dir}. "
            "Run the export stage first or use a different graphs directory."
        )

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


def _filter_rows_by_selection(rows: List[tuple], cfg: FullPipelineConfig) -> List[tuple]:
    if cfg.selected_debate_ids:
        selected = set(cfg.selected_debate_ids)
        return [row for row in rows if row[0] in selected]
    if cfg.debate_index_start is None and cfg.debate_index_end is None:
        return rows

    debate_ids = sorted({row[0] for row in rows})
    start = 0 if cfg.debate_index_start is None else max(0, int(cfg.debate_index_start))
    end = len(debate_ids) - 1 if cfg.debate_index_end is None else min(len(debate_ids) - 1, int(cfg.debate_index_end))
    chosen = set(debate_ids[start : end + 1]) if end >= start else set()
    return [row for row in rows if row[0] in chosen]


def _ensure_non_empty_rows(rows: List[tuple], cfg: FullPipelineConfig) -> None:
    if rows:
        return
    selector_bits: List[str] = []
    if cfg.selected_debate_ids:
        selector_bits.append(f"debates={cfg.selected_debate_ids}")
    if cfg.debate_index_start is not None or cfg.debate_index_end is not None:
        selector_bits.append(f"debate_range=({cfg.debate_index_start}, {cfg.debate_index_end})")
    selector = ", ".join(selector_bits) if selector_bits else "current pipeline settings"
    raise ValueError(f"No graph edge rows were available for the requested selection: {selector}.")


def _filter_graph_frames(cfg: FullPipelineConfig, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.selected_debate_ids:
        selected = set(cfg.selected_debate_ids)
    elif cfg.debate_index_start is not None or cfg.debate_index_end is not None:
        debate_ids = sorted(nodes_df["debate_id"].astype(str).unique().tolist())
        start = 0 if cfg.debate_index_start is None else max(0, int(cfg.debate_index_start))
        end = len(debate_ids) - 1 if cfg.debate_index_end is None else min(len(debate_ids) - 1, int(cfg.debate_index_end))
        selected = set(debate_ids[start : end + 1]) if end >= start else set()
    else:
        return nodes_df, edges_df

    return (
        nodes_df[nodes_df["debate_id"].isin(selected)].copy(),
        edges_df[edges_df["debate_id"].isin(selected)].copy(),
    )


def _reset_graph_outputs(cfg: FullPipelineConfig) -> None:
    for path in [cfg.graphs_dir / "nodes.tsv", cfg.graphs_dir / "edges.tsv"]:
        if path.exists():
            path.unlink()


def _export_graph_with_graph_aware_model(
    cfg: FullPipelineConfig,
    rows: List[tuple],
    claims_by_debate: Dict[str, Dict[str, str]],
    nli: HFNLIModel,
    stance: HFStanceModel,
    base_features_path: Path,
    graph_features_path: Path,
    model_path: Path,
) -> Dict[str, Any]:
    from debate_graph.stance import meta_benchmark as meta_bench
    from debate_graph.stance import syncialo_model_benchmarks as sync_bench

    _reset_graph_outputs(cfg)
    graph_feats = sync_bench.build_graph_feature_matrix_from_rows(
        rows,
        claims_by_debate,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )
    if graph_features_path.exists():
        data = np.load(graph_features_path, allow_pickle=True)
        X_graph = data["X"]
    else:
        X_graph = graph_feats.astype(np.float32)
        np.savez_compressed(graph_features_path, X=X_graph)

    X_base, y_base = meta_bench.featurize_graph_rows(
        rows=rows,
        nli=nli,
        stance=stance,
        cache_path=(str(base_features_path) if cfg.reuse_graph_features else None),
        max_examples=cfg.max_examples,
    )
    active_rows = rows[: cfg.max_examples]
    X_graph = X_graph[: len(active_rows)]
    y_true = y_base[: len(active_rows)]
    X_full = np.concatenate([X_base[: len(active_rows)], X_graph], axis=1)

    if cfg.skip_training:
        if not model_path.exists():
            raise FileNotFoundError(f"--skip-training was requested, but no saved graph-aware model was found at {model_path}")
        clf = joblib.load(model_path)
    else:
        clf = sync_bench.train_graph_aware_model(
            X_base[: len(active_rows)],
            X_graph,
            y_true,
            model_path,
            cfg.force_retrain_meta,
        )

    logger = meta_bench.GraphResultLogger(out_dir=str(cfg.graphs_dir))
    try:
        for debate_id, claim_map in claims_by_debate.items():
            for cid, txt in claim_map.items():
                logger.log_node(debate_id, cid, txt)

        y_pred = clf.predict(X_full)
        probs = clf.predict_proba(X_full)
        for row, pred_id, prob in zip(active_rows, y_pred, probs):
            debate_id, src_id, tgt_id, premise, hypothesis, gold = row
            logger.log_edge(
                debate_id=debate_id,
                src_id=src_id,
                tgt_id=tgt_id,
                gold_label=gold,
                pred_label=sync_bench.ID2LABEL[int(pred_id)],
                pred_score=float(np.max(prob)),
            )
    finally:
        logger.close()

    return {
        "graph_model_path": str(model_path),
        "graph_features_path": str(graph_features_path),
        "graphs_dir": str(cfg.graphs_dir),
        "edge_rows": len(active_rows),
    }


def _run_training_and_graph_export(cfg: FullPipelineConfig) -> Dict[str, Any]:
    from debate_graph.stance import meta_benchmark as meta_bench
    from debate_graph.stance.meta_nli import HFNLIModel, HFStanceModel

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
        selected_debate_ids=cfg.selected_debate_ids,
        debate_index_start=cfg.debate_index_start,
        debate_index_end=cfg.debate_index_end,
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

    cache_paths = _cache_paths(cfg)
    features_path = cache_paths["features"]
    lr_path = cache_paths["model"]
    graph_features_path = cache_paths["graph_features"]
    graph_model_path = cache_paths["graph_model"]
    graph_features_cache_preexisting = graph_features_path.exists()

    training_summary: Dict[str, Any]
    if cfg.model_choice == "meta":
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
    else:
        training_summary = {
            "pairs": len(pairs),
            "feature_shape": None,
            "group_count": None,
            "meta_model_loaded_or_trained": False,
            "meta_classes": None,
            "features_path": str(features_path),
            "model_path": str(graph_model_path),
            "training_skipped": bool(cfg.skip_training),
            "features_recomputed": False,
        }
        imported_from = None
        if cfg.skip_training and not graph_model_path.exists():
            imported_from = _maybe_import_benchmark_graph_model(cfg, graph_model_path)
            if imported_from is None:
                raise FileNotFoundError(
                    f"--skip-training was requested, but no saved graph-aware model was found at {graph_model_path}"
                )
        training_summary["graph_model_imported_from"] = (str(imported_from) if imported_from else None)

    rows, claims_by_debate = meta_bench.build_debate_edge_dataset(
        local_dir=local_dir,
        corpus_id=cfg.corpus_id,
        split=cfg.split,
        neutrals_per_pos=cfg.neutrals_per_pos,
        max_edges=cfg.max_edges,
        max_debates=cfg.max_debates,
        seed=cfg.seed,
        selected_debate_ids=cfg.selected_debate_ids,
        debate_index_start=cfg.debate_index_start,
        debate_index_end=cfg.debate_index_end,
    )
    _ensure_non_empty_rows(rows, cfg)
    filtered_claims = {did: claims for did, claims in claims_by_debate.items() if did in {row[0] for row in rows}}
    claims_by_debate = filtered_claims

    _reset_graph_outputs(cfg)
    if cfg.model_choice == "graph-aware":
        export_summary = _export_graph_with_graph_aware_model(
            cfg=cfg,
            rows=rows,
            claims_by_debate=claims_by_debate,
            nli=nli,
            stance=stance,
            base_features_path=features_path,
            graph_features_path=graph_features_path,
            model_path=graph_model_path,
        )
    else:
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
        export_summary = {"graphs_dir": str(cfg.graphs_dir), "edge_rows": min(len(rows), cfg.max_examples)}

    training_summary.update({
        "edge_rows": export_summary["edge_rows"],
        "graphs_dir": export_summary["graphs_dir"],
        "graph_features_path": str(graph_features_path),
        "graph_features_cache_enabled": bool(cfg.reuse_graph_features),
        "graph_features_cache_preexisting": bool(graph_features_cache_preexisting),
        "graph_model_path": str(graph_model_path),
        "model_choice": cfg.model_choice,
        "cache_key": _cache_key(cfg),
        "cache_manifest": str(cache_paths["manifest"]),
    })
    _write_cache_manifest(cfg, cache_paths)
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


def _run_generic_visualization(cfg: FullPipelineConfig, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> Dict[str, Any]:
    nodes_path = cfg.root_dir / "filtered_nodes_for_generic_viz.tsv"
    edges_path = cfg.root_dir / "filtered_edges_for_generic_viz.tsv"
    nodes_df.to_csv(nodes_path, sep="\t", index=False)
    edges_df.to_csv(edges_path, sep="\t", index=False)

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

    training_summary: Dict[str, Any] | None = None
    if not cfg.use_existing_graphs:
        training_summary = _run_training_and_graph_export(cfg)
    nodes_df, edges_df = _load_graph_frames(cfg)
    nodes_df, edges_df = _filter_graph_frames(cfg, nodes_df, edges_df)

    debate_id = gb.pick_single_debate(edges_df)

    basic_viz_summary: Dict[str, Any] | None = None
    generic_viz_summary: Dict[str, Any] | None = None
    algo_viz_summary: Dict[str, Any] | None = None
    if not cfg.skip_visuals:
        basic_viz_summary = _run_basic_graph_visualizations(cfg, nodes_df, edges_df)
        debate_id = basic_viz_summary["debate_id"]
        if cfg.run_generic_visualization:
            generic_viz_summary = _run_generic_visualization(cfg, nodes_df, edges_df)
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
            "model_choice": cfg.model_choice,
            "skip_training": cfg.skip_training,
            "skip_visuals": cfg.skip_visuals,
            "small_mode": cfg.small_mode,
            "use_existing_graphs": cfg.use_existing_graphs,
        },
    }

    cfg.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
