# viz_after_benchmark.py
"""
Visualize graphs AFTER your benchmark runs.

Assumes your benchmark writes:
1) benchmark file (TSV/CSV) with at least:
   debate_id, src_id, tgt_id, pred_label, pred_score
2) nodes file (JSONL/JSON/TSV) with at least:
   debate_id, claim_id, text

This script:
- loads both
- builds one DiGraph per debate_id
- saves:
    out/<debate_id>.html   (interactive)
    out/<debate_id>.png    (static)
    out/<debate_id>.gexf   (Gephi)
    out/summary.csv        (stats)

Install:
  pip install pandas networkx matplotlib pyvis
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import networkx as nx

DEFAULT_LABEL_COLORS = {
    "support": "#2ca02c",
    "attack": "#d62728",
    "neutral": "#7f7f7f",
}

def _read_nodes(nodes_path: str) -> pd.DataFrame:
    ext = os.path.splitext(nodes_path)[1].lower()
    if ext in [".jsonl"]:
        rows = []
        with open(nodes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if ext in [".json"]:
        with open(nodes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    # default: csv/tsv
    sep = "\t" if ext in [".tsv", ".txt"] else ","
    return pd.read_csv(nodes_path, sep=sep)

def _read_edges(edges_path: str) -> pd.DataFrame:
    ext = os.path.splitext(edges_path)[1].lower()
    sep = "\t" if ext in [".tsv", ".txt"] else ","
    return pd.read_csv(edges_path, sep=sep)

def build_graph_for_debate(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    debate_id: Any,
    *,
    min_score: float = 0.0,
    drop_neutral: bool = False,
    keep_top_k_out: Optional[int] = None,
    label_colors: Optional[Dict[str, str]] = None,
    node_id_col: str = "claim_id",
    node_text_col: str = "text",
    edge_src_col: str = "src_id",
    edge_tgt_col: str = "tgt_id",
    edge_label_col: str = "pred_label",
    edge_score_col: str = "pred_score",
) -> nx.DiGraph:
    label_colors = label_colors or DEFAULT_LABEL_COLORS

    ndf = nodes_df[nodes_df["debate_id"] == debate_id].copy()
    edf = edges_df[edges_df["debate_id"] == debate_id].copy()

    # basic sanitization
    edf[edge_score_col] = pd.to_numeric(edf[edge_score_col], errors="coerce").fillna(0.0)

    # filter
    edf = edf[edf[edge_score_col] >= float(min_score)]
    if drop_neutral:
        edf = edf[edf[edge_label_col] != "neutral"]

    # top-k outgoing per src
    if keep_top_k_out is not None:
        edf = (
            edf.sort_values([edge_src_col, edge_score_col], ascending=[True, False])
               .groupby(edge_src_col, as_index=False)
               .head(int(keep_top_k_out))
        )

    # build graph
    G = nx.DiGraph(debate_id=debate_id)

    # nodes
    for _, r in ndf.iterrows():
        nid = r[node_id_col]
        text = str(r.get(node_text_col, ""))
        G.add_node(
            nid,
            label=str(nid),
            text=text,
            title=text,
        )

    node_set = set(G.nodes())

    # edges
    for _, r in edf.iterrows():
        u = r[edge_src_col]
        v = r[edge_tgt_col]
        if u not in node_set or v not in node_set or u == v:
            continue
        lab = str(r.get(edge_label_col, "neutral"))
        score = float(r.get(edge_score_col, 0.0))
        color = label_colors.get(lab, "#000000")
        width = 1.0 + 4.0 * max(0.0, min(1.0, score))
        G.add_edge(
            u, v,
            label=lab,
            score=score,
            color=color,
            width=width,
            title=f"{lab} | score={score:.3f}",
        )

    return G

def graph_stats(G: nx.DiGraph) -> Dict[str, Any]:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dens = (m / (n * (n - 1))) if n > 1 else 0.0
    counts: Dict[str, int] = {}
    for _, _, a in G.edges(data=True):
        lab = a.get("label", "unknown")
        counts[lab] = counts.get(lab, 0) + 1
    has_cycle = False
    try:
        nx.find_cycle(G, orientation="original")
        has_cycle = True
    except Exception:
        has_cycle = False
    return {
        "debate_id": G.graph.get("debate_id"),
        "nodes": n,
        "edges": m,
        "density": dens,
        "has_cycle": has_cycle,
        "support_edges": counts.get("support", 0),
        "attack_edges": counts.get("attack", 0),
        "neutral_edges": counts.get("neutral", 0),
    }

def save_pyvis_html(G: nx.DiGraph, out_html: str, *, show_buttons: bool = True) -> None:
    from pyvis.network import Network
    net = Network(height="750px", width="100%", directed=True)

    # stable-ish defaults for argument graphs
    net.barnes_hut(gravity=-25000, central_gravity=0.3, spring_length=120, spring_strength=0.02, damping=0.09)

    for nid, a in G.nodes(data=True):
        text = a.get("text", "")
        words = str(text).strip().split()
        preview = " ".join(words[:6]) + ("…" if len(words) > 6 else "")
        net.add_node(nid, label=f"{nid}: {preview}", title=a.get("title", text))

    for u, v, a in G.edges(data=True):
        net.add_edge(
            u, v,
            label=a.get("label", ""),
            title=a.get("title", ""),
            color=a.get("color", "#000000"),
            width=float(a.get("width", 1.0)),
            arrows="to",
        )

    if show_buttons:
        net.show_buttons(filter_=["physics", "interaction"])

    net.write_html(out_html)

def save_static_png(G: nx.DiGraph, out_png: str, *, seed: int = 7) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=seed)

    nx.draw_networkx_nodes(G, pos, node_size=650, alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_size=8)

    edge_colors = [a.get("color", "#000000") for _, _, a in G.edges(data=True)]
    widths = [float(a.get("width", 1.0)) for _, _, a in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=14, width=widths, edge_color=edge_colors, alpha=0.9)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def export_gexf(G: nx.DiGraph, out_gexf: str) -> None:
    nx.write_gexf(G, out_gexf)

def run_visualization(
    nodes_path: str,
    edges_path: str,
    out_dir: str = "out_graphs",
    *,
    min_score: float = 0.6,
    drop_neutral: bool = True,
    keep_top_k_out: Optional[int] = 5,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    nodes_df = _read_nodes(nodes_path)
    edges_df = _read_edges(edges_path)

    # required columns check (fail early with clear message)
    for c in ["debate_id", "claim_id", "text"]:
        if c not in nodes_df.columns:
            raise ValueError(f"nodes file missing required column: {c}")
    for c in ["debate_id", "src_id", "tgt_id", "pred_label", "pred_score"]:
        if c not in edges_df.columns:
            raise ValueError(f"edges file missing required column: {c}")

    debate_ids = sorted(set(edges_df["debate_id"].unique()).intersection(set(nodes_df["debate_id"].unique())))
    if not debate_ids:
        raise ValueError("No overlapping debate_id between nodes and edges files.")

    stats_rows = []
    for did in debate_ids:
        G = build_graph_for_debate(
            nodes_df, edges_df, did,
            min_score=min_score,
            drop_neutral=drop_neutral,
            keep_top_k_out=keep_top_k_out,
        )

        stats_rows.append(graph_stats(G))

        html_path = os.path.join(out_dir, f"{did}.html")
        png_path = os.path.join(out_dir, f"{did}.png")
        gexf_path = os.path.join(out_dir, f"{did}.gexf")

        save_pyvis_html(G, html_path)
        save_static_png(G, png_path)
        export_gexf(G, gexf_path)

    pd.DataFrame(stats_rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)


if __name__ == "__main__":
    # EDIT THESE PATHS to match your benchmark outputs
    NODES_PATH = "nodes.tsv"        # or nodes.jsonl
    EDGES_PATH = "benchmark.tsv"    # your benchmark predictions file
    run_visualization(
        NODES_PATH,
        EDGES_PATH,
        out_dir="out_graphs",
        min_score=0.60,
        drop_neutral=True,
        keep_top_k_out=5,
    )
