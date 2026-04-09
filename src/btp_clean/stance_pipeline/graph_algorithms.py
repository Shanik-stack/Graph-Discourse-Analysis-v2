# graph_algorithms.py
"""
Implements the graph algorithms you listed on ONE debate graph.

Input (TSV):
  out_graphs/nodes.tsv : debate_id, claim_id, text
  out_graphs/edges.tsv : debate_id, src_id, tgt_id, pred_label, pred_score
                         (optional gold_label ignored)

Outputs (folder OUT_DIR):
  - support_indegree.csv
  - pagerank_support.csv
  - vulnerability_mincut.json
  - clusters.json
  - cluster_themes.json

Algorithms implemented:
1) Support in-degree (local strength/weakness)
2) PageRank on support graph (global influence)
3) Minimum cut / vulnerability of support graph (global edge min-cut + articulation points)
4) Graph clustering (support graph communities) + themes/subtopics (TF-IDF keywords per cluster)
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

from btp_clean.paths import ARTIFACTS_DIR


# -----------------------
# CONFIG
# -----------------------
NODES_PATH = str(ARTIFACTS_DIR / "stance_pipeline" / "out_graphs" / "nodes.tsv")
EDGES_PATH = str(ARTIFACTS_DIR / "stance_pipeline" / "out_graphs" / "edges.tsv")
OUT_DIR = str(ARTIFACTS_DIR / "stance_pipeline" / "out_graphs_analysis")

# Edge filtering for analysis
MIN_SCORE = 0.55
USE_ONLY = {"support", "attack"}  # we will build support-subgraph separately

# Clustering/theme extraction
MAX_FEATURES = 2000
STOP_WORDS = "english"
TOP_K_KEYWORDS = 10


# -----------------------
# Loading + single debate selection
# -----------------------
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(NODES_PATH, sep="\t")
    edges = pd.read_csv(EDGES_PATH, sep="\t")

    nodes["debate_id"] = nodes["debate_id"].astype(str)
    nodes["claim_id"] = nodes["claim_id"].astype(str)
    nodes["text"] = nodes["text"].astype(str)

    edges["debate_id"] = edges["debate_id"].astype(str)
    edges["src_id"] = edges["src_id"].astype(str)
    edges["tgt_id"] = edges["tgt_id"].astype(str)
    edges["pred_label"] = edges["pred_label"].astype(str)
    edges["pred_score"] = pd.to_numeric(edges["pred_score"], errors="coerce").fillna(0.0)

    # basic filter
    edges = edges[edges["pred_score"] >= float(MIN_SCORE)]
    edges = edges[edges["pred_label"].isin(list(USE_ONLY))]

    return nodes, edges


def pick_single_debate(edges_df: pd.DataFrame) -> str:
    # choose the debate with most support/attack edges
    return edges_df["debate_id"].value_counts().idxmax()


def restrict_to_debate(
    nodes_df: pd.DataFrame, edges_df: pd.DataFrame, debate_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ndf = nodes_df[nodes_df["debate_id"] == debate_id].copy()
    edf = edges_df[edges_df["debate_id"] == debate_id].copy()
    return ndf, edf


# -----------------------
# Graph builders
# -----------------------
def build_directed_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()

    for _, r in nodes_df.iterrows():
        nid = r["claim_id"]
        G.add_node(nid, text=r["text"])

    for _, r in edges_df.iterrows():
        u, v = r["src_id"], r["tgt_id"]
        if u == v:
            continue
        if u in G and v in G:
            G.add_edge(u, v, label=r["pred_label"], score=float(r["pred_score"]))
    # remove isolates
    isolates = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    G.remove_nodes_from(isolates)
    return G


def support_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, a in G.edges(data=True):
        if a.get("label") == "support":
            H.add_edge(u, v, **a)
    isolates = [n for n in H.nodes() if H.in_degree(n) == 0 and H.out_degree(n) == 0]
    H.remove_nodes_from(isolates)
    return H


# -----------------------
# 1) Support in-degree (weakness/strength)
# -----------------------
def compute_support_indegree(S: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for n in S.nodes():
        indeg = S.in_degree(n)
        rows.append(
            {
                "claim_id": n,
                "support_in_degree": int(indeg),
                "text": S.nodes[n].get("text", ""),
            }
        )
    df = pd.DataFrame(rows).sort_values(["support_in_degree"], ascending=False).reset_index(drop=True)

    # Weak claims (potentially “non sequitur-ish”) = low support_in_degree
    # Strong claims = high support_in_degree
    return df


# -----------------------
# 2) PageRank on support graph (global influence)
# -----------------------
def compute_pagerank_support(S: nx.DiGraph) -> pd.DataFrame:
    if S.number_of_nodes() == 0:
        return pd.DataFrame(columns=["claim_id", "pagerank", "support_in_degree", "text"])

    # Weight PageRank by edge confidence (pred_score)
    for u, v, a in S.edges(data=True):
        a["weight"] = float(a.get("score", 1.0))

    pr = nx.pagerank(S, alpha=0.85, weight="weight")

    rows = []
    for n, val in pr.items():
        rows.append(
            {
                "claim_id": n,
                "pagerank": float(val),
                "support_in_degree": int(S.in_degree(n)),
                "text": S.nodes[n].get("text", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)


# -----------------------
# 3) Min-cut vulnerability (support graph)
# -----------------------
def mincut_vulnerability(S: nx.DiGraph) -> Dict[str, Any]:
    """
    Interpretable “vulnerability” for a support subgraph.

    We compute:
    - Global edge min-cut on an UNDIRECTED projection of support graph
      (Stoer-Wagner requires undirected).
      Interpretation: minimum number of support edges to remove to disconnect the graph.

    - Articulation points on undirected projection:
      nodes whose removal increases number of connected components (weak points / bottlenecks).

    Notes:
    - If graph is disconnected already, min-cut is 0 for some components; we report per largest component.
    """
    if S.number_of_nodes() == 0:
        return {"status": "empty_support_graph"}

    U = nx.Graph()
    U.add_nodes_from(S.nodes())

    # Edge capacity: higher confidence => harder to “break”.
    # Use capacity = score (0..1). Stoer-Wagner finds min total capacity cut.
    # If you want “min number of edges”, set all capacities=1.0.
    for u, v, a in S.edges(data=True):
        cap = float(a.get("score", 1.0))
        if U.has_edge(u, v):
            U[u][v]["capacity"] += cap
        else:
            U.add_edge(u, v, capacity=cap)

    # Focus on largest connected component for stable interpretation
    comps = list(nx.connected_components(U))
    comps_sorted = sorted(comps, key=lambda c: len(c), reverse=True)
    largest = comps_sorted[0]
    Uc = U.subgraph(largest).copy()

    # articulation points (node vulnerability)
    arts = list(nx.articulation_points(Uc))

    # global min cut (edge vulnerability)
    # returns (cut_value, (A, B))
    cut_value, partition = nx.stoer_wagner(Uc, weight="capacity")
    A, B = partition
    cut_edges = []
    for u, v, data in Uc.edges(data=True):
        if (u in A and v in B) or (u in B and v in A):
            cut_edges.append({"u": u, "v": v, "capacity": float(data.get("capacity", 0.0))})

    return {
        "status": "ok",
        "nodes_support": int(S.number_of_nodes()),
        "edges_support": int(S.number_of_edges()),
        "largest_component_nodes": int(len(largest)),
        "articulation_points": arts,
        "mincut_capacity": float(cut_value),
        "mincut_edges": sorted(cut_edges, key=lambda x: x["capacity"]),
        "partition_sizes": {"A": int(len(A)), "B": int(len(B))},
        "note": "Min-cut computed on undirected projection (Stoer-Wagner). Capacity=sum(pred_score).",
    }


# -----------------------
# 4) Clustering (support graph) + themes
# -----------------------
def cluster_support_graph(S: nx.DiGraph) -> List[List[str]]:
    """
    Community detection on UNDIRECTED projection using greedy modularity.
    Output: list of clusters, each is a list of claim_ids.
    """
    if S.number_of_nodes() == 0:
        return []

    U = nx.Graph()
    U.add_nodes_from(S.nodes(data=True))
    for u, v, a in S.edges(data=True):
        w = float(a.get("score", 1.0))
        if U.has_edge(u, v):
            U[u][v]["weight"] += w
        else:
            U.add_edge(u, v, weight=w)

    communities = list(nx.algorithms.community.greedy_modularity_communities(U, weight="weight"))
    clusters = [sorted(list(c)) for c in communities]
    clusters.sort(key=len, reverse=True)
    return clusters


def cluster_themes(S: nx.DiGraph, clusters: List[List[str]]) -> Dict[str, Any]:
    """
    Extract TF-IDF keywords per cluster (themes/subtopics).
    """
    id_to_text = {n: S.nodes[n].get("text", "") for n in S.nodes()}
    cluster_texts = []
    cluster_ids = []
    for idx, c in enumerate(clusters):
        texts = [id_to_text.get(n, "") for n in c]
        doc = " ".join(t for t in texts if t)
        cluster_texts.append(doc)
        cluster_ids.append(idx)

    if not cluster_texts:
        return {"status": "empty"}

    vec = TfidfVectorizer(stop_words=STOP_WORDS, max_features=MAX_FEATURES)
    X = vec.fit_transform(cluster_texts)
    feats = vec.get_feature_names_out()

    out = {"status": "ok", "clusters": []}
    for i, cid in enumerate(cluster_ids):
        row = X[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:TOP_K_KEYWORDS]
        keywords = [{"term": feats[j], "tfidf": float(row[j])} for j in top_idx if row[j] > 0]
        out["clusters"].append(
            {
                "cluster_id": int(cid),
                "size": int(len(clusters[i])),
                "keywords": keywords,
                "nodes": clusters[i],
            }
        )

    return out


# -----------------------
# Main: run everything
# -----------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    nodes_df, edges_df = load_data()
    debate_id = pick_single_debate(edges_df)
    nodes_df, edges_df = restrict_to_debate(nodes_df, edges_df, debate_id)

    G = build_directed_graph(nodes_df, edges_df)
    S = support_subgraph(G)

    # 1) support in-degree
    df_indeg = compute_support_indegree(S)
    df_indeg.to_csv(os.path.join(OUT_DIR, "support_indegree.csv"), index=False)

    # 2) pagerank
    df_pr = compute_pagerank_support(S)
    df_pr.to_csv(os.path.join(OUT_DIR, "pagerank_support.csv"), index=False)

    # 3) mincut vulnerability
    vuln = mincut_vulnerability(S)
    with open(os.path.join(OUT_DIR, "vulnerability_mincut.json"), "w", encoding="utf-8") as f:
        json.dump({"debate_id": debate_id, **vuln}, f, indent=2)

    # 4) clustering + themes
    clusters = cluster_support_graph(S)
    with open(os.path.join(OUT_DIR, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump({"debate_id": debate_id, "clusters": clusters}, f, indent=2)

    themes = cluster_themes(S, clusters)
    with open(os.path.join(OUT_DIR, "cluster_themes.json"), "w", encoding="utf-8") as f:
        json.dump({"debate_id": debate_id, **themes}, f, indent=2)

    print("Debate:", debate_id)
    print("Full graph: nodes", G.number_of_nodes(), "edges", G.number_of_edges())
    print("Support graph: nodes", S.number_of_nodes(), "edges", S.number_of_edges())
    print("Wrote analysis to:", OUT_DIR)


if __name__ == "__main__":
    main()
