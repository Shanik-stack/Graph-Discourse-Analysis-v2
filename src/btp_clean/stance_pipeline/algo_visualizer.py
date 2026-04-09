# algo_visualizer.py
"""
Graph-algorithm visualizers (ONE debate) for:
1) Support in-degree (support graph)
2) PageRank (support graph) + PageRank (attack graph) + Net influence (support - attack)
3) Min-cut on EVERY support-cluster subgraph (articulation points + min-cut edges)
4) Clustering (support graph): colors clusters + shows cluster keywords (TF-IDF)

Reads:
  out_graphs/nodes.tsv : debate_id, claim_id, text
  out_graphs/edges.tsv : debate_id, src_id, tgt_id, pred_label, pred_score

Writes HTML (OUT_DIR):
  <debate>_support_indegree.html
  <debate>_pagerank_support.html
  <debate>_pagerank_attack.html
  <debate>_pagerank_net.html
  <debate>_clusters.html
  <debate>_mincut_all_clusters.json
  <debate>_mincut_cluster_<k>.html   (per support cluster)

Notes:
- Uses ONE debate (auto-picked by most support/attack edges).
- Only uses SUPPORT/ATTACK edges for visuals (neutral ignored).
- Removes isolated nodes.
- Adds:
  - node-label slider (top-right) to expand/shrink visible text
  - legend panel (top-left) explaining each visualization
"""

import os
import re
import json
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import networkx as nx
from pyvis.network import Network
from sklearn.feature_extraction.text import TfidfVectorizer

from btp_clean.paths import ARTIFACTS_DIR

# -----------------------
# CONFIG
# -----------------------
NODES_PATH = str(ARTIFACTS_DIR / "stance_pipeline" / "out_graphs" / "nodes.tsv")
EDGES_PATH = str(ARTIFACTS_DIR / "stance_pipeline" / "out_graphs" / "edges.tsv")
OUT_DIR = str(ARTIFACTS_DIR / "stance_pipeline" / "out_graphs_algo_viz")

MIN_SCORE = 0.55
KEEP_LABELS = {"support", "attack"}

LAYOUT = "force"          # "force" (spread) or "hierarchical" (structured)
STABILIZE_ITERS = 2000

DEFAULT_LABEL_WORDS = 7
MAX_LABEL_WORDS = 50

EDGE_LABELS = True

EDGE_COLOR = {"support": "#2ca02c", "attack": "#d62728"}

# Clustering themes
STOP_WORDS = "english"
MAX_FEATURES = 2500
TOP_K_KEYWORDS = 10

# Legend colors (kept separate for clarity)
EDGE_COLOR_SUPPORT = "#2ca02c"
EDGE_COLOR_ATTACK = "#d62728"


# -----------------------
# Utilities
# -----------------------
def _shorten_words(text: str, k: int) -> str:
    w = (text or "").strip().split()
    if not w:
        return ""
    s = " ".join(w[:k])
    return s + ("…" if len(w) > k else "")


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _scale(val: float, vmin: float, vmax: float, out_min: float, out_max: float) -> float:
    if vmax <= vmin:
        return (out_min + out_max) / 2.0
    t = (val - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    return out_min + t * (out_max - out_min)


# -----------------------
# HTML inject: node label slider
# -----------------------
def _inject_label_slider(html_path: str):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    if "id=\"labelSlider\"" in html:
        return

    slider_panel = f"""
<div style="position:fixed; top:12px; right:12px; z-index:9999; background:#fff; border:1px solid #ddd;
            border-radius:10px; padding:10px 12px; box-shadow:0 6px 18px rgba(0,0,0,0.08); font-family:Arial;">
  <div style="font-size:13px; font-weight:600; margin-bottom:6px;">Node label length</div>
  <input id="labelSlider" type="range" min="3" max="{MAX_LABEL_WORDS}" value="{DEFAULT_LABEL_WORDS}" step="1" style="width:220px;">
  <span id="labelVal" style="margin-left:8px; font-size:12px;">{DEFAULT_LABEL_WORDS} words</span>
  <div style="margin-top:6px; font-size:11px; color:#555;">Hover nodes for full text.</div>
</div>
"""
    html = re.sub(r"(<body[^>]*>)", r"\1\n" + slider_panel, html, count=1)

    js_hook = r"""
<script>
(function() {
  function shortenWords(text, k) {
    if (!text) return "";
    var words = text.trim().split(/\s+/);
    if (words.length <= k) return words.join(" ");
    return words.slice(0, k).join(" ") + "…";
  }

  function tryInit() {
    if (typeof nodes === "undefined" || !nodes.get) return false;

    var all = nodes.get();
    all.forEach(function(n) {
      var tmp = document.createElement("div");
      tmp.innerHTML = n.title || "";
      var plain = (tmp.textContent || tmp.innerText || "").trim();
      n.__fulltext = plain;
      nodes.update(n);
    });

    var slider = document.getElementById("labelSlider");
    var labelVal = document.getElementById("labelVal");

    function apply(k) {
      var cur = nodes.get();
      cur.forEach(function(n) {
        var text = n.__fulltext || n.label || "";
        n.label = shortenWords(text, k) || n.id;
        nodes.update(n);
      });
      labelVal.textContent = k + " words";
    }

    slider.addEventListener("input", function() {
      apply(parseInt(slider.value, 10));
    });

    apply(parseInt(slider.value, 10));
    return true;
  }

  var tries = 0;
  var t = setInterval(function() {
    tries += 1;
    if (tryInit() || tries > 40) clearInterval(t);
  }, 100);
})();
</script>
"""
    html = html.replace("</body>", js_hook + "\n</body>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# -----------------------
# HTML inject: legend
# -----------------------
def _inject_legend(html_path: str, legend_title: str, legend_lines: List[str]):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    if "id=\"algoLegend\"" in html:
        return

    items_html = "".join([f"<li style='margin:4px 0;'>{line}</li>" for line in legend_lines])

    legend_panel = f"""
<div id="algoLegend" style="
  position:fixed; top:12px; left:12px; z-index:9999;
  background:#ffffff; border:1px solid #ddd; border-radius:10px;
  padding:10px 12px; box-shadow:0 6px 18px rgba(0,0,0,0.08);
  font-family:Arial; max-width:380px;">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
    <div style="font-size:13px; font-weight:700;">{legend_title}</div>
    <button id="legendToggleBtn" style="
      border:1px solid #ccc; background:#f7f7f7; border-radius:8px;
      padding:3px 8px; font-size:12px; cursor:pointer;">Hide</button>
  </div>

  <div id="legendBody" style="margin-top:8px;">
    <div style="font-size:12px; font-weight:700; margin-bottom:6px;">Edges</div>
    <div style="display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:8px;">
      <span style="display:flex; align-items:center; gap:6px;">
        <span style="display:inline-block; width:18px; height:4px; background:{EDGE_COLOR_SUPPORT}; border-radius:2px;"></span>
        <span style="font-size:12px;">support</span>
      </span>
      <span style="display:flex; align-items:center; gap:6px;">
        <span style="display:inline-block; width:18px; height:4px; background:{EDGE_COLOR_ATTACK}; border-radius:2px;"></span>
        <span style="font-size:12px;">attack</span>
      </span>
    </div>

    <div style="font-size:12px; font-weight:700; margin-bottom:6px;">Nodes</div>
    <ul style="margin:0 0 8px 16px; padding:0; font-size:12px; color:#111;">
      {items_html}
    </ul>

    <div style="font-size:11px; color:#555; border-top:1px solid #eee; padding-top:8px;">
      <div>Use the <b>Node label length</b> slider (top-right) to expand/shrink visible text.</div>
      <div>Hover nodes/edges for full details.</div>
    </div>
  </div>
</div>

<script>
(function() {{
  var btn = document.getElementById("legendToggleBtn");
  var body = document.getElementById("legendBody");
  if (!btn || !body) return;

  btn.addEventListener("click", function() {{
    var hidden = body.style.display === "none";
    body.style.display = hidden ? "block" : "none";
    btn.textContent = hidden ? "Hide" : "Show";
  }});
}})();
</script>
"""
    html = re.sub(r"(<body[^>]*>)", r"\1\n" + legend_panel, html, count=1)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def legend_lines_support_indegree():
    return [
        "<b>Size</b> ∝ support in-degree (more incoming supports ⇒ stronger claim).",
        "<b>Shade</b>: darker ⇒ higher support in-degree.",
        "Only claims participating in support edges are shown.",
    ]


def legend_lines_pagerank_support():
    return [
        "<b>Size</b> ∝ PageRank on <b>support</b> graph (global influence).",
        "<b>Color</b>: darker blue ⇒ higher PageRank.",
        "PageRank depends on how influential the supporting neighbors are.",
    ]


def legend_lines_pagerank_attack():
    return [
        "<b>Size</b> ∝ PageRank on <b>attack</b> graph (global influence of attacking).",
        "<b>Color</b>: darker blue ⇒ higher PageRank.",
        "Higher score ⇒ the claim attacks influential claims (directly/indirectly).",
    ]


def legend_lines_pagerank_net():
    return [
        "<b>Green</b> nodes: more globally supported (PR_support > PR_attack).",
        "<b>Red</b> nodes: more globally attacked (PR_attack > PR_support).",
        "<b>Size</b> ∝ |PR_support − PR_attack| (magnitude of net influence).",
    ]


def legend_lines_mincut_cluster(cid: int):
    return [
        f"<b>Cluster {cid}</b> support subgraph.",
        "<b>Orange nodes</b>: articulation points (removing them disconnects cluster).",
        "<b>Orange edges</b>: edges in the minimum cut (weakest links splitting cluster).",
        "Min-cut computed on undirected projection (capacity = confidence score).",
    ]


def legend_lines_clusters():
    return [
        "<b>Node color</b>: community/cluster in support graph (dense mutual support).",
        "<b>Hover</b> a node to see cluster keywords (TF-IDF) summarizing sub-topic.",
        "<b>Size</b> ∝ degree in support graph (connectedness).",
    ]


# -----------------------
# Load + build ONE debate graphs
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

    edges = edges[edges["pred_score"] >= float(MIN_SCORE)]
    edges = edges[edges["pred_label"].isin(list(KEEP_LABELS))]

    return nodes, edges


def pick_single_debate(edges_df: pd.DataFrame) -> str:
    return edges_df["debate_id"].value_counts().idxmax()


def build_combined_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, debate_id: str) -> nx.DiGraph:
    ndf = nodes_df[nodes_df["debate_id"] == debate_id].copy()
    edf = edges_df[edges_df["debate_id"] == debate_id].copy()

    G = nx.DiGraph(debate_id=debate_id)
    for _, r in ndf.iterrows():
        G.add_node(r["claim_id"], text=r["text"])

    for _, r in edf.iterrows():
        u, v = r["src_id"], r["tgt_id"]
        if u == v:
            continue
        if u in G and v in G:
            G.add_edge(u, v, label=r["pred_label"], score=float(r["pred_score"]))

    isolates = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    G.remove_nodes_from(isolates)
    return G


def support_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    S = nx.DiGraph(debate_id=G.graph.get("debate_id"))
    S.add_nodes_from(G.nodes(data=True))
    for u, v, a in G.edges(data=True):
        if a.get("label") == "support":
            S.add_edge(u, v, **a)
    isolates = [n for n in S.nodes() if S.in_degree(n) == 0 and S.out_degree(n) == 0]
    S.remove_nodes_from(isolates)
    return S


def attack_subgraph(G: nx.DiGraph) -> nx.DiGraph:
    A = nx.DiGraph(debate_id=G.graph.get("debate_id"))
    A.add_nodes_from(G.nodes(data=True))
    for u, v, a in G.edges(data=True):
        if a.get("label") == "attack":
            A.add_edge(u, v, **a)
    isolates = [n for n in A.nodes() if A.in_degree(n) == 0 and A.out_degree(n) == 0]
    A.remove_nodes_from(isolates)
    return A


# -----------------------
# Algorithms
# -----------------------
def algo_support_indegree(S: nx.DiGraph) -> Dict[str, int]:
    return {n: int(S.in_degree(n)) for n in S.nodes()}


def pagerank_on_label_subgraph(G: nx.DiGraph, label: str) -> Dict[str, float]:
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, a in G.edges(data=True):
        if a.get("label") == label:
            H.add_edge(u, v, weight=float(a.get("score", 1.0)))

    isolates = [n for n in H.nodes() if H.in_degree(n) == 0 and H.out_degree(n) == 0]
    H.remove_nodes_from(isolates)

    if H.number_of_nodes() == 0:
        return {}
    return nx.pagerank(H, alpha=0.85, weight="weight")


def algo_clusters_and_themes(S: nx.DiGraph) -> Dict[str, Any]:
    if S.number_of_nodes() == 0:
        return {"status": "empty", "clusters": []}

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

    id_to_text = {n: S.nodes[n].get("text", "") for n in S.nodes()}
    docs = [" ".join([id_to_text.get(n, "") for n in cl]) for cl in clusters]
    themes = []

    if docs:
        vec = TfidfVectorizer(stop_words=STOP_WORDS, max_features=MAX_FEATURES)
        X = vec.fit_transform(docs)
        feats = vec.get_feature_names_out()
        for i, cl in enumerate(clusters):
            row = X[i].toarray().ravel()
            top = row.argsort()[::-1][:TOP_K_KEYWORDS]
            kw = [feats[j] for j in top if row[j] > 0]
            themes.append({"cluster_id": i, "size": len(cl), "keywords": kw})

    node_cluster = {}
    for i, cl in enumerate(clusters):
        for n in cl:
            node_cluster[n] = i

    return {"status": "ok", "clusters": clusters, "themes": themes, "node_cluster": node_cluster}


# ---- Min-cut helpers (per cluster) ----
def _undirected_capacity_graph(D: nx.DiGraph, weight_attr: str = "score") -> nx.Graph:
    U = nx.Graph()
    U.add_nodes_from(D.nodes(data=True))
    for u, v, a in D.edges(data=True):
        cap = float(a.get(weight_attr, 1.0))
        if U.has_edge(u, v):
            U[u][v]["capacity"] += cap
        else:
            U.add_edge(u, v, capacity=cap)
    return U


def _mincut_on_graph_undirected(U: nx.Graph) -> Dict[str, Any]:
    if U.number_of_nodes() == 0:
        return {"status": "empty"}

    comps = list(nx.connected_components(U))
    comps_sorted = sorted(comps, key=len, reverse=True)
    out = {"status": "ok", "components": []}

    for comp in comps_sorted:
        if len(comp) < 2:
            continue
        Uc = U.subgraph(comp).copy()

        arts = list(nx.articulation_points(Uc))
        cut_value, partition = nx.stoer_wagner(Uc, weight="capacity")
        A, B = partition

        cut_edges = []
        for u, v, d in Uc.edges(data=True):
            if (u in A and v in B) or (u in B and v in A):
                cut_edges.append((u, v, float(d.get("capacity", 0.0))))

        out["components"].append({
            "nodes": int(len(comp)),
            "articulation_points": arts,
            "mincut_capacity": float(cut_value),
            "mincut_edges": [{"u": u, "v": v, "capacity": c} for (u, v, c) in sorted(cut_edges, key=lambda x: x[2])],
            "partition_sizes": {"A": int(len(A)), "B": int(len(B))},
        })

    return out


def mincut_every_subgraph_by_clusters(S: nx.DiGraph, clusters: List[List[str]]) -> Dict[str, Any]:
    results = {"status": "ok", "clusters": []}
    for cid, nodes in enumerate(clusters):
        if len(nodes) < 2:
            continue
        sub = S.subgraph(nodes).copy()
        U = _undirected_capacity_graph(sub, weight_attr="score")
        info = _mincut_on_graph_undirected(U)
        results["clusters"].append({
            "cluster_id": int(cid),
            "cluster_nodes": int(len(nodes)),
            "support_edges_in_cluster": int(sub.number_of_edges()),
            "mincut": info,
        })
    return results


# -----------------------
# Visualization core
# -----------------------
def _set_layout_options(net: Network):
    # Do NOT call net.show_buttons() (pyvis bug with set_options dict)
    if LAYOUT == "hierarchical":
        net.set_options(f"""
        {{
          "layout": {{
            "hierarchical": {{
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "levelSeparation": 220,
              "nodeSpacing": 220,
              "treeSpacing": 260,
              "edgeMinimization": true
            }}
          }},
          "physics": {{ "enabled": false }},
          "interaction": {{
            "hover": true,
            "tooltipDelay": 120,
            "navigationButtons": true,
            "keyboard": true,
            "hideEdgesOnDrag": true
          }}
        }}
        """)
    else:
        net.set_options(f"""
        {{
          "physics": {{
            "enabled": true,
            "stabilization": {{ "enabled": true, "iterations": {int(STABILIZE_ITERS)} }},
            "barnesHut": {{
              "gravitationalConstant": -22000,
              "springLength": 280,
              "springConstant": 0.015,
              "damping": 0.12,
              "avoidOverlap": 0.80
            }}
          }},
          "interaction": {{
            "hover": true,
            "tooltipDelay": 120,
            "navigationButtons": true,
            "keyboard": true,
            "hideEdgesOnDrag": true
          }}
        }}
        """)


def make_net_base(height="900px") -> Network:
    net = Network(height=height, width="100%", bgcolor="#ffffff", font_color="#111111", directed=True)
    _set_layout_options(net)
    return net


def add_edges_support_attack(
    net: Network,
    G: nx.DiGraph,
    *,
    edge_override: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None
):
    edge_override = edge_override or {}
    for u, v, a in G.edges(data=True):
        lab = a.get("label", "")
        score = float(a.get("score", 0.0))

        color = EDGE_COLOR.get(lab, "#000000")
        width = 2.0 + 5.5 * max(0.0, min(1.0, score))

        o = edge_override.get((u, v))
        if o:
            color = o.get("color", color)
            width = o.get("width", width)

        net.add_edge(
            u, v,
            label=(lab if EDGE_LABELS else ""),
            title=f"{lab} | score={score:.3f}",
            color=color,
            width=width,
            arrows="to",
            smooth={"type": "dynamic"},
            font={"size": 16, "strokeWidth": 5, "strokeColor": "#ffffff"},
        )


def write_html(net: Network, out_html: str, legend_title: str, legend_lines: List[str]):
    net.write_html(out_html)
    _inject_label_slider(out_html)
    _inject_legend(out_html, legend_title, legend_lines)


# -----------------------
# Visualizers
# -----------------------
def viz_support_indegree(S: nx.DiGraph, out_html: str):
    indeg = algo_support_indegree(S)
    net = make_net_base()

    if not indeg:
        write_html(net, out_html, "Support in-degree (Support graph)", legend_lines_support_indegree())
        return

    vals = list(indeg.values())
    vmin, vmax = min(vals), max(vals)

    for n, a in S.nodes(data=True):
        text = a.get("text", "")
        k = indeg.get(n, 0)

        size = _scale(k, vmin, vmax, 16, 42)
        shade = int(_scale(k, vmin, vmax, 240, 60))  # light -> dark
        color = f"rgb({shade},{shade},{shade})"

        net.add_node(
            n,
            label=_shorten_words(text, DEFAULT_LABEL_WORDS) or n,
            title=f"<b>support_in_degree={k}</b><br>{text}",
            size=float(size),
            shape="dot",
            color=color,
        )

    add_edges_support_attack(net, S)
    write_html(net, out_html, "Support in-degree (Support graph)", legend_lines_support_indegree())


def viz_pagerank_custom(H: nx.DiGraph, scores: Dict[str, float], out_html: str, title: str, legend_lines: List[str]):
    net = make_net_base()
    vals = list(scores.values()) if scores else [0.0]
    vmin, vmax = min(vals), max(vals)

    for n, a in H.nodes(data=True):
        text = a.get("text", "")
        val = float(scores.get(n, 0.0))
        size = _scale(val, vmin, vmax, 16, 46)
        intensity = int(_scale(val, vmin, vmax, 220, 40))
        color = f"rgb(60,90,{intensity})"

        net.add_node(
            n,
            label=_shorten_words(text, DEFAULT_LABEL_WORDS) or n,
            title=f"<b>{title}: {val:.6f}</b><br>{text}",
            size=float(size),
            shape="dot",
            color=color,
        )

    add_edges_support_attack(net, H)
    write_html(net, out_html, title, legend_lines)


def viz_net_influence(G: nx.DiGraph, net_scores: Dict[str, float], out_html: str):
    net = make_net_base()
    vals = list(net_scores.values()) if net_scores else [0.0]
    vmin, vmax = min(vals), max(vals)
    denom = max(abs(vmin), abs(vmax), 1e-12)

    for n, a in G.nodes(data=True):
        text = a.get("text", "")
        val = float(net_scores.get(n, 0.0))

        size = _scale(abs(val), 0.0, denom, 16, 46)
        color = "#2ca02c" if val > 0 else ("#d62728" if val < 0 else "#bbbbbb")

        net.add_node(
            n,
            label=_shorten_words(text, DEFAULT_LABEL_WORDS) or n,
            title=f"<b>net(PR_support - PR_attack)={val:.6f}</b><br>{text}",
            size=float(size),
            shape="dot",
            color=color,
        )

    add_edges_support_attack(net, G)
    write_html(net, out_html, "Net influence (PR_support - PR_attack)", legend_lines_pagerank_net())


def viz_pagerank_support_and_attack(G: nx.DiGraph, out_dir: str):
    debate_id = G.graph.get("debate_id", "debate")

    pr_sup = pagerank_on_label_subgraph(G, "support")
    pr_att = pagerank_on_label_subgraph(G, "attack")

    S = support_subgraph(G)
    A = attack_subgraph(G)

    viz_pagerank_custom(
        S, pr_sup,
        os.path.join(out_dir, f"{debate_id}_pagerank_support.html"),
        title="PageRank (Support graph)",
        legend_lines=legend_lines_pagerank_support()
    )
    viz_pagerank_custom(
        A, pr_att,
        os.path.join(out_dir, f"{debate_id}_pagerank_attack.html"),
        title="PageRank (Attack graph)",
        legend_lines=legend_lines_pagerank_attack()
    )

    net_scores = {n: float(pr_sup.get(n, 0.0) - pr_att.get(n, 0.0)) for n in G.nodes()}
    viz_net_influence(G, net_scores, os.path.join(out_dir, f"{debate_id}_pagerank_net.html"))


def viz_clusters(S: nx.DiGraph, out_html: str):
    out = algo_clusters_and_themes(S)
    net = make_net_base()

    if out.get("status") != "ok":
        write_html(net, out_html, "Graph clustering + themes (Support graph)", legend_lines_clusters())
        return

    node_cluster = out["node_cluster"]
    themes = out.get("themes", [])

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"
    ]
    theme_map = {t["cluster_id"]: t.get("keywords", []) for t in themes}

    deg = dict(S.degree())
    dmin, dmax = (min(deg.values()), max(deg.values())) if deg else (0, 0)

    for n, a in S.nodes(data=True):
        text = a.get("text", "")
        cid = int(node_cluster.get(n, -1))
        color = palette[cid % len(palette)] if cid >= 0 else "#cccccc"

        size = _scale(deg.get(n, 0), dmin, dmax, 16, 40)
        kws = theme_map.get(cid, [])
        theme_line = ", ".join(kws[:8]) if kws else ""

        net.add_node(
            n,
            label=_shorten_words(text, DEFAULT_LABEL_WORDS) or n,
            title=f"<b>cluster={cid}</b><br><b>keywords:</b> {theme_line}<br>{text}",
            size=float(size),
            shape="dot",
            color=color,
        )

    add_edges_support_attack(net, S)
    write_html(net, out_html, "Graph clustering + themes (Support graph)", legend_lines_clusters())

    side_json = out_html.replace(".html", "_themes.json")
    with open(side_json, "w", encoding="utf-8") as f:
        json.dump({"themes": themes, "clusters": out.get("clusters", [])}, f, indent=2)

# Replace viz_mincut_all_clusters(S, out_dir) with this ONE-FILE version.
# It creates:
#  - <debate>_mincut_clusters.html   (single HTML showing all clusters)
#  - <debate>_mincut_all_clusters.json (summary)
#
# Visualization rules:
#  - Node color = cluster color (same palette as clustering viz)
#  - Articulation points (within their cluster) get ORANGE border + larger size
#  - Min-cut edges (within their cluster) are highlighted ORANGE + thicker
#  - Normal support edges remain green
#
# Drop-in: paste this function into your file and in main() call:
#   viz_mincut_all_clusters_one_html(S, OUT_DIR)

def viz_mincut_all_clusters_one_html(S: nx.DiGraph, out_dir: str):
    debate_id = S.graph.get("debate_id", "debate")
    out = algo_clusters_and_themes(S)
    if out.get("status") != "ok":
        return

    clusters = out["clusters"]
    node_cluster = out["node_cluster"]

    # Compute min-cut per cluster
    results = mincut_every_subgraph_by_clusters(S, clusters)

    # Save JSON summary
    with open(os.path.join(out_dir, f"{debate_id}_mincut_all_clusters.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Collect highlights across ALL clusters
    # - articulation points union
    # - mincut edges union
    art_points = set()
    cut_edges = set()

    for item in results["clusters"]:
        cid = item["cluster_id"]
        comps = item["mincut"].get("components", [])
        if not comps:
            continue
        comp0 = comps[0]  # largest component inside cluster

        for n in comp0.get("articulation_points", []):
            art_points.add(n)

        for e in comp0.get("mincut_edges", []):
            u, v = e["u"], e["v"]
            cut_edges.add((u, v))
            cut_edges.add((v, u))  # treat as undirected highlight

    # Cluster palette (same as viz_clusters)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78"
    ]

    # Build ONE visualization
    net = make_net_base()

    # Degree for sizing baseline
    deg = dict(S.degree())
    dmin, dmax = (min(deg.values()), max(deg.values())) if deg else (0, 0)

    for n, a in S.nodes(data=True):
        text = a.get("text", "")
        cid = int(node_cluster.get(n, -1))
        base_color = palette[cid % len(palette)] if cid >= 0 else "#cccccc"

        size = _scale(deg.get(n, 0), dmin, dmax, 16, 38)

        # Highlight articulation points with orange border + bigger size
        if n in art_points:
            net.add_node(
                n,
                label=_shorten_words(text, DEFAULT_LABEL_WORDS) or n,
                title=f"<b>ARTICULATION POINT (cluster {cid})</b><br>{text}",
                size=float(size + 10),
                shape="dot",
                color={
                    "background": base_color,
                    "border": "#ff7f0e",
                    "highlight": {"background": base_color, "border": "#ff7f0e"},
                },
                borderWidth=5,
            )
        else:
            net.add_node(
                n,
                label=_shorten_words(text, DEFAULT_LABEL_WORDS) or n,
                title=f"<b>cluster {cid}</b><br>{text}",
                size=float(size),
                shape="dot",
                color=base_color,
            )

    # Edge overrides: highlight min-cut edges orange and thick
    overrides = {}
    for u, v in S.edges():
        if (u, v) in cut_edges:
            overrides[(u, v)] = {"color": "#ff7f0e", "width": 8.0}

    add_edges_support_attack(net, S, edge_override=overrides)

    out_html = os.path.join(out_dir, f"{debate_id}_mincut_clusters.html")
    write_html(net, out_html, "Min-cut vulnerability (per cluster, Support graph)", [
        "<b>Node color</b>: cluster/community in support graph.",
        "<b>Orange border</b>: articulation points within their cluster (node bottlenecks).",
        "<b>Orange thick edges</b>: min-cut edges within their cluster (weakest links).",
        "Min-cut computed per cluster on undirected projection (capacity = confidence score).",
    ])


# ---- In main(), replace the old call:
# viz_mincut_all_clusters(S, OUT_DIR)
# with:
# viz_mincut_all_clusters_one_html(S, OUT_DIR)

# -----------------------
# Main
# -----------------------
def main():
    _ensure_dir(OUT_DIR)

    nodes_df, edges_df = load_data()
    debate_id = pick_single_debate(edges_df)
    print("Using ONE debate:", debate_id)

    G = build_combined_graph(nodes_df, edges_df, debate_id)
    S = support_subgraph(G)

    # 1) Support in-degree (support graph)
    viz_support_indegree(S, os.path.join(OUT_DIR, f"{debate_id}_support_indegree.html"))

    # 2) PageRank for support + attack + net influence (combined)
    viz_pagerank_support_and_attack(G, OUT_DIR)

    # 3) Min-cut on EVERY support cluster subgraph (writes many htmls + one json)
    viz_mincut_all_clusters_one_html(S, OUT_DIR)

    # 4) Clustering + themes (support graph)
    viz_clusters(S, os.path.join(OUT_DIR, f"{debate_id}_clusters.html"))

    print("Wrote algorithm visualizers to:", OUT_DIR)
    print("Support graph nodes/edges:", S.number_of_nodes(), S.number_of_edges())



if __name__ == "__main__":
    main()
