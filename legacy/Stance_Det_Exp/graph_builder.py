# graph_builder.py
# Produces 3 HTMLs for ONE debate:
#  1) combined (support + attack)
#  2) attack-only
#  3) support-only
#
# Also adds an on-page slider to EXPAND/SHRINK node label text length (words shown).
# Full claim text is always available on hover.
#
# Yes: graphs are built from ONE debate (selected by pick_single_debate()).

import os
import re
import pandas as pd
import networkx as nx
from pyvis.network import Network

# -----------------------
# CONFIG (EDIT HERE)
# -----------------------
NODES_PATH = "out_graphs/nodes.tsv"
EDGES_PATH = "out_graphs/edges.tsv"
OUT_DIR = "out_graphs_viz"

MIN_SCORE = 0.55
TOP_K_OUT = None  # None = keep all outgoing edges
KEEP_LABELS = {"support", "attack"}

LAYOUT = "force"          # "force" spreads; "hierarchical" structured
STABILIZE_ITERS = 1800

DEFAULT_LABEL_WORDS = 7   # initial words shown (slider can change)
MAX_LABEL_WORDS = 40      # upper bound for slider

LABEL_COLORS = {"support": "#2ca02c", "attack": "#d62728", "neutral": "#7f7f7f"}


def load_data():
    nodes = pd.read_csv(NODES_PATH, sep="\t")
    edges = pd.read_csv(EDGES_PATH, sep="\t")

    nodes["debate_id"] = nodes["debate_id"].astype(str)
    nodes["claim_id"] = nodes["claim_id"].astype(str)

    edges["debate_id"] = edges["debate_id"].astype(str)
    edges["src_id"] = edges["src_id"].astype(str)
    edges["tgt_id"] = edges["tgt_id"].astype(str)
    edges["pred_score"] = pd.to_numeric(edges["pred_score"], errors="coerce").fillna(0.0)
    edges["pred_label"] = edges["pred_label"].astype(str)

    return nodes, edges


def pick_single_debate(edges_df: pd.DataFrame) -> str:
    # pick debate with most (support/attack) edges -> informative
    tmp = edges_df[edges_df["pred_label"].isin(list(KEEP_LABELS))]
    if len(tmp) == 0:
        return edges_df["debate_id"].value_counts().idxmax()
    return tmp["debate_id"].value_counts().idxmax()


def _shorten_words(text: str, k: int) -> str:
    words = (text or "").strip().split()
    if not words:
        return ""
    s = " ".join(words[:k])
    return s + ("…" if len(words) > k else "")


def build_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, debate_id: str, keep_labels: set) -> nx.DiGraph:
    ndf = nodes_df[nodes_df["debate_id"] == debate_id].copy()
    edf = edges_df[edges_df["debate_id"] == debate_id].copy()

    # filter edges: score + labels
    edf = edf[edf["pred_score"] >= float(MIN_SCORE)]
    edf = edf[edf["pred_label"].isin(list(keep_labels))]

    if TOP_K_OUT is not None:
        edf = (
            edf.sort_values(["src_id", "pred_score"], ascending=[True, False])
               .groupby("src_id", as_index=False)
               .head(int(TOP_K_OUT))
        )

    # build graph with all nodes first
    G = nx.DiGraph(debate_id=debate_id)
    for _, r in ndf.iterrows():
        G.add_node(r["claim_id"], text=str(r.get("text", "")))

    # add edges
    for _, r in edf.iterrows():
        u, v = r["src_id"], r["tgt_id"]
        if u == v:
            continue
        if u in G and v in G:
            G.add_edge(u, v, label=r["pred_label"], score=float(r["pred_score"]))

    # remove isolated nodes after filtering
    isolates = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    G.remove_nodes_from(isolates)

    return G


def _set_layout_options(net: Network):
    # Do NOT call net.show_buttons() (pyvis bug when set_options turns options into dict)
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
              "springLength": 260,
              "springConstant": 0.015,
              "damping": 0.12,
              "avoidOverlap": 0.70
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


def _inject_label_slider(html_path: str):
    """
    Adds a slider UI to expand/contract node label text.
    Implementation: store full text in node.title and node.__fulltext,
    then replace node labels on slider input.

    This is a simple string injection into the exported HTML.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Only inject once
    if "id=\"labelSlider\"" in html:
        return

    # Insert slider panel near body start
    slider_panel = f"""
<div style="position:fixed; top:12px; right:12px; z-index:9999; background:#fff; border:1px solid #ddd;
            border-radius:10px; padding:10px 12px; box-shadow:0 6px 18px rgba(0,0,0,0.08); font-family:Arial;">
  <div style="font-size:13px; font-weight:600; margin-bottom:6px;">Node label length</div>
  <input id="labelSlider" type="range" min="3" max="{MAX_LABEL_WORDS}" value="{DEFAULT_LABEL_WORDS}" step="1" style="width:220px;">
  <span id="labelVal" style="margin-left:8px; font-size:12px;">{DEFAULT_LABEL_WORDS} words</span>
  <div style="margin-top:6px; font-size:11px; color:#555;">
    Hover nodes for full text.
  </div>
</div>
"""

    # Insert right after <body> tag
    html = re.sub(r"(<body[^>]*>)", r"\1\n" + slider_panel, html, count=1)

    # Add JS hook: after network is created, modify labels on slider changes.
    # PyVis defines: var nodes = new vis.DataSet([...]); var edges = new vis.DataSet([...]); var network = new vis.Network(...)
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
    // nodes is a vis.DataSet in pyvis output
    if (typeof nodes === "undefined" || !nodes.get) return false;

    // store full text for each node (from title HTML -> strip tags)
    var all = nodes.get();
    all.forEach(function(n) {
      // pyvis "title" may contain HTML; keep as-is for hover, but also store plain text
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

    // initial apply (ensure consistent)
    apply(parseInt(slider.value, 10));
    return true;
  }

  // retry a few times until pyvis globals exist
  var tries = 0;
  var t = setInterval(function() {
    tries += 1;
    if (tryInit() || tries > 40) clearInterval(t);
  }, 100);
})();
</script>
"""

    # Insert before closing </body>
    html = html.replace("</body>", js_hook + "\n</body>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def visualize(G: nx.DiGraph, out_html: str, title: str):
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#111111",
        directed=True,
    )

    _set_layout_options(net)

    # node size by degree
    deg = dict(G.degree())
    dmin, dmax = (min(deg.values()), max(deg.values())) if deg else (0, 0)

    def node_size(n: str) -> int:
        if dmin == dmax:
            return 18
        return int(14 + 26 * (deg[n] - dmin) / (dmax - dmin))

    # nodes: label is short initially, hover shows full text
    for n, a in G.nodes(data=True):
        full = str(a.get("text", ""))
        label = _shorten_words(full, DEFAULT_LABEL_WORDS) or n
        net.add_node(
            n,
            label=label,
            title=full,
            size=node_size(n),
            shape="dot",
        )

    # edges: curved + thick + big arrows + white stroke behind label
    for u, v, a in G.edges(data=True):
        lab = a.get("label", "")
        score = float(a.get("score", 0.0))
        color = LABEL_COLORS.get(lab, "#000000")
        width = 2.0 + 5.5 * max(0.0, min(1.0, score))

        net.add_edge(
            u, v,
            label=lab,  # always show attack/support
            title=f"{lab} | score={score:.3f}",
            color=color,
            width=width,
            arrows="to",
            smooth={"type": "dynamic"},
            font={"size": 16, "strokeWidth": 5, "strokeColor": "#ffffff"},
        )

    # write html
    net.write_html(out_html)

    # inject slider UI
    _inject_label_slider(out_html)

    print(f"Wrote {out_html}")
    print(f"{title} | Debate: {G.graph.get('debate_id')} | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    nodes_df, edges_df = load_data()
    debate_id = pick_single_debate(edges_df)

    # YES: ONE debate only
    print("Using ONE debate:", debate_id)

    # 1) combined
    G_all = build_graph(nodes_df, edges_df, debate_id, keep_labels={"support", "attack"})
    visualize(G_all, os.path.join(OUT_DIR, f"{debate_id}_combined.html"), "COMBINED (support+attack)")

    # 2) attack only
    G_att = build_graph(nodes_df, edges_df, debate_id, keep_labels={"attack"})
    visualize(G_att, os.path.join(OUT_DIR, f"{debate_id}_attack_only.html"), "ATTACK ONLY")

    # 3) support only
    G_sup = build_graph(nodes_df, edges_df, debate_id, keep_labels={"support"})
    visualize(G_sup, os.path.join(OUT_DIR, f"{debate_id}_support_only.html"), "SUPPORT ONLY")


if __name__ == "__main__":
    main()
