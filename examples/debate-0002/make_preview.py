"""Render a readable excerpt of the included debate graph for the README."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = Path(__file__).resolve().parent
GRAPH_DIR = ROOT / "outputs" / "pipeline" / "graphs"
DEBATE_ID = "node_link_data-debate-eval-0002"


def main() -> None:
    nodes = pd.read_csv(GRAPH_DIR / "nodes.tsv", sep="\t").astype({"claim_id": str})
    edges = pd.read_csv(GRAPH_DIR / "edges.tsv", sep="\t").astype({"src_id": str, "tgt_id": str})
    edges = edges[
        (edges["debate_id"] == DEBATE_ID)
        & (edges["pred_label"].isin(["support", "attack"]))
        & (edges["pred_score"] >= 0.55)
    ]

    graph = nx.from_pandas_edgelist(edges, "src_id", "tgt_id", edge_attr="pred_label", create_using=nx.DiGraph)
    hub = max(graph.degree, key=lambda item: item[1])[0]
    neighbors = set(graph.predecessors(hub)) | set(graph.successors(hub))
    chosen = [hub, *sorted(neighbors)]
    excerpt = graph.subgraph(chosen).copy()
    claim_text = nodes[nodes["debate_id"] == DEBATE_ID].set_index("claim_id")["text"].to_dict()
    labels = {claim_id: str(index + 1) for index, claim_id in enumerate(chosen)}

    fig = plt.figure(figsize=(14, 7.8), facecolor="#f8fafc")
    ax = fig.add_axes([0.04, 0.13, 0.52, 0.72])
    ax.set_facecolor("#f8fafc")
    positions = nx.spring_layout(excerpt, seed=17, k=1.8)
    edge_colors = ["#12805c" if excerpt[u][v]["pred_label"] == "support" else "#d14343" for u, v in excerpt.edges]
    nx.draw_networkx_edges(excerpt, positions, ax=ax, edge_color=edge_colors, width=2.5, arrows=True, arrowsize=20, node_size=1200, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(excerpt, positions, ax=ax, nodelist=[hub], node_color="#1d4ed8", node_size=1500)
    nx.draw_networkx_nodes(excerpt, positions, ax=ax, nodelist=chosen[1:], node_color="#dbeafe", edgecolors="#3b82f6", linewidths=1.5, node_size=1100)
    nx.draw_networkx_labels(excerpt, positions, labels=labels, ax=ax, font_size=13, font_weight="bold", font_color="#0f172a")
    ax.set_axis_off()

    fig.text(0.04, 0.94, "How claims connect", fontsize=24, weight="bold", color="#0f172a")
    fig.text(0.04, 0.89, "One high-connection claim and its immediate neighbors from debate 0002", fontsize=12, color="#475569")
    fig.legend(handles=[Line2D([0], [0], color="#12805c", lw=3, label="Support"), Line2D([0], [0], color="#d14343", lw=3, label="Attack")], loc="lower left", bbox_to_anchor=(0.04, 0.025), ncol=2, frameon=False, fontsize=11)

    for index, claim_id in enumerate(chosen):
        preview = textwrap.shorten(str(claim_text.get(claim_id, "")), width=73, placeholder="…")
        y = 0.80 - index * 0.09
        fig.text(0.61, y, f"{index + 1}.", fontsize=11, weight="bold", color="#1d4ed8")
        fig.text(0.64, y, textwrap.fill(preview, width=45), fontsize=10, color="#334155", va="top")

    fig.savefig(EXAMPLE / "graph.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
