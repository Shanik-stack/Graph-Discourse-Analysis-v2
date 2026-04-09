# benchmark_with_graph_outputs.py
"""
Benchmark + save outputs for graph visualization.

What this script saves (TSV):
  out_graphs/nodes.tsv : debate_id, claim_id, text
  out_graphs/edges.tsv : debate_id, src_id, tgt_id, gold_label, pred_label, pred_score

Notes:
- This uses the ORIGINAL graph direction from the dataset:
    src -> tgt with valence PRO/CON mapped to support/attack.
- For NLI/Stance models that take (premise, hypothesis), we keep your prompt format:
    premise   = "TARGET CLAIM: <tgt text>"
    hypothesis= "CURRENT CLAIM: <src text>"
  but we still log edges as src_id -> tgt_id so visualization matches dataset orientation.
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from huggingface_hub import HfApi
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from nli_relations import StanceRelation, DeBerta

# ---- disable SDPA/flash (your settings) ----
os.environ["TORCH_SDPA"] = "0"
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
try:
    torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass


# -----------------------------
# Graph logger (for visualizer)
# -----------------------------
class GraphResultLogger:
    """
    Streaming-safe logger.
    Creates:
      - nodes.tsv: debate_id, claim_id, text
      - edges.tsv: debate_id, src_id, tgt_id, gold_label, pred_label, pred_score
    """
    def __init__(self, out_dir: str = "out_graphs"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.nodes_path = os.path.join(out_dir, "nodes.tsv")
        self.edges_path = os.path.join(out_dir, "edges.tsv")

        self._nodes_fp = None
        self._edges_fp = None
        self._nodes_writer = None
        self._edges_writer = None

        self._seen_nodes = set()  # (debate_id, claim_id)

        self._open()

    def _open(self):
        new_nodes = not os.path.exists(self.nodes_path)
        new_edges = not os.path.exists(self.edges_path)

        self._nodes_fp = open(self.nodes_path, "a", newline="", encoding="utf-8")
        self._edges_fp = open(self.edges_path, "a", newline="", encoding="utf-8")

        self._nodes_writer = csv.DictWriter(
            self._nodes_fp, delimiter="\t",
            fieldnames=["debate_id", "claim_id", "text"],
        )
        self._edges_writer = csv.DictWriter(
            self._edges_fp, delimiter="\t",
            fieldnames=["debate_id", "src_id", "tgt_id", "gold_label", "pred_label", "pred_score"],
        )

        if new_nodes:
            self._nodes_writer.writeheader()
            self._nodes_fp.flush()
        if new_edges:
            self._edges_writer.writeheader()
            self._edges_fp.flush()

    def log_node(self, debate_id: str, claim_id: str, text: str):
        key = (str(debate_id), str(claim_id))
        if key in self._seen_nodes:
            return
        self._seen_nodes.add(key)
        self._nodes_writer.writerow({
            "debate_id": str(debate_id),
            "claim_id": str(claim_id),
            "text": (text or "").replace("\r\n", "\n").replace("\r", "\n"),
        })
        self._nodes_fp.flush()

    def log_edge(
        self,
        debate_id: str,
        src_id: str,
        tgt_id: str,
        gold_label: str,
        pred_label: str,
        pred_score: float,
    ):
        self._edges_writer.writerow({
            "debate_id": str(debate_id),
            "src_id": str(src_id),
            "tgt_id": str(tgt_id),
            "gold_label": str(gold_label),
            "pred_label": str(pred_label),
            "pred_score": float(pred_score),
        })
        self._edges_fp.flush()

    def close(self):
        try:
            if self._nodes_fp:
                self._nodes_fp.close()
            if self._edges_fp:
                self._edges_fp.close()
        finally:
            self._nodes_fp = None
            self._edges_fp = None


# -----------------------------
# Download split locally
# -----------------------------
def snapshot_download_syncialo(
    repo_id: str = "DebateLabKIT/syncialo-raw",
    corpus_id: str = "synthetic_corpus-001",
    split: str = "eval",
    local_dir: str = "syncialo_snapshot",
) -> str:
    hfapi = HfApi()
    hfapi.snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"data/{corpus_id}/{split}/**/*.json",
        local_dir=local_dir,
    )
    return local_dir


def iter_files(local_dir: str, corpus_id: str, split: str) -> List[Path]:
    root = Path(local_dir)
    return list(root.glob(f"data/{corpus_id}/{split}/**/*.json"))


# -----------------------------
# Parse THIS schema
# -----------------------------
def load_debate_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _node_claim_text(n: Dict) -> str:
    if isinstance(n.get("claim"), str):
        return n["claim"]
    data = n.get("data", {})
    if isinstance(data, dict) and isinstance(data.get("claim"), str):
        return data["claim"]
    return ""


def _node_id(n: Dict) -> Optional[str]:
    if isinstance(n.get("id"), str):
        return n["id"]
    data = n.get("data", {})
    if isinstance(data, dict) and isinstance(data.get("id"), str):
        return data["id"]
    return None


def parse_nodes_and_links(obj: Dict) -> Tuple[Dict[str, str], List[Tuple[str, str, str]]]:
    """
    Returns:
      claims: dict[node_id] = claim_text
      edges: list[(source_id, target_id, gold_label)] where gold_label in {"support","attack"}
    """
    nodes = obj.get("nodes") or obj.get("data", {}).get("nodes") or obj.get("elements", {}).get("nodes")
    links = obj.get("links") or obj.get("data", {}).get("links") or obj.get("elements", {}).get("edges")
    if nodes is None or links is None:
        raise KeyError(f"Missing 'nodes' or 'links'. Top-level keys: {list(obj.keys())}")

    claims: Dict[str, str] = {}
    for n in nodes:
        nid = _node_id(n)
        txt = _node_claim_text(n).strip()
        if nid and txt:
            claims[nid] = txt

    edges: List[Tuple[str, str, str]] = []
    for e in links:
        ed = e.get("data", e) if isinstance(e, dict) else {}
        src = ed.get("source")
        tgt = ed.get("target")
        val = str(ed.get("valence", "")).strip().upper()
        if not src or not tgt:
            continue
        if src not in claims or tgt not in claims:
            continue
        if val == "PRO":
            edges.append((src, tgt, "support"))
        elif val == "CON":
            edges.append((src, tgt, "attack"))

    return claims, edges


# -----------------------------
# Pair generation (within debate)
# -----------------------------
def sample_non_edges(
    node_ids: List[str],
    edge_set: set,
    n_samples: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    out = set()
    max_tries = n_samples * 500
    tries = 0
    while len(out) < n_samples and tries < max_tries:
        u = rng.choice(node_ids)
        v = rng.choice(node_ids)
        tries += 1
        if u == v:
            continue
        if (u, v) in edge_set:
            continue
        out.add((u, v))
    return list(out)


def build_edges_with_neutrals(
    claims: Dict[str, str],
    edges: List[Tuple[str, str, str]],
    neutrals_per_pos: float,
    rng: random.Random,
) -> List[Tuple[str, str, str]]:
    """
    Returns edge list with (src_id, tgt_id, gold_label) where gold_label includes neutral samples.
    """
    pos = [(src, tgt, y) for src, tgt, y in edges]
    if not pos:
        return []

    node_ids = list(claims.keys())
    edge_set = set((src, tgt) for src, tgt, _ in edges)

    n_neg = int(len(pos) * neutrals_per_pos)
    neg_uv = sample_non_edges(node_ids, edge_set, n_neg, rng)
    neg = [(u, v, "neutral") for (u, v) in neg_uv]

    all_edges = pos + neg
    rng.shuffle(all_edges)
    return all_edges


def build_debate_edge_dataset(
    local_dir: str,
    corpus_id: str = "synthetic_corpus-001",
    split: str = "eval",
    neutrals_per_pos: float = 1.0,
    max_edges: int = 5000,
    max_debates: Optional[int] = 30,
    seed: int = 0,
) -> List[Tuple[str, str, str, str, str]]:
    """
    Returns list of:
      (debate_id, src_id, tgt_id, premise, hypothesis, gold_label)
    """
    rng = random.Random(seed)
    files = iter_files(local_dir, corpus_id, split)
    rng.shuffle(files)

    out = []
    used = 0
    for p in files:
        if max_debates is not None and used >= max_debates:
            break

        obj = load_debate_json(p)
        claims, gold_edges = parse_nodes_and_links(obj)

        all_edges = build_edges_with_neutrals(claims, gold_edges, neutrals_per_pos, rng)
        if not all_edges:
            continue

        debate_id = p.stem

        # emit nodes (for visualization)
        # (caller may log them; we keep claims here for prompt construction)
        for src_id, tgt_id, gold in all_edges:
            premise = f"TARGET CLAIM: {claims[tgt_id]}"
            hypothesis = f"CURRENT CLAIM: {claims[src_id]}"
            out.append((debate_id, src_id, tgt_id, premise, hypothesis, gold))

        used += 1
        if len(out) >= max_edges:
            break

    rng.shuffle(out)
    return out[:max_edges]


# -----------------------------
# Benchmark runners that also log edges/nodes
# -----------------------------
def run_benchmark_deberta_with_logs(
    dataset_rows: List[Tuple[str, str, str, str, str, str]],
    claims_by_debate: Dict[str, Dict[str, str]],
    *,
    out_dir: str = "out_graphs",
    model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    max_examples: int = 2000,
    p_min: float = 0.0,
) -> None:
    logger = GraphResultLogger(out_dir=out_dir)
    nli = DeBerta(model_name=model_name, max_length=256)

    y_true, y_pred = [], []

    try:
        for i, (debate_id, src_id, tgt_id, premise, hypothesis, gold) in enumerate(dataset_rows[:max_examples], start=1):
            # ensure nodes exist in nodes.tsv
            debate_claims = claims_by_debate[debate_id]
            logger.log_node(debate_id, src_id, debate_claims[src_id])
            logger.log_node(debate_id, tgt_id, debate_claims[tgt_id])

            scores = nli.scores(premise, hypothesis)
            pred, conf = nli.decide(scores, p_min=p_min)

            # save per-edge outputs
            logger.log_edge(debate_id, src_id, tgt_id, gold, pred, float(conf))

            y_true.append(gold)
            y_pred.append(pred)

            if i % 200 == 0:
                print(i)

    finally:
        logger.close()

        labels = ["support", "attack", "neutral"]
        print("N evaluated:", len(y_true))
        if y_true:
            print(confusion_matrix(y_true, y_pred, labels=labels))
            print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
            print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")


def run_benchmark_stance_with_logs(
    dataset_rows: List[Tuple[str, str, str, str, str, str]],
    claims_by_debate: Dict[str, Dict[str, str]],
    *,
    out_dir: str = "out_graphs",
    model_name: str = "krishnagarg09/stance-detection-semeval2016",
    max_examples: int = 2000,
) -> None:
    logger = GraphResultLogger(out_dir=out_dir)
    stance = StanceRelation(model_name, max_length=128)

    y_true, y_pred = [], []

    try:
        for i, (debate_id, src_id, tgt_id, premise, hypothesis, gold) in enumerate(dataset_rows[:max_examples], start=1):
            debate_claims = claims_by_debate[debate_id]
            logger.log_node(debate_id, src_id, debate_claims[src_id])
            logger.log_node(debate_id, tgt_id, debate_claims[tgt_id])

            scores = stance.scores(premise, hypothesis)
            pred, conf = stance.decide(scores)

            logger.log_edge(debate_id, src_id, tgt_id, gold, pred, float(conf))

            y_true.append(gold)
            y_pred.append(pred)

            if i % 200 == 0:
                print(i)

    finally:
        logger.close()

        labels = ["support", "attack", "neutral"]
        print("N evaluated:", len(y_true))
        if y_true:
            print(confusion_matrix(y_true, y_pred, labels=labels))
            print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
            print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    local_dir = snapshot_download_syncialo(
        corpus_id="synthetic_corpus-001",
        split="eval",
        local_dir="syncialo_snapshot",
    )

    # Build per-edge dataset (keeps src_id and tgt_id)
    rows = build_debate_edge_dataset(
        local_dir=local_dir,
        corpus_id="synthetic_corpus-001",
        split="eval",
        neutrals_per_pos=1.0,
        max_edges=5000,
        max_debates=30,
        seed=0,
    )

    print("Edges sampled:", len(rows))

    # Build claims_by_debate for logging nodes
    # (Re-scan debates used in rows; cheap and keeps logic simple)
    debate_ids = sorted(set(r[0] for r in rows))
    claims_by_debate: Dict[str, Dict[str, str]] = {}
    files = iter_files(local_dir, "synthetic_corpus-001", "eval")
    file_map = {p.stem: p for p in files}
    for did in debate_ids:
        obj = load_debate_json(file_map[did])
        claims, _ = parse_nodes_and_links(obj)
        claims_by_debate[did] = claims

    # Run ONE benchmark (choose)
    run_benchmark_deberta_with_logs(
        rows,
        claims_by_debate,
        out_dir="out_graphs",
        model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        max_examples=2000,
        p_min=0.0,
    )

    # Or stance:
    # run_benchmark_stance_with_logs(
    #     rows,
    #     claims_by_debate,
    #     out_dir="out_graphs",
    #     model_name="krishnagarg09/stance-detection-semeval2016",
    #     max_examples=2000,
    # )

    print("Wrote:")
    print("  out_graphs/nodes.tsv")
    print("  out_graphs/edges.tsv")


if __name__ == "__main__":
    main()
