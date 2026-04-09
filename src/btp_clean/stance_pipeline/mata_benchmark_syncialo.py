# benchmark_meta_nli_fast.py
import json
import random
import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib

import numpy as np
from huggingface_hub import HfApi

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from btp_clean.paths import ARTIFACTS_DIR

from .meta_nli import HFNLIModel, HFStanceModel


Pair = Tuple[str, str, str, str]  # (debate_id, premise, hypothesis, gold)


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
    def __init__(self, out_dir: str = str(ARTIFACTS_DIR / "stance_pipeline" / "out_graphs")):
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
    local_dir: str = str(ARTIFACTS_DIR / "stance_pipeline" / "syncialo_snapshot"),
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
# Parse schema (valence PRO/CON)
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
        val = str(ed.get("valence", "")).strip().upper()  # PRO/CON

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
# Pair generation
# -----------------------------
def sample_non_edges(node_ids: List[str], edge_set: set, n_samples: int, rng: random.Random) -> List[Tuple[str, str]]:
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


def build_pairs_from_debate(
    claims: Dict[str, str],
    edges: List[Tuple[str, str, str]],
    debate_id: str,
    neutrals_per_pos: float,
    rng: random.Random,
) -> List[Pair]:
    pos: List[Pair] = []
    for src, tgt, y in edges:
        pos.append((debate_id, claims[tgt], claims[src], y))

    if not pos:
        return []

    node_ids = list(claims.keys())
    edge_set = set((src, tgt) for src, tgt, _ in edges)

    n_neg = int(len(pos) * neutrals_per_pos)
    neg_uv = sample_non_edges(node_ids, edge_set, n_neg, rng)

    neg: List[Pair] = []
    for u, v in neg_uv:
        neg.append((debate_id, claims[v], claims[u], "neutral"))

    all_pairs = pos + neg
    rng.shuffle(all_pairs)
    return all_pairs


def build_dataset(
    local_dir: str,
    corpus_id: str = "synthetic_corpus-001",
    split: str = "eval",
    neutrals_per_pos: float = 1.0,
    max_pairs: int = 5000,
    max_debates: Optional[int] = 30,
    seed: int = 0,
) -> List[Pair]:
    rng = random.Random(seed)
    files = iter_files(local_dir, corpus_id, split)
    rng.shuffle(files)

    pairs: List[Pair] = []
    used = 0
    for p in files:
        if max_debates is not None and used >= max_debates:
            break

        obj = load_debate_json(p)
        claims, edges = parse_nodes_and_links(obj)
        debate_pairs = build_pairs_from_debate(claims, edges, debate_id=p.stem, neutrals_per_pos=neutrals_per_pos, rng=rng)

        if debate_pairs:
            pairs.extend(debate_pairs)
            used += 1
        if len(pairs) >= max_pairs:
            break

    rng.shuffle(pairs)
    return pairs[:max_pairs]


# -----------------------------
# Featurize (cache logits once)
# -----------------------------
LABEL2ID = {"support": 0, "attack": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def featurize_all(
    pairs: List[Pair],
    nli: HFNLIModel,
    stance: HFStanceModel,
    cache_path: str = str(ARTIFACTS_DIR / "stance_pipeline" / "meta_features.npz"),
    verbose_every: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cp = Path(cache_path)
    if cp.exists():
        data = np.load(cp, allow_pickle=True)
        return data["X"], data["y"], data["groups"]

    X = np.zeros((len(pairs), 6), dtype=np.float32)
    y = np.zeros((len(pairs),), dtype=np.int64)
    groups = np.empty((len(pairs),), dtype=object)

    for i, (debate_id, premise, hypothesis, gold) in enumerate(pairs):
        nli_l = nli.logits(premise, hypothesis)          # [3] contra, neutral, entail
        stance_l = stance.logits(premise, hypothesis)    # target=premise, claim=hypothesis -> [3] against, none, favor
        X[i, :3] = nli_l
        X[i, 3:] = stance_l
        y[i] = LABEL2ID[gold]
        groups[i] = debate_id

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"[featurize] {i+1}/{len(pairs)}")

    np.savez_compressed(cp, X=X, y=y, groups=groups)
    print(f"Saved cached features -> {cache_path}")
    return X, y, groups


# -----------------------------
# Train/eval only on cached logits
# -----------------------------
def train_eval_on_cached_logits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    seed: int = 0,
    lr_path: str = str(ARTIFACTS_DIR / "stance_pipeline" / "meta_lr.joblib"),
    force_retrain: bool = False,
) -> LogisticRegression:
    lr_file = Path(lr_path)

    splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    if lr_file.exists() and not force_retrain:
        payload = joblib.load(lr_file)
        clf: LogisticRegression = payload["clf"]
        saved = payload.get("meta", {})
        print(f"[meta] Loaded LR from: {lr_path}")
        if saved.get("seed") != seed or saved.get("test_size") != float(test_size):
            print(
                f"[meta] WARNING: loaded model was saved with seed={saved.get('seed')} "
                f"test_size={saved.get('test_size')} (current seed={seed}, test_size={test_size})"
            )
    else:
        Xtr, ytr = X[train_idx], y[train_idx]
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            multi_class="multinomial",
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr)

        payload = {
            "clf": clf,
            "meta": {
                "seed": int(seed),
                "test_size": float(test_size),
                "n_features": int(X.shape[1]),
                "label2id": LABEL2ID,
            },
        }
        joblib.dump(payload, lr_file)
        print(f"[meta] Saved LR to: {lr_path}")

    Xte, yte = X[test_idx], y[test_idx]
    pred = clf.predict(Xte)

    labels = [0, 1, 2]
    print("Accuracy:", float(accuracy_score(yte, pred)))
    print(confusion_matrix(yte, pred, labels=labels))
    print(
        classification_report(
            yte,
            pred,
            labels=labels,
            target_names=["support", "attack", "neutral"],
            digits=4,
            zero_division=0,
        )
    )

    return clf


# -----------------------------
# Graph logging: build full edge rows
# -----------------------------
EdgeRow = Tuple[str, str, str, str, str, str]  # debate_id, src_id, tgt_id, premise, hypothesis, gold_label


def build_edges_with_neutrals(
    claims: Dict[str, str],
    gold_edges: List[Tuple[str, str, str]],
    neutrals_per_pos: float,
    rng: random.Random,
) -> List[Tuple[str, str, str]]:
    """
    Returns list of (src_id, tgt_id, gold_label) including sampled neutrals (non-edges).
    """
    pos = [(src, tgt, lab) for (src, tgt, lab) in gold_edges]
    if not pos:
        return []

    node_ids = list(claims.keys())
    edge_set = set((src, tgt) for src, tgt, _ in gold_edges)

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
) -> Tuple[List[EdgeRow], Dict[str, Dict[str, str]]]:
    """
    Returns:
      rows: [(debate_id, src_id, tgt_id, premise, hypothesis, gold_label), ...]
      claims_by_debate: {debate_id: {claim_id: text}}
    """
    rng = random.Random(seed)
    files = iter_files(local_dir, corpus_id, split)
    rng.shuffle(files)

    rows: List[EdgeRow] = []
    claims_by_debate: Dict[str, Dict[str, str]] = {}

    used = 0
    for p in files:
        if max_debates is not None and used >= max_debates:
            break

        obj = load_debate_json(p)
        claims, gold_edges = parse_nodes_and_links(obj)
        if not claims:
            continue

        debate_id = p.stem
        claims_by_debate[debate_id] = claims

        all_edges = build_edges_with_neutrals(claims, gold_edges, neutrals_per_pos, rng)
        if not all_edges:
            continue

        for src_id, tgt_id, gold in all_edges:
            premise = claims[tgt_id]
            hypothesis = claims[src_id]
            rows.append((debate_id, src_id, tgt_id, premise, hypothesis, gold))
            if len(rows) >= max_edges:
                break

        used += 1
        if len(rows) >= max_edges:
            break

    rng.shuffle(rows)
    return rows[:max_edges], claims_by_debate


def featurize_graph_rows(
    rows: List[EdgeRow],
    nli: HFNLIModel,
    stance: HFStanceModel,
    cache_path: Optional[str] = None,
    max_examples: int = 2000,
    verbose_every: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build or reuse cached graph-eval features.

    Returns:
      X: (N, 6) feature matrix
      y: (N,) gold label ids
    """
    active_rows = rows[:max_examples]
    cp = Path(cache_path) if cache_path else None

    if cp is not None and cp.exists():
        data = np.load(cp, allow_pickle=True)
        return data["X"], data["y"]

    X = np.zeros((len(active_rows), 6), dtype=np.float32)
    y = np.zeros((len(active_rows),), dtype=np.int64)

    for i, (_, _, _, premise, hypothesis, gold) in enumerate(active_rows):
        nli_l = nli.logits(premise, hypothesis)
        stance_l = stance.logits(premise, hypothesis)
        X[i, :3] = nli_l
        X[i, 3:] = stance_l
        y[i] = LABEL2ID[gold]

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"[graph-featurize] {i+1}/{len(active_rows)}")

    if cp is not None:
        np.savez_compressed(cp, X=X, y=y)
        print(f"Saved cached graph-eval features -> {cache_path}")

    return X, y


def run_meta_with_graph_logs(
    rows: List[EdgeRow],
    claims_by_debate: Dict[str, Dict[str, str]],
    nli: HFNLIModel,
    stance: HFStanceModel,
    lr_path: str = "meta_lr.joblib",
    out_dir: str = str(ARTIFACTS_DIR / "stance_pipeline" / "out_graphs"),
    max_examples: int = 2000,
    graph_features_path: Optional[str] = None,
    force_retrain_meta: bool = False,
) -> None:
    """
    Uses cached LR (or trains it once from cached features) to predict on rows,
    prints metrics, and writes nodes.tsv / edges.tsv for visualization.
    """
    # Load meta classifier
    payload = joblib.load(lr_path)
    clf: LogisticRegression = payload["clf"]

    logger = GraphResultLogger(out_dir=out_dir)

    y_true: List[int] = []
    y_pred: List[int] = []

    # Log all nodes for debates appearing in rows (small, safe)
    for debate_id, claim_map in claims_by_debate.items():
        for cid, txt in claim_map.items():
            logger.log_node(debate_id, cid, txt)

    X, y_true_ids = featurize_graph_rows(
        rows=rows,
        nli=nli,
        stance=stance,
        cache_path=graph_features_path,
        max_examples=max_examples,
    )

    # Predict edges
    for i, (row, feats, gold_id) in enumerate(zip(rows[:max_examples], X, y_true_ids)):
        debate_id, src_id, tgt_id, premise, hypothesis, gold = row
        feats = feats.reshape(1, -1)
        probs = clf.predict_proba(feats)[0]  # [3]
        pred_id = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_id]
        pred_score = float(np.max(probs))

        y_true.append(int(gold_id))
        y_pred.append(pred_id)

        logger.log_edge(
            debate_id=debate_id,
            src_id=src_id,
            tgt_id=tgt_id,
            gold_label=gold,
            pred_label=pred_label,
            pred_score=pred_score,
        )

        if (i + 1) % 200 == 0:
            print(f"[graph_eval] {i+1}/{min(len(rows), max_examples)}")

    logger.close()

    labels = [0, 1, 2]
    print("Accuracy:", float(accuracy_score(y_true, y_pred)))
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=["support", "attack", "neutral"],
            digits=4,
            zero_division=0,
        )
    )

    print(f"[graph_eval] wrote: {Path(out_dir) / 'nodes.tsv'}")
    print(f"[graph_eval] wrote: {Path(out_dir) / 'edges.tsv'}")


def main() -> None:
    local_dir = snapshot_download_syncialo(
        corpus_id="synthetic_corpus-001",
        split="eval",
        local_dir=str(ARTIFACTS_DIR / "stance_pipeline" / "syncialo_snapshot"),
    )

    pairs = build_dataset(
        local_dir=local_dir,
        corpus_id="synthetic_corpus-001",
        split="eval",
        neutrals_per_pos=1.0,
        max_pairs=5000,
        max_debates=30,
        seed=0,
    )

    print("Pairs:", len(pairs))
    counts = {k: sum(1 for _, _, _, y in pairs if y == k) for k in ["support", "attack", "neutral"]}
    print("Label counts:", counts)

    # Base models
    nli = HFNLIModel(
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        max_length=256,
    )
    stance = HFStanceModel(
        "krishnagarg09/stance-detection-semeval2016",
        max_length=128,     # keep <=128 to avoid position-id crash
        device="cpu",       # avoid CUDA device-side assert you saw
        force_eager_attention=True,
    )

    # Compute once, then sklearn is fast
    X, y, groups = featurize_all(
        pairs,
        nli=nli,
        stance=stance,
        cache_path=str(ARTIFACTS_DIR / "stance_pipeline" / "meta_features.npz"),
        verbose_every=200,
    )

    train_eval_on_cached_logits(
        X, y, groups,
        test_size=0.2,
        seed=0,
        lr_path=str(ARTIFACTS_DIR / "stance_pipeline" / "meta_lr.joblib"),
        force_retrain=False,
    )

    # Build edge rows with IDs for visualization
    rows, claims_by_debate = build_debate_edge_dataset(
        local_dir=local_dir,
        corpus_id="synthetic_corpus-001",
        split="eval",
        neutrals_per_pos=1.0,
        max_edges=5000,
        max_debates=30,
        seed=0,
    )

    # Run meta classifier and log nodes/edges TSVs
    run_meta_with_graph_logs(
        rows=rows,
        claims_by_debate=claims_by_debate,
        nli=nli,
        stance=stance,
        lr_path=str(ARTIFACTS_DIR / "stance_pipeline" / "meta_lr.joblib"),
        out_dir=str(ARTIFACTS_DIR / "stance_pipeline" / "out_graphs"),
        max_examples=2000,
    )


if __name__ == "__main__":
    main()
