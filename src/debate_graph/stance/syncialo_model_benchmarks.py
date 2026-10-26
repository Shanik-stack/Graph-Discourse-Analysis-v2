from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity

from debate_graph.paths import DATA_DIR, MODELS_DIR, ARTIFACTS_DIR

from . import meta_benchmark as meta_bench
from .meta_nli import HFNLIModel, HFStanceModel
from .nli_relations import DeBerta, StanceRelation


LABEL2ID = {"support": 0, "attack": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class BenchmarkConfig:
    dataset_name: str = "syncialo"
    corpus_id: str = "synthetic_corpus-001"
    split: str = "eval"
    neutrals_per_pos: float = 1.0
    max_debates: int = 30
    test_size: float = 0.25
    seed: int = 0
    small_mode: bool = False

    nli_model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    stance_model_name: str = "krishnagarg09/stance-detection-semeval2016"
    nli_max_length: int = 256
    stance_max_length: int = 128
    stance_device: str = "cpu"

    similarity_top_k: int = 8
    similarity_threshold: float = 0.15

    @property
    def snapshot_dir(self) -> Path:
        return DATA_DIR / "syncialo_snapshot"

    @property
    def createdebate_cache_dir(self) -> Path:
        return Path.home() / ".cache" / "huggingface" / "datasets" / "azza1625___createdebate-arguments"

    @property
    def model_dir(self) -> Path:
        return MODELS_DIR / "syncialo_benchmarks"

    @property
    def report_dir(self) -> Path:
        return ARTIFACTS_DIR / "benchmark_reports"

    @property
    def split_path(self) -> Path:
        return self.report_dir / f"{self.dataset_name}_train_test_split.json"


@dataclass
class EdgeExample:
    debate_id: str
    src_id: str
    tgt_id: str
    premise: str
    hypothesis: str
    gold_label: str


def _ensure_dirs(cfg: BenchmarkConfig) -> None:
    cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)


def _apply_small_mode(cfg: BenchmarkConfig) -> BenchmarkConfig:
    if not cfg.small_mode:
        return cfg
    cfg.max_debates = min(cfg.max_debates, 8)
    return cfg


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _dataset_slug(cfg: BenchmarkConfig) -> str:
    if cfg.dataset_name == "createdebate":
        return f"createdebate_train_{cfg.max_debates}_seed{cfg.seed}"
    return f"{cfg.corpus_id}_{cfg.split}_{cfg.max_debates}_seed{cfg.seed}"


def load_syncialo_debates(cfg: BenchmarkConfig) -> List[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]]:
    local_dir = meta_bench.snapshot_download_syncialo(
        corpus_id=cfg.corpus_id,
        split=cfg.split,
        local_dir=str(cfg.snapshot_dir),
    )
    files = meta_bench.iter_files(local_dir, cfg.corpus_id, cfg.split)
    rng = random.Random(cfg.seed)
    rng.shuffle(files)

    debates: List[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]] = []
    for p in files[: cfg.max_debates]:
        obj = meta_bench.load_debate_json(p)
        claims, gold_edges = meta_bench.parse_nodes_and_links(obj)
        if claims and gold_edges:
            debates.append((p.stem, claims, gold_edges))
    return debates


def _createdebate_local_arrow(cfg: BenchmarkConfig) -> Optional[Path]:
    root = cfg.createdebate_cache_dir / "default" / "0.0.0"
    if not root.exists():
        return None
    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue
        arrow = subdir / "createdebate-arguments-train.arrow"
        if arrow.exists():
            return arrow
    return None


def _normalize_createdebate_parent_id(value: Any) -> Optional[str]:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    if ivalue < 0:
        return None
    return f"arg{ivalue}"


def _load_createdebate_rows(cfg: BenchmarkConfig) -> Dataset:
    local_arrow = _createdebate_local_arrow(cfg)
    if local_arrow is not None:
        return Dataset.from_file(str(local_arrow))
    ds = load_dataset("azza1625/createdebate-arguments")
    return ds["train"]


def load_createdebate_debates(cfg: BenchmarkConfig) -> List[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]]:
    ds = _load_createdebate_rows(cfg)
    grouped_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in ds:
        debate_key = row.get("debateUrl") or row.get("debateTitle") or row.get("topic") or "createdebate_unknown"
        grouped_rows[str(debate_key)].append(dict(row))

    debate_keys = list(grouped_rows.keys())
    rng = random.Random(cfg.seed)
    rng.shuffle(debate_keys)

    debates: List[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]] = []
    for debate_key in debate_keys[: cfg.max_debates]:
        rows = grouped_rows[debate_key]
        claims: Dict[str, str] = {}
        side_by_id: Dict[str, str] = {}

        for row in rows:
            arg_id = str(row.get("argumentId", "")).strip()
            body = str(row.get("argumentBody", "")).strip()
            if not arg_id or not body:
                continue
            claims[arg_id] = body
            side_by_id[arg_id] = str(row.get("argumentSide", "")).strip().lower()

        gold_edges: List[Tuple[str, str, str]] = []
        for row in rows:
            src_id = str(row.get("argumentId", "")).strip()
            tgt_id = _normalize_createdebate_parent_id(row.get("parentId"))
            if not src_id or not tgt_id:
                continue
            if src_id not in claims or tgt_id not in claims:
                continue
            src_side = side_by_id.get(src_id, "")
            tgt_side = side_by_id.get(tgt_id, "")
            if not src_side or not tgt_side:
                continue
            label = "support" if src_side == tgt_side else "attack"
            gold_edges.append((src_id, tgt_id, label))

        if claims and gold_edges:
            debate_id = _safe_slug(rows[0].get("debateTitle") or rows[0].get("topic") or debate_key)
            debates.append((debate_id, claims, gold_edges))
    return debates


def load_benchmark_debates(cfg: BenchmarkConfig) -> List[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]]:
    if cfg.dataset_name == "createdebate":
        return load_createdebate_debates(cfg)
    return load_syncialo_debates(cfg)


def split_debates(
    debates: Sequence[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]],
    test_size: float,
    seed: int,
) -> Tuple[List[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]], List[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]]]:
    debates = list(debates)
    rng = random.Random(seed)
    rng.shuffle(debates)
    n_test = max(1, int(round(len(debates) * test_size)))
    test = debates[:n_test]
    train = debates[n_test:]
    if not train:
        train, test = debates[:-1], debates[-1:]
    return train, test


def save_split(train_debates: Sequence[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]], test_debates: Sequence[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]], path: Path) -> None:
    payload = {
        "train_debate_ids": [d[0] for d in train_debates],
        "test_debate_ids": [d[0] for d in test_debates],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _token_overlap(a: str, b: str) -> float:
    ta = {t for t in a.lower().split() if t}
    tb = {t for t in b.lower().split() if t}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def build_similarity_feature_lookup(
    claims: Dict[str, str],
    top_k: int,
    sim_threshold: float,
) -> Dict[Tuple[str, str], np.ndarray]:
    claim_ids = list(claims.keys())
    texts = [claims[cid] for cid in claim_ids]

    if len(texts) <= 1:
        return {}

    vec = TfidfVectorizer(stop_words="english", max_features=4000)
    X = vec.fit_transform(texts)
    sims = cosine_similarity(X)

    neighbor_sets: Dict[str, set[str]] = {}
    degree_norm: Dict[str, float] = {}
    n = len(claim_ids)

    for i, cid in enumerate(claim_ids):
        order = np.argsort(-sims[i])
        nbrs: List[str] = []
        for j in order:
            if i == j:
                continue
            if float(sims[i, j]) < sim_threshold:
                continue
            nbrs.append(claim_ids[j])
            if len(nbrs) >= top_k:
                break
        neighbor_sets[cid] = set(nbrs)
        degree_norm[cid] = len(nbrs) / max(1, top_k)

    feature_lookup: Dict[Tuple[str, str], np.ndarray] = {}
    for i, src_id in enumerate(claim_ids):
        for j, tgt_id in enumerate(claim_ids):
            if i == j:
                continue
            src_neighbors = neighbor_sets[src_id]
            tgt_neighbors = neighbor_sets[tgt_id]
            union = src_neighbors | tgt_neighbors
            jacc = (len(src_neighbors & tgt_neighbors) / len(union)) if union else 0.0
            feature_lookup[(src_id, tgt_id)] = np.asarray(
                [
                    float(sims[i, j]),
                    degree_norm[src_id],
                    degree_norm[tgt_id],
                    1.0 if tgt_id in src_neighbors else 0.0,
                    1.0 if src_id in tgt_neighbors else 0.0,
                    float(jacc),
                    _token_overlap(claims[src_id], claims[tgt_id]),
                ],
                dtype=np.float32,
            )
    return feature_lookup


def _build_parent_depth_maps(
    claims: Dict[str, str],
    gold_edges: List[Tuple[str, str, str]],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, int]]:
    parent_of: Dict[str, str] = {}
    children_of: Dict[str, List[str]] = {cid: [] for cid in claims}
    for src, tgt, _ in gold_edges:
        if src in claims and tgt in claims and src not in parent_of:
            parent_of[src] = tgt
            children_of.setdefault(tgt, []).append(src)

    roots = [cid for cid in claims if cid not in parent_of]
    depth: Dict[str, int] = {cid: 0 for cid in claims}
    queue: List[str] = roots[:]
    while queue:
        node = queue.pop(0)
        for child in children_of.get(node, []):
            depth[child] = depth.get(node, 0) + 1
            queue.append(child)
    return parent_of, children_of, depth


def _cosine_dense(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_context_feature_lookup(
    claims: Dict[str, str],
    gold_edges: List[Tuple[str, str, str]],
) -> Dict[Tuple[str, str], np.ndarray]:
    claim_ids = list(claims.keys())
    texts = [claims[cid] for cid in claim_ids]
    if len(texts) <= 1:
        return {}

    vec = TfidfVectorizer(stop_words="english", max_features=4000)
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    idx_of = {cid: i for i, cid in enumerate(claim_ids)}
    parent_of, _, depth = _build_parent_depth_maps(claims, gold_edges)
    max_depth = max(1, max(depth.values(), default=0))

    ancestor_vectors: Dict[str, np.ndarray] = {}
    ancestor_counts: Dict[str, int] = {}
    for cid in claim_ids:
        cur = cid
        dist = 1
        weighted = np.zeros(X.shape[1], dtype=np.float32)
        total_w = 0.0
        count = 0
        while cur in parent_of:
            anc = parent_of[cur]
            w = 1.0 / float(2 ** (dist - 1))
            weighted += X[idx_of[anc]] * w
            total_w += w
            count += 1
            cur = anc
            dist += 1
        if total_w > 0:
            weighted /= total_w
        ancestor_vectors[cid] = weighted
        ancestor_counts[cid] = count

    feature_lookup: Dict[Tuple[str, str], np.ndarray] = {}
    for src_id in claim_ids:
        src_vec = X[idx_of[src_id]]
        src_anc = ancestor_vectors[src_id]
        src_depth = depth.get(src_id, 0) / max_depth
        src_anc_count = ancestor_counts[src_id] / max_depth
        for tgt_id in claim_ids:
            if src_id == tgt_id:
                continue
            tgt_vec = X[idx_of[tgt_id]]
            tgt_anc = ancestor_vectors[tgt_id]
            tgt_depth = depth.get(tgt_id, 0) / max_depth
            tgt_anc_count = ancestor_counts[tgt_id] / max_depth
            feature_lookup[(src_id, tgt_id)] = np.asarray(
                [
                    src_depth,
                    tgt_depth,
                    abs(src_depth - tgt_depth),
                    src_anc_count,
                    tgt_anc_count,
                    _cosine_dense(src_vec, src_anc),
                    _cosine_dense(tgt_vec, tgt_anc),
                    _cosine_dense(src_vec, tgt_anc),
                    _cosine_dense(tgt_vec, src_anc),
                    _cosine_dense(src_anc, tgt_anc),
                ],
                dtype=np.float32,
            )
    return feature_lookup


def build_examples_and_graph_features(
    debates: Sequence[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]],
    neutrals_per_pos: float,
    seed: int,
    similarity_top_k: int,
    similarity_threshold: float,
) -> Tuple[List[EdgeExample], np.ndarray]:
    rng = random.Random(seed)
    examples: List[EdgeExample] = []
    graph_feats: List[np.ndarray] = []

    for debate_id, claims, gold_edges in debates:
        all_edges = meta_bench.build_edges_with_neutrals(claims, gold_edges, neutrals_per_pos, rng)
        feature_lookup = build_similarity_feature_lookup(claims, similarity_top_k, similarity_threshold)

        for src_id, tgt_id, gold in all_edges:
            examples.append(
                EdgeExample(
                    debate_id=debate_id,
                    src_id=src_id,
                    tgt_id=tgt_id,
                    premise=claims[tgt_id],
                    hypothesis=claims[src_id],
                    gold_label=gold,
                )
            )
            graph_feats.append(feature_lookup.get((src_id, tgt_id), np.zeros(7, dtype=np.float32)))

    if graph_feats:
        return examples, np.stack(graph_feats)
    return examples, np.zeros((0, 7), dtype=np.float32)


def build_examples_with_feature_sets(
    debates: Sequence[Tuple[str, Dict[str, str], List[Tuple[str, str, str]]]],
    neutrals_per_pos: float,
    seed: int,
    similarity_top_k: int,
    similarity_threshold: float,
) -> Tuple[List[EdgeExample], np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    examples: List[EdgeExample] = []
    graph_feats: List[np.ndarray] = []
    context_feats: List[np.ndarray] = []

    for debate_id, claims, gold_edges in debates:
        all_edges = meta_bench.build_edges_with_neutrals(claims, gold_edges, neutrals_per_pos, rng)
        graph_lookup = build_similarity_feature_lookup(claims, similarity_top_k, similarity_threshold)
        context_lookup = build_context_feature_lookup(claims, gold_edges)

        for src_id, tgt_id, gold in all_edges:
            examples.append(
                EdgeExample(
                    debate_id=debate_id,
                    src_id=src_id,
                    tgt_id=tgt_id,
                    premise=claims[tgt_id],
                    hypothesis=claims[src_id],
                    gold_label=gold,
                )
            )
            graph_feats.append(graph_lookup.get((src_id, tgt_id), np.zeros(7, dtype=np.float32)))
            context_feats.append(context_lookup.get((src_id, tgt_id), np.zeros(10, dtype=np.float32)))

    if examples:
        return examples, np.stack(graph_feats), np.stack(context_feats)
    return examples, np.zeros((0, 7), dtype=np.float32), np.zeros((0, 10), dtype=np.float32)


def build_graph_feature_matrix_from_rows(
    rows: Sequence[Tuple[str, str, str, str, str, str]],
    claims_by_debate: Dict[str, Dict[str, str]],
    similarity_top_k: int,
    similarity_threshold: float,
) -> np.ndarray:
    per_debate_lookup: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {}
    for debate_id, claims in claims_by_debate.items():
        per_debate_lookup[debate_id] = build_similarity_feature_lookup(
            claims,
            top_k=similarity_top_k,
            sim_threshold=similarity_threshold,
        )

    feats: List[np.ndarray] = []
    for debate_id, src_id, tgt_id, _, _, _ in rows:
        lookup = per_debate_lookup.get(debate_id, {})
        feats.append(lookup.get((src_id, tgt_id), np.zeros(7, dtype=np.float32)))

    if feats:
        return np.stack(feats)
    return np.zeros((0, 7), dtype=np.float32)


def _feature_cache_paths(cfg: BenchmarkConfig) -> Dict[str, Path]:
    slug = _dataset_slug(cfg)
    return {
        "train_base": cfg.model_dir / f"{slug}_train_base_features.npz",
        "test_base": cfg.model_dir / f"{slug}_test_base_features.npz",
        "meta_model": cfg.model_dir / f"{slug}_meta_lr.joblib",
        "graph_model": cfg.model_dir / f"{slug}_graph_aware_rf.joblib",
        "context_model": cfg.model_dir / f"{slug}_context_aware_rf.joblib",
        "predictions": cfg.report_dir / f"{slug}_predictions.tsv",
        "summary": cfg.report_dir / f"{slug}_benchmark_summary.json",
        "ablation_summary": cfg.report_dir / f"{slug}_ablation_summary.json",
        "ablation_table": cfg.report_dir / f"{slug}_ablation_table.tsv",
        "graph_feature_ablation_summary": cfg.report_dir / f"{slug}_graph_feature_ablation_summary.json",
        "graph_feature_ablation_table": cfg.report_dir / f"{slug}_graph_feature_ablation_table.tsv",
    }


def featurize_base_examples(
    examples: Sequence[EdgeExample],
    nli: HFNLIModel,
    stance: HFStanceModel,
    cache_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"]

    X = np.zeros((len(examples), 6), dtype=np.float32)
    y = np.zeros((len(examples),), dtype=np.int64)
    for i, ex in enumerate(examples):
        X[i, :3] = nli.logits(ex.premise, ex.hypothesis)
        X[i, 3:] = stance.logits(ex.premise, ex.hypothesis)
        y[i] = LABEL2ID[ex.gold_label]
        if (i + 1) % 200 == 0:
            print(f"[base-featurize] {i+1}/{len(examples)}")
    np.savez_compressed(cache_path, X=X, y=y)
    return X, y


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    labels = [0, 1, 2]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=["support", "attack", "neutral"],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }


def benchmark_old_deberta(test_examples: Sequence[EdgeExample], model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    model = DeBerta(model_name=model_name, max_length=256)
    y_true = np.zeros((len(test_examples),), dtype=np.int64)
    y_pred = np.zeros((len(test_examples),), dtype=np.int64)
    for i, ex in enumerate(test_examples):
        scores = model.scores(ex.premise, ex.hypothesis)
        label, _ = model.decide(scores)
        y_true[i] = LABEL2ID[ex.gold_label]
        y_pred[i] = LABEL2ID[label]
    return y_true, y_pred


def benchmark_old_stance(test_examples: Sequence[EdgeExample], model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    model = StanceRelation(model_name=model_name, max_length=128, verbose=False)
    y_true = np.zeros((len(test_examples),), dtype=np.int64)
    y_pred = np.zeros((len(test_examples),), dtype=np.int64)
    for i, ex in enumerate(test_examples):
        scores = model.scores(ex.premise, ex.hypothesis)
        label, _ = model.decide(scores)
        y_true[i] = LABEL2ID[ex.gold_label]
        y_pred[i] = LABEL2ID[label]
    return y_true, y_pred


def train_meta_model(X_train: np.ndarray, y_train: np.ndarray, model_path: Path, force_retrain: bool) -> LogisticRegression:
    if model_path.exists() and not force_retrain:
        return joblib.load(model_path)
    clf = LogisticRegression(max_iter=2500, class_weight="balanced", multi_class="multinomial", n_jobs=-1)
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
    return clf


def train_graph_aware_model(
    X_train_base: np.ndarray,
    X_train_graph: np.ndarray,
    y_train: np.ndarray,
    model_path: Path,
    force_retrain: bool,
) -> RandomForestClassifier:
    if model_path.exists() and not force_retrain:
        return joblib.load(model_path)
    X_train = np.concatenate([X_train_base, X_train_graph], axis=1)
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=0,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
    return clf


def train_context_aware_model(
    X_train_base: np.ndarray,
    X_train_context: np.ndarray,
    y_train: np.ndarray,
    model_path: Path,
    force_retrain: bool,
) -> RandomForestClassifier:
    if model_path.exists() and not force_retrain:
        return joblib.load(model_path)
    X_train = np.concatenate([X_train_base, X_train_context], axis=1)
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=0,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
    return clf


def _load_existing_model_only(model_path: Path, mode_name: str) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(
            f"--test-only was requested, but no saved {mode_name} model was found at {model_path}"
        )
    return joblib.load(model_path)


def _feature_subset_matrix(
    X_base: np.ndarray,
    X_graph: np.ndarray,
    subset_name: str,
) -> np.ndarray:
    if subset_name == "nli":
        return X_base[:, :3]
    if subset_name == "stance":
        return X_base[:, 3:]
    if subset_name == "graph":
        return X_graph
    if subset_name == "nli+stance":
        return X_base
    if subset_name == "nli+graph":
        return np.concatenate([X_base[:, :3], X_graph], axis=1)
    if subset_name == "stance+graph":
        return np.concatenate([X_base[:, 3:], X_graph], axis=1)
    if subset_name == "nli+stance+graph":
        return np.concatenate([X_base, X_graph], axis=1)
    raise ValueError(f"Unknown ablation subset: {subset_name}")


def _train_ablation_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_path: Path,
    force_retrain: bool,
) -> RandomForestClassifier:
    if model_path.exists() and not force_retrain:
        clf = joblib.load(model_path)
        expected = int(getattr(clf, "n_features_in_", X_train.shape[1]))
        if expected == X_train.shape[1]:
            return clf
    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=0,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
    return clf


GRAPH_FEATURE_NAMES = [
    "cosine_similarity",
    "normalized_source_degree",
    "normalized_target_degree",
    "target_in_source",
    "source_in_target",
    "neighbor_jaccard_overlap",
    "token_jaccard_overlap",
]

CONTEXT_FEATURE_NAMES = [
    "normalized_source_depth",
    "normalized_target_depth",
    "normalized_depth_gap",
    "normalized_source_ancestor_count",
    "normalized_target_ancestor_count",
    "source_to_own_ancestor_context",
    "target_to_own_ancestor_context",
    "source_to_target_ancestor_context",
    "target_to_source_ancestor_context",
    "ancestor_context_to_ancestor_context",
]


def _graph_feature_ablation_matrix(
    X_base: np.ndarray,
    X_graph: np.ndarray,
    dropped_feature_idx: int,
) -> np.ndarray:
    kept_cols = [i for i in range(X_graph.shape[1]) if i != dropped_feature_idx]
    X_graph_kept = X_graph[:, kept_cols]
    return np.concatenate([X_base, X_graph_kept], axis=1)


def _graph_feature_only_matrix(
    X_base: np.ndarray,
    X_graph: np.ndarray,
    kept_feature_idx: int,
) -> np.ndarray:
    return np.concatenate([X_base, X_graph[:, [kept_feature_idx]]], axis=1)


def _context_subset_matrix(
    X_base: np.ndarray,
    X_context: np.ndarray,
    subset_name: str,
) -> np.ndarray:
    if subset_name == "nli":
        return X_base[:, :3]
    if subset_name == "stance":
        return X_base[:, 3:]
    if subset_name == "context":
        return X_context
    if subset_name == "nli+stance":
        return X_base
    if subset_name == "nli+context":
        return np.concatenate([X_base[:, :3], X_context], axis=1)
    if subset_name == "stance+context":
        return np.concatenate([X_base[:, 3:], X_context], axis=1)
    if subset_name == "nli+stance+context":
        return np.concatenate([X_base, X_context], axis=1)
    raise ValueError(f"Unknown context ablation subset: {subset_name}")


def _context_feature_ablation_matrix(
    X_base: np.ndarray,
    X_context: np.ndarray,
    dropped_feature_idx: int,
) -> np.ndarray:
    kept_cols = [i for i in range(X_context.shape[1]) if i != dropped_feature_idx]
    X_context_kept = X_context[:, kept_cols]
    return np.concatenate([X_base, X_context_kept], axis=1)


def _context_feature_only_matrix(
    X_base: np.ndarray,
    X_context: np.ndarray,
    kept_feature_idx: int,
) -> np.ndarray:
    return np.concatenate([X_base, X_context[:, [kept_feature_idx]]], axis=1)


def run_graph_feature_ablation_study(
    cfg: BenchmarkConfig,
    *,
    force_retrain: bool = False,
    test_only: bool = False,
) -> Dict[str, Any]:
    cfg = _apply_small_mode(cfg)
    _ensure_dirs(cfg)

    debates = load_benchmark_debates(cfg)
    train_debates, test_debates = split_debates(debates, cfg.test_size, cfg.seed)
    save_split(train_debates, test_debates, cfg.split_path)

    train_examples, train_graph_feats = build_examples_and_graph_features(
        train_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )
    test_examples, test_graph_feats = build_examples_and_graph_features(
        test_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed + 1,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )

    cache_paths = _feature_cache_paths(cfg)
    nli = HFNLIModel(cfg.nli_model_name, max_length=cfg.nli_max_length)
    stance = HFStanceModel(
        cfg.stance_model_name,
        max_length=cfg.stance_max_length,
        device=cfg.stance_device,
        force_eager_attention=True,
    )

    X_train_base, y_train = featurize_base_examples(train_examples, nli, stance, cache_paths["train_base"])
    X_test_base, y_test = featurize_base_examples(test_examples, nli, stance, cache_paths["test_base"])

    full_model_path = cfg.model_dir / f"{cfg.corpus_id}_{cfg.split}_{cfg.max_debates}_seed{cfg.seed}_graph_feature_full.joblib"
    if test_only:
        full_clf = _load_existing_model_only(full_model_path, "graph-feature ablation full-reference")
    else:
        full_clf = _train_ablation_classifier(
            np.concatenate([X_train_base, train_graph_feats], axis=1),
            y_train,
            full_model_path,
            force_retrain,
        )
    y_pred_full = full_clf.predict(np.concatenate([X_test_base, test_graph_feats], axis=1))
    full_metrics = _metrics_dict(y_test, y_pred_full)

    ablation_rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {
        "full_graph_aware_reference": {
            "kept_graph_features": GRAPH_FEATURE_NAMES,
            "dropped_graph_feature": None,
            "n_features": int(X_train_base.shape[1] + train_graph_feats.shape[1]),
            "model_path": str(full_model_path),
            "metrics": full_metrics,
        }
    }

    for idx, feature_name in enumerate(GRAPH_FEATURE_NAMES):
        X_train = _graph_feature_ablation_matrix(X_train_base, train_graph_feats, idx)
        X_test = _graph_feature_ablation_matrix(X_test_base, test_graph_feats, idx)
        model_path = cfg.model_dir / (
            f"{cfg.corpus_id}_{cfg.split}_{cfg.max_debates}_seed{cfg.seed}_graph_feature_drop_{_safe_slug(feature_name)}.joblib"
        )
        if test_only:
            clf = _load_existing_model_only(model_path, f"graph-feature ablation '{feature_name}'")
        else:
            clf = _train_ablation_classifier(X_train, y_train, model_path, force_retrain)
        y_pred = clf.predict(X_test)
        metrics = _metrics_dict(y_test, y_pred)
        detailed[feature_name] = {
            "dropped_graph_feature": feature_name,
            "kept_graph_features": [name for j, name in enumerate(GRAPH_FEATURE_NAMES) if j != idx],
            "n_features": int(X_train.shape[1]),
            "model_path": str(model_path),
            "metrics": metrics,
            "delta_vs_full": {
                "accuracy": float(metrics["accuracy"] - full_metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"] - full_metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"] - full_metrics["weighted_f1"]),
            },
        }
        ablation_rows.append(
            {
                "dropped_graph_feature": feature_name,
                "n_features": int(X_train.shape[1]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "delta_accuracy_vs_full": metrics["accuracy"] - full_metrics["accuracy"],
                "delta_macro_f1_vs_full": metrics["macro_f1"] - full_metrics["macro_f1"],
                "delta_weighted_f1_vs_full": metrics["weighted_f1"] - full_metrics["weighted_f1"],
            }
        )

    ablation_table = pd.DataFrame(ablation_rows).sort_values(by=["delta_macro_f1_vs_full", "delta_accuracy_vs_full"])
    ablation_table.to_csv(cache_paths["graph_feature_ablation_table"], sep="\t", index=False)

    summary = {
        "config": asdict(cfg),
        "train_debate_ids": [d[0] for d in train_debates],
        "test_debate_ids": [d[0] for d in test_debates],
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "graph_feature_names": GRAPH_FEATURE_NAMES,
        "reference_model": detailed["full_graph_aware_reference"],
        "feature_drop_ablations": {k: v for k, v in detailed.items() if k != "full_graph_aware_reference"},
        "ablation_table": str(cache_paths["graph_feature_ablation_table"]),
        "split_path": str(cfg.split_path),
    }
    cache_paths["graph_feature_ablation_summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_graph_feature_only_study(
    cfg: BenchmarkConfig,
    *,
    force_retrain: bool = False,
    test_only: bool = False,
) -> Dict[str, Any]:
    cfg = _apply_small_mode(cfg)
    _ensure_dirs(cfg)

    debates = load_benchmark_debates(cfg)
    train_debates, test_debates = split_debates(debates, cfg.test_size, cfg.seed)
    save_split(train_debates, test_debates, cfg.split_path)

    train_examples, train_graph_feats = build_examples_and_graph_features(
        train_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )
    test_examples, test_graph_feats = build_examples_and_graph_features(
        test_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed + 1,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )

    cache_paths = _feature_cache_paths(cfg)
    nli = HFNLIModel(cfg.nli_model_name, max_length=cfg.nli_max_length)
    stance = HFStanceModel(
        cfg.stance_model_name,
        max_length=cfg.stance_max_length,
        device=cfg.stance_device,
        force_eager_attention=True,
    )
    X_train_base, y_train = featurize_base_examples(train_examples, nli, stance, cache_paths["train_base"])
    X_test_base, y_test = featurize_base_examples(test_examples, nli, stance, cache_paths["test_base"])

    study_slug = _dataset_slug(cfg)
    summary_path = cfg.report_dir / f"{study_slug}_graph_feature_only_summary.json"
    table_path = cfg.report_dir / f"{study_slug}_graph_feature_only_table.tsv"

    rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {}
    for idx, feature_name in enumerate(GRAPH_FEATURE_NAMES):
        X_train = _graph_feature_only_matrix(X_train_base, train_graph_feats, idx)
        X_test = _graph_feature_only_matrix(X_test_base, test_graph_feats, idx)
        model_path = cfg.model_dir / (
            f"{study_slug}_graph_feature_only_v2_{_safe_slug(feature_name)}.joblib"
        )
        if test_only:
            clf = _load_existing_model_only(model_path, f"graph-feature-only '{feature_name}'")
        else:
            clf = _train_ablation_classifier(X_train, y_train, model_path, force_retrain)
        y_pred = clf.predict(X_test)
        metrics = _metrics_dict(y_test, y_pred)
        detailed[feature_name] = {
            "kept_graph_feature": feature_name,
            "n_features": int(X_train.shape[1]),
            "base_features": "6 textual features: 3 NLI logits + 3 stance logits",
            "model_path": str(model_path),
            "metrics": metrics,
        }
        rows.append(
            {
                "kept_graph_feature": feature_name,
                "n_features": int(X_train.shape[1]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

    table = pd.DataFrame(rows).sort_values(by=["macro_f1", "accuracy"], ascending=False)
    table.to_csv(table_path, sep="\t", index=False)

    summary = {
        "config": asdict(cfg),
        "train_debate_ids": [d[0] for d in train_debates],
        "test_debate_ids": [d[0] for d in test_debates],
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "graph_feature_names": GRAPH_FEATURE_NAMES,
        "feature_only_runs": detailed,
        "table_path": str(table_path),
        "split_path": str(cfg.split_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_feature_ablation_study(
    cfg: BenchmarkConfig,
    *,
    force_retrain: bool = False,
    test_only: bool = False,
) -> Dict[str, Any]:
    cfg = _apply_small_mode(cfg)
    _ensure_dirs(cfg)

    debates = load_benchmark_debates(cfg)
    train_debates, test_debates = split_debates(debates, cfg.test_size, cfg.seed)
    save_split(train_debates, test_debates, cfg.split_path)

    train_examples, train_graph_feats = build_examples_and_graph_features(
        train_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )
    test_examples, test_graph_feats = build_examples_and_graph_features(
        test_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed + 1,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )

    cache_paths = _feature_cache_paths(cfg)
    nli = HFNLIModel(cfg.nli_model_name, max_length=cfg.nli_max_length)
    stance = HFStanceModel(
        cfg.stance_model_name,
        max_length=cfg.stance_max_length,
        device=cfg.stance_device,
        force_eager_attention=True,
    )

    X_train_base, y_train = featurize_base_examples(train_examples, nli, stance, cache_paths["train_base"])
    X_test_base, y_test = featurize_base_examples(test_examples, nli, stance, cache_paths["test_base"])

    subsets = [
        "nli",
        "stance",
        "graph",
        "nli+stance",
        "nli+graph",
        "stance+graph",
        "nli+stance+graph",
    ]

    ablation_rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {}
    for subset in subsets:
        X_train = _feature_subset_matrix(X_train_base, train_graph_feats, subset)
        X_test = _feature_subset_matrix(X_test_base, test_graph_feats, subset)
        model_path = cfg.model_dir / f"{cfg.corpus_id}_{cfg.split}_{cfg.max_debates}_seed{cfg.seed}_ablation_{_safe_slug(subset)}.joblib"
        if test_only:
            clf = _load_existing_model_only(model_path, f"ablation subset '{subset}'")
        else:
            clf = _train_ablation_classifier(X_train, y_train, model_path, force_retrain)
        y_pred = clf.predict(X_test)
        metrics = _metrics_dict(y_test, y_pred)
        detailed[subset] = {
            "metrics": metrics,
            "n_features": int(X_train.shape[1]),
            "model_path": str(model_path),
        }
        ablation_rows.append(
            {
                "subset": subset,
                "n_features": int(X_train.shape[1]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

    ablation_table = pd.DataFrame(ablation_rows).sort_values(by=["macro_f1", "accuracy"], ascending=False)
    ablation_table.to_csv(cache_paths["ablation_table"], sep="\t", index=False)

    summary = {
        "config": asdict(cfg),
        "train_debate_ids": [d[0] for d in train_debates],
        "test_debate_ids": [d[0] for d in test_debates],
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "feature_groups": {
            "nli": "3 DeBERTa NLI logits",
            "stance": "3 stance logits",
            "graph": "7 debate-local graph features",
        },
        "ablations": detailed,
        "ablation_table": str(cache_paths["ablation_table"]),
        "split_path": str(cfg.split_path),
    }
    cache_paths["ablation_summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_context_ablation_study(
    cfg: BenchmarkConfig,
    *,
    force_retrain: bool = False,
    test_only: bool = False,
) -> Dict[str, Any]:
    cfg = _apply_small_mode(cfg)
    _ensure_dirs(cfg)

    debates = load_benchmark_debates(cfg)
    train_debates, test_debates = split_debates(debates, cfg.test_size, cfg.seed)
    save_split(train_debates, test_debates, cfg.split_path)

    train_examples, _, train_context_feats = build_examples_with_feature_sets(
        train_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )
    test_examples, _, test_context_feats = build_examples_with_feature_sets(
        test_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed + 1,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )

    cache_paths = _feature_cache_paths(cfg)
    nli = HFNLIModel(cfg.nli_model_name, max_length=cfg.nli_max_length)
    stance = HFStanceModel(
        cfg.stance_model_name,
        max_length=cfg.stance_max_length,
        device=cfg.stance_device,
        force_eager_attention=True,
    )

    X_train_base, y_train = featurize_base_examples(train_examples, nli, stance, cache_paths["train_base"])
    X_test_base, y_test = featurize_base_examples(test_examples, nli, stance, cache_paths["test_base"])

    study_slug = _dataset_slug(cfg)
    summary_path = cfg.report_dir / f"{study_slug}_context_ablation_summary.json"
    table_path = cfg.report_dir / f"{study_slug}_context_ablation_table.tsv"

    subsets = [
        "nli",
        "stance",
        "context",
        "nli+stance",
        "nli+context",
        "stance+context",
        "nli+stance+context",
    ]

    rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {}
    for subset in subsets:
        X_train = _context_subset_matrix(X_train_base, train_context_feats, subset)
        X_test = _context_subset_matrix(X_test_base, test_context_feats, subset)
        model_path = cfg.model_dir / f"{study_slug}_context_ablation_{_safe_slug(subset)}.joblib"
        if test_only:
            clf = _load_existing_model_only(model_path, f"context ablation subset '{subset}'")
        else:
            clf = _train_ablation_classifier(X_train, y_train, model_path, force_retrain)
        y_pred = clf.predict(X_test)
        metrics = _metrics_dict(y_test, y_pred)
        detailed[subset] = {
            "metrics": metrics,
            "n_features": int(X_train.shape[1]),
            "model_path": str(model_path),
        }
        rows.append(
            {
                "subset": subset,
                "n_features": int(X_train.shape[1]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

    table = pd.DataFrame(rows).sort_values(by=["macro_f1", "accuracy"], ascending=False)
    table.to_csv(table_path, sep="\t", index=False)

    summary = {
        "config": asdict(cfg),
        "train_debate_ids": [d[0] for d in train_debates],
        "test_debate_ids": [d[0] for d in test_debates],
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "feature_groups": {
            "nli": "3 DeBERTa NLI logits",
            "stance": "3 stance logits",
            "context": "10 ancestor/tree-context features",
        },
        "ablations": detailed,
        "ablation_table": str(table_path),
        "split_path": str(cfg.split_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_context_feature_ablation_study(
    cfg: BenchmarkConfig,
    *,
    force_retrain: bool = False,
    test_only: bool = False,
) -> Dict[str, Any]:
    cfg = _apply_small_mode(cfg)
    _ensure_dirs(cfg)

    debates = load_benchmark_debates(cfg)
    train_debates, test_debates = split_debates(debates, cfg.test_size, cfg.seed)
    save_split(train_debates, test_debates, cfg.split_path)

    train_examples, _, train_context_feats = build_examples_with_feature_sets(
        train_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )
    test_examples, _, test_context_feats = build_examples_with_feature_sets(
        test_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed + 1,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )

    cache_paths = _feature_cache_paths(cfg)
    nli = HFNLIModel(cfg.nli_model_name, max_length=cfg.nli_max_length)
    stance = HFStanceModel(
        cfg.stance_model_name,
        max_length=cfg.stance_max_length,
        device=cfg.stance_device,
        force_eager_attention=True,
    )
    X_train_base, y_train = featurize_base_examples(train_examples, nli, stance, cache_paths["train_base"])
    X_test_base, y_test = featurize_base_examples(test_examples, nli, stance, cache_paths["test_base"])

    study_slug = _dataset_slug(cfg)
    summary_path = cfg.report_dir / f"{study_slug}_context_feature_ablation_summary.json"
    table_path = cfg.report_dir / f"{study_slug}_context_feature_ablation_table.tsv"
    full_model_path = cfg.model_dir / f"{study_slug}_context_feature_full.joblib"

    if test_only:
        full_clf = _load_existing_model_only(full_model_path, "context-feature ablation full-reference")
    else:
        full_clf = _train_ablation_classifier(
            np.concatenate([X_train_base, train_context_feats], axis=1),
            y_train,
            full_model_path,
            force_retrain,
        )
    y_pred_full = full_clf.predict(np.concatenate([X_test_base, test_context_feats], axis=1))
    full_metrics = _metrics_dict(y_test, y_pred_full)

    rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {
        "full_context_aware_reference": {
            "kept_context_features": CONTEXT_FEATURE_NAMES,
            "dropped_context_feature": None,
            "n_features": int(X_train_base.shape[1] + train_context_feats.shape[1]),
            "model_path": str(full_model_path),
            "metrics": full_metrics,
        }
    }

    for idx, feature_name in enumerate(CONTEXT_FEATURE_NAMES):
        X_train = _context_feature_ablation_matrix(X_train_base, train_context_feats, idx)
        X_test = _context_feature_ablation_matrix(X_test_base, test_context_feats, idx)
        model_path = cfg.model_dir / f"{study_slug}_context_feature_drop_{_safe_slug(feature_name)}.joblib"
        if test_only:
            clf = _load_existing_model_only(model_path, f"context-feature ablation '{feature_name}'")
        else:
            clf = _train_ablation_classifier(X_train, y_train, model_path, force_retrain)
        y_pred = clf.predict(X_test)
        metrics = _metrics_dict(y_test, y_pred)
        detailed[feature_name] = {
            "dropped_context_feature": feature_name,
            "kept_context_features": [name for j, name in enumerate(CONTEXT_FEATURE_NAMES) if j != idx],
            "n_features": int(X_train.shape[1]),
            "model_path": str(model_path),
            "metrics": metrics,
            "delta_vs_full": {
                "accuracy": float(metrics["accuracy"] - full_metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"] - full_metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"] - full_metrics["weighted_f1"]),
            },
        }
        rows.append(
            {
                "dropped_context_feature": feature_name,
                "n_features": int(X_train.shape[1]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "delta_accuracy_vs_full": metrics["accuracy"] - full_metrics["accuracy"],
                "delta_macro_f1_vs_full": metrics["macro_f1"] - full_metrics["macro_f1"],
                "delta_weighted_f1_vs_full": metrics["weighted_f1"] - full_metrics["weighted_f1"],
            }
        )

    table = pd.DataFrame(rows).sort_values(by=["delta_macro_f1_vs_full", "delta_accuracy_vs_full"])
    table.to_csv(table_path, sep="\t", index=False)

    summary = {
        "config": asdict(cfg),
        "train_debate_ids": [d[0] for d in train_debates],
        "test_debate_ids": [d[0] for d in test_debates],
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "context_feature_names": CONTEXT_FEATURE_NAMES,
        "reference_model": detailed["full_context_aware_reference"],
        "feature_drop_ablations": {k: v for k, v in detailed.items() if k != "full_context_aware_reference"},
        "ablation_table": str(table_path),
        "split_path": str(cfg.split_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_context_feature_only_study(
    cfg: BenchmarkConfig,
    *,
    force_retrain: bool = False,
    test_only: bool = False,
) -> Dict[str, Any]:
    cfg = _apply_small_mode(cfg)
    _ensure_dirs(cfg)

    debates = load_benchmark_debates(cfg)
    train_debates, test_debates = split_debates(debates, cfg.test_size, cfg.seed)
    save_split(train_debates, test_debates, cfg.split_path)

    train_examples, _, train_context_feats = build_examples_with_feature_sets(
        train_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )
    test_examples, _, test_context_feats = build_examples_with_feature_sets(
        test_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed + 1,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )

    cache_paths = _feature_cache_paths(cfg)
    nli = HFNLIModel(cfg.nli_model_name, max_length=cfg.nli_max_length)
    stance = HFStanceModel(
        cfg.stance_model_name,
        max_length=cfg.stance_max_length,
        device=cfg.stance_device,
        force_eager_attention=True,
    )
    X_train_base, y_train = featurize_base_examples(train_examples, nli, stance, cache_paths["train_base"])
    X_test_base, y_test = featurize_base_examples(test_examples, nli, stance, cache_paths["test_base"])

    study_slug = _dataset_slug(cfg)
    summary_path = cfg.report_dir / f"{study_slug}_context_feature_only_summary.json"
    table_path = cfg.report_dir / f"{study_slug}_context_feature_only_table.tsv"

    rows: List[Dict[str, Any]] = []
    detailed: Dict[str, Any] = {}
    for idx, feature_name in enumerate(CONTEXT_FEATURE_NAMES):
        X_train = _context_feature_only_matrix(X_train_base, train_context_feats, idx)
        X_test = _context_feature_only_matrix(X_test_base, test_context_feats, idx)
        model_path = cfg.model_dir / f"{study_slug}_context_feature_only_{_safe_slug(feature_name)}.joblib"
        if test_only:
            clf = _load_existing_model_only(model_path, f"context-feature-only '{feature_name}'")
        else:
            clf = _train_ablation_classifier(X_train, y_train, model_path, force_retrain)
        y_pred = clf.predict(X_test)
        metrics = _metrics_dict(y_test, y_pred)
        detailed[feature_name] = {
            "kept_context_feature": feature_name,
            "n_features": int(X_train.shape[1]),
            "base_features": "6 textual features: 3 NLI logits + 3 stance logits",
            "model_path": str(model_path),
            "metrics": metrics,
        }
        rows.append(
            {
                "kept_context_feature": feature_name,
                "n_features": int(X_train.shape[1]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

    table = pd.DataFrame(rows).sort_values(by=["macro_f1", "accuracy"], ascending=False)
    table.to_csv(table_path, sep="\t", index=False)

    summary = {
        "config": asdict(cfg),
        "train_debate_ids": [d[0] for d in train_debates],
        "test_debate_ids": [d[0] for d in test_debates],
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "context_feature_names": CONTEXT_FEATURE_NAMES,
        "feature_only_runs": detailed,
        "table_path": str(table_path),
        "split_path": str(cfg.split_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_context_all_ablations(
    cfg: BenchmarkConfig,
    *,
    force_retrain: bool = False,
    test_only: bool = False,
) -> Dict[str, Any]:
    return {
        "context_ablation": run_context_ablation_study(cfg, force_retrain=force_retrain, test_only=test_only),
        "context_feature_ablation": run_context_feature_ablation_study(cfg, force_retrain=force_retrain, test_only=test_only),
        "context_feature_only": run_context_feature_only_study(cfg, force_retrain=force_retrain, test_only=test_only),
    }


def benchmark_syncialo_models(
    cfg: BenchmarkConfig,
    model_choice: str = "compare",
    force_retrain: bool = False,
    test_only: bool = False,
) -> Dict[str, Any]:
    cfg = _apply_small_mode(cfg)
    _ensure_dirs(cfg)

    debates = load_benchmark_debates(cfg)
    train_debates, test_debates = split_debates(debates, cfg.test_size, cfg.seed)
    save_split(train_debates, test_debates, cfg.split_path)

    train_examples, train_graph_feats, train_context_feats = build_examples_with_feature_sets(
        train_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )
    test_examples, test_graph_feats, test_context_feats = build_examples_with_feature_sets(
        test_debates,
        neutrals_per_pos=cfg.neutrals_per_pos,
        seed=cfg.seed + 1,
        similarity_top_k=cfg.similarity_top_k,
        similarity_threshold=cfg.similarity_threshold,
    )

    cache_paths = _feature_cache_paths(cfg)

    nli = HFNLIModel(cfg.nli_model_name, max_length=cfg.nli_max_length)
    stance = HFStanceModel(
        cfg.stance_model_name,
        max_length=cfg.stance_max_length,
        device=cfg.stance_device,
        force_eager_attention=True,
    )
    X_train_base, y_train = featurize_base_examples(train_examples, nli, stance, cache_paths["train_base"])
    X_test_base, y_test = featurize_base_examples(test_examples, nli, stance, cache_paths["test_base"])

    results: Dict[str, Any] = {
        "config": asdict(cfg),
        "train_debate_ids": [d[0] for d in train_debates],
        "test_debate_ids": [d[0] for d in test_debates],
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "models": {},
    }

    predictions_table = pd.DataFrame(
        {
            "debate_id": [ex.debate_id for ex in test_examples],
            "src_id": [ex.src_id for ex in test_examples],
            "tgt_id": [ex.tgt_id for ex in test_examples],
            "gold_label": [ex.gold_label for ex in test_examples],
        }
    )

    if model_choice in {"deberta", "compare"}:
        y_true_old, y_pred_old = benchmark_old_deberta(test_examples, cfg.nli_model_name)
        results["models"]["deberta"] = _metrics_dict(y_true_old, y_pred_old)
        predictions_table["pred_deberta"] = [ID2LABEL[int(i)] for i in y_pred_old]

    if model_choice in {"stance", "compare"}:
        y_true_old, y_pred_old = benchmark_old_stance(test_examples, cfg.stance_model_name)
        results["models"]["stance"] = _metrics_dict(y_true_old, y_pred_old)
        predictions_table["pred_stance"] = [ID2LABEL[int(i)] for i in y_pred_old]

    if model_choice in {"meta", "compare"}:
        if test_only:
            meta_clf = _load_existing_model_only(cache_paths["meta_model"], "meta")
        else:
            meta_clf = train_meta_model(X_train_base, y_train, cache_paths["meta_model"], force_retrain)
        y_pred_meta = meta_clf.predict(X_test_base)
        results["models"]["meta"] = _metrics_dict(y_test, y_pred_meta)
        predictions_table["pred_meta"] = [ID2LABEL[int(i)] for i in y_pred_meta]

    if model_choice in {"graph-aware", "compare"}:
        if test_only:
            graph_clf = _load_existing_model_only(cache_paths["graph_model"], "graph-aware")
        else:
            graph_clf = train_graph_aware_model(
                X_train_base,
                train_graph_feats,
                y_train,
                cache_paths["graph_model"],
                force_retrain,
            )
        X_test_graph_model = np.concatenate([X_test_base, test_graph_feats], axis=1)
        y_pred_graph = graph_clf.predict(X_test_graph_model)
        results["models"]["graph-aware"] = _metrics_dict(y_test, y_pred_graph)
        predictions_table["pred_graph_aware"] = [ID2LABEL[int(i)] for i in y_pred_graph]

    if model_choice in {"context-aware", "compare"}:
        if test_only:
            context_clf = _load_existing_model_only(cache_paths["context_model"], "context-aware")
        else:
            context_clf = train_context_aware_model(
                X_train_base,
                train_context_feats,
                y_train,
                cache_paths["context_model"],
                force_retrain,
            )
        X_test_context_model = np.concatenate([X_test_base, test_context_feats], axis=1)
        y_pred_context = context_clf.predict(X_test_context_model)
        results["models"]["context-aware"] = _metrics_dict(y_test, y_pred_context)
        predictions_table["pred_context_aware"] = [ID2LABEL[int(i)] for i in y_pred_context]

    predictions_table.to_csv(cache_paths["predictions"], sep="\t", index=False)
    results["prediction_table"] = str(cache_paths["predictions"])
    results["split_path"] = str(cfg.split_path)
    results["model_paths"] = {
        "meta": str(cache_paths["meta_model"]),
        "graph-aware": str(cache_paths["graph_model"]),
        "context-aware": str(cache_paths["context_model"]),
        "train_base_features": str(cache_paths["train_base"]),
        "test_base_features": str(cache_paths["test_base"]),
    }

    cache_paths["summary"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _print_summary(results: Dict[str, Any]) -> None:
    print("Train debates:", len(results["train_debate_ids"]))
    print("Test debates:", len(results["test_debate_ids"]))
    print("Train examples:", results["train_examples"])
    print("Test examples:", results["test_examples"])
    for model_name, metrics in results["models"].items():
        print(f"\n[{model_name}]")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")


def _print_ablation_summary(results: Dict[str, Any]) -> None:
    print("Train debates:", len(results["train_debate_ids"]))
    print("Test debates:", len(results["test_debate_ids"]))
    print("Train examples:", results["train_examples"])
    print("Test examples:", results["test_examples"])
    rows = []
    for subset, payload in results["ablations"].items():
        metrics = payload["metrics"]
        rows.append((subset, payload["n_features"], metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"]))
    rows.sort(key=lambda item: (item[3], item[2]), reverse=True)
    print("\n[Ablation study]")
    for subset, n_features, acc, macro_f1, weighted_f1 in rows:
        print(f"{subset:18s} feats={n_features:2d}  acc={acc:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")


def _print_graph_feature_ablation_summary(results: Dict[str, Any]) -> None:
    print("Train debates:", len(results["train_debate_ids"]))
    print("Test debates:", len(results["test_debate_ids"]))
    print("Train examples:", results["train_examples"])
    print("Test examples:", results["test_examples"])
    ref = results["reference_model"]["metrics"]
    print("\n[Graph-feature ablation reference]")
    print(
        f"full-graph-aware      feats={results['reference_model']['n_features']:2d}  "
        f"acc={ref['accuracy']:.4f}  macro_f1={ref['macro_f1']:.4f}  weighted_f1={ref['weighted_f1']:.4f}"
    )
    rows = []
    for feature_name, payload in results["feature_drop_ablations"].items():
        metrics = payload["metrics"]
        delta = payload["delta_vs_full"]
        rows.append(
            (
                feature_name,
                payload["n_features"],
                metrics["accuracy"],
                metrics["macro_f1"],
                metrics["weighted_f1"],
                delta["macro_f1"],
            )
        )
    rows.sort(key=lambda item: item[5])
    print("\n[Graph-feature ablation: drop one feature]")
    for feature_name, n_features, acc, macro_f1, weighted_f1, delta_macro_f1 in rows:
        print(
            f"drop={feature_name:26s} feats={n_features:2d}  acc={acc:.4f}  "
            f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}  delta_macro_f1={delta_macro_f1:+.4f}"
        )


def _print_graph_feature_only_summary(results: Dict[str, Any]) -> None:
    print("Train debates:", len(results["train_debate_ids"]))
    print("Test debates:", len(results["test_debate_ids"]))
    print("Train examples:", results["train_examples"])
    print("Test examples:", results["test_examples"])
    rows = []
    for feature_name, payload in results["feature_only_runs"].items():
        metrics = payload["metrics"]
        rows.append((feature_name, payload["n_features"], metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"]))
    rows.sort(key=lambda item: (item[3], item[2]), reverse=True)
    print("\n[Graph-feature-only study]")
    for feature_name, n_features, acc, macro_f1, weighted_f1 in rows:
        print(
            f"keep={feature_name:26s} feats={n_features:2d}  acc={acc:.4f}  "
            f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}"
        )


def _print_context_ablation_summary(results: Dict[str, Any]) -> None:
    print("Train debates:", len(results["train_debate_ids"]))
    print("Test debates:", len(results["test_debate_ids"]))
    print("Train examples:", results["train_examples"])
    print("Test examples:", results["test_examples"])
    rows = []
    for subset, payload in results["ablations"].items():
        metrics = payload["metrics"]
        rows.append((subset, payload["n_features"], metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"]))
    rows.sort(key=lambda item: (item[3], item[2]), reverse=True)
    print("\n[Context ablation study]")
    for subset, n_features, acc, macro_f1, weighted_f1 in rows:
        print(f"{subset:22s} feats={n_features:2d}  acc={acc:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")


def _print_context_feature_ablation_summary(results: Dict[str, Any]) -> None:
    print("Train debates:", len(results["train_debate_ids"]))
    print("Test debates:", len(results["test_debate_ids"]))
    print("Train examples:", results["train_examples"])
    print("Test examples:", results["test_examples"])
    ref = results["reference_model"]["metrics"]
    print("\n[Context-feature ablation reference]")
    print(
        f"full-context-aware    feats={results['reference_model']['n_features']:2d}  "
        f"acc={ref['accuracy']:.4f}  macro_f1={ref['macro_f1']:.4f}  weighted_f1={ref['weighted_f1']:.4f}"
    )
    rows = []
    for feature_name, payload in results["feature_drop_ablations"].items():
        metrics = payload["metrics"]
        delta = payload["delta_vs_full"]
        rows.append((feature_name, payload["n_features"], metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"], delta["macro_f1"]))
    rows.sort(key=lambda item: item[5])
    print("\n[Context-feature ablation: drop one feature]")
    for feature_name, n_features, acc, macro_f1, weighted_f1, delta_macro_f1 in rows:
        print(
            f"drop={feature_name:30s} feats={n_features:2d}  acc={acc:.4f}  "
            f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}  delta_macro_f1={delta_macro_f1:+.4f}"
        )


def _print_context_feature_only_summary(results: Dict[str, Any]) -> None:
    print("Train debates:", len(results["train_debate_ids"]))
    print("Test debates:", len(results["test_debate_ids"]))
    print("Train examples:", results["train_examples"])
    print("Test examples:", results["test_examples"])
    rows = []
    for feature_name, payload in results["feature_only_runs"].items():
        metrics = payload["metrics"]
        rows.append((feature_name, payload["n_features"], metrics["accuracy"], metrics["macro_f1"], metrics["weighted_f1"]))
    rows.sort(key=lambda item: (item[3], item[2]), reverse=True)
    print("\n[Context-feature-only study]")
    for feature_name, n_features, acc, macro_f1, weighted_f1 in rows:
        print(
            f"keep={feature_name:30s} feats={n_features:2d}  acc={acc:.4f}  "
            f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark debate models with explicit debate-level train/test splits.")
    parser.add_argument(
        "--dataset",
        choices=["syncialo", "createdebate"],
        default="syncialo",
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "deberta",
            "stance",
            "meta",
            "graph-aware",
            "context-aware",
            "compare",
            "ablation",
            "graph-feature-ablation",
            "graph-feature-only",
            "context-ablation",
            "context-feature-ablation",
            "context-feature-only",
            "context-all-ablations",
        ],
        default="compare",
    )
    parser.add_argument("--max-debates", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--test-only", action="store_true", help="Do not train benchmark classifiers; only evaluate using already saved models.")
    args = parser.parse_args(argv)

    cfg = BenchmarkConfig(
        dataset_name=args.dataset,
        max_debates=args.max_debates,
        test_size=args.test_size,
        seed=args.seed,
        small_mode=args.small,
    )
    if args.model == "ablation":
        results = run_feature_ablation_study(cfg, force_retrain=args.force_retrain, test_only=args.test_only)
        _print_ablation_summary(results)
        print(f"\nAblation report: {cfg.report_dir}")
    elif args.model == "graph-feature-ablation":
        results = run_graph_feature_ablation_study(cfg, force_retrain=args.force_retrain, test_only=args.test_only)
        _print_graph_feature_ablation_summary(results)
        print(f"\nGraph-feature ablation report: {cfg.report_dir}")
    elif args.model == "graph-feature-only":
        results = run_graph_feature_only_study(cfg, force_retrain=args.force_retrain, test_only=args.test_only)
        _print_graph_feature_only_summary(results)
        print(f"\nGraph-feature-only report: {cfg.report_dir}")
    elif args.model == "context-ablation":
        results = run_context_ablation_study(cfg, force_retrain=args.force_retrain, test_only=args.test_only)
        _print_context_ablation_summary(results)
        print(f"\nContext ablation report: {cfg.report_dir}")
    elif args.model == "context-feature-ablation":
        results = run_context_feature_ablation_study(cfg, force_retrain=args.force_retrain, test_only=args.test_only)
        _print_context_feature_ablation_summary(results)
        print(f"\nContext-feature ablation report: {cfg.report_dir}")
    elif args.model == "context-feature-only":
        results = run_context_feature_only_study(cfg, force_retrain=args.force_retrain, test_only=args.test_only)
        _print_context_feature_only_summary(results)
        print(f"\nContext-feature-only report: {cfg.report_dir}")
    elif args.model == "context-all-ablations":
        results = run_context_all_ablations(cfg, force_retrain=args.force_retrain, test_only=args.test_only)
        _print_context_ablation_summary(results["context_ablation"])
        print()
        _print_context_feature_ablation_summary(results["context_feature_ablation"])
        print()
        _print_context_feature_only_summary(results["context_feature_only"])
        print(f"\nContext all-ablations report: {cfg.report_dir}")
    else:
        results = benchmark_syncialo_models(cfg, model_choice=args.model, force_retrain=args.force_retrain, test_only=args.test_only)
        _print_summary(results)
        print(f"\nDetailed report: {cfg.report_dir}")


if __name__ == "__main__":
    main()
