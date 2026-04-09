# meta_nli.py
"""
Meta (stacked) relation classifier.

This file is SELF-CONTAINED:
- Includes HFNLIModel (MNLI-style) and HFStanceModel (SemEval stance)
- MetaRelation stacks their logits -> sklearn classifier

Features:
- NLI logits:    [contradiction, neutral, entailment]  (3)
- Stance logits: [AGAINST, NONE, FAVOR]                (3)
- Meta logits:   6-dim feature -> LogisticRegression -> {support, attack, neutral}

Pairs format:
  (debate_id, premise, hypothesis, gold)
Where:
  premise    = TARGET claim (edge target)
  hypothesis = CURRENT claim (edge source)
  gold in {"support","attack","neutral"}

Recommended stance settings to avoid your earlier crash:
- stance_max_length <= 128
- stance_device="cpu" for krishnagarg09/stance-detection-semeval2016
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import os
import numpy as np
import joblib

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


Label = str
Pair = Tuple[str, str, str, str]  # (debate_id, premise, hypothesis, gold)


# =========================================================
# Base model wrappers (included in this file)
# =========================================================

class HFNLIModel:
    """
    Generic MNLI-style classifier.
    Returns logits ordered as [contradiction, neutral, entailment] (canonical).
    """
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 256,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.max_length = int(max_length)

        self.cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        # Map model indices -> canonical MNLI labels
        self.idx2canon = self._make_idx2canon(self.cfg)
        if verbose:
            print("[HFNLIModel] idx2canon:", self.idx2canon)

    @staticmethod
    def _make_idx2canon(cfg: Any) -> Dict[int, str]:
        id2label = getattr(cfg, "id2label", None)
        idx2canon: Dict[int, str] = {}

        if isinstance(id2label, dict) and len(id2label) >= 3:
            for i, lab in id2label.items():
                s = str(lab).lower()
                if "contra" in s:
                    idx2canon[int(i)] = "contradiction"
                elif "neutral" in s:
                    idx2canon[int(i)] = "neutral"
                elif "entail" in s:
                    idx2canon[int(i)] = "entailment"

        # fall back to common MNLI convention
        if len(idx2canon) < 3:
            idx2canon = {0: "contradiction", 1: "neutral", 2: "entailment"}

        return idx2canon

    @torch.no_grad()
    def logits(self, premise: str, hypothesis: str) -> np.ndarray:
        enc = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        raw = self.model(**enc).logits[0].detach().cpu().numpy()  # (num_labels,)
        # reorder into canonical [contra, neutral, entail]
        out = np.zeros(3, dtype=np.float32)
        for idx, canon in self.idx2canon.items():
            if canon == "contradiction":
                out[0] = float(raw[idx])
            elif canon == "neutral":
                out[1] = float(raw[idx])
            elif canon == "entailment":
                out[2] = float(raw[idx])
        return out


class HFStanceModel:
    """
    Target-dependent stance model.
    Returns logits ordered as [AGAINST, NONE, FAVOR] (canonical).

    Tokenizer call order:
      (claim, target)
    """
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_length: int = 128,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
        force_eager_attention: bool = True,
    ):
        self.model_name = model_name
        self.max_length = int(max_length)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)

        # Avoid SDPA-related GPU asserts for some envs/checkpoints
        if force_eager_attention:
            try:
                self.model.config.attn_implementation = "eager"
            except Exception:
                pass

        self.model.eval()

        chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(chosen)

        # robust device fallback
        try:
            self.model.to(self.device)
        except RuntimeError as e:
            if "cuda" in str(e).lower():
                self.device = torch.device("cpu")
                self.model.to(self.device)
                if verbose:
                    print("[HFStanceModel] switched to CPU due to CUDA error")
            else:
                raise

        # label mapping from config
        raw_id2label = getattr(self.model.config, "id2label", None)
        self.id2label = {int(i): str(lbl).upper() for i, lbl in (raw_id2label or {}).items()}

        # build canonical index order
        self._canon_index = self._make_canon_index(self.id2label)
        if verbose:
            print("[HFStanceModel] id2label:", self.id2label)
            print("[HFStanceModel] canon_index:", self._canon_index)

    @staticmethod
    def _make_canon_index(id2label: Dict[int, str]) -> Dict[str, int]:
        """
        Return mapping from canonical stance labels -> model index.
        """
        canon = {}
        for i, lab in id2label.items():
            if lab == "AGAINST":
                canon["AGAINST"] = i
            elif lab == "NONE":
                canon["NONE"] = i
            elif lab == "FAVOR":
                canon["FAVOR"] = i
        # If checkpoint uses different names, you can extend here.
        if len(canon) < 3:
            # assume a common ordering if missing
            canon = {"AGAINST": 0, "NONE": 1, "FAVOR": 2}
        return canon

    @torch.no_grad()
    def logits(self, target: str, claim: str) -> np.ndarray:
        enc = self.tokenizer(
            claim,   # claim first
            target,  # target second
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        raw = self.model(**enc).logits[0].detach().cpu().numpy()
        out = np.zeros(3, dtype=np.float32)
        out[0] = float(raw[self._canon_index["AGAINST"]])
        out[1] = float(raw[self._canon_index["NONE"]])
        out[2] = float(raw[self._canon_index["FAVOR"]])
        return out


# =========================================================
# Meta model
# =========================================================

@dataclass
class MetaConfig:
    label2id: Dict[Label, int] = None
    max_iter: int = 2000
    class_weight: str = "balanced"

    def __post_init__(self):
        if self.label2id is None:
            self.label2id = {"support": 0, "attack": 1, "neutral": 2}


class MetaRelation:
    """
    Stacking classifier over base-model logits.
    """

    def __init__(
        self,
        nli_model: HFNLIModel,
        stance_model: HFStanceModel,
        config: Optional[MetaConfig] = None,
    ):
        self.nli = nli_model
        self.stance = stance_model

        self.cfg = config or MetaConfig()
        self.id2label = {v: k for k, v in self.cfg.label2id.items()}

        self.clf: Optional[LogisticRegression] = None
        self.is_fitted: bool = False

    def featurize(self, premise: str, hypothesis: str) -> np.ndarray:
        """
        Returns (6,):
          [nli_logits(3: contra, neutral, entail), stance_logits(3: against, none, favor)]
        """
        nli_l = self.nli.logits(premise, hypothesis)              # (3,)
        stance_l = self.stance.logits(premise, hypothesis)        # target=premise, claim=hypothesis
        return np.concatenate([nli_l, stance_l], axis=0).astype(np.float32)

    def featurize_pairs(
        self,
        pairs: Sequence[Pair],
        max_examples: Optional[int] = None,
        verbose_every: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X: List[np.ndarray] = []
        y: List[int] = []
        groups: List[str] = []

        for i, (debate_id, premise, hypothesis, gold) in enumerate(pairs):
            if max_examples is not None and i >= max_examples:
                break
            X.append(self.featurize(premise, hypothesis))
            y.append(self.cfg.label2id[gold])
            groups.append(debate_id)

            if verbose_every and (i + 1) % verbose_every == 0:
                print(f"[meta_nli] featurized {i+1} examples")

        return np.stack(X), np.asarray(y, dtype=np.int64), np.asarray(groups, dtype=object)

    def fit(
        self,
        pairs: Sequence[Pair],
        test_size: float = 0.2,
        seed: int = 0,
        max_examples: Optional[int] = None,
        report: bool = True,
        verbose_every: int = 0,
    ) -> "MetaRelation":
        X, y, groups = self.featurize_pairs(pairs, max_examples=max_examples, verbose_every=verbose_every)

        splitter = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        self.clf = LogisticRegression(
            max_iter=int(self.cfg.max_iter),
            class_weight=self.cfg.class_weight,
            multi_class="multinomial",
            n_jobs=-1,
        )
        self.clf.fit(Xtr, ytr)
        self.is_fitted = True

        if report:
            pred = self.clf.predict(Xte)
            labels = [self.cfg.label2id["support"], self.cfg.label2id["attack"], self.cfg.label2id["neutral"]]
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
        return self

    def predict(
        self,
        premise: str,
        hypothesis: str,
        return_info: bool = False,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self.is_fitted or self.clf is None:
            raise RuntimeError("MetaRelation is not fitted. Call .fit(...) first or load a saved meta model.")

        x = self.featurize(premise, hypothesis).reshape(1, -1)
        probs = self.clf.predict_proba(x)[0]
        pred_id = int(np.argmax(probs))
        pred_label = self.id2label[pred_id]

        if not return_info:
            return pred_label, None

        info = {
            "meta_probs": {self.id2label[i]: float(probs[i]) for i in range(len(probs))},
            "features": x[0].astype(float).tolist(),
        }
        return pred_label, info

    def save(self, path: str) -> None:
        """
        Saves ONLY the sklearn meta-classifier + label config.
        Base HF models are not saved (load them separately).
        """
        if not self.is_fitted or self.clf is None:
            raise RuntimeError("Nothing to save: MetaRelation is not fitted.")
        payload = {
            "config": self.cfg,
            "clf": self.clf,
            "nli_model_name": getattr(self.nli, "model_name", None),
            "stance_model_name": getattr(self.stance, "model_name", None),
            "nli_max_length": getattr(self.nli, "max_length", None),
            "stance_max_length": getattr(self.stance, "max_length", None),
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(
        path: str,
        nli_model: HFNLIModel,
        stance_model: HFStanceModel,
    ) -> "MetaRelation":
        payload = joblib.load(path)
        cfg: MetaConfig = payload["config"]
        obj = MetaRelation(nli_model=nli_model, stance_model=stance_model, config=cfg)
        obj.clf = payload["clf"]
        obj.is_fitted = True
        return obj


# =========================================================
# Convenience constructors (optional)
# =========================================================

def build_default_meta(
    nli_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    stance_name: str = "krishnagarg09/stance-detection-semeval2016",
    nli_max_length: int = 256,
    stance_max_length: int = 128,
    stance_device: str = "cpu",
    cache_dir: Optional[str] = None,
    verbose: bool = False,
) -> MetaRelation:
    nli = HFNLIModel(nli_name, max_length=nli_max_length, cache_dir=cache_dir, verbose=verbose)
    stance = HFStanceModel(stance_name, max_length=stance_max_length, device=stance_device, cache_dir=cache_dir, verbose=verbose)
    return MetaRelation(nli, stance)
