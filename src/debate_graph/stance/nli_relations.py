# stance_relations.py
from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class StanceRelation:
    """
    Target-dependent stance detection:
      FAVOR   -> support
      AGAINST -> attack
      NONE    -> neutral
    """

    def __init__(
        self,
        model_name: str = "krishnagarg09/stance-detection-semeval2016",
        device: Optional[str] = None,
        max_length: int = 128,
        verbose: bool = True,
    ):

        self.model_name = model_name
        self.max_length = max_length

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model (load ONCE)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        print("USING StanceRelation FROM:", __file__)
        print("attn_implementation before:", getattr(self.model.config, "attn_implementation", None))
        # Force safe attention path (Transformers 4.36+)
        if hasattr(self.model.config, "attn_implementation"):
            self.model.config.attn_implementation = "eager"

        self.model.eval()

        # Device selection + safe CUDA fallback
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        try:
            self.model.to(self.device)
        except RuntimeError as e:
            if "cuda" in str(e).lower():
                self.device = "cpu"
                self.model.to("cpu")
                if verbose:
                    print("HF Stance Model switched to CPU")
            else:
                raise

        # id2label mapping
        self.id2label = {int(i): str(lbl).upper() for i, lbl in self.model.config.id2label.items()}

    @torch.no_grad()
    def scores(self, target: str, claim: str) -> Dict[str, float]:
        """
        target = TARGET CLAIM (edge target)
        claim  = CURRENT CLAIM (edge source)
        """
        enc = self.tokenizer(
            claim,
            target,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**enc).logits[0]
        probs = F.softmax(logits, dim=-1).detach().cpu().tolist()

        return {self.id2label[i]: float(p) for i, p in enumerate(probs)}

    @staticmethod
    def decide(scores: Dict[str, float]) -> Tuple[str, float]:
        lab, p = max(scores.items(), key=lambda x: x[1])
        if lab == "FAVOR":
            return "support", float(p)
        if lab == "AGAINST":
            return "attack", float(p)
        return "neutral", float(p)

# nli_relations.py
from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


class DeBerta:
    """
    NLI-based relation:
      entailment    -> support
      contradiction -> attack
      neutral       -> neutral
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device: Optional[str] = None,
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.max_length = max_length

        self.cfg = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Robust id -> canonical label mapping
        self.idx2canon = self._make_idx2canon()

    def _make_idx2canon(self) -> Dict[int, str]:
        id2label = getattr(self.cfg, "id2label", None)
        idx2canon = {}

        if isinstance(id2label, dict) and len(id2label) >= 3:
            for i, lab in id2label.items():
                s = str(lab).lower()
                if "contra" in s:
                    idx2canon[int(i)] = "contradiction"
                elif "neutral" in s:
                    idx2canon[int(i)] = "neutral"
                elif "entail" in s:
                    idx2canon[int(i)] = "entailment"

        # fallback MNLI convention
        if len(idx2canon) < 3:
            idx2canon = {0: "contradiction", 1: "neutral", 2: "entailment"}

        return idx2canon

    @torch.no_grad()
    def scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        enc = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**enc).logits[0]
        probs = F.softmax(logits, dim=-1).detach().cpu().tolist()

        out: Dict[str, float] = {}
        for i, p in enumerate(probs):
            canon = self.idx2canon.get(i)
            if canon is not None:
                out[canon] = float(p)
        return out

    @staticmethod
    def decide(scores: Dict[str, float], p_min: float = 0.0) -> Tuple[str, float]:
        lab, p = max(scores.items(), key=lambda x: x[1])

        if p < p_min:
            return "neutral", p

        if lab == "entailment":
            return "support", p
        if lab == "contradiction":
            return "attack", p
        return "neutral", p
