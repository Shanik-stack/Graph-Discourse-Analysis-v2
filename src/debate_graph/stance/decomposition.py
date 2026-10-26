import re
from typing import List, Set, Tuple
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner"])

NEGATION_WORDS = {"not", "no", "never", "without"}

def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" \t\n\r,;:-")
    return s

def _is_wellformed(s: str, min_tokens: int = 6) -> bool:
    return len(s.split()) >= min_tokens

def _sent_has_verb(sent_doc) -> bool:
    return any(t.pos_ in ("VERB", "AUX") for t in sent_doc)

def _get_subject_span(root) -> Tuple[int, int] | None:
    # Find syntactic subject subtree indices
    for ch in root.children:
        if ch.dep_ in ("nsubj", "nsubjpass", "csubj"):
            return (ch.left_edge.i, ch.right_edge.i + 1)
    return None

def _predicate_span(tok) -> Tuple[int, int]:
    # Predicate phrase from token to its right edge
    return (tok.i, tok.right_edge.i + 1)

def _has_own_complements(pred_tok) -> bool:
    # Heuristic: predicate has object/complement/advcl/ccomp etc.
    for ch in pred_tok.children:
        if ch.dep_ in ("dobj", "obj", "attr", "oprd", "ccomp", "xcomp", "advcl", "prep", "acomp"):
            return True
    return False

def decompose_facts(text: str, max_facts: int = 12, min_tokens: int = 6) -> List[str]:
    """
    Extractive, high-precision atomic fact decomposition.
    - Sentence split
    - Extract ccomp/xcomp (embedded propositions)
    - Extract advcl (conditional/causal subclaims)
    - Split coordinated predicates only when likely separate propositions
    """
    doc = nlp(_clean(text))
    facts: List[str] = []

    for sent in doc.sents:
        sent_doc = sent.as_doc()
        if not _sent_has_verb(sent_doc):
            continue

        base = _clean(sent.text)
        if _is_wellformed(base, min_tokens):
            facts.append(base)

        # embedded propositions: "X claims that Y" -> Y
        for tok in sent_doc:
            if tok.dep_ in ("ccomp", "xcomp"):
                span = sent_doc[tok.left_edge.i : tok.right_edge.i + 1].text
                span = _clean(span)
                if _is_wellformed(span, min_tokens):
                    facts.append(span)

        # adverbial clauses: "Because..., ..." -> keep clause as a fact too
        for tok in sent_doc:
            if tok.dep_ == "advcl":
                span = sent_doc[tok.left_edge.i : tok.right_edge.i + 1].text
                span = _clean(span)
                if _is_wellformed(span, min_tokens):
                    facts.append(span)

        # predicate coordination split: root + conj predicates, with shared subject
        root = next((t for t in sent_doc if t.dep_ == "ROOT"), None)
        if root is None:
            continue

        subj_span = _get_subject_span(root)
        if subj_span is None:
            continue

        subj_txt = sent_doc[subj_span[0] : subj_span[1]].text
        preds = [root] + list(root.conjuncts)

        # Only split if predicate heads are verbs/aux (or adjectival pred with copula)
        split_preds = []
        for p in preds:
            if p.pos_ in ("VERB", "AUX") and _has_own_complements(p):
                split_preds.append(p)

        # If we got 2+ meaningful predicates, create separate facts
        if len(split_preds) >= 2:
            for p in split_preds:
                ps = _predicate_span(p)
                pred_txt = sent_doc[ps[0] : ps[1]].text
                candidate = _clean(f"{subj_txt} {pred_txt}")
                if _is_wellformed(candidate, min_tokens):
                    facts.append(candidate)

    # Deduplicate (case-insensitive), preserve order
    seen: Set[str] = set()
    out: List[str] = []
    for f in facts:
        key = f.lower()
        if key not in seen:
            seen.add(key)
            out.append(f)

    return out[:max_facts]
