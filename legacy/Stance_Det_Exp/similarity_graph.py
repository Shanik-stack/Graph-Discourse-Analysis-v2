from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

class SimilarityIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

def build_similarity_graph(
    emb: np.ndarray,
    k: int = 8,
    sim_threshold: float = 0.45,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    emb: (N, d) normalized embeddings
    adjacency: node -> [(neighbor, sim), ...] sorted by sim desc
    """
    N = emb.shape[0]
    sims = emb @ emb.T
    graph: Dict[int, List[Tuple[int, float]]] = {}

    for i in range(N):
        # get top k+1 including self, then drop self
        idx = np.argpartition(-sims[i], range(min(N, k + 1)))[: min(N, k + 1)]
        idx = [j for j in idx if j != i]
        idx = sorted(idx, key=lambda j: float(sims[i, j]), reverse=True)[:k]
        nbrs = [(int(j), float(sims[i, j])) for j in idx if float(sims[i, j]) >= sim_threshold]
        graph[i] = nbrs

    return graph
