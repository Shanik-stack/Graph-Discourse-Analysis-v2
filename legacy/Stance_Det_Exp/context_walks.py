from typing import Dict, List, Tuple, Set
import numpy as np

def walk_context(
    graph: Dict[int, List[Tuple[int, float]]],
    start: int,
    allowed_nodes: Set[int] | None = None,
    n_walks: int = 15,
    walk_len: int = 2,
    top_m: int = 4,
    restart_p: float = 0.3,
    rng_seed: int = 0,
) -> List[int]:
    """
    Weighted random walks with restarts over a similarity graph.
    allowed_nodes: restrict traversal to a set (e.g., same debate topic, earlier nodes).
    """
    rng = np.random.default_rng(rng_seed)
    counts: Dict[int, int] = {}

    def filtered_neighbors(u: int):
        nbrs = graph.get(u, [])
        if allowed_nodes is None:
            return nbrs
        return [(v, w) for (v, w) in nbrs if v in allowed_nodes]

    for _ in range(n_walks):
        cur = start
        for _ in range(walk_len):
            if rng.random() < restart_p:
                cur = start
            nbrs = filtered_neighbors(cur)
            if not nbrs:
                break
            nodes = np.array([n for n, _ in nbrs], dtype=int)
            weights = np.array([w for _, w in nbrs], dtype=float)
            weights = np.maximum(weights, 1e-6)
            probs = weights / weights.sum()
            cur = int(rng.choice(nodes, p=probs))
            if cur != start:
                counts[cur] = counts.get(cur, 0) + 1

    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in ranked[:top_m]]
