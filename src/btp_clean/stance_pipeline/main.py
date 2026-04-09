from btp_clean.paths import DATA_DIR

from .decomposition import decompose_facts
from .graph_builder import build_graph, load_data
from .visualize import save_static_png

def parse_debate(path: str):
    motion = None
    turns = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("MOTION:"):
                motion = line.split(":", 1)[1].strip()
                continue
            if ":" not in line:
                continue
            speaker, text = line.split(":", 1)
            speaker = speaker.strip().upper()
            text = text.strip()
            turns.append((speaker, text))
    if motion is None:
        raise ValueError("Missing MOTION: line in debate.txt")
    return motion, turns

def main():
    motion, turns = parse_debate(str(DATA_DIR / "stance_pipeline" / "debate.txt"))
    facts = []
    speakers = []
    origin = []  # (turn_idx, speaker)

    for t, (spk, text) in enumerate(turns):
        fs = decompose_facts(text)
        print(f"\nTURN {t} [{spk}] RAW: {text}")
        print("  Atomic facts:")
        for k, f in enumerate(fs):
            print(f"   - ({t}.{k}) {f}")
            facts.append(f)
            speakers.append(spk)
            origin.append((t, spk))

    print(f"\nExtracted {len(facts)} atomic facts from {len(turns)} turns.")

    print(f"Motion: {motion}")
    print("Legacy note: this entrypoint now visualizes the first available benchmark graph artifact.")
    nodes_df, edges_df = load_data()
    debate_id = edges_df["debate_id"].astype(str).value_counts().idxmax()
    G = build_graph(nodes_df, edges_df, debate_id, keep_labels={"support", "attack"})
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (non-neutral).")

    out_path = DATA_DIR.parent / "artifacts" / "stance_pipeline" / "debate_graph.png"
    save_static_png(G, str(out_path))
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
