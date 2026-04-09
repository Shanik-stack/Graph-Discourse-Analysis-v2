from decomposition import decompose_facts
from graph_builder import build_debate_graph
from visualize import draw_graph

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
    motion, turns = parse_debate("debate.txt")
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

    print(f"Extracted {len(facts)} atomic facts from {len(turns)} turns.")
    G = build_debate_graph(facts, speakers, motion)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (non-neutral).")

    draw_graph(G, "debate_graph.png")
    print("Saved: debate_graph.png")

if __name__ == "__main__":
    main()
