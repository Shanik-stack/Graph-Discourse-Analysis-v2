from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from debate_graph.stance.syncialo_model_benchmarks import main

if __name__ == "__main__":
    main()
