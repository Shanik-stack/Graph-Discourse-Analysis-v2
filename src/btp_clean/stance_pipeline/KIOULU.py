import datasets
from huggingface_hub import HfApi
import json
import networkx as nx
from pathlib import Path

from btp_clean.paths import ARTIFACTS_DIR

repo_id = "DebateLabKIT/syncialo-raw"
corpus_id = "synthetic_corpus-001"
split = "eval"

hfapi = HfApi()
hfapi.snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=f"data/{corpus_id}/{split}/**/*",
    local_dir=str(ARTIFACTS_DIR / "stance_pipeline" / "syncialo_snapshot"),
  )

argmaps = []
for f in (ARTIFACTS_DIR / "stance_pipeline" / "syncialo_snapshot").glob(pattern=f"data/{corpus_id}/{split}/**/*.json"):
  argmap = nx.node_link_graph(json.loads(f.read_text()))
  argmaps.append(argmap)

print(f"Loaded {len(argmaps)} from split {split} in corpus {corpus_id}.")

i = 1
print(f"Inpecting debate at index {i}:")
print(f"* Number of nodes: {argmaps[i].number_of_nodes()}")
print(f"* Number of edges: {argmaps[i].number_of_edges()}")


# Distillation

def instructions_from_argmaps():
  for argmap in argmaps:
    for u, v, data in argmap.edges(data=True):
      ul = str(argmap.nodes[u])
      vl = str(argmap.nodes[v])
      yield {
        "prompt": f"{ul} > {vl}: support or attack?",
        "answer": f"{data['valence']}",
      }

# ds_distilled = datasets.Dataset.from_generator(instructions_from_argmaps)
# ds_distilled
