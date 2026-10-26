# Legacy audit

The active pipeline does not import this directory. A source comparison found these migrations:

| Historical code | Current location | Audit result |
| --- | --- | --- |
| `front.py` | `src/debate_graph/demos/argument_graph_demo.py` | Identical source |
| `x.py` | `src/debate_graph/runtime/core.py` | Identical source |
| `backend.py` | `src/debate_graph/runtime/api.py` | Same API functions, packaged imports |
| `Stance_Det_Exp/` graph and stance modules | `src/debate_graph/stance/` | Most modules copied or adapted; current orchestration and paths differ |
| `New_Igraph/Discource.py`, `debate_datasets.py` | `src/debate_graph/discourse/` | Core code adapted for package imports and data paths |

The original notebooks, `New_Igraph` Git repository, sample images and graph files, and experimental entry points have not been shown to have full functional equivalents. In particular, the older `Stance_Det_Exp/main.py` and the active `stance_pipeline/main.py` follow different graph construction paths. This prevents a confident whole-folder deletion. Keep `legacy/` until those workflows are either ported and tested or explicitly retired.

The supported path for the current project is [`run_pipeline.py`](../run_pipeline.py). Historical files are for provenance and are not part of the documented run command.
