# Project Layout V2

```text
clean_project_v2/
├── README.md
├── PROJECT_LAYOUT.md
├── pyproject.toml
├── requirements.txt
├── current_pipeline/
│   ├── README.md
│   ├── FULL_PIPELINE_WALKTHROUGH.md
│   └── run_pipeline.py
├── data/
│   ├── discourse_graph/
│   ├── syncialo_snapshot/
│   └── stance_pipeline/
├── outputs/
│   ├── README.md
│   ├── full_pipeline/
│   ├── graphs/
│   ├── analysis/
│   └── visualizations/
├── legacy/
│   ├── New_Igraph/
│   ├── Stance_Det_Exp/
│   ├── backend.py
│   ├── front.py
│   ├── x.py
│   └── notebooks...
└── src/
    └── btp_clean/
        ├── paths.py
        ├── full_pipeline/
        ├── stance_pipeline/
        ├── discourse_graph/
        ├── debate_runtime/
        └── apps/
```

## Current vs legacy

- `current_pipeline/`
  - what you should read and run
- `src/btp_clean/full_pipeline/`
  - orchestration logic used by the current runner
- `legacy/`
  - historical folders copied from the original workspace

## Recommended reading order

1. `current_pipeline/FULL_PIPELINE_WALKTHROUGH.md`
2. `current_pipeline/run_pipeline.py`
3. `src/btp_clean/full_pipeline/pipeline.py`
4. `src/btp_clean/stance_pipeline/`
