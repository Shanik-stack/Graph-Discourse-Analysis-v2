# Clean Project V2

This is a second, clearer project layout that separates the active end-to-end workflow from legacy experiments.

## Top-level folders

- `current_pipeline/`
  - the user-facing place to run and understand the active end-to-end pipeline
- `src/btp_clean/`
  - the Python package code that powers the pipeline
- `data/`
  - input datasets and local supporting files
- `outputs/`
  - the intended location for generated outputs from this layout
- `legacy/`
  - archived older project branches and original files kept for reference

## Recommended starting point

If you want to understand or run the current workflow, start here:

- [current_pipeline/README.md](C:\All Codes\BTP\clean_project_v2\current_pipeline\README.md)
- [current_pipeline/FULL_PIPELINE_WALKTHROUGH.md](C:\All Codes\BTP\clean_project_v2\current_pipeline\FULL_PIPELINE_WALKTHROUGH.md)
- [current_pipeline/run_pipeline.py](C:\All Codes\BTP\clean_project_v2\current_pipeline\run_pipeline.py)

## Setup

From `clean_project_v2/`:

```bash
pip install -e .
pip install -r requirements.txt
```

## Design intent

- `current_pipeline` is the canonical path.
- `legacy` exists so nothing important is lost, but it is not the recommended execution path.
- `outputs` is where v2-generated artifacts should live conceptually.
- the downloaded Syncialo snapshot now lives under `data/syncialo_snapshot/`

## Notes

- The package name remains `btp_clean` to avoid rewriting imports unnecessarily.
- This v2 layout is mainly a structural cleanup and navigation improvement.
