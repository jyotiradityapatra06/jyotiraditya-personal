# Hackathon modularization plan (exoplanet_pipeline.py)

- [ ] Create project folder structure: `exoplanet_pipeline_mod/`
- [ ] Move code into focused modules: `data_acquisition.py`, `synthetic.py`, `preprocessing.py`, `bls.py`, `classifier.py`, `fitting.py`, `visualization.py`, `report.py`, `config.py`, `types.py` (if needed)
- [ ] Implement `main.py` (entrypoint) that reproduces `run_pipeline()` behavior and outputs the same JSON/PDF/PNGs
- [ ] Keep a small top-level `exoplanet_pipeline.py` shim that calls the new `main.py` so `python exoplanet_pipeline.py` still works
- [x] Run a quick syntax/import check (and optionally execute pipeline) to ensure everything works


