# Agent notes

- Read `README.md`, then `docs/architecture.md` (the contracts) and `docs/ROADMAP.md`.
- Never commit challenge PDFs, raw data or model checkpoints; the R&S material is company restricted.
- Every pose method must write TUM format to `results/<seq>/<method>.txt`, `T_world_cam`, seconds.
- ORB-SLAM3 is GPLv3: build it externally, never vendor its source.
- Experiment logs go in each track's `notes.md`.
