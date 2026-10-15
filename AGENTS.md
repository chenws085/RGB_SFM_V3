# Local research handoff

- Before concluding that checkpoints or datasets are unavailable, inspect both
  this repo's `runs/` and `data/` and the persistent HDD inbox at
  `/mnt/d/RGB_SFM_artifacts/` (`D:\RGB_SFM_artifacts` on Windows).
- Remote agents may add new files at any time. Inspect `weights/`, `datasets/`
  and recent `notes/` files when resuming work; verify hashes/configuration before
  reusing an artifact. Received files are not automatically active experiment data.
- Shared agent notes live on the HDD in `notes/local/` and `notes/remote140165/`.
  Follow `agent_notes/README.md`: create uniquely named messages, do not overwrite
  another agent's note, and promote durable conclusions into `plan/` and
  `RESEARCH_CONTEXT.md`. Credentials remain outside Git and shared notes.
- Research progress and model lineage are summarized in `RESEARCH_CONTEXT.md`.
  Preserve unrelated uncommitted work when committing experiment changes.
