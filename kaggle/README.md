# Multi-seed runs on Kaggle (for the Computers & Security manuscript)

What it does: trains the guard with seeds 0–4 for each encoder (hashing, then mpnet). Each seed is calibrated at a
1 % budget and re-run on every real test set:
- AgentDojo-Steps 7B and 14B: overall, per suite, and at a 0.5 threshold;
- the four stress sets;
- all five SWE-Inject runs: the learned guard alone and with the provenance rule.

The script then writes mean / sd / min / max for each metric to `seeds/seed_summary.md`.

## Steps

1. On your PC, build the upload bundle (about 17 MB). The SWE-Inject goal-only runs must first be copied from
   the swe-inject repo into `data/swe_inject_final/` (git ignores `data/`):
   ```
   cd C:\Projects\tracewarden
   mkdir data\swe_inject_final
   copy ..\swe-inject\data\final\*_goalonly.jsonl data\swe_inject_final\
   .venv\Scripts\python.exe kaggle\make_bundle.py
   ```
2. On Kaggle, upload `kaggle/tracewarden_kaggle.zip` (**Datasets → New dataset, private**).
3. Create a new notebook with **GPU T4**, **Internet on**, and add the dataset. Paste the cells from
   `kaggle/multiseed_notebook.py` one per cell and run them in order.
   - Cell 2 is a sanity check: the old checkpoint must give 32B blocked 0.037 and AUROC 0.399. If it doesn't, stop
     and tell Claude.
   - Cell 3 (hashing) takes roughly 30–45 min and cell 4 (mpnet) roughly 1–1.5 h. Kaggle sessions last up to 12 h.
   - If a session stops, re-run the same cell: finished steps are skipped.
4. Download `seeds_results.zip` from the Output panel and unzip it into `C:\Projects\tracewarden\results\`.
5. Ask Claude to fill the red `[TBD]` cells in `paper/cose/main.tex` from `results/seeds/seed_summary.md`.

## Optional: more SWE-Inject trajectories

The SWE-Inject counts are small (48 hijacks; 18 URL-free). With spare GPU time, extra generation seeds for the 14B
and 32B coders would roughly double them. Use the existing generation notebook
(`swe-inject/notebooks/kaggle/week2_generate.py`) with
`MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"` (then 32B), parser `hermes`, and change the run cell to
`--variants --phrasings-per-goal 2 --max-per-repo 6 --strengths weak,strong --seed 1` (then `--seed 2`). Then:
- relabel the new files with `swe-inject/scripts/relabel_goal_only.py`;
- audit 30 trajectories per file with `scripts/audit_runs.py --sample 30`.

Bring the files back so the tables can be updated. This is a larger change to the paper: decide before starting,
and keep the seed-0 numbers as the main result if you run out of time.
