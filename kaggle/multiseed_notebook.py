# Kaggle notebook for the multi-seed study (paper/cose, Table "seeds").
# Settings: Accelerator GPU T4 x1 (x2 is fine), Internet ON, attach the dataset built by kaggle/make_bundle.py.
# Paste each cell into its own notebook cell. Every step resumes if the session restarts.

# %% CELL 1: unpack the bundle and install --------------------------------------------------------
import glob, os, shutil, subprocess, sys, zipfile
TW = "/kaggle/working/tw"
if not os.path.exists(f"{TW}/pyproject.toml"):
    shutil.rmtree(TW, ignore_errors=True)
    zips = glob.glob("/kaggle/input/**/tracewarden_kaggle.zip", recursive=True)
    roots = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)
             if os.path.exists(os.path.join(os.path.dirname(p), "scripts", "multiseed.py"))]
    if zips:      # Kaggle kept the zip as a file
        zipfile.ZipFile(zips[0]).extractall(TW)
    elif roots:   # Kaggle unpacked it; the folder depth varies, so find the bundle root
        shutil.copytree(roots[0], TW)
    else:
        raise SystemExit("bundle not found under /kaggle/input -- is the dataset attached?")
os.chdir(TW)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
print(os.listdir("."))

# %% CELL 2: sanity check -- the headline checkpoint must reproduce the paper ---------------------
# Expected: 32B learned guard blocks 1 of 27 (blocked_before_execution 0.037), AUROC 0.399.
subprocess.run([sys.executable, "scripts/eval_swe_inject.py", "--trajectories",
                "data/swe_inject_final/coder32b_goalonly.jsonl", "--checkpoint", "runs/sg.pt",
                "--out", "/kaggle/working/sanity_32b.json"], check=True)
import json
r = json.load(open("/kaggle/working/sanity_32b.json"))
print("blocked", r["prevent/blocked_before_execution"], "auroc", r["scores/auroc"])

# %% CELL 3: hashing encoder, seeds 0-4 (CPU work, about 5-8 min per seed) ------------------------
subprocess.run([sys.executable, "scripts/multiseed.py", "--encoders", "hashing-768",
                "--seeds", "0,1,2,3,4", "--out", "/kaggle/working/seeds"], check=True)

# %% CELL 4: sentence encoder, seeds 0-4 (GPU for encoding and SWE-Inject replay) ------------------
subprocess.run([sys.executable, "scripts/multiseed.py", "--encoders", "sentence-transformers/all-mpnet-base-v2",
                "--device", "cuda", "--seeds", "0,1,2,3,4", "--out", "/kaggle/working/seeds"], check=True)

# %% CELL 5: summary and download ------------------------------------------------------------------
subprocess.run([sys.executable, "scripts/multiseed.py", "--aggregate-only", "--out", "/kaggle/working/seeds"], check=True)
subprocess.run("cd /kaggle/working && zip -qr seeds_results.zip seeds sanity_32b.json", shell=True, check=True)
print(open("/kaggle/working/seeds/seed_summary.md").read())
# Download /kaggle/working/seeds_results.zip from the Output panel and unzip it into
# C:\Projects\tracewarden\results\ , then ask Claude to fill the red [TBD] cells in paper/cose/main.tex.
