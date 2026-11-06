"""Build tracewarden_kaggle.zip: the code and data the multi-seed run needs, nothing else.

    python kaggle/make_bundle.py            # writes kaggle/tracewarden_kaggle.zip

Upload the zip as a private Kaggle dataset (Datasets -> New dataset), then attach it to the notebook.
Caches, virtualenvs, checkpoints (except the headline sg.pt for a sanity check) and logs are left out:
the run re-encodes everything itself.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = [
    "pyproject.toml", "README.md", "LICENSE", "src", "scripts",
    "data/processed", "data/real_qwen", "data/real_qwen14b", "data/stress", "data/swe_inject_final",
    "runs/sg.pt",
]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}

out = ROOT / "kaggle" / "tracewarden_kaggle.zip"
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for item in INCLUDE:
        p = ROOT / item
        if not p.exists():
            raise SystemExit(f"missing: {p}")
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        for f in files:
            if SKIP_PARTS & set(f.parts):
                continue
            z.write(f, f.relative_to(ROOT).as_posix())
            n += 1
print(f"wrote {out} ({n} files, {out.stat().st_size / 1e6:.1f} MB)")
