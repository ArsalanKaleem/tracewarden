"""Sentence encoders and the on-disk embedding cache.

* ``SentenceTransformerEncoder`` - the real encoder (default all-mpnet-base-v2, 768-d), frozen.
* ``HashingEncoder``             - dependency-free, offline, deterministic 768-d bag-of-n-grams encoder.
  Used by the tests, by CI, and as a sanity baseline ("does semantics matter at all?").

All encoders return L2-normalized float32 arrays of shape (n, dim).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .features import serialize as S
from .schema import Trajectory

DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
FAST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEncoder:
    def __init__(self, dim: int = 768):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self.name = f"hashing-{dim}"
        self._v = HashingVectorizer(n_features=dim, ngram_range=(1, 2), alternate_sign=True, norm="l2",
                                    token_pattern=r"(?u)\b\w+\b|[@:/]")

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._v.transform(texts).todense(), dtype=np.float32)


class SentenceTransformerEncoder:
    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 64):
        from sentence_transformers import (
            SentenceTransformer,  # optional dependency: pip install tracewarden[st]
        )

        self.name = model
        self._m = SentenceTransformer(model, device=device)
        if device and device.startswith("cuda"):
            self._m = self._m.half()  # T4: fp16, not bf16
        self.dim = self._m.get_sentence_embedding_dimension()
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._m.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=len(texts) > 2000).astype(np.float32)


class OnnxEncoder:
    """Encoder exported by scripts/export_onnx.py (N10). Expects `model.onnx` + tokenizer files in a folder."""

    def __init__(self, folder: str):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.name = f"onnx:{folder}"
        self._tok = AutoTokenizer.from_pretrained(folder)
        self._s = ort.InferenceSession(str(Path(folder) / "model.onnx"), providers=["CPUExecutionProvider"])
        self.dim = int(json.loads((Path(folder) / "tw_encoder.json").read_text())["dim"])

    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self._tok(texts, padding=True, truncation=True, max_length=384, return_tensors="np")
        feeds = {i.name: enc[i.name] for i in self._s.get_inputs() if i.name in enc}
        tok = self._s.run(None, feeds)[0]
        m = enc["attention_mask"][..., None].astype(np.float32)
        e = (tok * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return (e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-9)).astype(np.float32)


def get_encoder(name: str, device: str | None = None) -> Encoder:
    if name.startswith("hashing"):
        dim = int(name.split("-")[1]) if "-" in name else 768
        return HashingEncoder(dim)
    if name.startswith("onnx:"):
        return OnnxEncoder(name[5:])
    return SentenceTransformerEncoder(name, device=device)


# ---------------------------------------------------------------------------------------------
# Embedding cache: encode every step of a corpus once (Kaggle GPU), reuse everywhere (laptop CPU)
# ---------------------------------------------------------------------------------------------
class EmbeddingCache:
    """Per-trajectory access to cached embeddings. Views: action, obs, full (per step) and task (per traj)."""

    def __init__(self, folder: str | Path, mmap: bool = True):
        self.folder = Path(folder)
        mode = "r" if mmap else None
        self.meta = json.loads((self.folder / "meta.json").read_text())
        self.offsets = {k: tuple(v) for k, v in self.meta["offsets"].items()}  # traj_id -> (start, n_steps)
        self.task_row = self.meta["task_row"]
        self.views = {v: np.load(self.folder / f"{v}.npy", mmap_mode=mode) for v in self.meta["views"]}
        self.task = np.load(self.folder / "task.npy", mmap_mode=mode)
        self.dim = int(self.meta["dim"])
        self.encoder = self.meta["encoder"]

    def get(self, traj_id: str, view: str) -> np.ndarray:
        a, n = self.offsets[traj_id]
        return np.asarray(self.views[view][a:a + n], dtype=np.float32)

    def get_task(self, traj_id: str) -> np.ndarray:
        return np.asarray(self.task[self.task_row[traj_id]], dtype=np.float32)

    def __contains__(self, traj_id: str) -> bool:
        return traj_id in self.offsets


def encode_corpus(trajs: list[Trajectory], encoder: Encoder, out: str | Path,
                  views: tuple[str, ...] = ("action", "obs", "full"), chunk: int = 4096) -> Path:
    """Encode all steps of `trajs` and write <out>/{view}.npy (float16), task.npy and meta.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    offsets, task_row, pos = {}, {}, 0
    for i, t in enumerate(trajs):
        offsets[t.id] = (pos, len(t.steps))
        task_row[t.id] = i
        pos += len(t.steps)
    steps = [s for t in trajs for s in t.steps]

    def enc_all(texts: list[str]) -> np.ndarray:
        parts = [encoder.encode(texts[i:i + chunk]) for i in range(0, len(texts), chunk)]
        return np.concatenate(parts).astype(np.float16) if parts else np.zeros((0, encoder.dim), np.float16)

    for v in views:
        np.save(out / f"{v}.npy", enc_all([S.VIEWS[v](s) for s in steps]))
    np.save(out / "task.npy", enc_all([S.task(t) for t in trajs]))
    (out / "meta.json").write_text(json.dumps({
        "encoder": encoder.name, "dim": encoder.dim, "views": list(views),
        "offsets": offsets, "task_row": task_row, "n_steps": pos, "n_traj": len(trajs),
    }))
    return out
