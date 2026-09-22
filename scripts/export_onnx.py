"""Day 19 - CPU-first deployment (N10): export the StreamGuard trunk to ONNX and quantize to int8.

python scripts/export_onnx.py --checkpoint runs/sg.pt --out runs/onnx
Then benchmark with scripts/benchmark_latency.py. The sentence encoder dominates latency; export it with
Hugging Face Optimum if needed:
    optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 runs/onnx/encoder
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from tracewarden.training import load_checkpoint

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="runs/onnx")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_checkpoint(a.checkpoint)
    assert ck["kind"] == "streamguard"
    d_in = ck["config"]["d_in"]
    E = 12
    x, et, m = torch.randn(1, E, d_in), torch.tensor([[0, 1] * (E // 2)]), torch.ones(1, E, dtype=torch.bool)
    fp32 = out / "streamguard.onnx"
    torch.onnx.export(model, (x, et, m), str(fp32), input_names=["x", "etype", "mask"],
                      output_names=["hijack", "poison"], opset_version=17, dynamo=False,
                      dynamic_axes={"x": {0: "B", 1: "E"}, "etype": {0: "B", 1: "E"}, "mask": {0: "B", 1: "E"},
                                    "hijack": {0: "B", 1: "E"}, "poison": {0: "B", 1: "E"}})
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = out / "streamguard.int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    import onnxruntime as ort

    feeds = {"x": x.numpy(), "etype": et.numpy(), "mask": m.numpy()}
    with torch.no_grad():
        ref = torch.sigmoid(model(x, et, m)[0]).numpy()
    for p in (fp32, int8):
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        got = 1 / (1 + np.exp(-s.run(None, feeds)[0]))
        t0 = time.perf_counter()
        for _ in range(200):
            s.run(None, feeds)
        ms = (time.perf_counter() - t0) / 200 * 1000
        print(f"{p.name}: {p.stat().st_size / 1e6:.2f} MB | max |diff| vs torch {np.abs(got - ref).max():.4f} "
              f"| {ms:.2f} ms per 12-event forward")
