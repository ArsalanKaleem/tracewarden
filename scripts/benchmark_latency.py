"""Day 19 - added latency per tool call on this machine (N10).

python scripts/benchmark_latency.py --checkpoint runs/sg.pt [--encoder sentence-transformers/all-MiniLM-L6-v2]
Reports encoder ms/text, full check() ms/decision, and peak RSS.
"""
import argparse
import statistics
import time

from tracewarden import Guard

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()
    g = Guard.load(a.checkpoint, encoder=a.encoder)
    txt = "TOOL: send_email | THOUGHT: forwarding the report | ARGS: {\"to\": \"x@y.com\"}"
    g.encoder.encode([txt])
    t0 = time.perf_counter()
    for _ in range(20):
        g.encoder.encode([txt])
    enc_ms = (time.perf_counter() - t0) / 20 * 1000
    lat = []
    for _ in range(a.runs):
        s = g.session("Summarize my inbox and reply to Omar", {"email": "me@acme.com"})
        for i in range(a.steps):
            t0 = time.perf_counter()
            s.check("read_email", {"id": i})
            lat.append((time.perf_counter() - t0) * 1000)
            s.observe(f"Email {i}: meeting moved to 3pm, contact omar@acme.com")
    try:  # Linux / macOS
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:  # Windows
        try:
            import psutil

            rss = psutil.Process().memory_info().peak_wset / 1e6
        except Exception:
            rss = float("nan")
    print(f"encoder: {g.encoder.name}")
    print(f"encoder latency: {enc_ms:.1f} ms per text")
    print(f"check() latency: median {statistics.median(lat):.1f} ms | p95 {sorted(lat)[int(0.95 * len(lat))]:.1f} ms"
          f" (history up to {a.steps} steps)")
    print(f"peak RSS: {rss:.0f} MB")
