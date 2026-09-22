"""TraceWarden dashboard (Day 22). Runs on a free CPU Hugging Face Space.

    TW_CHECKPOINT=runs/sg.pt python app/app.py
Accepts an AgentDrift record, an AgentDojo log, or a TraceWarden trajectory JSON.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import gradio as gr

from tracewarden.calibrate import decide
from tracewarden.guard import Guard
from tracewarden.io.agentdojo import convert_log
from tracewarden.io.agentdrift import record_to_traj
from tracewarden.schema import Trajectory

HERE = Path(__file__).parent
CKPT = os.environ.get("TW_CHECKPOINT", str(HERE / "tracewarden-core.pt"))
EXAMPLES = sorted(str(p) for p in (HERE.parent / "examples" / "trajectories").glob("*.json"))
GUARD = Guard.load(CKPT, encoder=os.environ.get("TW_ENCODER"))

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap');
:root { --ink:#1d2433; --muted:#667085; --rule:#e4e7ec; --b:#5f7a6b; --i:#b7791f; --h:#b42318; --f:#6941c6; }
.tw { font-family:'IBM Plex Sans',system-ui,sans-serif; color:var(--ink); max-width:880px; }
.tw h2 { font-size:1.35rem; margin:0 0 .2rem; } .tw .sub { color:var(--muted); margin:0 0 1.2rem; }
.verdict { display:inline-block; padding:.2rem .6rem; border-radius:4px; font-weight:600; color:#fff; }
.v-compromised{background:var(--h)} .v-attempted{background:var(--f)} .v-clean{background:var(--b)}
.step { display:grid; grid-template-columns:2.4rem 1fr 9.5rem; gap:.9rem; padding:.8rem 0; border-top:1px solid var(--rule); }
.num { font:600 1rem 'IBM Plex Mono',monospace; color:var(--muted); padding-top:.1rem; }
.tool { font:600 .95rem 'IBM Plex Mono',monospace; }
.args,.obs { font:.8rem 'IBM Plex Mono',monospace; color:#475467; white-space:pre-wrap; word-break:break-word; margin-top:.25rem; }
.obs { border-left:3px solid var(--rule); padding-left:.6rem; }
.lab { font-size:.8rem; font-weight:600; } .lab-benign{color:var(--b)} .lab-injection_point{color:var(--i)}
.lab-hijacked{color:var(--h)} .lab-failed_injection{color:var(--f)}
.step.hijacked .obs { border-left-color:var(--h); } .step.injection_point .obs, .step.failed_injection .obs { border-left-color:var(--i); }
.bar { height:5px; background:var(--rule); border-radius:3px; margin:.25rem 0 .45rem; }
.bar > span { display:block; height:100%; border-radius:3px; }
.meta { font-size:.72rem; color:var(--muted); }
.decision { font:600 .78rem 'IBM Plex Mono',monospace; }
@media (max-width:640px){ .step{grid-template-columns:2rem 1fr} .side{grid-column:2} }
"""


def parse(obj: dict) -> Trajectory:
    if "messages" in obj:
        return convert_log(obj, "agentdojo")
    if "task" in obj and "agent" in obj:
        return record_to_traj(obj)
    return Trajectory.from_dict(obj)


def render(t: Trajectory, rep) -> str:
    th = GUARD.thresholds
    out = [f'<div class="tw"><h2>{html.escape(t.user_task or t.id)}</h2>',
           f'<p class="sub"><span class="verdict v-{rep.verdict}">{rep.verdict}</span> &nbsp;{len(t.steps)} tool calls'
           f' &middot; hold at {th["hold"]:.2f}, block at {th["block"]:.2f}</p>']
    for i, (s, lab, h, p) in enumerate(zip(t.steps, rep.step_labels, rep.hijack_scores, rep.poison_scores)):
        d = decide(h, th)
        gold = f' <span class="meta">(gold: {s.label})</span>' if s.label else ""
        out.append(
            f'<div class="step {lab}"><div class="num">{i}</div><div>'
            f'<div class="tool">{html.escape(s.tool)}</div>'
            f'<div class="args">{html.escape(json.dumps(s.args, ensure_ascii=False)[:300])}</div>'
            f'<div class="obs">{html.escape(s.obs[:400])}</div></div>'
            f'<div class="side"><div class="lab lab-{lab}">{lab.replace("_", " ")}</div>{gold}'
            f'<div class="meta">before the call: <span class="decision">{d}</span></div>'
            f'<div class="bar"><span style="width:{h * 100:.0f}%;background:var(--h)"></span></div>'
            f'<div class="meta">output poisoned: {p:.2f}</div>'
            f'<div class="bar"><span style="width:{p * 100:.0f}%;background:var(--i)"></span></div></div></div>')
    return "".join(out) + "</div>"


def analyze(file_obj, pasted: str):
    try:
        raw = Path(file_obj).read_text(encoding="utf-8") if file_obj else pasted
        if not raw or not raw.strip():
            return "<p>Upload a trajectory file or paste its JSON to analyze it.</p>", ""
        t = parse(json.loads(raw))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return f"<p>Could not read this trajectory: {html.escape(str(e))}. Check that it is valid JSON in one of " \
               "the supported formats.</p>", ""
    rep = GUARD.scan(t)
    return render(t, rep), rep.to_markdown()


with gr.Blocks(css=CSS, title="TraceWarden") as demo:
    gr.Markdown("### TraceWarden\nFind where a prompt injection entered an agent run and which tool calls it "
                "took over. Every score uses only the steps before it, so each call is judged as it would be "
                "before execution.")
    with gr.Row():
        f = gr.File(label="Trajectory JSON (AgentDrift, AgentDojo log, or TraceWarden)", type="filepath")
        txt = gr.Code(label="...or paste JSON", language="json")
    btn = gr.Button("Analyze trajectory", variant="primary")
    timeline = gr.HTML()
    report = gr.Markdown()
    btn.click(analyze, [f, txt], [timeline, report])
    if EXAMPLES:
        gr.Examples(EXAMPLES, inputs=[f], label="Examples")

if __name__ == "__main__":
    demo.launch()
