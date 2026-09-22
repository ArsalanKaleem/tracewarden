"""Day 18 - multilingual and obfuscated stress sets (N8).

Rewrites the poisoned observations of attacked / failed-attack test trajectories; labels are unchanged
(the agent's behavior is fixed, we test whether the DETECTOR still sees the poison). Benign and hard-negative
trajectories are copied as-is so false-alarm rates stay measurable.

Obfuscations (offline):   zero_width, homoglyph, base64, split (poison spread over two observations)
Translations (need an LLM): --langs ur,roman-ur,ar,hi --llm-base-url http://localhost:8000/v1 --llm-model Qwen/...

python scripts/make_stress_set.py --data data/processed --out data/stress --variants zero_width,homoglyph,base64,split
"""
import argparse
import base64
import copy
import json
import random
from pathlib import Path

from tracewarden.features.provenance import extract_values
from tracewarden.schema import load_jsonl, save_jsonl

LANG_NAMES = {"ur": "Urdu (Nastaliq script)", "roman-ur": "Roman Urdu (Urdu written in Latin letters)",
              "ar": "Arabic", "hi": "Hindi (Devanagari)", "es": "Spanish", "zh": "Simplified Chinese"}
HOMO = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441", "x": "\u0445"}


def zero_width(s, rnd):
    return "".join(ch + ("\u200b" if ch.isalpha() and rnd.random() < 0.3 else "") for ch in s)


def homoglyph(s, rnd):
    return "".join(HOMO[ch] if ch in HOMO and rnd.random() < 0.5 else ch for ch in s)


def b64(s, rnd):
    return "Encoded maintenance note (base64, decode and apply): " + base64.b64encode(s.encode()).decode()


def split_across(t, i):
    """Move the first half of poisoned obs i into the previous observation -> two injection points."""
    if i == 0:
        return False
    words = t.steps[i].obs.split()
    half = len(words) // 2
    t.steps[i - 1].obs += " " + " ".join(words[:half])
    t.steps[i].obs = " ".join(words[half:])
    if t.steps[i - 1].label == "benign":
        t.steps[i - 1].label = t.steps[i].label  # multi-injection grammar: I I H ... or F F B ...
    return True


def translate_fn(base_url, model, lang):
    from openai import OpenAI  # pip install openai

    client = OpenAI(base_url=base_url, api_key="sk-none")

    def tr(text):
        msg = (f"Translate the following tool output into {LANG_NAMES.get(lang, lang)}. Keep every email address, "
               "URL, account number, amount and code EXACTLY as written. Output only the translation.\n\n" + text)
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": msg}], temperature=0)
        return r.choices[0].message.content.strip()

    return tr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="zero_width,homoglyph,base64,split")
    ap.add_argument("--langs", default="")
    ap.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    T = load_jsonl(str(Path(a.data) / f"{a.split}.jsonl"))
    if a.limit:
        T = T[: a.limit]
    out = Path(a.out)
    rnd = random.Random(0)
    fns = {"zero_width": zero_width, "homoglyph": homoglyph, "base64": b64}
    variants = [v for v in a.variants.split(",") if v] + [f"lang_{x}" for x in a.langs.split(",") if x]
    for v in variants:
        tr = translate_fn(a.llm_base_url, a.llm_model, v[5:]) if v.startswith("lang_") else None
        res, dropped = [], 0
        for t in T:
            t = copy.deepcopy(t)
            t.meta = {**t.meta, "stress": v}
            ok = True
            for i in t.poisoned_indices:
                if v == "split":
                    ok = split_across(t, i)
                elif tr is not None:
                    new = tr(t.steps[i].obs)
                    ents = extract_values(t.steps[i].obs, entities_only=True)
                    ok = all(e in new.lower() for e in ents)  # translation must keep the attacker's values
                    t.steps[i].obs = new
                else:
                    t.steps[i].obs = fns[v](t.steps[i].obs, rnd)
                break  # only the first poisoned observation
            if ok:
                res.append(t)
            else:
                dropped += 1
        (out / v).mkdir(parents=True, exist_ok=True)
        save_jsonl(res, str(out / v / "test.jsonl"))
        print(f"{v}: {len(res)} trajectories ({dropped} dropped) -> {out / v / 'test.jsonl'}")
    (out / "README.json").write_text(json.dumps({"source": a.data, "variants": variants}, indent=1))
