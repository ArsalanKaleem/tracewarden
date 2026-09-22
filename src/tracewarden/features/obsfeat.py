"""Cheap lexical features of an incoming observation (OBS events)."""
from __future__ import annotations

import re

from .normalize import normalize
from .provenance import extract_values

IMPERATIVE = re.compile(
    r"\b(you must|you should|please (?:send|forward|transfer|add|delete|share|update|run|email)|"
    r"ignore (?:all |any |the )?(?:previous|prior|above)|immediately|do not tell|without (?:telling|notifying)|"
    r"new instructions?|system (?:note|override|message)|as an ai|assistant[:,])", re.I)
URGENCY = re.compile(r"\b(urgent|asap|within \d+ (?:minutes|hours)|lock(?:ed|down)|suspend|verify now|final notice)\b", re.I)


def obs_feats(obs: str, known_text: str) -> list[float]:
    o = normalize(obs)
    n_words = max(len(o.split()), 1)
    imp = len(IMPERATIVE.findall(o))
    urg = len(URGENCY.findall(o))
    ents = extract_values(o, entities_only=True)
    kt = known_text.lower()
    ext = [e for e in ents if e not in kt]
    return [min(imp * 20 / n_words, 1.0), min(urg * 20 / n_words, 1.0), float(bool(ext))]


N_OBS = 3
