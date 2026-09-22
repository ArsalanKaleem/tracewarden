"""DriftNet's four identity-free world features, computed from the user's known world."""
from __future__ import annotations

import json
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL = re.compile(r"https?://([^\s\"'/<>]+)")


def known_sets(world: dict) -> tuple[set[str], set[str]]:
    """Addresses and domains the user's world defines (user + contacts)."""
    emails = set()
    if isinstance(world.get("email"), str):
        emails.add(world["email"].lower())
    for c in world.get("contacts") or []:
        if isinstance(c, dict) and isinstance(c.get("email"), str):
            emails.add(c["email"].lower())
    for e in world.get("known_emails") or []:
        emails.add(str(e).lower())
    domains = {e.split("@")[-1] for e in emails} | {str(d).lower() for d in world.get("domains") or []}
    return emails, domains


def world_feats(args: dict, known: set[str], domains: set[str]) -> list[float]:
    a = json.dumps(args, default=str).lower()
    emails = EMAIL.findall(a)
    ext = [e for e in emails if e not in known and e.split("@")[-1] not in domains]
    urls = [u for u in URL.findall(a) if not any(u.endswith(d) for d in domains)]
    return [float(bool(emails)), float(bool(ext)), len(ext) / max(len(emails), 1), float(bool(urls))]


N_WORLD = 4
