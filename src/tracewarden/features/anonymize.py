"""Remove world-identity shortcuts - novel addition N4.

The AgentDrift README reports that a lookup on the world object (user, company, contacts) predicts the
attacked class with 93-98% accuracy in four of five domains. Any model that reads names or addresses can
learn the category from identity. Two transformations remove that signal while keeping the *relations*
an attack depends on (an address is either inside the user's organization or outside it):

* ``anonymize``        - deterministic placeholders (PERSON_1, ORG_1, user1@org1.example, ext1@ext1.example).
                         This is the procedure the AgentDrift authors recommend.
* ``identity_shuffle`` - consistent, random, realistic-looking replacements per trajectory. Used as data
                         augmentation, so the model sees each attack pattern under many identities.
"""
from __future__ import annotations

import copy
import random
import re
from collections.abc import Callable
from typing import Any

from ..schema import Trajectory
from .world import EMAIL, known_sets

FIRST = ["Amina", "Bilal", "Carmen", "Dario", "Esra", "Farid", "Grace", "Hamza", "Ines", "Jonas", "Kiran",
         "Layla", "Mateo", "Nadia", "Omar", "Priya", "Quentin", "Rania", "Samir", "Tariq", "Uma", "Viktor",
         "Wen", "Yusuf", "Zara", "Arjun", "Beatriz", "Chen", "Dmitri", "Elif"]
LAST = ["Qureshi", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Haddad", "Novak", "Ferreira", "Iqbal", "Kowalski",
        "Mensah", "Rossi", "Sato", "Varga", "Yilmaz", "Zhou", "Abbasi", "Brennan", "Castillo", "Duarte"]
ORG_A = ["Northwind", "Bluefield", "Crescent", "Meridian", "Ironbark", "Sablewood", "Tidewater", "Halcyon"]
ORG_B = ["Systems", "Holdings", "Labs", "Partners", "Group", "Health", "Financial", "Works"]
TLD = ["com", "net", "io", "org", "co"]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower()) or "x"


def _map_strings(obj: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_strings(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_strings(v, fn) for v in obj]
    return obj


def _all_text(t: Trajectory) -> str:
    parts = [t.user_task, str(t.world)]
    for s in t.steps:
        parts += [s.thought, str(s.args), s.obs]
    return "\n".join(parts)


def _rewrite(t: Trajectory, person_new: Callable[[int], str], org_new: Callable[[int], str],
             internal_domain: Callable[[int], str], external_domain: Callable[[int], str],
             local_new: Callable[[int, bool], str]) -> Trajectory:
    t = copy.deepcopy(t)
    w = t.world or {}
    known, own = known_sets(w)
    text = _all_text(t)

    # people: full names, then first names on their own
    people = [w.get("user")] + [c.get("name") for c in w.get("contacts") or [] if isinstance(c, dict)]
    people = [p for p in dict.fromkeys(people) if isinstance(p, str) and p.strip()]
    name_map: dict[str, str] = {}
    for i, p in enumerate(people):
        new = person_new(i)
        name_map[p] = new
        first = p.split()[0]
        if len(first) >= 3 and first not in name_map:
            name_map[first] = new.split()[0]
    orgs = [w.get("company")] if isinstance(w.get("company"), str) and w.get("company") else []
    org_map = {o: org_new(i) for i, o in enumerate(orgs)}

    # addresses: map every domain seen anywhere, keeping inside/outside the organization
    dom_map: dict[str, str] = {}
    n_int = n_ext = 0
    for e in dict.fromkeys(m.group(0).lower() for m in EMAIL.finditer(text)):
        d = e.split("@")[-1]
        if d not in dom_map:
            if d in own:
                dom_map[d] = internal_domain(n_int)
                n_int += 1
            else:
                dom_map[d] = external_domain(n_ext)
                n_ext += 1
    email_map: dict[str, str] = {}
    counters = {True: 0, False: 0}
    for e in dict.fromkeys(m.group(0).lower() for m in EMAIL.finditer(text)):
        d = e.split("@")[-1]
        inside = d in own
        email_map[e] = f"{local_new(counters[inside], inside)}@{dom_map[d]}"
        counters[inside] += 1

    # one single-pass regex so a replacement is never replaced again
    table = {k.lower(): v for k, v in {**dom_map, **name_map, **org_map}.items()}
    table.update(email_map)
    keys = sorted(table, key=len, reverse=True)
    rx = re.compile(r"(?<![\w.@-])(?:" + "|".join(re.escape(k) for k in keys) + r")(?![\w@-])", re.I) if keys else None

    def fn(s: str) -> str:
        return rx.sub(lambda m: table.get(m.group(0).lower(), m.group(0)), s) if rx else s

    t.user_task = fn(t.user_task)
    t.world = _map_strings(w, fn)
    for st in t.steps:
        st.thought, st.obs = fn(st.thought), fn(st.obs)
        st.args = _map_strings(st.args, fn)
    t.meta = {**t.meta, "identity": "rewritten"}
    return t


def anonymize(t: Trajectory) -> Trajectory:
    return _rewrite(
        t,
        person_new=lambda i: f"Person{i + 1} Surname{i + 1}",
        org_new=lambda i: f"Org{i + 1}",
        internal_domain=lambda i: f"org{i + 1}.example",
        external_domain=lambda i: f"ext{i + 1}.example",
        local_new=lambda i, inside: f"{'user' if inside else 'contact'}{i + 1}",
    )


def identity_shuffle(t: Trajectory, seed: int) -> Trajectory:
    rnd = random.Random(seed)
    firsts, lasts = rnd.sample(FIRST, len(FIRST)), rnd.sample(LAST, len(LAST))
    org = f"{rnd.choice(ORG_A)} {rnd.choice(ORG_B)}"
    ext_words = rnd.sample(ORG_A + ["secure", "verify", "audit", "support", "billing", "notice"], 12)

    def person(i: int) -> str:
        return f"{firsts[i % len(firsts)]} {lasts[i % len(lasts)]}"

    return _rewrite(
        t,
        person_new=person,
        org_new=lambda i: org,
        internal_domain=lambda i: f"{_slug(org)}{'' if i == 0 else i}.{TLD[0]}",
        external_domain=lambda i: f"{ext_words[i % 12].lower()}-{rnd.randint(10, 99)}.{rnd.choice(TLD)}",
        local_new=lambda i, inside: (f"{firsts[i % 30].lower()}.{lasts[i % 20].lower()}" if inside
                                     else rnd.choice(["info", "team", "desk", "ops", "admin", "help"]) + str(i)),
    )


def world_fingerprint(t: Trajectory) -> tuple:
    """Key used by the shortcut audit: identical worlds share a fingerprint."""
    w = t.world or {}
    return (w.get("user"), w.get("company"),
            tuple(sorted(c.get("email", "") for c in w.get("contacts") or [] if isinstance(c, dict))))
