"""Text normalization that defeats cheap obfuscation (N8) before anything is encoded."""
from __future__ import annotations

import re
import unicodedata

ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
# a few Cyrillic/Greek homoglyphs commonly used to dodge keyword filters
HOMOGLYPHS = str.maketrans({
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c", "\u0443": "y", "\u0445": "x",
    "\u0410": "A", "\u0415": "E", "\u041e": "O", "\u0420": "P", "\u0421": "C", "\u0425": "X",
    "\u03bf": "o", "\u03b1": "a", "\u0391": "A", "\u039f": "O",
})


def normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = ZERO_WIDTH.sub("", t)
    t = t.translate(HOMOGLYPHS)
    return re.sub(r"[ \t]+", " ", t)
