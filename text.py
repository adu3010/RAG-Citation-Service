"""Text normalisation helpers shared by the lexical and dense retrieval paths.

Both retrievers must agree on what a "token" is, otherwise scores drift apart in
ways that are very hard to debug. Keeping one implementation here is deliberate.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Small, explicit stop list, in two parts.
#
# The first group is ordinary function words. The second is query scaffolding:
# interrogatives and auxiliaries. These matter more than they look. Leaving "what"
# and "how" in place means a question like "What is the best pizza in Naples?"
# matches any passage containing the word "what", the retriever returns something,
# and the generator answers an out-of-domain question instead of abstaining. That
# is a silent correctness bug, not a ranking nuance.
#
# Negations ("no", "not", "without") are deliberately absent: on short technical
# and clinical queries they carry meaning and dropping them inverts intent.
STOPWORDS = frozenset(
    """
    a an and are as at be by for from has have in into is it its of on or that
    the to was were will with
    how what when where which who whom whose why does do did done
    this these those there here am been being can could should would
    i me my we our you your they them their he she his her
    """.split()  # noqa: SIM905 - a multi-line string keeps this list reviewable
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lowercase alphanumeric tokenisation.

    >>> tokenize("The ICB's Band-5 role, 2026!")
    ['icb', 's', 'band', '5', 'role', '2026']
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def bigrams(tokens: list[str]) -> list[str]:
    # strict=False is intentional: this is a sliding window, so the second
    # sequence is deliberately one shorter than the first.
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter.

    Deliberately not using a model here: chunk boundaries need to be reproducible
    across machines and CI runs, and a regex is auditable.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


def token_overlap(a: str, b: str) -> float:
    """Jaccard-style overlap of ``a``'s tokens against ``b``. Used for groundedness."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)
