"""The public tree must not carry material that does not belong to it.

This repository is published openly. Three classes of text must never enter
it: references to a commercial simulator used during private development,
references to private company material, and local machine paths. A forbidden
term is checked here, in the suite that every pull request runs, so that it
is caught where the damage would happen (a merge), not in a review someone
may skip.

The gate is only worth something if it discriminates: the self-tests below
prove it fires on an injected term and stays silent on clean text.
"""

from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
THIS_FILE = Path(__file__).resolve()

# Term -> why it is forbidden. Matched case-insensitively on word boundaries.
# The terms are stored base64-encoded so that this file itself does not name
# what it keeps out of the repository.
_ENCODED = {
    "cGxlY3M=": "commercial simulator; the public evidence base is ngspice and closed forms",
    "cGxleGlt": "commercial simulator vendor",
    "Y3Rv": "private company material",
    "cGVhci1icmFpbg==": "private planning material",
    "YW52aWw=": "unrelated private project",
    "Y3Rlb3M=": "unrelated private project",
}
FORBIDDEN = {base64.b64decode(key).decode(): why for key, why in _ENCODED.items()}
_FORBIDDEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(term) for term in FORBIDDEN) + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_LOCAL_PATH_RE = re.compile(r"/Users/[A-Za-z0-9._-]+|[A-Za-z]:\\\\Users\\\\")
_POLISH_RE = re.compile(r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]")

# Files that are allowed to break a rule, with the reason. Keep this short.
_SKIP = {"uv.lock"}


def scan_text(text: str) -> list[str]:
    """Return a description of every forbidden occurrence in ``text``."""

    hits = [f"forbidden term {m.group(1)!r}" for m in _FORBIDDEN_RE.finditer(text)]
    hits += [f"local path {m.group(0)!r}" for m in _LOCAL_PATH_RE.finditer(text)]
    hits += [f"Polish text {m.group(0)!r}" for m in _POLISH_RE.finditer(text)]
    return hits


def _tracked_files() -> list[Path]:
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
        ).stdout.decode()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / name for name in listing.split("\0") if name]


# --- self-tests: the gate must discriminate ---------------------------------


def test_the_gate_fires_on_each_forbidden_term() -> None:
    for term in FORBIDDEN:
        assert scan_text(f"compared against {term.upper()} last week"), term


def test_the_gate_fires_on_a_local_path_and_on_polish_text() -> None:
    assert scan_text("see /Users/someone/project/run.json")
    assert scan_text("napięcie wyjściowe")


def test_the_gate_stays_silent_on_clean_text() -> None:
    clean = (
        "A buck converter at 20 kHz; the ngspice oracle agrees within 1e-3. "
        "Factor, vector, octopus and doctor contain the letters but not the term."
    )
    assert scan_text(clean) == []


# --- the gate itself ---------------------------------------------------------


def test_no_tracked_file_carries_forbidden_material() -> None:
    offenders: list[str] = []
    for path in _tracked_files():
        if path.resolve() == THIS_FILE or path.name in _SKIP or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary
        hits = scan_text(text) + scan_text(str(path.relative_to(ROOT)))
        if hits:
            offenders.append(f"{path.relative_to(ROOT)}: {sorted(set(hits))[:5]}")
    assert not offenders, "forbidden material in the public tree:\n" + "\n".join(offenders)
