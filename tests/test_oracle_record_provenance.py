# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Evidence in the repository must be claimed by a test that regenerates it.

An oracle record is a claim: two independent engines were compared, and here
is how far apart they were. The record is only worth its weight if the
comparison can be RUN AGAIN — otherwise the repository carries a number whose
provenance is a session that no longer exists.

The failure this guards against is a record produced by a one-off script
whose procedure was later lost: the number survives in git while the ability
to reproduce it does not. Every record must name a case that
some test declares, and every declared case must own a record or be plainly
absent (never silently orphaned).

The guard runs WITHOUT the oracles installed — it reads the records and the
test sources, not the engines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RECORDS_DIR = ROOT / "validation" / "oracle-records"
TESTS_DIR = ROOT / "tests"


def _records() -> list[tuple[Path, dict]]:
    found = []
    for path in sorted(RECORDS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict), path.name
        found.append((path, payload))
    return found


def _test_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(TESTS_DIR.glob("*.py"))
    )


def test_records_exist_at_all() -> None:
    # A tripwire: if the records directory is empty the rest of this module
    # would pass vacuously.
    assert len(_records()) >= 4


@pytest.mark.parametrize("path,payload", _records(), ids=lambda value: getattr(value, "name", ""))
def test_every_record_declares_its_case_and_verdict(path: Path, payload: dict) -> None:
    for field in ("schema", "case_id", "verdict"):
        assert field in payload, f"{path.name} lacks {field}"
    assert payload["verdict"] in {"pass", "fail"}, path.name
    assert payload["schema"] == "pear-sim/oracle-comparison/v0.1", path.name


@pytest.mark.parametrize("path,payload", _records(), ids=lambda value: getattr(value, "name", ""))
def test_record_file_is_named_after_its_case(path: Path, payload: dict) -> None:
    """The harness writes ``<case_id>.json``, so a mismatch means hand-made.

    Both records that turned out to be unreproducible had this fingerprint:
    a file name that the persisting helper could never have produced. Holding
    the invariant makes the harness the only way evidence enters the tree.
    """

    assert path.stem == payload["case_id"], (
        f"{path.name} was not written by the oracle harness (case_id "
        f"{payload['case_id']!r}), so nothing regenerates it in place"
    )


def test_no_record_is_an_orphan() -> None:
    """Every retained record's case_id must appear in a test source.

    The test that mentions the case is the one that can regenerate the record,
    so this is the link between a number in the repository and the procedure
    that produced it.
    """

    sources = _test_sources()
    orphans = [
        path.name
        for path, payload in _records()
        if str(payload.get("case_id")) not in sources
    ]
    assert not orphans, (
        "these records carry no test that can regenerate them, so their "
        f"provenance is unverifiable: {orphans}"
    )


def test_a_failing_verdict_would_never_be_retained_silently() -> None:
    """A retained record with a failing verdict must be an explicit exception.

    Nothing in the repository should be able to keep a red comparison quietly;
    if a case is knowingly failing it belongs in the validation status prose,
    not only in a record file.
    """

    failing = [
        path.name for path, payload in _records() if payload.get("verdict") != "pass"
    ]
    assert not failing, f"failing oracle records retained without a note: {failing}"
