"""Tests for the tamper-evident audit trail (Step 4).

The key test mutates a past log entry and confirms the chain validator catches
it -- this is the property that makes the record-keeping meaningful under
Article 12.
"""

import json
from pathlib import Path

from compliance import audit_trail as at


def _write_log(path: Path, n: int = 5) -> None:
    rows = []
    for i in range(n):
        rows.append(json.dumps({
            "timestamp": f"2026-07-05T12:{i:02d}:00+00:00",
            "features": {"age": 50 + i, "sex": i % 2},
            "risk_score": round(0.1 * i, 4),
            "prediction": int(i % 2 == 0),
        }))
    path.write_text("\n".join(rows) + "\n")


def test_build_and_verify_roundtrip(tmp_path):
    src = tmp_path / "predictions.jsonl"
    chain = tmp_path / "chain.jsonl"
    _write_log(src, 6)
    n = at.build_chain(src, chain)
    assert n == 6
    res = at.verify_chain(chain)
    assert res.valid
    assert res.n_entries == 6
    assert res.broken_at is None


def test_mutation_is_detected(tmp_path):
    """Mutate a past entry's payload -> chain must break exactly there."""
    src = tmp_path / "predictions.jsonl"
    chain = tmp_path / "chain.jsonl"
    _write_log(src, 6)
    at.build_chain(src, chain)

    lines = chain.read_text().splitlines()
    tampered = json.loads(lines[2])
    tampered["risk_score"] = 0.99          # silently change a historical score
    lines[2] = json.dumps(tampered, separators=(",", ":"))
    chain.write_text("\n".join(lines) + "\n")

    res = at.verify_chain(chain)
    assert res.valid is False
    assert res.broken_at == 2               # detected at the mutated entry
    assert "altered" in res.reason.lower()


def test_deletion_is_detected(tmp_path):
    """Remove an entry -> the link check must fail."""
    src = tmp_path / "predictions.jsonl"
    chain = tmp_path / "chain.jsonl"
    _write_log(src, 6)
    at.build_chain(src, chain)

    lines = chain.read_text().splitlines()
    del lines[3]                            # drop a record
    chain.write_text("\n".join(lines) + "\n")

    res = at.verify_chain(chain)
    assert res.valid is False
    assert res.broken_at == 3


def test_reorder_is_detected(tmp_path):
    src = tmp_path / "predictions.jsonl"
    chain = tmp_path / "chain.jsonl"
    _write_log(src, 6)
    at.build_chain(src, chain)

    lines = chain.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]  # swap two records
    chain.write_text("\n".join(lines) + "\n")

    res = at.verify_chain(chain)
    assert res.valid is False


def test_real_prediction_log_chains_and_verifies(tmp_path):
    """The real repo prediction log must build and verify cleanly.

    Copies the real log into tmp_path first so this test never writes to
    monitoring/logs/predictions_chained.jsonl -- that file is also rebuilt by
    the dashboard's "Rebuild + verify chain" button, and a test suite racing
    the dashboard on the same real file is exactly the kind of non-hermetic
    behaviour that makes CI flaky and can corrupt a file meant to be evidence.
    """
    if not at.SOURCE_LOG.exists():
        return  # nothing to check in a bare checkout
    src = tmp_path / "predictions.jsonl"
    dest = tmp_path / "predictions_chained.jsonl"
    src.write_text(at.SOURCE_LOG.read_text(encoding="utf-8"), encoding="utf-8")
    n = at.build_chain(src, dest)
    assert n > 0
    res = at.verify_chain(dest)
    assert res.valid, res.reason
