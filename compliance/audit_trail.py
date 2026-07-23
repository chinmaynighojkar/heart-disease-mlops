"""Step 4 -- Immutable (tamper-evident) audit trail over the prediction log.

Maps to Article 12 (record-keeping) and Article 19 (automatically generated
logs). The existing prediction log (monitoring/logs/predictions.jsonl) is an
append-only record but is not tamper-evident: a row could be edited after the
fact with no trace. This module builds a hash chain over it -- each entry stores
the SHA-256 of the previous entry, so altering ANY past record breaks the chain
from that point onward and the validator pinpoints where.

This is the same construction a blockchain uses for its block headers; here it
is applied to model prediction records to give traceability of functioning.

CLI:
  python -m compliance.audit_trail build     # write the chained log
  python -m compliance.audit_trail verify    # validate the chain
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOG = REPO_ROOT / "monitoring" / "logs" / "predictions.jsonl"
CHAIN_LOG = REPO_ROOT / "monitoring" / "logs" / "predictions_chained.jsonl"

GENESIS = "0" * 64  # prev_hash of the first entry


def _entry_hash(record: dict, prev_hash: str) -> str:
    """Deterministic hash of (payload + prev_hash). Excludes the entry's own
    hash fields so it can be recomputed for verification."""
    payload = {k: v for k, v in record.items() if k not in ("entry_hash", "prev_hash")}
    material = json.dumps(payload, sort_keys=True, separators=(",", ":")) + prev_hash
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_chain(source: Path = SOURCE_LOG, dest: Path = CHAIN_LOG) -> int:
    """Read the plain prediction log and write a hash-chained copy.
    Returns the number of chained entries."""
    if not source.exists():
        raise FileNotFoundError(f"No source log at {source}")
    lines = [ln for ln in source.read_text().splitlines() if ln.strip()]
    prev = GENESIS
    out_lines: list[str] = []
    for i, ln in enumerate(lines):
        record = json.loads(ln)
        record["seq"] = i
        record["prev_hash"] = prev
        h = _entry_hash(record, prev)
        record["entry_hash"] = h
        out_lines.append(json.dumps(record, separators=(",", ":")))
        prev = h
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return len(out_lines)


@dataclass
class ChainResult:
    valid: bool
    n_entries: int
    broken_at: int | None = None   # seq index of first tampered/broken entry
    reason: str = ""

    @property
    def head_hash(self) -> str:
        return self._head

    _head: str = GENESIS


def verify_chain(chain: Path = CHAIN_LOG) -> ChainResult:
    """Recompute every hash and confirm each entry links to the previous one.
    Detects: edited payloads, altered/removed entries, reordering."""
    if not chain.exists():
        return ChainResult(valid=False, n_entries=0, reason=f"No chain file at {chain}")
    lines = [ln for ln in chain.read_text().splitlines() if ln.strip()]
    prev = GENESIS
    for i, ln in enumerate(lines):
        record = json.loads(ln)
        # 1. link integrity
        if record.get("prev_hash") != prev:
            return ChainResult(False, len(lines), broken_at=i,
                               reason=f"prev_hash mismatch at seq {i}: chain does not link "
                                      "to the previous entry (entry removed, reordered, or edited).")
        # 2. payload integrity
        recomputed = _entry_hash(record, prev)
        if record.get("entry_hash") != recomputed:
            return ChainResult(False, len(lines), broken_at=i,
                               reason=f"entry_hash mismatch at seq {i}: this record's contents "
                                      "were altered after it was written.")
        prev = record["entry_hash"]
    res = ChainResult(True, len(lines), reason="Chain intact: all entries verified.")
    res._head = prev
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prediction-log hash chain (AI Act Art. 12).")
    ap.add_argument("command", choices=["build", "verify"])
    args = ap.parse_args(argv)
    if args.command == "build":
        n = build_chain()
        print(f"Built hash chain: {n} entries -> {CHAIN_LOG}")
        res = verify_chain()
        print(f"Self-check: {'VALID' if res.valid else 'INVALID'} ({res.reason})")
        print(f"Head hash: {res.head_hash}")
        return 0
    res = verify_chain()
    if res.valid:
        print(f"CHAIN VALID: {res.n_entries} entries. Head hash {res.head_hash[:16]}...")
        return 0
    print(f"CHAIN INVALID at seq {res.broken_at}: {res.reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
