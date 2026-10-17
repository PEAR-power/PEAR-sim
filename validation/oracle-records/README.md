# Oracle comparison records

Compact ledger of differential comparisons against an independent open
engine (ngspice). One JSON per case (`pear-sim/oracle-comparison/v0.1`):
metadata, per-signal max deviation vs the declared tolerance, verdict.
Written by the live tests in `tests/test_ngspice_oracle.py` on every opt-in
run and overwritten in place — **git history is the ledger**. Every record
must be regenerable by a test in this repository
(`tests/test_oracle_record_provenance.py` enforces it).

A record here is evidence that a specific comparison ran on a specific
version against a declared ngspice version — it is not a general validation
claim beyond its exact case, and agreement between two simulators is
verification, not validation against hardware.
