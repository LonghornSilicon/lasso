"""
Main simulation runner for LASSO sweep framework.

Runs all four block sweeps (KVE, TIU, MHC, LACU), prints a summary table,
writes findings.jsonl and summary.md, then exits with code 1 if any FAIL
findings are present.
"""

import sys
from sim.logger import SimLogger
from sim.sweep_kve import run_kve_sweeps
from sim.sweep_tiu import run_tiu_sweeps
from sim.sweep_mhc import run_mhc_sweeps
from sim.sweep_lacu import run_lacu_sweeps


def main():
    logger = SimLogger()
    run_kve_sweeps(logger)
    run_tiu_sweeps(logger)
    run_mhc_sweeps(logger)
    run_lacu_sweeps(logger)
    summary = logger.summary()
    logger.close()
    if summary.get("FAIL", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
