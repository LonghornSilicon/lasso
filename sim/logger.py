"""
Structured logger for LASSO simulation sweep findings.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict


class SimLogger:
    """
    Logs simulation findings to JSONL and produces a markdown summary.

    Severity levels: PASS, WARN, FAIL, OVERFLOW, EDGE
    """

    SEVERITIES = ["PASS", "WARN", "FAIL", "OVERFLOW", "EDGE"]

    def __init__(self, outdir: str = "sim/results"):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        log_path = os.path.join(outdir, "findings.jsonl")
        self._fh = open(log_path, "w", encoding="utf-8")
        self._findings = []  # list of dicts for all non-PASS findings
        self._counts: Dict[str, Dict[str, int]] = {}  # block -> severity -> count

    def log(
        self,
        block: str,
        test_name: str,
        status: str,
        params: Dict[str, Any],
        detail: str,
        severity: str = "INFO",
    ):
        """Append a JSON line to findings.jsonl."""
        # Normalize severity
        severity = severity.upper()
        if severity not in self.SEVERITIES:
            severity = "INFO"

        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "block": block,
            "test": test_name,
            "status": status,
            "severity": severity,
            "params": params,
            "detail": detail,
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

        # Track counts
        if block not in self._counts:
            self._counts[block] = {}
        self._counts[block][severity] = self._counts[block].get(severity, 0) + 1

        # Keep non-PASS findings for summary
        if severity != "PASS":
            self._findings.append(record)

    def summary(self) -> Dict[str, int]:
        """
        Print a table of counts by block and severity.
        Returns a flat dict of {severity: total_count}.
        """
        # Collect all severity types used
        all_severities = sorted(
            {sev for counts in self._counts.values() for sev in counts},
            key=lambda s: self.SEVERITIES.index(s) if s in self.SEVERITIES else 99,
        )

        header_parts = ["Block".ljust(12)] + [s.ljust(10) for s in all_severities]
        print("\n" + "=" * 70)
        print("SIMULATION SWEEP SUMMARY")
        print("=" * 70)
        print("  ".join(header_parts))
        print("-" * 70)

        totals: Dict[str, int] = {}
        for block in sorted(self._counts):
            row_parts = [block.ljust(12)]
            for sev in all_severities:
                cnt = self._counts[block].get(sev, 0)
                totals[sev] = totals.get(sev, 0) + cnt
                row_parts.append(str(cnt).ljust(10))
            print("  ".join(row_parts))

        print("-" * 70)
        total_row = ["TOTAL".ljust(12)]
        for sev in all_severities:
            total_row.append(str(totals.get(sev, 0)).ljust(10))
        print("  ".join(total_row))
        print("=" * 70 + "\n")

        return totals

    def close(self):
        """Close the log file and write summary.md."""
        self._fh.close()
        self._write_summary_md()

    def _write_summary_md(self):
        """Write sim/results/summary.md with a markdown table of WARN/FAIL/OVERFLOW/EDGE findings."""
        md_path = os.path.join(self.outdir, "summary.md")
        notable = [
            f for f in self._findings
            if f["severity"] in ("WARN", "FAIL", "OVERFLOW", "EDGE")
        ]

        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# LASSO Simulation Sweep Summary\n\n")
            fh.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")

            if not notable:
                fh.write("No notable findings (WARN/FAIL/OVERFLOW/EDGE).\n")
                return

            fh.write(
                "| Block | Test | Severity | Status | Params | Detail |\n"
                "|-------|------|----------|--------|--------|--------|\n"
            )
            for f in notable:
                params_str = ", ".join(
                    f"{k}={v}" for k, v in f["params"].items()
                )
                # Escape pipe characters
                detail = f["detail"].replace("|", "\\|")
                params_str = params_str.replace("|", "\\|")
                fh.write(
                    f"| {f['block']} | {f['test']} | {f['severity']} "
                    f"| {f['status']} | {params_str} | {detail} |\n"
                )
