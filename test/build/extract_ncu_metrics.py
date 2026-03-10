#!/usr/bin/env python3
import argparse
import csv
import io
import json
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


METRIC_ALIASES = {
    "occupancy": [
        "launch__occupancy_per_sm",
        "launch__occupancy_limit_active_warps_pct",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ],
    "memory_throughput": [
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ],
    "warp_stall": [
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    ],
    "tensor_pipe_utilization": [
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
    ],
}


def run_ncu_import(ncu_path: str, report_path: str) -> str:
    commands = [
        [ncu_path, "--import", report_path, "--page", "raw", "--csv"],
        [ncu_path, "--import", report_path, "--csv"],
    ]
    last_error = None
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            if completed.stdout.strip():
                return completed.stdout
        except subprocess.CalledProcessError as exc:
            last_error = exc.stderr or exc.stdout or str(exc)
    raise RuntimeError(last_error or "failed to import ncu report")


def parse_float(text: str) -> Optional[float]:
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def extract_metric_rows(csv_text: str) -> Dict[str, List[Tuple[List[str], Optional[float]]]]:
    rows = list(csv.reader(io.StringIO(csv_text)))
    all_metric_names = {
        metric_name
        for names in METRIC_ALIASES.values()
        for metric_name in names
    }

    extracted: Dict[str, List[Tuple[List[str], Optional[float]]]] = {}
    for row in rows:
        normalized = [cell.strip() for cell in row]
        for idx, cell in enumerate(normalized):
            if cell not in all_metric_names:
                continue
            value = None
            for candidate in normalized[idx + 1:]:
                value = parse_float(candidate)
                if value is not None:
                    break
            extracted.setdefault(cell, []).append((normalized, value))
    return extracted


def choose_metric(metric_rows: Dict[str, List[Tuple[List[str], Optional[float]]]], aliases: List[str]) -> Dict[str, object]:
    for alias in aliases:
        rows = metric_rows.get(alias)
        if not rows:
            continue
        row, value = rows[0]
        return {
            "name": alias,
            "value": value,
            "row": row,
        }
    return {
        "name": None,
        "value": None,
        "row": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract high-level metrics from an Nsight Compute report.")
    parser.add_argument("report_path", help="Path to .ncu-rep report")
    parser.add_argument("--ncu-path", default="/usr/local/cuda/bin/ncu", help="Path to ncu binary")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    csv_text = run_ncu_import(args.ncu_path, args.report_path)
    metric_rows = extract_metric_rows(csv_text)

    summary = {
        "report_path": args.report_path,
        "metrics": {
            summary_name: choose_metric(metric_rows, aliases)
            for summary_name, aliases in METRIC_ALIASES.items()
        },
        "metric_aliases": METRIC_ALIASES,
    }

    output_text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output_text)
            handle.write("\n")
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
