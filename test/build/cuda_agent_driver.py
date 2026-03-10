#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_TRIALS_PATH = PROJECT_ROOT / "test" / "nsys" / "cuda_agent_trials.jsonl"


def run_command(command: List[str], cwd: Path, env: Dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def parse_trial_output(output: str) -> Dict[str, object]:
    parsed: Dict[str, object] = {
        "trial_case": None,
        "trial_metrics": [],
        "launch_config": None,
        "trial_profile": None,
    }

    case_pattern = re.compile(
        r"\[TrialCase\] program=(?P<program>\S+) role=(?P<role>\S+) trial_id=(?P<trial_id>\S+) "
        r"sms=(?P<sms>\S+) tile_config=(?P<tile_config>\S+) M=(?P<M>\d+) K=(?P<K>\d+) N=(?P<N>\d+) SPLIT_K=(?P<SPLIT_K>\d+)"
    )
    metric_pattern = re.compile(
        r"\[TrialMetric\] kernel=(?P<kernel>\S+) time_ms=(?P<time_ms>[-+0-9.eE]+) "
        r"tflops=(?P<tflops>[-+0-9.eE]+) error=(?P<error>[-+0-9.eE]+)"
    )
    launch_pattern = re.compile(
        r"\[SpMM\] launch_config mode=(?P<mode>\S+) label=(?P<label>\S+) N=(?P<N>\d+) SplitK=(?P<SplitK>\d+) "
        r"tile=\(K=(?P<mma_k>\d+),row_warps=(?P<block_row_warps>\d+),col_warps=(?P<block_col_warps>\d+),"
        r"warp_col=(?P<warp_col_tensors>\d+),warp_row=(?P<warp_row_tensors>\d+),"
        r"block_k_steps=(?P<block_k_steps>\d+),stages=(?P<stages>\d+)\)"
    )
    profile_pattern = re.compile(
        r"\[TrialProfile\] mode=(?P<mode>\S+) program_path=(?P<program_path>\S+) report_path=(?P<report_path>\S+) "
        r"log_path=(?P<log_path>\S+) M=(?P<M>\d+) K=(?P<K>\d+) N=(?P<N>\d+) SPLIT_K=(?P<SPLIT_K>\d+) "
        r"kernel_filter=(?P<kernel_filter>.*)"
    )

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue

        case_match = case_pattern.match(line)
        if case_match:
            parsed["trial_case"] = {
                key: int(value) if key in {"M", "K", "N", "SPLIT_K"} else value
                for key, value in case_match.groupdict().items()
            }
            continue

        metric_match = metric_pattern.match(line)
        if metric_match:
            parsed["trial_metrics"].append(
                {
                    "kernel": metric_match.group("kernel"),
                    "time_ms": float(metric_match.group("time_ms")),
                    "tflops": float(metric_match.group("tflops")),
                    "error": float(metric_match.group("error")),
                }
            )
            continue

        launch_match = launch_pattern.match(line)
        if launch_match:
            parsed["launch_config"] = {
                key: int(value) if key not in {"mode", "label"} else value
                for key, value in launch_match.groupdict().items()
            }
            continue

        profile_match = profile_pattern.match(line)
        if profile_match:
            parsed["trial_profile"] = {
                key: int(value) if key in {"M", "K", "N", "SPLIT_K"} else value
                for key, value in profile_match.groupdict().items()
            }
            continue

    return parsed


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def maybe_extract_ncu_metrics(report_path: Optional[str], ncu_path: str) -> Optional[Dict[str, object]]:
    if not report_path:
        return None
    command = [
        sys.executable,
        str(SCRIPT_DIR / "extract_ncu_metrics.py"),
        report_path,
        "--ncu-path",
        ncu_path,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {
            "error": completed.stderr.strip() or completed.stdout.strip() or "metric extraction failed",
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"error": "invalid metrics json", "raw": completed.stdout}


def default_candidates_for_n(n: int) -> List[Dict[str, Optional[str]]]:
    candidates = [{"label": "compiled_default", "tile_config": None}]
    if n == 128:
        candidates.append({"label": "n128_wide_tile", "tile_config": "16,4,2,8,2,4,2"})
        candidates.append({"label": "n128_compact_tile", "tile_config": "16,2,2,4,4,4,2"})
    return candidates


def build_targets(args: argparse.Namespace, env: Dict[str, str]) -> None:
    if args.build == "skip":
        return

    targets: List[Path] = []
    if args.build in {"kernel", "both"}:
        targets.append(SCRIPT_DIR)
    if args.build in {"bench", "both"}:
        targets.append(PROJECT_ROOT / "bench")
    if args.build == "auto":
        targets.append(SCRIPT_DIR if args.program == "kernel" else PROJECT_ROOT / "bench")

    for directory in targets:
        make_env = env.copy()
        make_env["SPMM_BUILD_SMS"] = str(args.sms)
        command = ["make", "clean", "all", f"SMS={args.sms}"]
        completed = run_command(command, directory, make_env)
        if completed.returncode != 0:
            raise RuntimeError(
                f"build failed in {directory}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )


def run_case(
    args: argparse.Namespace,
    env: Dict[str, str],
    candidate: Dict[str, Optional[str]],
) -> Dict[str, object]:
    trial_id = f"{candidate['label']}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    candidate_env = env.copy()
    candidate_env["SPMM_TRIAL_ID"] = trial_id
    candidate_env["SPMM_BUILD_SMS"] = str(args.sms)
    candidate_env["SPMM_PRINT_CONFIG"] = "1"
    candidate_env["SPMM_PREVIEW_COUNT"] = str(args.preview_count)
    if candidate["tile_config"]:
        candidate_env["SPMM_TILE_CONFIG"] = candidate["tile_config"]
    else:
        candidate_env.pop("SPMM_TILE_CONFIG", None)

    record: Dict[str, object] = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "program_mode": args.program,
        "case": {"M": args.m, "K": args.k, "N": args.n, "SPLIT_K": args.split_k},
        "build": {"sms": args.sms},
        "candidate": candidate,
    }

    if args.mode in {"correctness", "full"}:
        command = [str(SCRIPT_DIR / "kernelTest.sh"), args.program, str(args.m), str(args.k), str(args.n), str(args.split_k)]
        completed = run_command(command, SCRIPT_DIR, candidate_env)
        record["correctness"] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "parsed": parse_trial_output(completed.stdout + "\n" + completed.stderr),
        }

    if args.mode in {"profile", "full"}:
        ncu_env = candidate_env.copy()
        if args.no_sudo:
            ncu_env["NCU_NO_SUDO"] = "1"
        ncu_env["REPORT_PREFIX"] = args.report_prefix
        command = [str(SCRIPT_DIR / "ncu.sh"), args.program, str(args.m), str(args.k), str(args.n), str(args.split_k)]
        completed = run_command(command, SCRIPT_DIR, ncu_env)
        parsed_output = parse_trial_output(completed.stdout + "\n" + completed.stderr)
        trial_profile = parsed_output.get("trial_profile") or {}
        report_path = trial_profile.get("report_path") if isinstance(trial_profile, dict) else None
        record["profile"] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "parsed": parsed_output,
            "report_metrics": maybe_extract_ncu_metrics(report_path, args.ncu_path),
        }

    append_jsonl(Path(args.trials_file), record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="CUDA-Agent style driver for cudaCode_v0.")
    parser.add_argument("--program", choices=["kernel", "bench"], default="bench")
    parser.add_argument("--mode", choices=["correctness", "profile", "full"], default="full")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--split-k", type=int, default=4)
    parser.add_argument("--sms", type=int, default=86)
    parser.add_argument("--build", choices=["auto", "kernel", "bench", "both", "skip"], default="auto")
    parser.add_argument("--preview-count", type=int, default=0)
    parser.add_argument("--tile-config", action="append", default=[], help="Override tile config as csv tuple K,row_warps,col_warps,warp_col,warp_row,block_k_steps,stages")
    parser.add_argument("--trials-file", default=str(DEFAULT_TRIALS_PATH))
    parser.add_argument("--report-prefix", default="cuda_agent")
    parser.add_argument("--ncu-path", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--no-sudo", action="store_true")
    args = parser.parse_args()

    base_env = os.environ.copy()
    build_targets(args, base_env)

    candidates = (
        [{"label": f"candidate_{idx}", "tile_config": tile_config} for idx, tile_config in enumerate(args.tile_config, start=1)]
        if args.tile_config
        else default_candidates_for_n(args.n)
    )

    records = [run_case(args, base_env, candidate) for candidate in candidates]

    print(json.dumps({"trials_written": len(records), "trials_file": args.trials_file}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
