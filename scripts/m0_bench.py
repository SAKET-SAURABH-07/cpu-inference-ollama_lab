"""Runner for M0 baseline benchmarks using llama-bench.

Automates environment recording and raw benchmark runs with pre-run
system state captures.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M0 benchmark runner")
    parser.add_argument(
        "--llama-dir",
        type=Path,
        default=Path(r"tools\llama.cpp\b11871"),
        help="Directory containing llama.cpp binaries",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(r"models\qwen2.5-0.5b-instruct-q4_k_m.gguf"),
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of benchmark repetitions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands, check executable and model exist, and exit",
    )
    return parser.parse_args()


def find_llama_bench(llama_dir: Path) -> Path | None:
    """Find llama-bench executable in the given directory or its subdirectories."""
    direct = llama_dir / "llama-bench.exe"
    if direct.is_file():
        return direct
    direct_no_ext = llama_dir / "llama-bench"
    if direct_no_ext.is_file():
        return direct_no_ext
    for p in llama_dir.rglob("llama-bench.exe"):
        if p.is_file():
            return p
    for p in llama_dir.rglob("llama-bench"):
        if p.is_file():
            return p
    return None


def safe_write_text(path: Path, content: str) -> None:
    """Write text to path without overwriting if it already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Target file already exists: {path}")
    with open(path, "x", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    args = parse_args()

    bench_exe = find_llama_bench(args.llama_dir)
    model_path = args.model

    if args.dry_run:
        print("[DRY RUN] Verifying files:")
        if bench_exe is not None and bench_exe.is_file():
            print(f"  Executable: FOUND -> {bench_exe}")
        else:
            print(f"  Executable: NOT FOUND under {args.llama_dir}")

        if model_path.is_file():
            print(f"  Model:      FOUND -> {model_path}")
        else:
            print(f"  Model:      NOT FOUND -> {model_path}")

        print("\n[DRY RUN] Planned commands:")
        env_cmd = [
            sys.executable,
            str(Path("scripts/record_env.py")),
            "--llama-dir",
            str(args.llama_dir),
            "--label",
            "m0-bench",
        ]
        print(f"  Step 1 (Env Record): {' '.join(env_cmd)}")

        print(f"  Step 2 ({args.runs} benchmark runs):")
        bench_cmd = [
            str(bench_exe or (args.llama_dir / "llama-bench.exe")),
            "-m",
            str(model_path),
            "-p",
            "512",
            "-n",
            "128",
            "-t",
            "6",
            "-ngl",
            "0",
            "-r",
            "5",
            "-o",
            "json",
        ]
        for i in range(1, args.runs + 1):
            print(f"    Run {i}: {' '.join(bench_cmd)}")

        print("\n[DRY RUN] Planned output paths:")
        print("  experiments/results/raw/m0_<timestamp>_run<i>.json")
        print("  experiments/results/raw/m0_<timestamp>_run<i>.stderr.log")
        print("  experiments/results/raw/m0_<timestamp>_meta.json")
        sys.exit(0)

    print("Run from a plain terminal with other apps closed. Press Enter to start.")
    input()

    if bench_exe is None or not bench_exe.is_file():
        raise FileNotFoundError(f"llama-bench executable not found in {args.llama_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments/results/raw")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Record environment
    env_cmd = [
        sys.executable,
        str(Path("scripts/record_env.py")),
        "--llama-dir",
        str(args.llama_dir),
        "--label",
        "m0-bench",
    ]
    env_start = datetime.now().astimezone().isoformat()
    env_t0 = time.perf_counter()
    env_proc = subprocess.run(env_cmd)
    env_t1 = time.perf_counter()
    env_end = datetime.now().astimezone().isoformat()
    print(f"Step 1 complete (return code: {env_proc.returncode})")

    # Step 2: Benchmark runs
    bench_cmd = [
        str(bench_exe),
        "-m",
        str(model_path),
        "-p",
        "512",
        "-n",
        "128",
        "-t",
        "6",
        "-ngl",
        "0",
        "-r",
        "5",
        "-o",
        "json",
    ]

    runs_meta = []
    for i in range(1, args.runs + 1):
        pre_ram_available_bytes = psutil.virtual_memory().available
        pre_cpu_percent = psutil.cpu_percent(interval=1)

        run_start_iso = datetime.now().astimezone().isoformat()
        run_t0 = time.perf_counter()
        proc = subprocess.run(
            bench_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        run_t1 = time.perf_counter()
        run_end_iso = datetime.now().astimezone().isoformat()
        wall_seconds = run_t1 - run_t0

        stdout_path = results_dir / f"m0_{timestamp}_run{i}.json"
        stderr_path = results_dir / f"m0_{timestamp}_run{i}.stderr.log"

        safe_write_text(stdout_path, proc.stdout)
        safe_write_text(stderr_path, proc.stderr)

        print(f"Run {i}: wrote {stdout_path} (return code: {proc.returncode})")
        print(f"Run {i}: wrote {stderr_path}")

        runs_meta.append({
            "run_index": i,
            "command": bench_cmd,
            "start_time": run_start_iso,
            "end_time": run_end_iso,
            "wall_seconds": wall_seconds,
            "returncode": proc.returncode,
            "pre_run_available_ram_bytes": pre_ram_available_bytes,
            "pre_run_cpu_percent": pre_cpu_percent,
            "stdout_file": str(stdout_path),
            "stderr_file": str(stderr_path),
        })

    meta_path = results_dir / f"m0_{timestamp}_meta.json"
    meta = {
        "session_timestamp": timestamp,
        "step1_env_record": {
            "command": env_cmd,
            "start_time": env_start,
            "end_time": env_end,
            "wall_seconds": env_t1 - env_t0,
            "returncode": env_proc.returncode,
        },
        "step2_runs": runs_meta,
    }
    safe_write_text(meta_path, json.dumps(meta, indent=2))
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
