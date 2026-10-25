"""Summarize an M0 llama-bench session from its raw JSON files.

Reads experiments/results/raw/<session>_run*.json (llama-bench -o json output).
For each test (pp512, tg128, ...) it prints per-process and pooled statistics
computed from the raw per-repetition samples (samples_ts, tokens/s).

It does not compare against predictions; that belongs in docs/postmortem.md.
Stdlib only. Standard deviation is the sample standard deviation (n - 1).

Usage (from the repo root):
    python scripts/analyze_m0.py
    python scripts/analyze_m0.py --session m0_20261025_123707
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

# Settings that must match across every row that gets pooled together.
CONSISTENCY_KEYS = (
    "build_commit",
    "model_filename",
    "n_threads",
    "n_batch",
    "n_ubatch",
    "type_k",
    "type_v",
    "n_gpu_layers",
    "load_mode",
    "repack",
    "flash_attn",
)


def test_name(row: dict) -> str:
    """Label a llama-bench row, e.g. pp512, tg128, tg128@d2048."""
    n_prompt, n_gen = row["n_prompt"], row["n_gen"]
    depth = f"@d{row['n_depth']}" if row.get("n_depth") else ""
    if n_prompt and n_gen:
        return f"pp{n_prompt}+tg{n_gen}{depth}"
    if n_prompt:
        return f"pp{n_prompt}{depth}"
    return f"tg{n_gen}{depth}"


def summarize(samples: list[float]) -> dict:
    n = len(samples)
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if n > 1 else float("nan")
    return {
        "n": n,
        "mean": mean,
        "stdev": stdev,
        "cv_pct": 100.0 * stdev / mean if n > 1 else float("nan"),
        "min": min(samples),
        "max": max(samples),
    }


def run_index(path: Path) -> int:
    m = re.search(r"_run(\d+)\.json$", path.name)
    return int(m.group(1)) if m else 0


def latest_session(raw_dir: Path) -> str:
    metas = sorted(raw_dir.glob("m0_*_meta.json"))
    if not metas:
        raise FileNotFoundError(f"no m0_*_meta.json found in {raw_dir}")
    return metas[-1].name.removesuffix("_meta.json")


def load_session(raw_dir: Path, session: str) -> dict[str, list[dict]]:
    """Return {test name: [{"run": index, "row": llama-bench row}, ...]}."""
    files = sorted(raw_dir.glob(f"{session}_run*.json"), key=run_index)
    if not files:
        raise FileNotFoundError(f"no {session}_run*.json files in {raw_dir}")
    tests: dict[str, list[dict]] = {}
    for f in files:
        rows = json.loads(f.read_text(encoding="utf-8-sig"))
        for row in rows:
            tests.setdefault(test_name(row), []).append(
                {"run": run_index(f), "row": row}
            )
    return tests


def check_consistency(tests: dict[str, list[dict]]) -> list[str]:
    rows = [e["row"] for entries in tests.values() for e in entries]
    problems = []
    for key in CONSISTENCY_KEYS:
        values = {json.dumps(r.get(key)) for r in rows}
        if len(values) > 1:
            problems.append(f"{key} differs between rows: {sorted(values)}")
    return problems


def check_against_tool(entry: dict) -> list[str]:
    """Compare recomputed mean/stdev with llama-bench's own avg_ts/stddev_ts."""
    row = entry["row"]
    s = summarize(row["samples_ts"])
    msgs = []
    if not math.isclose(s["mean"], row["avg_ts"], rel_tol=1e-3):
        msgs.append(
            f"run{entry['run']}: recomputed mean {s['mean']:.4f} "
            f"!= llama-bench avg_ts {row['avg_ts']:.4f}"
        )
    if s["n"] > 1 and not math.isclose(s["stdev"], row["stddev_ts"], rel_tol=1e-2):
        msgs.append(
            f"run{entry['run']}: recomputed stdev {s['stdev']:.4f} "
            f"!= llama-bench stddev_ts {row['stddev_ts']:.4f}"
        )
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=Path("experiments/results/raw"))
    ap.add_argument("--session", default=None,
                    help="e.g. m0_20261025_123707 (default: latest meta file)")
    ap.add_argument("--peak-gbs", type=float, default=21.336,
                    help="theoretical peak DRAM bandwidth in GB/s (default: "
                         "single-channel DDR4-2667)")
    args = ap.parse_args()

    session = args.session or latest_session(args.raw_dir)
    tests = load_session(args.raw_dir, session)
    print(f"session: {session}")
    for problem in check_consistency(tests):
        print(f"WARNING: {problem}")

    for name, entries in tests.items():
        print(f"\n== {name} (tokens/s, from raw samples_ts) ==")
        print(f"{'process':<14}{'n':>3}{'mean':>10}{'stdev':>9}{'CV%':>7}"
              f"{'min':>10}{'max':>10}")
        pooled: list[float] = []
        means: list[float] = []
        for e in entries:
            s = summarize(e["row"]["samples_ts"])
            print(f"{'run' + str(e['run']):<14}{s['n']:>3}{s['mean']:>10.2f}"
                  f"{s['stdev']:>9.2f}{s['cv_pct']:>7.2f}"
                  f"{s['min']:>10.2f}{s['max']:>10.2f}")
            for msg in check_against_tool(e):
                print(f"WARNING: {msg}")
            pooled += e["row"]["samples_ts"]
            means.append(s["mean"])
        p = summarize(pooled)
        print(f"{'pooled':<14}{p['n']:>3}{p['mean']:>10.2f}{p['stdev']:>9.2f}"
              f"{p['cv_pct']:>7.2f}{p['min']:>10.2f}{p['max']:>10.2f}")
        if len(means) > 1:
            m = summarize(means)
            print(f"between-process means: {min(means):.2f} .. {max(means):.2f}"
                  f" (CV of the {m['n']} means: {m['cv_pct']:.2f}%)")

        if name.startswith("tg"):
            size = entries[0]["row"]["model_size"]
            implied = p["mean"] * size / 1e9
            print("ASSUMPTION-DEPENDENT (not a measurement of DRAM traffic):")
            print(f"  if every model byte ({size:,} B, llama-bench model_size) were read once "
                  f"per token, bandwidth would be {implied:.2f} GB/s, "
                  f"{100 * implied / args.peak_gbs:.1f}% of the {args.peak_gbs:.1f} GB/s "
                  f"theoretical peak")


if __name__ == "__main__":
    main()
