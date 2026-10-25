"""Record the hardware/software environment for a benchmark session.

Run this at the start of every benchmarking session and keep the JSON next
to the results it describes. Every probe is best-effort: if one fails, its
field records the error instead of crashing the script.

Usage (from the repo root):
    python scripts/record_env.py --llama-dir C:\\tools\\llama.cpp --label baseline

Output: experiments/env/env_<timestamp>_<label>.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# SMBIOS memory type codes. Verify against your own output; some BIOSes
# report 0 or 2 (unknown) even for DDR4/DDR5 modules.
SMBIOS_MEMORY_TYPES = {20: "DDR", 21: "DDR2", 24: "DDR3", 26: "DDR4", 34: "DDR5"}


def run(cmd: list[str], timeout: int = 30) -> dict:
    """Run a command, return {ok, stdout, stderr} without ever raising."""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, errors="replace"
        )
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except Exception as e:  # missing binary, timeout, permissions...
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def powershell_json(command: str) -> object:
    """Run a PowerShell command that ends in ConvertTo-Json and parse it."""
    r = run(["powershell", "-NoProfile", "-Command", command])
    if not r.get("ok") or not r.get("stdout"):
        return {"error": r.get("error") or r.get("stderr") or "no output"}
    try:
        return json.loads(r["stdout"])
    except json.JSONDecodeError as e:
        return {"error": f"bad JSON: {e}", "raw": r["stdout"][:500]}


def as_list(x: object) -> list:
    """PowerShell emits a bare object for one item and a list for several."""
    if isinstance(x, list):
        return x
    return [x]


def memory_type_label(code: object) -> str:
    """Map an SMBIOS memory type code to a label without assuming its type."""
    try:
        code_int = int(code)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return f"unknown (raw value: {code!r})"
    return SMBIOS_MEMORY_TYPES.get(code_int, f"unknown (code {code_int})")


def probe_cpu() -> dict:
    info: dict = {
        "platform_processor": platform.processor(),
        "os_cpu_count": os.cpu_count(),
    }
    if platform.system() == "Windows":
        info["win32_processor"] = powershell_json(
            "Get-CimInstance Win32_Processor | "
            "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | "
            "ConvertTo-Json"
        )
    return info


def probe_memory() -> dict:
    info: dict = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["total_gb"] = round(vm.total / 2**30, 2)
        info["available_gb"] = round(vm.available / 2**30, 2)
        info["psutil_version"] = psutil.__version__
    except ImportError:
        info["psutil"] = "not installed (pip install psutil)"

    if platform.system() == "Windows":
        raw = powershell_json(
            "Get-CimInstance Win32_PhysicalMemory | "
            "Select-Object Capacity,Speed,ConfiguredClockSpeed,SMBIOSMemoryType,"
            "DeviceLocator,Manufacturer | ConvertTo-Json"
        )
        info["modules_raw"] = raw
        if isinstance(raw, (dict, list)) and "error" not in (raw if isinstance(raw, dict) else {}):
            mods = [m for m in as_list(raw) if isinstance(m, dict)]
            info["module_count"] = len(mods)
            info["modules_summary"] = [
                {
                    "capacity_gb": round(int(m.get("Capacity") or 0) / 2**30, 1),
                    "speed_mts": m.get("Speed"),
                    "configured_mts": m.get("ConfiguredClockSpeed"),
                    "type": memory_type_label(m.get("SMBIOSMemoryType")),
                    "slot": m.get("DeviceLocator"),
                }
                for m in mods
            ]
            info["note"] = (
                "Stick count is not channel count; confirm dual-channel from the "
                "slot labels and the motherboard manual."
            )
    return info


def probe_power_plan() -> dict:
    if platform.system() != "Windows":
        return {"skipped": "not Windows"}
    return run(["powercfg", "/getactivescheme"])


def find_exe(root: Path, stem: str) -> Path | None:
    """Find an executable directly in root, else in any subfolder."""
    name = f"{stem}.exe" if platform.system() == "Windows" else stem
    direct = root / name
    if direct.is_file():
        return direct
    try:
        for p in root.rglob(name):
            if p.is_file():
                return p
    except OSError:
        pass
    return None


def probe_llama(llama_dir: Path | None) -> dict:
    """Capture raw version output; do not assume its format."""
    if llama_dir is None:
        return {"skipped": "pass --llama-dir to record the llama.cpp build"}
    out: dict = {"dir": str(llama_dir)}
    for name in ("llama-bench", "llama-cli"):
        exe = find_exe(llama_dir, name)
        if exe is not None:
            out[name] = {"path": str(exe), **run([str(exe), "--version"])}
        else:
            out[name] = {"ok": False, "error": f"{name} not found under {llama_dir}"}
    out["note"] = (
        "Also record the release tag you downloaded (e.g. from the zip name) "
        "in your README. Check the CPU feature line in llama.cpp's startup log "
        "for AVX2/AVX-512 rather than assuming."
    )
    return out


def probe_git() -> dict:
    head = run(["git", "rev-parse", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])
    if not head.get("ok"):
        return {"available": False}
    return {
        "available": True,
        "commit": head["stdout"],
        "dirty": bool(dirty.get("stdout")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llama-dir", type=Path, default=None,
                    help="folder containing llama-bench / llama-cli")
    ap.add_argument("--label", default="session",
                    help="short tag in the filename, e.g. baseline or tuned")
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/env"))
    args = ap.parse_args()

    record = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": args.label,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {"version": sys.version, "executable": sys.executable},
        "cpu": probe_cpu(),
        "memory": probe_memory(),
        "power_plan": probe_power_plan(),
        "llama_cpp": probe_llama(args.llama_dir),
        "git": probe_git(),
    }

    warnings: list[str] = []
    cpu_proc = record.get("cpu", {}).get("win32_processor")
    if isinstance(cpu_proc, dict) and "error" in cpu_proc:
        warnings.append(f"cpu.win32_processor: {cpu_proc['error']}")
    mem_raw = record.get("memory", {}).get("modules_raw")
    if isinstance(mem_raw, dict) and "error" in mem_raw:
        warnings.append(f"memory.modules_raw: {mem_raw['error']}")
    if not record.get("power_plan", {}).get("ok"):
        power = record.get("power_plan", {})
        warnings.append(f"power_plan: ok is false ({power.get('error') or power.get('stderr') or 'failed'})")
    llama_bench = record.get("llama_cpp", {}).get("llama-bench")
    if not isinstance(llama_bench, dict) or not llama_bench.get("ok"):
        err = llama_bench.get("error") if isinstance(llama_bench, dict) else "llama-bench missing"
        warnings.append(f"llama-bench: {err}")
    record["warnings"] = warnings

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.out_dir / f"env_{stamp}_{args.label}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    print(json.dumps({k: record[k] for k in ("os", "power_plan")}, indent=2))
    for w in warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
