<div align="center">

# 🧪 CPU Inference Lab

**What actually limits LLM speed on a plain desktop CPU? Predict first, measure second, explain the gap.**

A small, reproducible benchmarking lab for running quantized LLMs on CPU with [llama.cpp](https://github.com/ggml-org/llama.cpp). Every number comes with its environment, raw output and a prediction written *before* the run.

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![llama.cpp](https://img.shields.io/badge/llama.cpp-b11871-1c5cab)
![Model](https://img.shields.io/badge/model-Qwen2.5--0.5B%20Q4__K__M-eb6834)
![Platform](https://img.shields.io/badge/platform-Windows%2011-0078D4?logo=windows&logoColor=white)
![Status](https://img.shields.io/badge/status-M0%20baseline%20done-1b7f4b)

</div>

---

## ✨ Highlights

- 🎯 **Prediction-first methodology.** Hypotheses are committed in [`docs/predictions.md`](docs/predictions.md) before any benchmark runs, and that file is never edited afterwards. Corrections go in a separate postmortem.
- 📦 **Fully reproducible.** Pinned llama.cpp build, SHA-256 hashes for every downloaded artifact, and a JSON snapshot of the hardware and software environment for each session.
- 🗂️ **Raw data preserved.** Unmodified `llama-bench` JSON and stderr logs are kept for every run, with per-run metadata (timestamps, free RAM, CPU load before the run).
- 🧮 **Honest statistics.** The analysis recomputes mean, sample standard deviation and CV from the raw per-repetition samples and cross-checks them against llama-bench's own figures.
- 🧪 **Tested analysis code.** Unit tests use small synthetic values that can be verified by hand.

## 📊 M0 baseline results

**Test machine:** Intel Core i5-12400F (6C/12T) · 16 GB DDR4-2667, **single stick (single-channel)** · Windows 11 · Balanced power plan

**Command:** `llama-bench -m <model> -p 512 -n 128 -t 6 -ngl 0 -r 5 -o json`, run as 3 separate processes × 5 repetitions each.

| Test | What it measures | Mean (tokens/s) | Std dev | CV | Range |
|---|---|---:|---:|---:|---|
| `pp512` | Prefill, 512-token prompt | **344.5** | 15.7 | 4.55% | 326.6 – 380.0 |
| `tg128` | Decode, 128 generated tokens | **45.4** | 1.5 | 3.23% | 42.8 – 46.9 |

Pooled over 15 samples per test. Per-process breakdown is in the analysis output (see [Analyze](#-analyze-results)).

### Predictions vs. observed

| # | Prediction (written 2026-10-25, pre-run) | Observed | Outcome |
|---|---|---|---|
| 1 | Decode between 25 and 38 tok/s | 45.4 tok/s | ❌ Above range |
| 2 | Prefill between 300 and 1,000 tok/s | 344.5 tok/s | ✅ In range |
| 3 | Decode CV below 5% | 3.23% | ✅ In range |
| 4 | Decode stops scaling at ~4 threads | not yet run | ⏳ Planned |
| 5 | Q4_K_M / Q8_0 decode ratio in 1.0 – 1.375 | not yet run | ⏳ Planned |

### 🔍 The interesting miss

The bandwidth arithmetic in the predictions suggested a decode ceiling of roughly 43 tok/s (21.3 GB/s single-channel DDR4-2667 ÷ ~0.49 GB model). The measured decode rate is *above* that. If every model byte were read from DRAM once per token, 45.4 tok/s would imply about **22.1 GB/s, or 103% of theoretical peak**, which is not physically possible.

So the "one full model read from DRAM per token" assumption doesn't hold for this setup. That is exactly the failure mode flagged as prediction #6 (weight repacking, access patterns, cache reuse). The mechanism is **not yet established**; explaining it is the job of `docs/postmortem.md`, which is the next deliverable.

> ⚠️ Note that the implied-bandwidth figure is assumption-dependent and is *not* a measurement of DRAM traffic. The analysis script labels it that way.

## 🗺️ Repository layout

```
.
├── scripts/
│   ├── record_env.py        # snapshot CPU, RAM, OS, power plan, llama.cpp version → JSON
│   ├── m0_bench.py          # runner: record env, then N separate llama-bench processes
│   └── analyze_m0.py        # stats from raw samples, consistency + cross-checks
├── tests/
│   └── test_analyze_m0.py   # hand-checkable unit tests
├── experiments/
│   ├── env/                 # environment snapshots, one per session
│   ├── results/raw/         # untouched llama-bench JSON, stderr logs, run metadata
│   └── configs/             # (reserved for future experiment configs)
├── docs/
│   ├── predictions.md       # frozen, pre-registered predictions
│   ├── artifact_hashes.txt  # size + SHA-256 + source URL of every download
│   ├── llama-version_*.txt  # captured llama.cpp version output
│   ├── llama-bench-help_*.txt
│   └── index (9).html       # standalone showcase page of the M0 write-up
├── data/                    # (reserved)
└── requirements.txt         # psutil, pytest
```

`models/` and `tools/` (the GGUF and the llama.cpp binaries) are git-ignored. Fetch them as described below.

## 🚀 Quickstart

The scripts default to Windows paths, matching the machine this was measured on. Use `--llama-dir` / `--model` to point elsewhere.

### 1. Set up Python

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Fetch the pinned artifacts

| Artifact | Source | Size (bytes) | SHA-256 |
|---|---|---:|---|
| `llama-b11871-bin-win-cpu-x64.zip` | [llama.cpp release b11871](https://github.com/ggml-org/llama.cpp/releases/download/b11871/llama-b11871-bin-win-cpu-x64.zip) | 24,216,651 | `A39AE0A5…F0CCFA` |
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf) | 491,400,032 | `74A4DA8C…93D7A9DB` |

Full hashes are in [`docs/artifact_hashes.txt`](docs/artifact_hashes.txt). Verify them before running (`Get-FileHash <file> -Algorithm SHA256`).

Expected layout:

```
tools\llama.cpp\b11871\llama-bench.exe
models\qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### 3. Run the benchmark

Close other applications first, and run from a plain terminal.

```powershell
# Preview what will run, and check the files are found
python scripts\m0_bench.py --dry-run

# Record the environment, then run 3 separate llama-bench processes
python scripts\m0_bench.py
```

Output lands in `experiments/env/` and `experiments/results/raw/`. Existing files are never overwritten.

### 4. 📈 Analyze results

```powershell
python scripts\analyze_m0.py                                # latest session
python scripts\analyze_m0.py --session m0_20261025_123707   # a specific one
```

The script prints per-process and pooled statistics for each test, warns if settings differ between runs, and verifies its recomputed mean and stdev against llama-bench's own `avg_ts` / `stddev_ts`. It deliberately does **not** compare against the predictions.

### 5. Run the tests

```powershell
pytest
```

## 🧭 Method in five rules

1. **Write predictions first.** Freeze them in `docs/predictions.md`, with the mechanism you expect.
2. **Record the environment.** Every session gets a `record_env.py` snapshot stored beside its results.
3. **Pin and hash everything.** Exact build, exact model file, SHA-256 recorded.
4. **Keep raw output.** Derived numbers are always recomputable from files in `experiments/results/raw/`.
5. **Explain misses separately.** Wrong predictions are analyzed in `docs/postmortem.md`, never by editing the original.

## 🛣️ Roadmap

- [x] **M0:** baseline prefill and decode, 3 processes × 5 reps
- [ ] Postmortem for M0, starting with the decode result above the bandwidth estimate
- [ ] Thread scaling: 1, 2, 4, 6, 12 threads
- [ ] Quantization comparison: Q4_K_M vs. Q8_0 (675,710,816 bytes)
- [ ] GGUF tensor inspection (is `output.weight` tied to the embeddings?)

## 📝 Caveats

- Results are for **one machine** (single-channel memory, Balanced power plan). Do not generalize them to other hardware.
- llama-bench labels the model `qwen2 1B Q4_K - Medium` and reports ~630M parameters (485,452,288 bytes), which differs from the "0.5B" in the file name. The tensor inventory check in the roadmap should clarify this.
- This lab uses **llama.cpp's `llama-bench` directly**, not Ollama.

## 🙏 Acknowledgements

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) for the inference engine and `llama-bench`
- [Qwen team](https://huggingface.co/Qwen) for Qwen2.5-0.5B-Instruct

---

<div align="center">
<sub>Built by Saket Saurabh · Predict, measure, explain.</sub>
</div>
