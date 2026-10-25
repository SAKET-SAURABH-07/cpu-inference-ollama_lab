# Predictions (written before any benchmark run)

Date written: 2026-10-25 · Written by: Saket Saurabh

Rule: This file is not edited after the first benchmark. Corrections go in a separate file called `docs/postmortem.md`.

## Setup (from experiments/env/ JSON, measured)

- CPU: Intel Core i5-12400F, 6 cores / 12 threads
- RAM: one 16 GB Kingston DDR4 stick, 2667 MT/s → single-channel
- OS: Windows 11 (10.0.26300), power plan: Balanced
- llama.cpp: build 11871 (commit b3fa69955), CPU backend `ggml-cpu-alderlake.dll`
- Model: Qwen2.5-0.5B-Instruct Q4_K_M, 491,400,032 bytes (SHA256 in `docs/artifact_hashes.txt`)

## Arithmetic (not a measurement)

- Theoretical single-channel bandwidth: 2667 × 10⁶ transfers/s × 8 bytes ≈ 21.3 GB/s.
- Idealized decode estimate: 21.3 GB/s ÷ 0.4914 GB ≈ 43 tokens/s.
- This estimate assumes approximately one full model-file equivalent is read from DRAM per generated token. It is an idealized reference, not a strict bound: actual DRAM traffic depends on tensor layout, cache reuse, and any runtime repacking.

## Benchmark config

`llama-bench -m <model> -p 512 -n 128 -t 6 -ngl 0 -r 5 -o json`, run as 3 separate processes.

## My predictions

1. **Decode (tg128, 6 threads): between 25 and 38 tokens/s.**

   Dominant mechanism: model-weight traffic through single-channel DRAM. The idealized bandwidth estimate is approximately 43 tokens/s, but sustained usable bandwidth should be lower than theoretical peak. I therefore predict throughput below that estimate. This assumes roughly one model-file equivalent of DRAM traffic per token; the GGUF tensor inventory and runtime logs will help assess that assumption.

2. **Prefill (pp512, 6 threads): between 300 and 1,000 tokens/s.**

   Dominant mechanism: parallel matrix computation across the 512 prompt tokens. Unlike single-token decode, prefill can reuse weights across multiple token positions and exploit batched matrix operations. I expect substantially higher throughput than decode, although the achieved rate depends on the CPU's quantized kernels.

3. **Run-to-run variation (coefficient of variation) for decode: below 5%.**

   Dominant mechanism: operating-system and background-process interference. With the same executable, model, and settings, I expect relatively stable throughput on an otherwise idle PC. Windows background activity could push the variation above 5%. I will calculate CV from the repeated measurements instead of treating the prediction as a result.

4. **Later, thread-scaling experiment — decode stops improving at about 4 threads.**

   Dominant mechanism: saturation of the single-channel memory subsystem. Additional threads can increase concurrent work, but they cannot increase DRAM bandwidth indefinitely. I predict diminishing returns around four threads; this is a testable hypothesis, not an assumption that four threads is optimal. I will compare 1, 2, 4, 6, and 12 threads.

5. **Later — Q4_K_M/Q8_0 decode-speed ratio: between 1.0 and 1.375.**

   The Q8_0 model is 675,710,816 bytes, while Q4_K_M is 491,400,032 bytes. Their file-size ratio is approximately 1.375. I predict Q4_K_M will decode faster, but by less than the inverse file-size ratio suggests, because quantized kernels and dequantization work affect throughput. The ratio will be calculated as Q4_K_M tokens/s divided by Q8_0 tokens/s under otherwise identical conditions.

6. **Biggest thing that could make these predictions wrong: actual DRAM traffic per generated token differs from the model-file-size assumption.**

   In particular, llama.cpp's weight repacking and tensor access patterns may change how many bytes are fetched from DRAM. Two other concrete confounds are the 0.5B model's per-token runtime overhead, which may be significant relative to weight-reading time, and CPU boost-clock behaviour under the Balanced power plan, which may change across runs.

## Verification notes

- Check Intel's CPU specifications before relying on cache size or instruction-set claims.
- Use the GGUF inspector to determine whether `output.weight` exists and whether the output projection shares weights with the input embeddings. Do not assume tied embeddings from the model name alone.
- Preserve the raw benchmark JSON and logs. If the observed results contradict these predictions, explain the discrepancy in `docs/postmortem.md` rather than changing this file.
