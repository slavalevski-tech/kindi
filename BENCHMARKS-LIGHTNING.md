# Lightning V10 — saved K6000 speed measurements

**Kindi 0.3.0 + custom Lightning Backend V10 · Quadro K6000 12 GB · CUDA 10.2**

Ternary-Nemotron-3.5-Lightning-30B-A3B (boro64). FP16 direct KV, FP32 Mamba,
chunk 256, capacity 65,536 total tokens. **256 output tokens; one observation per row.**

This particular owner-overclocked board reports maximum graphics/SM clock
**993 MHz**, memory clock **3350 MHz** (as reported by `nvidia-smi`), and
**250 W power limit**. These values were read after the measurements; there is
no per-row locked-clock log. Frequencies vary dynamically. The power limit is
not measured consumption, and these results must not be labelled stock K6000 results.
The backend/installer does not set these clocks or require this overclock.

| Input tokens | Read / prefill (tok/s) | Generation (tok/s) |
| ---: | ---: | ---: |
| 1,000 | 117.86 | 36.49 |
| 4,000 | 114.19 | 35.28 |
| 8,000 | 108.99 | 33.96 |
| 16,000 | 99.81 | 31.84 |
| 64,000 | 65.75 | 22.74 |

Measurements were saved on 2026-11-02. The four shorter cases share one Engine
invocation; 64K was completed in a separate invocation after the first session
stopped before starting that case. No measured case was repeated. Each invocation
excluded a 1,000-input/64-output warmup; there was no full warmup for each length.
Prefix reuse was disabled. Generation used 256 fixed greedy forward steps,
including GPU argmax/D2H, with EOS ignored. Prefill is synchronized inclusive
Engine wall time, including allocations, dispatch, chunk synchronization, final
head and progress writes; model load, reset, warmup and validation/export are excluded.
These are engine rates, not HTTP throughput.

This is a single-observation series: no median, spread or statistical-significance
claim. It measures the accepted development image, not a fresh installation of
the public packaging. No stock-clock or matched V6/V10 speedup is inferred.
The source/image/profile identities, five row arithmetic checks and all five
256-token sequences were checked against the saved audit.

See [machine-readable data](BENCHMARKS-LIGHTNING.json) for unrounded values,
timing definitions and source artifact hashes. Private host IDs, credentials
and raw request histories are not included.
