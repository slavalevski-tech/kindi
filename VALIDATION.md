# Lightning V6 validation and historical V3 installation record

This document separates the accepted V6 development build, the public packaging
and the earlier V3 clean-install test. Saved evidence is summarized without
publishing private configuration, API keys or infrastructure logs. No GPU run
or production-service change was performed for the 2026-11-01 publication audit.

## Accepted V6 development build — 2026-10-31

Accepted source: `3f03ba3fad655088c5d7819f30f18a3da109d17d`, based on V5
`2cf39e731fdffa0fb2b6c4c9a0c3f227ff528a43`. Tested local development image:
`sha256:57b8ca40350a77e2d5dc1afd0b841c4ce5b32e2159886de85646db45c2d1e4f6`.
This is not a published registry image.

The profile is CUDA 10.2/CuPy 9.6/K6000 sm_35, capacity 65536, FP16 direct KV,
FP32 recurrent state, tile24 dense/expert prefill, chunk 256, Q2 vec4/OR,
SSM auto, split exact Q8 head and `gqa16_vec2_32_bank4` attention with a
512 MiB workspace. The API was accepted with 4 GiB RAM and no swap. See
[BUILD.md](BUILD.md) for the twelve optimization variables and config fields.

The recorded acceptance covers:

- 53 Python/CUDA runtime files bound to SHA-256 values. An actual image metadata
  probe verified them without constructing another CUDA engine.
- Short numerical comparison of 246 full-array pairs at 1K/4K inputs.
- Two independent fresh reads of a 65280-token input at capacity 65536, with a
  65244-token common prefix. Both V5 reference and V6 candidate answered the
  declared three retrieval facts correctly with natural EOS.
- 354 full-used-array digest comparisons, 402 initial GPU-resident weight records
  and 46 initial positive-zero state/history arrays per role. Weight records are
  resident hash observations, not exported model-weight payloads.
- Eight saved raw-array pairs: 35145728 typed elements compared exactly, including
  signed zero, with no nonfinite values. Raw metrics cover those eight pairs;
  the larger 354-array cohort is compared by digests.
- Exact restoration and an eight-step replay at the total-token boundary.
- Eighteen named API/lifecycle record groups: authentication, ordinary/streamed
  completions, usage, invalid input/overflow, UTF-8/stops, prefix restoration,
  request isolation, cancellation, restart, actual V5 rollback and V6 activation.
  The arithmetic answer `42` is not a test count. Four process/GPU/cgroup probes
  accompanied these records.

The final API session records `completed_active`, `promoted:true`, no errors and
successful API validation. A stale preparation field in that original session
still reads `candidate_v6_64k_not_released`; it was preserved, not rewritten.
A separate post-run receipt records `accepted_v6_native64k` after successful
postconditions. These metadata fields must not be confused.

## V6 publication audit — 2026-11-01

All 53 accepted runtime hashes were rechecked against both the clean development
tree and its Git blobs: all matched. Saved API-session, acceptance-receipt,
bound-contract/manifest and independent-audit SHA links were also verified.

A fresh CPU replay of the saved cold 64K audit passed. It rechecked the 31-file
evidence closure, 354 digest comparisons, 402 resident-weight records, 46 initial
zero arrays per role and all sixteen NPY files. The eight raw pairs again matched
exactly across 35145728 typed elements. Re-running the original CPU comparator
produced the same comparison dictionary. Only output/local-file bindings were
adapted; numerical checks were unchanged. This validates saved GPU evidence and
is not another GPU execution.

Fresh checks against frozen development source passed seven V6 configuration/
dispatch contract groups, four older-GQA CPU tests and eight prefix-capture tests.
A portable V6 checker passed six contract groups; its internal-parent/frozen-file
provenance group is instead covered by the private source audit.

Fresh checks against the 146-file public source tree passed locally on Windows
with existing Python 3.12/3.14 environments:

- Source manifest, archive generation, Python syntax and selected secret/path
  checks passed through `scripts/check_release.py`.
- Seven source-offer tests passed, including archive/path/legal-route behavior.
- Four older-GQA tests, eight prefix-capture tests and six portable vector2
  contract groups passed without CUDA execution.
- Four protocol checks and 288 independent seeded sampling comparisons passed.
- Eleven split-head and nine Q2-OR CPU contract groups passed; their metadata
  checks use the existing model header, not model-weight computation.

These are local CPU/source checks, not a new Linux/Python 3.6/CUDA container build
or a fresh public V6 GPU acceptance run. The public distribution keeps its own
source/profile fingerprints after line-ending normalization.

The public tree retains accepted computation kernels after line-ending
normalization and incorporates the accepted prefix-cache admission/lifetime changes. Public packaging differs through version/CLI attribution,
corresponding-source middleware/manifest and installation defaults. Thus the
complete public tree is not byte-identical to every internal development file.
Source/profile fingerprints are recomputed for the distributed files; no
compatibility override is used. The two API `cache.capture` calls and release of
the restored snapshot reference remain alongside the public source middleware.

## Performance and quality limits

Speed tables, measured rates, percentage comparisons and extrapolations are
withheld pending a complete comparable benchmark campaign. The historical short
V6 protocol used 64 greedy forwards at 1000/4000 input tokens; it does not provide
a measured 256-token result or accepted V6 warm 8K/16K benchmark. Its two long
reads are single cold diagnostics with possible first-use JIT and probe effects.
See [BENCHMARKS-LIGHTNING.md](BENCHMARKS-LIGHTNING.md) for the planned protocol.

Passing three declared retrieval facts does not establish universal long-context
quality. The original BF16 teacher was not tested. An earlier V5 64K state outlier
remains unexplained and its failed evidence is preserved. Historical 128K retrieval
was incomplete; V6 has no new 128K acceptance. The published strategy rejects
capacity above 65536. No routed16, persistent-attention, AoS/job-metadata or
chunk 512/1024 prototype is integrated. CUDA 11.4 was researched, not installed
or benchmarked for this release.

## Historical public V3 clean-install test — 2026-10-30

A separate Compose project was built from public `ubuntu:18.04` using
`docker compose build --no-cache lightning`. Existing GGUF weights were mounted
read-only, not downloaded or copied. No private base image was used.

- The clean build passed, including the then-current 135-file source manifest.
- The image used Python 3.6.9, CuPy 9.6.0, NumPy1.19.5, FastAPI0.79.1,
  Starlette0.19.1, Uvicorn0.16.0 and Pydantic1.9.2.
- Seven source-offer tests, protocol/cache checks and 288 independently seeded
  sampling comparisons passed inside that image.
- `convert` created a manifest pointing to the existing GGUF.
- K6000 `verify` passed for V3 tile16/split/chunk 128/FP32/ctx 32768. Maximum absolute
  errors on the three reference arrays were 6.67572021484375e-06,
  7.659196853637695e-06 and 3.814697265625e-06; restoration was bit-exact.
- Health, inference authentication, ordinary/SSE completions, public source/legal
  routes and restart passed. The source ZIP matched all 135 reviewed files.
- No new NVRM Xid/Oops/BUG entries were observed. The temporarily stopped working
  service was restored; other containers retained their identities and state.

Historical local image:
`sha256:8494986d0099f6f4f72eacc4690d5eaf680fa07a7640d467a84dd9dd005cb15b`.
Historical source ZIP SHA-256:
`2b52319a879bffb38a109392817cfeefcf6d9adf26d622252ef4fe08f9d23f9f`.
These are V3 test identities, not V6 publication identities.

The first clean-build attempt exceeded its temporary-disk reserve and was
cancelled. CUDA download/extraction moved to BuildKit tmpfs, and the second build
passed. The current recipe retains that fix. This installation smoke test did
not repeat the historical full 32K retrieval or stress benchmark.

## Source and distribution scope

Original copyright/SPDX lines, LICENSE and COMMERCIAL-LICENSE are retained.
The three NPZ fixtures contain numerical validation outputs, not model weights.
The repository distributes backend source rather than images, model weights,
private API configuration or experiment directories. Source-manifest and selected
secret/path checks have defined scope; pattern scanning cannot guarantee the
absence of every possible secret.
