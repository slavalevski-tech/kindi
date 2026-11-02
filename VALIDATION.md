# Lightning Backend V10 validation scope

Development acceptance on NVIDIA Quadro K6000 was recorded on 2026-11-02.
The accepted development manifest identifies 58 runtime source files. The
publication copy retains numerical source after line-ending normalization and
keeps its separate release identity, source-offer middleware and CLI notices.

## Saved development evidence

- Q2 AoS operator: 444 full-output tensor pairs independently replayed on CPU.
- AOS_SHORT and AOS_LONG: 354 complete tensor pairs versus frozen V7 across
  1,000/4,000/64,000 input tokens, before/after 16 greedy steps.
- Actual-image V9/V10/V10/V9 short comparison: 826 complete tensor pairs and all
  256 greedy output IDs equal in the declared 1K/4K cases.
- Fresh 64K V9/V10 comparison: 118 direct cross-image tensor pairs and 16 output
  IDs equal. One observation per image in separate windows; not a 64K ABBA run.
- Service acceptance: HTTP/SSE, authentication, UTF-8, prefix/cache isolation,
  A-B-A requests, concurrency, observed disconnect/recovery, invalid requests,
  restart, actual V9 rollback and final V10 startup.
- Saved service response comparisons replayed independently on CPU: 23 pairs.

The accepted profile uses CUDA 10.2, FP16 direct KV, FP32 Mamba state,
`split_lane4`, `or_aos_lb4`, `load4` and total context capacity 65,536.
These results do not prove universal bitwise equality to another inference engine,
model quality on arbitrary prompts, or correctness outside the tested profile.

## Public installation acceptance — 2026-11-02

The local preparation checks source equivalence, the assembled CUDA hash,
license/notice preservation, source archive completeness, Python syntax and
selected private-data patterns. Existing CPU source-offer and dispatch/format
checks are run separately; the local audit dossier records exact commands/results.
This is an AI-assisted technical review, not an independent human legal audit.

The public Dockerfile was built with `docker compose build --no-cache` from
the available public Ubuntu 18.04 base, installing CUDA 10.2 and resolving the
pinned Python dependencies. It used no private development image ancestor.
The clean build took about 15 minutes; Python 3.6.9 CPU/package checks passed.

On the physical K6000 the separately configured installation converted a new
manifest referring to the existing read-only GGUF, allocated the 65,536-capacity
profile and passed three full-vocabulary numerical references plus bit-exact
snapshot restoration. Thirteen real HTTP acceptance groups passed, including
tool-call/result continuation, SSE, cache, sampling, invalid inputs and concurrency.
A separate cancellation check observed real decode before disconnecting.
The API answered after restart; anonymous source/license routes served the matching
153-file source. All uncompressed archive members were compared byte-for-byte.

The existing private V10 was temporarily stopped only for the GPU tests and then
restored with its original container/image/configuration; its answer was checked.
The other 403 container configurations and running states were unchanged.
See VALIDATION-LIGHTNING.json for the immutable tested image and input archive.
Subsequent edits refresh documentation/receipts only; no runtime, CUDA, entrypoint,
Docker recipe or Compose profile is changed. Final package checks are recorded
separately. These installation tests are distinct from the earlier development
64K and speed measurements. Image bytes are not claimed reproducible across
future OS/transitive dependency resolutions.

Historical V3/V6 reports remain in Git history. They are not new V10 installation
results. The saved five-length V10 series is documented in BENCHMARKS-LIGHTNING.md:
one observation per input length, 256 output steps, 64K in a separate invocation,
and this particular overclocked board. It is not a fresh public-image benchmark.
