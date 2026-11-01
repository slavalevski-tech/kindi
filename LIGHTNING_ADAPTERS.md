# Native MoE adapters and Lightning V6 contracts

The Lightning adapter separates packed CUDA operations from checkpoint semantics.
`moe.py` and `lightning_batch.py` provide linear algebra, routing, dispatch,
normalization and bounded-memory execution. `lightning.py` maps the pinned Nota
checkpoint onto these operations. Another MoE model still requires its own
validated adapter.

## Supported operations

| Operation | Accepted contract |
| --- | --- |
| Packed linear | GGUF Q2_0 type 42, Q8_0 type 8, F32 type 0; aligned rows; FP32 accumulation |
| Router | FP32 sigmoid; selection-only correction bias; top-k; normalization of original selected scores; explicit scale; lower expert-ID tie break |
| Batched router | 1–256 experts; independent IDs/weights per token |
| Expert dispatch | GPU histogram, offsets and permutation; bounded jobs; no CPU transfer of selected IDs |
| Expert feed-forward | Up → ReLU squared → down, plus shared expert |
| Attention | Causal GQA, online softmax, NoPE; FP32 arithmetic/output; explicit KV/profile contract |
| Mamba | State `[heads, head_dim, state_size]`, history `[channels, kernel-1]`; state size 128 for chunk execution |
| Memory | Packed weights, recurrent state, layer-based KV, bounded scratch/reserve; no recurrent allocation for decoder-only graphs |

Expert matrices use `[expert, output_row, input_column]`. Results scatter to
assignment order `token * top_k + slot`. Routed contributions are summed in
slot order, then the shared expert and residual are added. Altered ordering
may change later routing.

`tests/test_generic_moe.py` describes independent small attention/MoE fixtures
without Mamba metadata. Primitive validation is not proof of support for another
real checkpoint.

## Published V6 numerical profile

[BUILD.md](BUILD.md) lists all twelve environment settings. Config uses 65536 total
tokens, `lightning_kv = fp16`, exact head, 1024 MiB host-cache budget, 8 prefix
entries and minimum prefix length 512. The service has one GPU executor and a
4 GiB container RAM limit with no additional swap.

The profile is optimized kernels, Q2 vec4 input/OR decode, tile24 dense/expert
prefill, chunk 256, SSM auto, split exact Q8 head and direct FP16 attention using
`gqa16_vec2_32_bank4` with 512 MiB workspace.

Vector2 attention requires all of:

- `KINDI_LIGHTNING_KERNELS=optimized` and `KINDI_LIGHTNING_ATTENTION=split`;
- `KINDI_LIGHTNING_KV_READ=direct` and FP16 KV storage;
- query/KV head geometry 32/2 with head dimension 128;
- total context capacity from 1 through 65536.

Invalid geometry, profile, storage or capacity is rejected. Engine validation
runs before weight upload/CUDA construction where possible, with allocation and
attention-range checks retaining the bound. FP32 or Q8_0 with vector2 remains
invalid. Older `gqa16_32`/`gqa16_64`, legacy, sub8, quad and automatic strategies
have separate contracts; larger allocation capability is not V6 128K acceptance.

The V6 kernel retains 128-key leaves and STAGE32 arithmetic, pairing FP32 shared
V loads. Global FP16 loads and partial-output stores remain scalar. Tile and
merge functions receive individual four-byte bank requests through
`cuFuncSetSharedMemConfig`; there is no context-wide setter. Invalid pointers
or failed requests reject the configuration instead of silently falling back.

Q2 OR decoding changes exact coefficient formation, including code 3 and positive
zero, while retaining accepted arithmetic/result ordering. Batched tile24 prefill
keeps its validated path. SSM auto selects the accepted fixed-model implementation
with FP32 recurrent state. No Tensor Cores, TF32, DP4A, fast-math flag or FP16 model
arithmetic is required by this profile.

The split Q8 head evaluates the exact full vocabulary. It retires the packed GPU
head after conversion, leaving 21 MiB steady extra storage; startup temporarily
needs 378 MiB before releasing the original 357 MiB head. This is a GPU layout
change and does not alter or duplicate model files.

## Storage, snapshots and prefix caching

FP16 storage uses round-to-nearest-even. Separate Q8_0 storage uses 32 signed codes
plus a half scale; codes use the unrounded FP32 scale and round-to-nearest-even.
Nonfinite values and scale overflow are rejected. Mamba state remains FP32.
Lossy storage is not claimed to preserve every FP32 result. Compressed reference
fixtures quantize K/V independently at matching storage boundaries; they are
aligned references rather than a free-running BF16 teacher comparison.

`KINDI_LIGHTNING_KV_READ=auto|direct|expanded` controls older compatible paths.
Auto may reuse one layer-sized expansion scratch. Published V6 uses direct reads
and reserves no expanded KV copy. Split-attention workspace bounds query
microbatches. Direct library callers must select the complete intended profile.

Snapshots preserve raw KV dtype and the used token prefix. Genuine fingerprints
include source hashes and numerical choices, attention, storage/read policy,
workspace, device and CUDA runtime. Restoration rejects incompatible fingerprints;
V6 does not override them to import another source/profile snapshot.

Cache admission estimates used snapshot size before copying GPU state. Disabled
or over-budget caches avoid the copy. The server drops its temporary restored
snapshot reference before later eviction and rechecks size at insertion.
Entries use host RAM and vanish on restart. Requests keep separate sampling RNGs.

## Validation and historical profiles

[VALIDATION.md](VALIDATION.md) separates V6 development evidence from publication
checks. The accepted 64K fixture uses genuine fresh 65280-token reads at capacity
65536, not an imported-prefix or allocation-only test. Three declared facts pass
in both roles; this is not universal-quality proof. Earlier V5 64K failed evidence
is preserved and its outlier cause unresolved. Historical 128K is not V6 acceptance.

Speed tables and numerical comparisons are withheld pending a full comparable
campaign. The historical short V6 protocol used 64 forwards at 1K/4K inputs, not
measured 256-token output or warm V6 8K/16K runs. See
[BENCHMARKS-LIGHTNING.md](BENCHMARKS-LIGHTNING.md).

Historical V3 used optimized tile16 dense/expert prefill, split attention,
chunk 128 and FP32 KV at capacity 32768. Its full sequential/batched checks and
retrieval results remain historical. A shared-prefix tail comparison must not
be called two independent full prefills.

Portable CPU checks include `tests/test_vector2_cpu.py`,
`tests/test_gqa_dispatch_cpu.py` and `tests/test_prefix_capture.py`. They validate
configuration, dispatch/bank policy and cache admission/lifetime with metadata
stubs. They do not compile CUDA, load weights or replace numerical GPU acceptance.

An unrecoverable CUDA error rejects new work and makes `/health` return 503 until
restart. Unhealthy status alone does not restart a running process. Legacy
`cuda-memcheck` is excluded after a historical `nvidia_uvm` oops. Preserve the
working driver and other workloads.

Rejected/unintegrated experiments include dense cuBLAS reduction changes,
routed16 row geometry, persistent attention merge, AoS/job metadata and chunk 512/
1024 proposals. An operator test or microbenchmark cannot establish whole-model
speed, quality or another GPU's performance.

## Adding another model

1. Pin model/config/tokenizer revisions and identify blocks, tensor layout,
   quantization and special-token rules; reject unsupported metadata early.
2. Provide layer schedule, tensor names and dimensions in an adapter; reuse only
   applicable packed loading and CUDA primitives.
3. Implement positional encoding, norms, softcaps, routing, grouping, shared gates,
   activation, scale and residual order explicitly for that checkpoint.
4. Define memory/state contracts and include numerical choices in fingerprints.
5. Establish independent reference checks, layer/logit parity, long-prefix
   behavior, restoration, cancellation and API acceptance before measuring speed.

Low-level dispatch assumes expert IDs from the validated router. External callers
must not pass arbitrary out-of-range GPU indices. The kernels do not silently
implement unsupported architectures.
