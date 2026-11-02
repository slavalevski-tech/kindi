# Changes from published V6 to Lightning Backend V10

This modified version is maintained by slavalevski-tech, 2026-11-02.
Original Kindi 0.3.0 remains copyright 2026 Amir Husain, under AGPL-3.0-only.
The release targets NVIDIA Quadro K6000; model weights are unchanged and excluded.

## Output head inherited from V7

`lightning.cu`, `lightning.py` and `moe.py` add the exact `split_lane4` path.
The original Q8 output tensor is converted directly at startup to lane4 code
and scale buffers. Conversion completes before KV/state allocations and retires
the packed output owner. A stream-ordered input workspace is created per call.
The full vocabulary is evaluated; this is not an approximate or shortened head.

## Q2 prefill inherited from V8 and updated by V10

`lightning_prefill.cu` adds opt-in OR-coefficient tile24 prefill. Dense/expert
dispatch in `lightning_batch.py` selects the same accepted tile geometry.
V10 adds `lightning_prefill_aos.cu`: shared inputs use vector `float4` AoS staging
instead of scalar SoA staging. The selected mode is `or_aos_lb4`.
The recorded FMA order, scale operation, K traversal and warp reduction are kept.
There is no model weight repack, new quantization or expert skipping.

## Attention staging inherited from V9

The optional `load2` and selected `load4` source files derive from the fork's
existing FP16 GQA vector2 attention. `load4` stages four halfwords per load while
preserving the tested arithmetic and tile geometry. `kv.py` validates compatible
FP16/direct/split configuration and pointer alignment. These modes require the
declared model geometry and total context at most 65,536.

## Configuration and compatibility

Compared with V6, set `KINDI_LIGHTNING_HEAD_LAYOUT=split_lane4`,
`KINDI_LIGHTNING_PREFILL_Q2=or_aos_lb4` and
`KINDI_LIGHTNING_ATTENTION_KV_LOAD=load4`. Retain the other settings in BUILD.md.
FP16 KV and FP32 Mamba state are retained. CUDA 10.2 remains the supported toolkit.
Start with a fresh pack manifest and private state directory; do not reuse live
snapshots from another version or mount a different source tree into a running API.

## Publication layer and limits

The public tree continues the published V6 Git history and retains attribution,
AGPL source download, CLI legal notices and source-only Docker packaging.
Numerical/runtime source is compared separately from release identity and these
publication changes. The public recipe subsequently passed a clean build and
isolated K6000 numerical/API installation acceptance on 2026-11-02; see
[VALIDATION.md](VALIDATION.md) for the tested identities and limits.

Finite exact-state, token and API checks support the declared test cases.
Saved absolute V10 rates and the particular overclocked-board configuration are
in BENCHMARKS-LIGHTNING.md. No matched V6/V10 speedup or statistical-significance
claim is made. See VALIDATION.md and PROVENANCE.md for the broader test limits.
