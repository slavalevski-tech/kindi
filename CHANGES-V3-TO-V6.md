# Changes from Lightning V3 to V6

This describes the fork-specific backend for Kindi 0.3.0 and the same pinned
Lightning GGUF on Quadro K6000. Original Kindi remains Amir Husain's work. The
model weights, model identity and upstream Kindi version are unchanged.
V6 includes the accepted optimizations developed after V3; intermediate V4/V5
work is not represented as separate public releases. Performance tables and
speedup claims are deferred until a complete matched benchmark is available.

## Profile changes

| Setting | Published V3 | V6 |
| --- | --- | --- |
| Total context capacity, prompt plus output | 32,768 | 65,536 |
| KV storage | FP32 default | FP16, direct reads |
| Recurrent state and arithmetic | FP32 | FP32 |
| Dense/expert prefill | tile16 | tile24 |
| Prefill chunk | 128 tokens | 256 tokens |
| Q2 input/decode | Scalar/native | vec4/OR |
| SSM execution selection | Legacy | Auto exact shared/per-head paths |
| Split attention strategy | Legacy | GQA16 vector2 stage32/bank4 |
| Attention workspace | 128 MiB | 512 MiB |
| Exact output head layout | Packed Q8 | Split codes/scales Q8 |
| Host prefix-cache budget | 512 MiB, eight entries | 1024 MiB, eight entries |
| Runtime | CUDA 10.2, CuPy 9.6, Python 3.6 | Same legacy runtime |

The explicit complete profile is in compose.yaml and release_entry.py; context,
KV dtype and host cache limits are in lightning.conf.example. Direct library
calls need the same settings and explicit FP16 KV selection to reproduce it.

## Longer context and FP16 KV

For this model, FP32 KV at 32K and FP16 KV at 64K both occupy 384 MiB. Storage uses
two bytes per value instead of four, allowing twice as many positions at that
KV size. This is not a claim that total GPU memory is unchanged: attention
workspace and head layout also changed. Recurrent state remains FP32, and
attention calculations use FP32 values after loading stored FP16 data.

FP16 storage rounds K/V values. It is not byte-identical to the V3 FP32 profile,
and the validation does not promise identical answers or universal model quality.
The accepted 64K fixture rendered 65,280 input tokens and recovered three declared
facts. Boundary/replay checks reached 65,536 total positions. The tested V6 GQA16
strategy rejects capacity above 65,536; 128K is not an accepted V6 target.

## Prefill, Q2 decoding and SSM execution

tile24 expands the native dense/expert prefill row tile. The chunk size is 256,
with bounded cancellation checks and actual last-chunk length. The new vec4 Q2
input path and OR coefficient decoding preserve the declared per-token FP32
accumulation order. Existing quantized weights are read in their original format;
there is no model retraining, requantization or external execution backend.

The SSM auto strategy selects the fork's exact shared/per-head paths for supported
shapes. The Mamba recurrent state and convolution history remain part of the
snapshot. These changes target execution/memory behavior; they are not evidence
of a measured full-model throughput improvement by themselves.

## V6 attention strategy

The accepted GQA16 vector2 kernel derives from this fork's prior GQA16 kernel and
keeps 128-key leaves and the stage32 arithmetic. It pairs FP32 shared-memory
value loads. Global FP16 reads remain scalar; this must not be described as
vectorized FP16 global loads. Per-function FOUR_BYTE shared-bank requests are
made for the tile and merge functions, without changing the context-wide bank setting.

Geometry, FP16 storage, direct reads, optimized/split attention and capacity are
validated before running this strategy. CPU contract tests cover invalid settings,
launch metadata, capacity guards, driver requests and profile fingerprints.

## Exact split Q8 head

The full 131,072-token output vocabulary is retained. Codes and scales are split
at startup into a native layout; the original packed GPU head is then retired
before state/KV allocation. The selected model's steady head storage increases
by 21 MiB; constructing the new layout temporarily needs 378 MiB in addition to
the original packed head. Startup memory guards cover this transient requirement.
There is no approximate vocabulary selection or fast-head mode in the V6 profile.

## Prefix cache and API behavior

Cache admission estimates snapshot size from metadata before copying GPU state
to the host. A checkpoint exceeding the budget is skipped without first making
the expensive host copy. A local reference to a restored checkpoint is released
before later LRU eviction. Fingerprint, token-prefix and LRU matching remain enforced.

The API retains authentication, plain/streamed completions, stop handling,
cancellation, error recovery and source/license routes. CPU tests cover source
isolation and ASGI pass-through. Saved K6000 development acceptance covered API
requests, actual restart/rollback and final V6 activation; publication-only checks
are distinguished in VALIDATION.md.

## Upgrade and rollback

Follow [INSTALLATION.md](INSTALLATION.md). Keep the V3 image/source and private
configuration while preparing an isolated V6 installation with separate state,
pack/config paths, project name and port. Reuse the existing model read-only;
do not duplicate its weights. Run numerical verification under the V6 profile.
One K6000 generally cannot hold both full models at once; coordinate switching
workloads rather than starting a second model over the running service.

Snapshots are guarded by source/profile fingerprints. Reuse of an incompatible
V3 prefix is not supported. Use separate state directories, including CUDA
compilation caches, when switching releases.
The previous public V3 remains at commit 51c134248e9ff48ff1fa051adefbf228134359cb.
This release does not change an existing running installation automatically.

## Provenance and release boundaries

The publication continues the upstream-derived public history and preserves
original author/license notices, the AI disclosure and the corresponding-source
offer. See [PROVENANCE.md](PROVENANCE.md) for documented references, dependencies,
expanded comparison and limitations, and [LICENSING.md](LICENSING.md) for AGPL/CLA scope.
No model weights, registry images, private host scripts, unfinished persistent
attention/routing experiments or unrelated projects are distributed.
