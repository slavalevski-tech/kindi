"""Opt-in aligned float4 decode candidate: byte equality and bounded microbench.

No Engine or full-model weights are loaded. Real packed matrices are read one at
a time from the existing GGUF. Compatible with Python 3.6/CuPy 9.6/CUDA 10.2.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import cupy as cp
from kindi.moe import CudaOps
from kindi.packed import PackedFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--loops', type=int, default=80)
    parser.add_argument('--rounds', type=int, default=7)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise RuntimeError('Refusing to replace existing evidence: ' + str(output))
    if not 1 <= args.loops <= 500 or not 3 <= args.rounds <= 15:
        raise ValueError('Invalid bounded benchmark repetitions')
    output.parent.mkdir(parents=True, exist_ok=True)
    ops = CudaOps()
    rng = np.random.RandomState(831)
    checks, records = [], []
    started = time.monotonic()

    def packed(rows, cols, experts=1):
        blocks = experts * rows * cols // 64
        raw = rng.randint(0, 256, (blocks, 18)).astype(np.uint8)
        raw[:, :2] = rng.uniform(.001, .05, blocks).astype(np.float16).view(np.uint8).reshape(-1, 2)
        return raw.reshape(-1)

    def verify(name, w, x, rows, cols, ids=None, selected=1, per_input=False):
        # The experiment deliberately preserves the existing aligned decode contract.
        assert x.flags.c_contiguous and x.dtype == cp.float32 and x.data.ptr % 16 == 0
        assert cols % 64 == 0 and (not per_input or x.size == selected * cols)
        grid, block = ((rows + 3) // 4, selected), (128,)
        guard = 64
        baseline = cp.full(selected * rows + 2 * guard, np.float32(6.25), cp.float32)
        candidate = cp.full_like(baseline, np.float32(6.25))
        old_y = baseline[guard:-guard]
        new_y = candidate[guard:-guard]
        params = (w, x, old_y, ops.null if ids is None else ids,
                  np.int32(rows), np.int32(cols), np.int32(selected), np.int32(per_input))
        ops.k['q2_mv'](grid, block, params)
        ops.k['q2_mv_vec4'](grid, block, params[:2] + (new_y,) + params[3:])
        reference, actual = baseline.get(), candidate.get()
        assert np.isfinite(reference).all() and np.isfinite(actual).all(), name + ' finite'
        np.testing.assert_array_equal(actual.view(np.uint8), reference.view(np.uint8), err_msg=name + ' bytes')
        assert np.all(actual[:guard] == 6.25) and np.all(actual[-guard:] == 6.25), name + ' output guard'
        checks.append({'name': name, 'rows': rows, 'cols': cols, 'selected': selected,
                       'per_input': per_input, 'byte_identical': True,
                       'output_sha256': hashlib.sha256(actual[guard:-guard].tobytes()).hexdigest()})
        return grid, block, params

    for cols in (64, 128, 1856, 2688, 3712):
        for rows in (1, 3, 4, 5, 19):
            w = cp.asarray(packed(rows, cols, 3))
            shared = cp.asarray(rng.randn(cols).astype(np.float32))
            xs = cp.asarray(rng.randn(3, cols).astype(np.float32))
            ids = cp.asarray(np.array([2, 0, 2], np.int32))
            verify('synthetic-dense', w, shared, rows, cols)
            verify('synthetic-selected-shared-input', w, shared, rows, cols, ids, 3, False)
            verify('synthetic-selected-per-input', w, xs, rows, cols, ids, 3, True)
            # Also exercise IDs=null with per-input=true (batched dense fallback).
            verify('synthetic-dense-per-input', w, xs, rows, cols, None, 3, True)
        del w, shared, xs, ids
        cp.get_default_memory_pool().free_all_blocks()

    raw = np.empty((256, 18), np.uint8)
    raw[:, :2] = np.array([1], np.float16).view(np.uint8)
    raw[:, 2:] = np.arange(256, dtype=np.uint8)[:, None]
    w = cp.asarray(raw.reshape(-1))
    for label, values in [('positive-zero', np.zeros(64, np.float32)),
                          ('negative-zero', -np.zeros(64, np.float32)),
                          ('small-values', np.full(64, np.float32(2 ** -120))),
                          ('mixed', rng.randn(64).astype(np.float32))]:
        verify('all-256-packed-bytes-' + label, w, cp.asarray(values), 256, 64)
    del w
    cp.get_default_memory_pool().free_all_blocks()

    file = PackedFile(args.model)
    names = ['blk.0.ssm_in.weight', 'blk.0.ssm_out.weight',
             'blk.1.ffn_up_exps.weight', 'blk.1.ffn_down_exps.weight',
             'blk.1.ffn_up_shexp.weight', 'blk.1.ffn_down_shexp.weight']
    names.extend(next(name for name in file.tensors if name.endswith('.' + suffix))
                 for suffix in ('attn_q.weight', 'attn_output.weight'))
    for name in names:
        tensor = file.tensors[name]
        assert tensor.kind == 42, name + ' expected Q2_0'
        rows, cols = tensor.shape[-2:]
        experts = len(tensor.shape) == 3
        selected = 6 if experts else 1
        per_input = name.endswith('ffn_down_exps.weight')
        ids = cp.asarray(np.array([127, 73, 32, 3, 0, 95], np.int32)) if experts else None
        w = cp.asarray(file.raw(name))
        x = cp.asarray(rng.randn(selected if per_input else 1, cols).astype(np.float32))
        grid, block, params = verify(name, w, x, rows, cols, ids, selected, per_input)
        # Include a genuine dense shape in the multi-input mode without expert IDs.
        if not experts:
            xx = cp.asarray(rng.randn(7, cols).astype(np.float32))
            verify(name + '-dense-per-input', w, xx, rows, cols, None, 7, True)
            del xx
        samples = {'scalar': [], 'vec4': []}
        kernels = {'scalar': ops.k['q2_mv'], 'vec4': ops.k['q2_mv_vec4']}
        for kernel in kernels.values():
            for _ in range(8):
                kernel(grid, block, params)
        cp.cuda.Stream.null.synchronize()
        for round_index in range(args.rounds):
            # Alternate order to reduce clock/order bias; use one output allocation.
            order = ('scalar', 'vec4') if round_index % 2 == 0 else ('vec4', 'scalar')
            for mode in order:
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                begin.record()
                for _ in range(args.loops):
                    kernels[mode](grid, block, params)
                end.record()
                end.synchronize()
                samples[mode].append(float(cp.cuda.get_elapsed_time(begin, end)) / args.loops)
        scalar = float(np.median(samples['scalar']))
        vec4 = float(np.median(samples['vec4']))
        record = {'tensor': name, 'shape': list(tensor.shape), 'selected': selected,
                  'per_input': per_input, 'packed_bytes': int(w.nbytes),
                  'scalar_median_ms': scalar, 'vec4_median_ms': vec4,
                  'speedup': scalar / vec4, 'samples_ms': samples, 'byte_identical': True}
        records.append(record)
        print(json.dumps({key: record[key] for key in ('tensor', 'scalar_median_ms', 'vec4_median_ms', 'speedup')}), flush=True)
        del w, x, ids, params
        cp.get_default_memory_pool().free_all_blocks()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    result = {'passed': True, 'comparison': 'finite same-shape/dtype uint8 bytes, including signed zero',
              'candidate': 'q2_mv_vec4 opt-in; default q2_mv unchanged',
              'input_contract': 'FP32 contiguous, 16-byte-aligned, cols divisible by 64',
              'device': str(properties['name']), 'cupy': cp.__version__,
              'runtime': cp.cuda.runtime.runtimeGetVersion(), 'source_code_sha256': ops.code_hash,
              'checks': checks, 'records': records,
              'kernel_attributes': {name: getattr(ops.k[name], 'attributes', {}) for name in ('q2_mv', 'q2_mv_vec4')},
              'benchmark': {'loops': args.loops, 'rounds': args.rounds, 'timing': 'CUDA events; allocation excluded'},
              'wall_seconds': time.monotonic() - started}
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'checks': len(checks), 'matrices': len(records), 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
