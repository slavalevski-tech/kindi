"""CPU-only cache admission/lifetime checks; no NumPy, CUDA, model or SSH required."""
import ast
import codecs
import importlib.util
from pathlib import Path
import threading
import time
import types
import unittest
import weakref


SOURCE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prefix_capture_cache', str(SOURCE / 'kindi/prefix_cache.py'))
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)
PrefixCache = cache_module.PrefixCache
snapshot_bytes = cache_module.snapshot_bytes
engine_snapshot_bytes = cache_module.engine_snapshot_bytes


class MetadataArray(object):
    def __init__(self, shape, itemsize):
        self.shape = tuple(shape)
        self.itemsize = itemsize
        self.nbytes = itemsize
        for dimension in shape:
            self.nbytes *= dimension

    def get(self, *args, **kwargs):
        raise AssertionError('metadata admission must not transfer array data')

    def __getitem__(self, index):
        raise AssertionError('metadata admission must not construct GPU views')


class Snapshot(dict):
    pass


class MetadataEngine(object):
    def __init__(self, mode='fp32', position=8, capacity=131072):
        self.position = position
        self.tokens = list(range(position))
        self.fingerprint = 'native-profile'
        self.states = {i: MetadataArray((64, 64, 128), 4) for i in range(23)}
        self.histories = {i: MetadataArray((6144, 3), 4) for i in range(23)}
        row = (2, 136) if mode == 'q8_0' else (2, 128)
        itemsize = {'fp32': 4, 'fp16': 2, 'q8_0': 1}[mode]
        self.keys = {i: MetadataArray((capacity,) + row, itemsize) for i in range(6)}
        self.values = {i: MetadataArray((capacity,) + row, itemsize) for i in range(6)}
        self.last_logits = MetadataArray((131072,), 4) if position else None
        self.snapshot_calls = 0
        self.on_snapshot = None

    def snapshot(self):
        self.snapshot_calls += 1
        if self.on_snapshot:
            self.on_snapshot(self.snapshot_calls)
        result = Snapshot(fingerprint=self.fingerprint, tokens=list(self.tokens), position=self.position,
                          states=dict(self.states), histories=dict(self.histories), logits=self.last_logits)
        for group in ('keys', 'values'):
            result[group] = {i: MetadataArray((self.position,) + array.shape[1:], array.itemsize)
                             for i, array in getattr(self, group).items()}
        return result

    def reset(self):
        self.position = 0
        self.tokens = []

    def restore(self, snapshot):
        self.tokens = list(snapshot['tokens'])
        self.position = len(self.tokens)
        self.last_logits = snapshot['logits']

    def prefill(self, tokens, **kwargs):
        self.tokens.extend(tokens)
        self.position = len(self.tokens)
        return self.last_logits


class PrefixCaptureTests(unittest.TestCase):
    def test_disabled_cache_does_not_inspect_engine_or_capture(self):
        class Untouchable(object):
            def __getattr__(self, name):
                raise AssertionError('disabled cache inspected engine: ' + name)
        self.assertFalse(PrefixCache(512 * 1024 ** 2, 0).capture(Untouchable()))
        self.assertFalse(PrefixCache(0, 8).capture(Untouchable()))

    def test_insufficient_budget_does_not_capture_or_evict(self):
        engine = MetadataEngine()
        size = engine_snapshot_bytes(engine)
        cache = PrefixCache(size - 1, 1)
        smaller = MetadataEngine(position=1).snapshot()
        self.assertTrue(cache.put(smaller))
        retained = list(cache.entries)
        self.assertFalse(cache.capture(engine))
        self.assertEqual(engine.snapshot_calls, 0)
        self.assertEqual(list(cache.entries), retained)
        self.assertIs(cache.best(smaller['tokens'], engine.fingerprint), smaller)

    def test_exact_budget_admits_one_snapshot(self):
        engine = MetadataEngine()
        size = engine_snapshot_bytes(engine)
        cache = PrefixCache(size, 1)
        self.assertTrue(cache.capture(engine))
        self.assertEqual(engine.snapshot_calls, 1)
        self.assertEqual(cache.used, size)
        self.assertEqual(len(cache.entries), 1)
        self.assertEqual(snapshot_bytes(cache.best(engine.tokens, engine.fingerprint)), size)

    def test_consumed_rows_match_snapshot_accounting_for_every_kv_format(self):
        for mode in ('fp32', 'fp16', 'q8_0'):
            for consumed in (0, 128, 32768, 131072):
                with self.subTest(mode=mode, consumed=consumed):
                    engine = MetadataEngine(mode, consumed)
                    estimate = engine_snapshot_bytes(engine)
                    self.assertEqual(engine.snapshot_calls, 0)
                    actual = engine.snapshot()
                    self.assertEqual(estimate, snapshot_bytes(actual))
                    bytes_per_token = {'fp32': 12288, 'fp16': 6144, 'q8_0': 3264}[mode]
                    fixed = 49930240 + (524288 if consumed else 0) + 1024
                    self.assertEqual(estimate, fixed + consumed * (bytes_per_token + 32))

    def test_short_prefix_is_admitted_despite_large_reserved_capacity(self):
        engine = MetadataEngine('fp32', position=128, capacity=131072)
        cache = PrefixCache(engine_snapshot_bytes(engine), 1)
        self.assertTrue(cache.capture(engine))
        self.assertGreater(sum(a.nbytes for a in engine.keys.values()), cache.budget)

    def test_put_rechecks_size_after_capture(self):
        engine = MetadataEngine()
        original_snapshot = engine.snapshot
        size = engine_snapshot_bytes(engine)

        def enlarged_snapshot():
            result = original_snapshot()
            result['logits'] = MetadataArray((131073,), 4)
            return result
        engine.snapshot = enlarged_snapshot
        cache = PrefixCache(size, 1)
        self.assertFalse(cache.capture(engine))
        self.assertEqual(engine.snapshot_calls, 1)
        self.assertEqual(cache.used, 0)

    def test_existing_prefix_fingerprint_lru_semantics(self):
        engine = MetadataEngine(position=2)
        first = engine.snapshot()
        size = snapshot_bytes(first)
        cache = PrefixCache(size * 2, 2)
        second = MetadataEngine(position=2).snapshot()
        second['tokens'] = [4, 5]
        self.assertTrue(cache.put(first))
        self.assertTrue(cache.put(second))
        self.assertIs(cache.best([0, 1, 9], engine.fingerprint), first)
        third = MetadataEngine(position=2).snapshot()
        third['tokens'] = [7, 8]
        self.assertTrue(cache.put(third))
        self.assertIsNone(cache.best([4, 5], engine.fingerprint))
        self.assertIsNone(cache.best([0, 1], 'another-profile'))
        self.assertIsNone(cache.best([0, 2], engine.fingerprint))
        self.assertLessEqual(cache.used, cache.budget)
        cache.clear()
        self.assertEqual(cache.used, 0)
        self.assertFalse(cache.entries)

    def test_runtime_drops_restored_entry_before_next_capture(self):
        # Execute the actual Runtime class with CPU stubs, without importing Engine/CUDA.
        server = SOURCE / 'kindi/lightning_server.py'
        tree = ast.parse(server.read_text(encoding='utf-8'), filename=str(server))
        runtime_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Runtime')
        module = ast.Module(body=[runtime_class])
        if 'type_ignores' in ast.Module._fields:
            module.type_ignores = []
        namespace = {'time': time, 'threading': threading, 'codecs': codecs,
                     'np': types.SimpleNamespace(random=types.SimpleNamespace(RandomState=lambda seed: None)),
                     'sample': lambda *args: 99, 'split_text': lambda *args: ('', '', [])}
        exec(compile(ast.fix_missing_locations(module), str(server), 'exec'), namespace)
        runtime = namespace['Runtime'].__new__(namespace['Runtime'])
        engine = MetadataEngine('q8_0', position=2)
        old = engine.snapshot()
        old_ref = weakref.ref(old)
        runtime.cache = PrefixCache(200 * 1024 ** 2, 1)
        self.assertTrue(runtime.cache.put(old))
        del old
        engine.reset()
        engine.snapshot_calls = 0
        engine.cp = types.SimpleNamespace(cuda=types.SimpleNamespace(
            Stream=types.SimpleNamespace(null=types.SimpleNamespace(synchronize=lambda: None))))

        def inspect_capture(number):
            if number == 1:
                self.assertIsNotNone(old_ref())
            else:
                self.assertIsNone(old_ref(), 'Runtime retained an evicted restored snapshot')
        engine.on_snapshot = inspect_capture
        runtime.engine = engine
        runtime.tok = types.SimpleNamespace(stop_ids={99})
        runtime.cfg = {'prefix_min': 0}
        runtime.prefill_chunk = 128
        runtime.requests = 0
        prepared = ([0, 1, 2], False, [], 0, 1, 0, 1, [])
        result = runtime.generate({}, threading.Event(), lambda delta: None, prepared)
        self.assertEqual(result['usage']['prompt_tokens_details']['cached_tokens'], 2)
        self.assertEqual(engine.snapshot_calls, 2)
        self.assertEqual(runtime.requests, 1)
        self.assertIsNone(old_ref())


if __name__ == '__main__':
    unittest.main(verbosity=2)
