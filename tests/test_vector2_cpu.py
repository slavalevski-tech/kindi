"""CPU-only V6 configuration and dispatch contracts; no GPU or model required.

Portable publication adaptation, 2026-11-01.
SPDX-License-Identifier: AGPL-3.0-only
"""
import argparse,ast,ctypes,hashlib,importlib,json,os,sys,types
from pathlib import Path
from unittest import mock

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
STRATEGY='gqa16_vec2_32_bank4'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 global SOURCE
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output');p.add_argument('--source',type=Path,default=SOURCE);a=p.parse_args()
 SOURCE=a.source.resolve()
 out=Path(a.output) if a.output else None
 if out is not None and out.exists():raise RuntimeError('Refusing existing CPU evidence')
 if 'cupy' in sys.modules or 'torch' in sys.modules:raise RuntimeError('CPU-only checker')
 sys.path.insert(0,str(SOURCE));kv=importlib.import_module('kindi.kv');light=importlib.import_module('kindi.lightning');moe=importlib.import_module('kindi.moe')
 checks=[]
 def check(name,fn):fn();checks.append({'name':name,'passed':True})
 def raises(fn):
  try:fn()
  except ValueError:return
  raise AssertionError('Expected configuration rejection')
 def config():
  args=(STRATEGY,'direct','split','optimized',(32,2,128))
  for ctx in (1,4032,65536):assert kv.validate_gqa_config(*args,kv_dtype='fp16',ctx=ctx)==32
  for ctx in (0,65537,131072):raises(lambda:kv.validate_gqa_config(*args,kv_dtype='fp16',ctx=ctx))
  for mode in ('fp32','q8_0',None):raises(lambda:kv.validate_gqa_config(*args,kv_dtype=mode,ctx=4096))
  for read in ('auto','expanded'):raises(lambda:kv.validate_gqa_config(STRATEGY,read,'split','optimized',(32,2,128),kv_dtype='fp16',ctx=4096))
  for geometry in ((16,2,128),(32,2,64),(32,4,128)):raises(lambda:kv.validate_gqa_config(STRATEGY,'direct','split','optimized',geometry,kv_dtype='fp16',ctx=4096))
  raises(lambda:kv.validate_gqa_config(STRATEGY,'direct','warp','optimized',(32,2,128),kv_dtype='fp16',ctx=4096))
  raises(lambda:kv.validate_gqa_config(STRATEGY,'direct','split','baseline',(32,2,128),kv_dtype='fp16',ctx=4096))
  assert kv.validate_gqa_config('gqa16_32','direct','split','optimized',(32,2,128),kv_dtype='fp16',ctx=131072)==32
 check('explicit_profile_rejects_unsupported_dtype_read_geometry_capacity',config)
 def early():
  class ReachedCuda(Exception):pass
  class File:
   def __init__(self,path):self.gguf=None
  spec=types.SimpleNamespace(max_context=262144,qheads=32,kvheads=2,hd=128)
  env={'KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY':STRATEGY,'KINDI_LIGHTNING_KV_READ':'direct','KINDI_LIGHTNING_ATTENTION':'split','KINDI_LIGHTNING_KERNELS':'optimized'}
  with mock.patch.dict(os.environ,env),mock.patch.object(light,'PackedFile',File),mock.patch.object(light,'ModelSpec',lambda _:spec),mock.patch.object(light.Engine,'_validate_tensors',lambda _:None),mock.patch.object(light,'CudaOps',side_effect=ReachedCuda):
   raises(lambda:light.Engine('metadata-mock',ctx=65537,kv_dtype='fp16'))
   raises(lambda:light.Engine('metadata-mock',ctx=4096,kv_dtype='q8_0'))
   try:light.Engine('metadata-mock',ctx=65536,kv_dtype='fp16')
   except ReachedCuda:pass
   else:raise AssertionError('Valid configuration should reach CUDA constructor only after validation')
   with mock.patch.dict(os.environ,{'KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY':'gqa16_32'}):
    try:light.Engine('metadata-mock',ctx=131072,kv_dtype='fp16')
    except ReachedCuda:pass
    else:raise AssertionError('Original GQA32 capacity remains available')
 check('Engine_rejects_new_128K_before_CudaOps_and_preserves_old_GQA128K',early)
 def allocation():
  class CP:
   int32='i32';float16='f16';float32='f32';uint8='u8'
   def __init__(self):self.calls=[]
   def zeros(self,*args):return None
   def empty(self,shape,dtype):self.calls.append((shape,dtype));return shape
  cp=CP();ops=types.SimpleNamespace(cp=cp,profile='optimized',configure_attention_bank4=lambda:({'requested_bytes':4},))
  with mock.patch.dict(os.environ,{'KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY':STRATEGY,'KINDI_LIGHTNING_ATTENTION':'split'}):
   cache=kv.KVCacheOps(ops,'fp16','direct');assert cache.allocate(65536,2,128)==(65536,2,128)
   raises(lambda:cache.allocate(131072,2,128));assert len(cp.calls)==1 and cache.bank_requests==({'requested_bytes':4},)
 check('direct_KV_capacity_cap_before_empty_allocation',allocation)
 def bank():
  rows=[]
  class Setter:
   status=0
   def __call__(self,pointer,enum):rows.append((pointer.value,enum));return self.status
  setter=Setter();driver=types.SimpleNamespace(cuFuncSetSharedMemConfig=setter)
  ops=object.__new__(moe.CudaOps);ops.k={name:types.SimpleNamespace(kernel=types.SimpleNamespace(ptr=n)) for n,name in enumerate(('attention_tiles_fp16_gqa16_vector2_typed32','attention_merge_tiles'),1)}
  loader='WinDLL' if os.name=='nt' else 'CDLL'
  with mock.patch.object(ctypes,loader,return_value=driver):
   result=ops.configure_attention_bank4();assert rows==[(1,1),(2,1)] and all(x['return_code']==0 and x['requested_bytes']==4 for x in result)
   setter.status=1
   try:ops.configure_attention_bank4()
   except RuntimeError:pass
   else:raise AssertionError('Failed request must not silently fall back')
  assert 'cuCtxSetSharedMemConfig' not in (SOURCE/'kindi/moe.py').read_text(encoding='utf8')
 check('only_two_per_function_bank4_requests_and_fail_closed_status',bank)
 def profile():
  tree=ast.parse((SOURCE/'kindi/lightning.py').read_text(encoding='utf8'))
  text=ast.dump(tree);assert all(name in text for name in ('attention_shared_bank_bytes_requested','attention_shared_bank_scope','attention_strategy_context_limit'))
  assert "self.source_sha256+json.dumps(self.numeric_profile,sort_keys=True)" in (SOURCE/'kindi/lightning.py').read_text(encoding='utf8')
  restore=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='restore')
  assert 'fingerprint' in ast.dump(restore)
 check('genuine_numeric_profile_fingerprint_and_existing_restore_gate',profile)
 def syntax():
  for name in ('lightning.py','kv.py','moe.py'):ast.parse((SOURCE/'kindi'/name).read_text(encoding='utf8'),feature_version=(3,6))
 check('changed_host_source_Python36_AST',syntax)
 report={'complete':True,'passed':True,'scope':'CPU mocks/source contracts only; no numerical GPU/fullmodel acceptance','checks':checks,'source_files':{str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'kindi').rglob('*') if p.is_file() and p.suffix in ('.py','.cu')},'cupy_imported':'cupy' in sys.modules,'torch_imported':'torch' in sys.modules,'GPU_or_model_payload_used':False}
 assert not report['cupy_imported'] and not report['torch_imported']
 if out is not None:
  out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
 print(json.dumps({'passed':True,'groups':len(checks),'scope':report['scope']}))

if __name__=='__main__':main()
