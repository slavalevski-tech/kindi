"""CPU-only split-head startup ownership, configuration and launch contracts."""
import argparse
import ast
import gc
import hashlib
import json
import os
import sys
import types
import weakref
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kindi import lightning
from kindi.moe import CudaOps
from kindi.packed import PackedFile


class Array(object):
    next_ptr=4096
    def __init__(self,shape,dtype=np.float32,ptr=None,contiguous=True):
        self.shape=(shape,) if isinstance(shape,int) else tuple(shape)
        self.size=int(np.prod(self.shape));self.dtype=np.dtype(dtype)
        self.nbytes=self.size*self.dtype.itemsize
        self.flags=types.SimpleNamespace(c_contiguous=contiguous)
        if ptr is None:
            ptr=Array.next_ptr;Array.next_ptr+=((self.nbytes+4095)//4096)*4096
        self.data=types.SimpleNamespace(ptr=ptr)
    def __getitem__(self,key):
        assert key==0
        return Array(self.shape[1:],self.dtype,self.data.ptr)
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],tuple):shape=shape[0]
        assert int(np.prod(shape))==self.size
        return Array(shape,self.dtype,self.data.ptr)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output');args=p.parse_args();checks=[];events=[];launches=[];loads=[];modules=[]
    assert 'cupy' not in sys.modules and 'torch' not in sys.modules
    real=PackedFile(args.model);spec=lightning.ModelSpec(real.gguf)
    def passed(name):checks.append({'name':name,'passed':True})
    def rejects(action,kind=ValueError):
        try:action()
        except kind:return
        raise AssertionError('Expected '+kind.__name__)
    class Kernel(object):
        def __init__(self,name):self.name=name
        def __call__(self,grid,block,values):
            if self.name=='head_q8_split':
                packed,codes,scales,blocks=values
                assert grid==(1376256,) and block==(256,) and int(blocks)==11010048
                assert packed.shape==(374341632,) and packed.dtype==np.uint8
                assert codes.shape==(131072,2688) and codes.dtype==np.int8
                assert scales.shape==(131072,84) and scales.dtype==np.float32
                assert not any(e=='state_alloc' for e in events)
                events.append('split_build')  # do NOT retain original packed arguments
            else:launches.append((self.name,grid,block,values))
    class Module(object):
        def __init__(self,code,options):
            assert options==('--std=c++11',);modules.append(len(code))
        def get_function(self,name):return Kernel(name)
    class Runtime(object):
        free_sequence=[]
        def memGetInfo(self):return (self.free_sequence.pop(0) if self.free_sequence else 20*2**30,24*2**30)
        def getDeviceProperties(self,device):return {'major':3,'minor':5}
        def runtimeGetVersion(self):return 10020
    runtime=Runtime()
    def sync():events.append('build_sync')
    class Pool(object):
        def free_all_blocks(self):
            assert FakeFile.head_ref() is None
            assert 'output.weight' not in FakeFile.last.resident
            events.append('packed_freed_before_state_KV')
    pool=Pool()
    def empty(shape,dtype):
        if FakeFile.last is not None and FakeFile.last.loaded:
            if tuple(np.atleast_1d(shape)) not in ((131072,2688),(131072,84)):
                if os.environ.get('KINDI_LIGHTNING_HEAD_LAYOUT')=='split':
                    assert 'packed_freed_before_state_KV' in events and FakeFile.head_ref() is None
                events.append('state_alloc')
        return Array(shape,dtype)
    fake=types.ModuleType('cupy');fake.ndarray=Array;fake.float32=np.float32;fake.float16=np.float16
    fake.int8=np.int8;fake.int32=np.int32;fake.uint8=np.uint8;fake.empty=empty;fake.zeros=empty
    fake.empty_like=lambda x:empty(x.shape,x.dtype);fake.RawModule=Module
    fake.get_default_memory_pool=lambda:pool
    fake.cuda=types.SimpleNamespace(runtime=runtime,Device=lambda:types.SimpleNamespace(id=0),
        get_current_stream=lambda:types.SimpleNamespace(synchronize=sync))
    class FakeFile(object):
        last=None;head_ref=None;alias=False
        def __init__(self,path):
            self.gguf=real.gguf;self.tensors=real.tensors;self.resident={};self.loaded=False
            FakeFile.last=self
        def load_gpu(self,cp,reserve_bytes):
            loads.append(reserve_bytes)
            for name,tensor in self.tensors.items():
                self.resident[name]=Array(tensor.shape,np.float32) if tensor.kind==0 else Array((tensor.nbytes,),np.uint8)
            FakeFile.head_ref=weakref.ref(self.resident['output.weight'])
            if self.alias:self.resident['unexpected_alias']=self.resident['output.weight']
            self.loaded=True;events.append('packed_loaded')
        def source_hash(self):return '38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3'
    keys=[k for k in os.environ if k.startswith('KINDI_LIGHTNING_')]
    saved={k:os.environ[k] for k in keys};original_file=lightning.PackedFile
    try:
        for key in keys:os.environ.pop(key)
        os.environ.update(KINDI_LIGHTNING_KERNELS='optimized',KINDI_LIGHTNING_Q2_INPUT='vec4',
            KINDI_LIGHTNING_Q2_DECODE='or',KINDI_LIGHTNING_SSM_STRATEGY='auto',
            KINDI_LIGHTNING_ATTENTION='split',KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY='gqa16_32',
            KINDI_LIGHTNING_ATTENTION_WORKSPACE_MB='512',KINDI_LIGHTNING_KV_READ='direct')
        sys.modules['cupy']=fake;lightning.PackedFile=FakeFile
        native=lightning.Engine(args.model,ctx=131072,kv_dtype='fp16',log=lambda x:None)
        assert native.head_layout=='native' and 'output.weight' in native.w
        assert native.head_codes is None and native.head_scales is None
        assert native.numeric_profile['head_layout']=='native'
        assert not any('split' in e for e in events)
        native_plan=native.memory.copy();native_fingerprint=native.fingerprint
        native.linear('output.weight',Array((2688,)));assert launches[-1][0]=='q8_mv'
        passed('native_default_no_conversion_original_dispatch_and_profile')
        events[:]=[];os.environ['KINDI_LIGHTNING_HEAD_LAYOUT']='split'
        split=lightning.Engine(args.model,ctx=131072,kv_dtype='fp16',log=lambda x:None)
        assert events.index('packed_loaded')<events.index('split_build')<events.index('build_sync')<events.index('packed_freed_before_state_KV')<events.index('state_alloc')
        assert FakeFile.head_ref() is None and split.w is split.file.resident and 'output.weight' not in split.w
        assert real.tensors['output.weight'].shape==(131072,2688) and real.tensors['output.weight'].nbytes==374341632
        assert split.w['token_embd.weight'].nbytes==374341632
        assert split.numeric_profile['head_layout']=='split' and split.fingerprint!=native_fingerprint
        assert {k:v for k,v in split.numeric_profile.items() if k!='head_layout'}=={k:v for k,v in native.numeric_profile.items() if k!='head_layout'}
        assert loads[-1]+split.memory['weights_packed']==split.memory['allocation_peak_with_reserve']
        passed('startup_build_sync_retire_all_dictionary_aliases_free_before_state_KV_original_descriptors_embedding_retained')
        plan=split.memory
        assert plan['head_split_bytes']==378*2**20 and plan['head_packed_retired_bytes']==357*2**20
        assert plan['head_steady_extra_bytes']==21*2**20 and plan['head_startup_transient_bytes']==378*2**20
        assert plan['weights']-native_plan['weights']==21*2**20
        assert plan['total_with_reserve']-native_plan['total_with_reserve']==21*2**20
        assert plan['startup_with_reserve']==plan['weights_packed']+378*2**20+plan['reserve']
        assert plan['allocation_peak_with_reserve']==max(plan['startup_with_reserve'],plan['total_with_reserve'])
        passed('steady_plus21MiB_startup_plus378MiB_and_capacity_peak_not_doublecounted')
        x=Array((2688,));out=split.linear('output.weight',x)
        name,grid,block,values=launches[-1]
        assert (name,grid,block)==('head_q8_split_mv',(32768,),(128,)) and out.shape==(131072,)
        assert values[:3]==(split.head_codes,split.head_scales,x) and values[3] is out
        assert tuple(int(v) for v in values[4:])==(131072,2688)
        split.linear('token_embd.weight',x);assert launches[-1][0]=='q8_mv'
        split.linear('blk.0.ssm_in.weight',x);assert launches[-1][0]=='q2_mv_vec4_or'
        rejects(lambda:split.linear('output.weight',x,ids=Array((1,),np.int32)))
        rejects(lambda:split.linear('output.weight',x,per_input=True))
        rejects(split._prepare_split_head,RuntimeError)
        passed('output_only_ABI_native_embedding_other_tensors_and_no_warm_conversion')
        for invalid in (Array((2688,),np.float16),Array((2687,)),Array((2688,),ptr=2),Array((2688,),contiguous=False)):
            rejects(lambda:split.linear('output.weight',invalid))
        rejects(lambda:split.ops.split_head_linear(Array((131071,2688),np.int8),split.head_scales,x))
        rejects(lambda:split.ops.split_head_linear(split.head_codes,Array((131072,84),np.float16),x))
        passed('strict_fixed_layout_input_and_unselected_row_guards')
        invalid=lightning.Engine.__new__(lightning.Engine);invalid.spec=spec;invalid.file=real
        for mode in ('bogus','SPLIT',''):
            invalid.head_layout=mode;rejects(invalid._validate_head_layout)
        invalid.head_layout='split';os.environ['KINDI_LIGHTNING_KERNELS']='baseline';rejects(invalid._validate_head_layout)
        os.environ['KINDI_LIGHTNING_KERNELS']='optimized'
        invalid.spec=types.SimpleNamespace(vocab=131071,width=2688);rejects(invalid._validate_head_layout)
        invalid.spec=spec;original_tensor=real.tensors['output.weight']
        for update in ({'shape':(131072,2687)},{'kind':42},{'nbytes':374341631}):
            bad=types.SimpleNamespace(shape=original_tensor.shape,kind=original_tensor.kind,nbytes=original_tensor.nbytes)
            for k,v in update.items():setattr(bad,k,v)
            real.tensors['output.weight']=bad;rejects(invalid._validate_head_layout)
        real.tensors['output.weight']=original_tensor
        passed('enum_optimized_and_exact_model_head_gates_before_device_allocation')
        before_modules=len(modules);before_loads=len(loads)
        for mode,profile in (('invalid','optimized'),('split','baseline')):
            os.environ['KINDI_LIGHTNING_HEAD_LAYOUT']=mode;os.environ['KINDI_LIGHTNING_KERNELS']=profile
            rejects(lambda:lightning.Engine(args.model,ctx=131072,kv_dtype='fp16',log=lambda x:None))
        assert len(modules)==before_modules and len(loads)==before_loads
        os.environ['KINDI_LIGHTNING_HEAD_LAYOUT']='split';os.environ['KINDI_LIGHTNING_KERNELS']='optimized'
        passed('actual_Engine_invalid_configuration_rejected_before_RawModule_and_weight_load')
        before=len(loads);runtime.free_sequence=[plan['allocation_peak_with_reserve']-1]
        rejects(lambda:lightning.Engine(args.model,ctx=131072,kv_dtype='fp16',log=lambda x:None),MemoryError)
        assert len(loads)==before
        events[:]=[];runtime.free_sequence=[20*2**30,378*2**20+plan['reserve']-1]
        rejects(lambda:lightning.Engine(args.model,ctx=131072,kv_dtype='fp16',log=lambda x:None),MemoryError)
        assert 'split_build' not in events
        FakeFile.alias=True;events[:]=[]
        rejects(lambda:lightning.Engine(args.model,ctx=131072,kv_dtype='fp16',log=lambda x:None))
        assert 'split_build' not in events;FakeFile.alias=False
        passed('initial_peak_conversion_reserve_and_unexpected_resident_alias_fail_closed')
        # Exercise the other max() arm with the supported128MiB workspace;
        # the target512MiB workspace already makes steady larger even atctx1.
        split.kv.split_workspace_bytes=128*2**20
        os.environ['KINDI_LIGHTNING_ATTENTION_WORKSPACE_MB']='128'
        small=split.memory_plan(1)
        assert small['startup_with_reserve']>small['total_with_reserve']
        assert small['allocation_peak_with_reserve']==small['startup_with_reserve']
        before=len(loads);runtime.free_sequence=[small['startup_with_reserve']-1]
        rejects(lambda:lightning.Engine(args.model,ctx=1,kv_dtype='fp16',log=lambda x:None),MemoryError)
        assert len(loads)==before
        split.kv.split_workspace_bytes=512*2**20
        os.environ['KINDI_LIGHTNING_ATTENTION_WORKSPACE_MB']='512'
        passed('startup_dominant_short_context_capacity_enforced_before_weight_load')
        syntax={'feature_version':(3,6)} if sys.version_info>=(3,8) else {}
        for path in (ROOT/'kindi').glob('*.py'):ast.parse(path.read_text(encoding='utf8'),**syntax)
        base=(ROOT/'kindi/lightning.py').read_text(encoding='utf8')
        assert "'head_layout':self.head_layout" in base and "snapshot['fingerprint']!=self.fingerprint" in base
        rejects(lambda:split.restore({'version':split.state_version,'fingerprint':native_fingerprint,
            'position':0,'tokens':[],'logits':None,'states':{},'histories':{},'keys':{},'values':{}}))
        passed('Python36_AST_profile_fingerprint_and_strict_snapshot_contract')
        code=(ROOT/'kindi/lightning.cu').read_bytes().replace(b'\r\n',b'\n')
        for name,expected in (('head_q8_split','370a3125cc9f1eebc38318e8447771c514d7c65ae929c5da97c3f1106b9db545'),
                ('head_q8_split_mv','5af87d4e09ff3009d1ea0d79996f29236b787e2769d5f5a35ec5146f5b414741')):
            start=code.index(b'extern "C" __global__ void '+name.encode()+b'(')
            end=code.index(b'{',start)+1;depth=1
            while depth:
                depth+=(code[end:end+1]==b'{')-(code[end:end+1]==b'}');end+=1
            assert hashlib.sha256(code[start:end]).hexdigest()==expected
        passed('two_CUDA_function_bodies_match_frozen_K6_operator_after_EOL_normalization')
    finally:
        lightning.PackedFile=original_file;sys.modules.pop('cupy',None)
        for key in list(os.environ):
            if key.startswith('KINDI_LIGHTNING_'):os.environ.pop(key)
        os.environ.update(saved)
    assert 'cupy' not in sys.modules and 'torch' not in sys.modules
    result={'complete':True,'passed':True,'checks':checks,'gpu_used':False,'CUDA_compiler_called':False,
        'model_header_only':True,'candidate_runtime_acceptance':False,'head_shape':[131072,2688],
        'steady_extra_bytes':21*2**20,'startup_transient_bytes':378*2**20}
    if args.output:Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passed':True,'checks':len(checks),'gpu_used':False}))


if __name__=='__main__':main()
