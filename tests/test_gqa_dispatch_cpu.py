"""CPU-only ABI/policy tests; frozen operator evidence supplies numerical proof.

Real CudaOps/Engine/KVCacheOps code runs with metadata-only fake CUDA storage,
so no GPU, model, weights or large buffers are used.
"""
import ast
import os
from pathlib import Path
import re
import sys
import types
import unittest
from unittest import mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
CALLS=[]
ALLOCATIONS=[]


class Array(object):
    def __init__(self,shape,dtype):
        self.shape=(shape,) if isinstance(shape,int) else tuple(shape)
        self.dtype=np.dtype(dtype);self.ndim=len(self.shape);self.size=int(np.prod(self.shape))
        self.flags=types.SimpleNamespace(c_contiguous=True);self.data=types.SimpleNamespace(ptr=4096)
    def reshape(self,*shape):
        if len(shape)==1 and isinstance(shape[0],(tuple,list)):shape=shape[0]
        return Array(shape,self.dtype)
    def __getitem__(self,item):
        first=item[0] if isinstance(item,tuple) else item
        if isinstance(first,slice):
            size=len(range(*first.indices(self.shape[0])))
            return Array((size,)+self.shape[1:],self.dtype)
        return Array(self.shape[1:],self.dtype)
    def get(self):
        if self.size>16:raise AssertionError('CPU ABI test must not materialize large data')
        return np.zeros(self.shape,self.dtype)


class Kernel(object):
    def __init__(self,name):self.name=name
    def __call__(self,grid,block,args):CALLS.append((self.name,grid,block,args))


class Module(object):
    def __init__(self,code,options):
        self.names=set(re.findall(r'extern\s+"C"\s+__global__\s+void\s+(\w+)\s*\(',code))
        assert options==('--std=c++11',)
    def get_function(self,name):
        if name not in self.names:raise AssertionError('Requested undeclared CUDA function: '+name)
        return Kernel(name)


def empty(shape,dtype):
    ALLOCATIONS.append((shape,np.dtype(dtype).str));return Array(shape,dtype)


CP=types.ModuleType('cupy')
CP.ndarray=Array;CP.float32=np.float32;CP.float16=np.float16;CP.uint8=np.uint8;CP.int32=np.int32
CP.empty=empty;CP.zeros=empty;CP.empty_like=lambda value:empty(value.shape,value.dtype);CP.RawModule=Module
CP.cuda=types.SimpleNamespace(Device=lambda:types.SimpleNamespace(id=0),runtime=types.SimpleNamespace(
    memGetInfo=lambda:(2**40,2**40),getDeviceProperties=lambda device:{'major':3,'minor':5},runtimeGetVersion=lambda:10020))
sys.modules['cupy']=CP
from kindi.kv import KVCacheOps,validate_gqa_config
from kindi.moe import CudaOps
import kindi.lightning as lightning


def spec(geometry=(32,2,128)):
    return types.SimpleNamespace(qheads=geometry[0],kvheads=geometry[1],hd=geometry[2],max_context=131072,
        blocks=['attention'],heads=64,head_dim=64,state_size=128,groups=8,conv_dim=6144,conv=4)


class File(object):
    loaded=0
    def __init__(self,path):self.gguf=object();self.tensors={};self.resident={}
    def load_gpu(self,cp,reserve_bytes):File.loaded+=1
    def source_hash(self):return '1'*64


class GQAContracts(unittest.TestCase):
    def setUp(self):
        self.saved={key:value for key,value in os.environ.items() if key.startswith('KINDI_LIGHTNING_')}
        for key in self.saved:os.environ.pop(key)
        os.environ.update(KINDI_LIGHTNING_KERNELS='optimized',KINDI_LIGHTNING_ATTENTION='split',
            KINDI_LIGHTNING_Q2_INPUT='vec4',KINDI_LIGHTNING_KV_READ='direct',
            KINDI_LIGHTNING_ATTENTION_WORKSPACE_MB='512')
        CALLS[:]=[];ALLOCATIONS[:]=[];File.loaded=0
    def tearDown(self):
        for key in list(os.environ):
            if key.startswith('KINDI_LIGHTNING_'):os.environ.pop(key)
        os.environ.update(self.saved)
    def engine(self,strategy,read='direct',geometry=(32,2,128)):
        os.environ['KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY']=strategy
        with mock.patch.object(lightning,'PackedFile',File),mock.patch.object(lightning,'ModelSpec',lambda ignored:spec(geometry)),mock.patch.object(lightning.Engine,'_validate_tensors',lambda ignored:None):
            return lightning.Engine('CPU-ABI-FIXTURE',ctx=600,kv_dtype='q8_0',kv_read=read,log=lambda ignored:None)

    def test_invalid_contract_before_cuda_or_weights(self):
        cases=(('auto','optimized','split',(32,2,128)),('expanded','optimized','split',(32,2,128)),
               ('direct','baseline','split',(32,2,128)),('direct','optimized','warp',(32,2,128)),
               ('direct','optimized','split',(16,2,128)))
        for read,profile,attention,geometry in cases:
            os.environ.update(KINDI_LIGHTNING_KERNELS=profile,KINDI_LIGHTNING_ATTENTION=attention)
            with mock.patch.object(lightning,'CudaOps',side_effect=AssertionError('CUDA initialized before rejection')):
                with self.assertRaises(ValueError):self.engine('gqa16_32',read,geometry)
        self.assertEqual(File.loaded,0);self.assertEqual(ALLOCATIONS,[])

    def test_kernel_abi_all_dtypes_both_stages_and_255_tail(self):
        for mode,suffix in (('fp32','fp32'),('fp16','fp16'),('q8_0','q8')):
            for stage in (32,64):
                os.environ['KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY']='gqa16_'+str(stage)
                ops=CudaOps();kv=KVCacheOps(ops,mode,'direct');n=131072;m=255;past=n-m
                key=kv.allocate(n,2,128);value=CP.empty_like(key);query=Array((m,32,128),np.float32)
                CALLS[:]=[];kv.attention(query,key,value,past,m,32,2,128)
                tiles=[call for call in CALLS if call[0]!='attention_merge_tiles']
                self.assertEqual(len(tiles),9);self.assertIsNone(kv.scratch)
                for index,(name,grid,block,args) in enumerate(tiles):
                    count=31 if index<8 else 7
                    self.assertEqual(name,'attention_tiles_'+suffix+'_gqa16_typed'+str(stage))
                    self.assertEqual(grid,(2,count,1024));self.assertEqual(block,(512,))
                    self.assertEqual(tuple(int(v) for v in args[4:]),(count,past+31*index,32,2,128,1024))
                    self.assertIs(args[1],key);self.assertIs(args[2],value)
                merges=[call for call in CALLS if call[0]=='attention_merge_tiles']
                self.assertEqual(len(merges),9);self.assertTrue(all(call[2]==(128,) for call in merges))

    def test_existing_auto_policy_and_short_fallback(self):
        os.environ['KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY']='auto';ops=CudaOps();kv=KVCacheOps(ops,'q8_0','auto')
        n=131072;m=255;key=kv.allocate(n,2,128);CALLS[:]=[]
        kv.attention(Array((m,32,128),np.float32),key,CP.empty_like(key),n-m,m,32,2,128)
        self.assertEqual([call[0] for call in CALLS[:2]],['kv_decode_q8','kv_decode_q8'])
        tiles=[call[0] for call in CALLS if call[0].startswith('attention_tiles_')]
        self.assertEqual(tiles,['attention_tiles_fp32_quad']*8+['attention_tiles_fp32_sub8_fixed'])
        os.environ['KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY']='gqa16_32';kv=KVCacheOps(ops,'q8_0','direct');key=kv.allocate(128,2,128);CALLS[:]=[]
        kv.attention(Array((7,32,128),np.float32),key,CP.empty_like(key),121,7,32,2,128)
        self.assertEqual([call[0] for call in CALLS],['attention_q8_warp']);self.assertIsNone(kv.split_scratch)

    def test_profile_fingerprint_and_nonadditive_shared_memory(self):
        first=self.engine('gqa16_32');second=self.engine('gqa16_64')
        self.assertEqual(first.numeric_profile['kv_read'],'direct');self.assertEqual(first.memory['kv_scratch'],0)
        self.assertEqual(first.numeric_profile['attention_gqa_shared_bytes'],24576)
        self.assertEqual(second.numeric_profile['attention_gqa_shared_bytes'],40960)
        self.assertEqual(first.numeric_profile['attention_gqa_block_threads'],512)
        self.assertNotEqual(first.fingerprint,second.fingerprint)
        total=sum(first.memory[key] for key in ('weights','state','kv','kv_scratch','attention_scratch','ssm_scratch','reserve'))
        self.assertEqual(first.memory['total_with_reserve'],total)
        self.assertEqual(first.memory['attention_static_shared_per_block'],24576)
        self.assertEqual(first.numeric_profile['q2_decode'],'native')
        os.environ['KINDI_LIGHTNING_Q2_DECODE']='or';third=self.engine('gqa16_32')
        self.assertEqual(third.numeric_profile['q2_decode'],'or');self.assertNotEqual(first.fingerprint,third.fingerprint)


if __name__=='__main__':unittest.main()
