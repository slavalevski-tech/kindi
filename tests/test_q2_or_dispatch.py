"""CPU-only Q2 OR policy/ABI contracts; fake device arrays, no CUDA compiler."""
import argparse,ast,json,os,sys,types
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from kindi import lightning
from kindi.moe import CudaOps
from kindi.lightning_batch import BatchOps
from kindi.packed import PackedFile


class Array(object):
    def __init__(self,shape,dtype=np.float32,ptr=16,contiguous=True):
        self.shape=tuple(shape);self.size=int(np.prod(shape));self.dtype=np.dtype(dtype)
        self.flags=types.SimpleNamespace(c_contiguous=contiguous);self.data=types.SimpleNamespace(ptr=ptr)
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
    p.add_argument('--output');a=p.parse_args();checks=[];calls=[];compiles=[]
    assert 'cupy' not in sys.modules and 'torch' not in sys.modules
    def passed(name):checks.append({'name':name,'passed':True})
    def rejects(call):
        try:call()
        except ValueError:return
        raise AssertionError('Expected invalid configuration/payload rejection')
    class Kernel(object):
        def __init__(self,name):self.name=name
        def __call__(self,grid,block,args):calls.append((self.name,grid,block,args))
    class Module(object):
        def __init__(self,code,options):compiles.append({'bytes':len(code),'options':options})
        def get_function(self,name):return Kernel(name)
    fake=types.ModuleType('cupy');fake.ndarray=Array;fake.float32=np.float32;fake.empty=lambda shape,dtype:Array(shape,dtype);fake.RawModule=Module
    env_keys=('KINDI_LIGHTNING_KERNELS','KINDI_LIGHTNING_Q2_DECODE','KINDI_LIGHTNING_Q2_INPUT','KINDI_LIGHTNING_SSM_STRATEGY')
    saved={key:os.environ.get(key) for key in env_keys};original_cudaops=lightning.CudaOps
    try:
        os.environ['KINDI_LIGHTNING_KERNELS']='optimized';os.environ['KINDI_LIGHTNING_SSM_STRATEGY']='legacy'
        os.environ.pop('KINDI_LIGHTNING_Q2_DECODE',None);sys.modules['cupy']=fake
        native=CudaOps();assert native.q2_decode=='native'
        os.environ['KINDI_LIGHTNING_Q2_DECODE']='invalid';count=len(compiles);rejects(CudaOps);assert len(compiles)==count
        os.environ['KINDI_LIGHTNING_Q2_DECODE']='or';os.environ['KINDI_LIGHTNING_KERNELS']='baseline';rejects(CudaOps);assert len(compiles)==count
        os.environ['KINDI_LIGHTNING_KERNELS']='optimized';ops=CudaOps();ops.q2_input='vec4';assert ops.q2_decode=='or'
        assert 'q2_mv_vec4_or' in ops.k;passed('native_default_enum_and_optimized_config_gate_before_fake_compiler')
        sys.modules.pop('cupy')
        os.environ['KINDI_LIGHTNING_Q2_INPUT']='scalar';lightning.CudaOps=lambda:ops
        rejects(lambda:lightning.Engine(a.model,ctx=131072))
        lightning.CudaOps=original_cudaops;passed('Engine_rejects_OR_scalar_before_weight_load_or_runtime_calls')
        source=PackedFile(a.model);eng=lightning.Engine.__new__(lightning.Engine)
        eng.file=source;eng.ops=ops;eng.w={name:object() for name in source.tensors};eng.cp=fake
        q2=[t for t in source.tensors.values() if t.kind==42];assert len(q2)==162
        for tensor in q2:
            rows,cols=tensor.shape[-2:];selected=6 if len(tensor.shape)==3 else 1
            ids=Array((selected,),np.int32) if selected>1 else None
            x=Array((selected,cols))
            for per_input in (False,True):
                result=eng.linear(tensor.name,x,ids,per_input)
                name,grid,block,args=calls[-1];assert name=='q2_mv_vec4_or' and grid==((rows+3)//4,selected) and block==(128,)
                assert args[0] is eng.w[tensor.name] and args[1] is x and args[3] is (ids if ids is not None else ops.null)
                assert tuple(int(x) for x in args[4:])==(rows,cols,selected,int(per_input))
                assert result.shape==((rows,) if selected==1 else (selected,rows))
                ops.linear(eng.w[tensor.name],x,rows,cols,42,ids,selected,per_input);assert calls[-1][0]=='q2_mv_vec4'
        passed('all162_actual_Q2_tensor_shapes_ids_selected_per_input_launch_ABI_Engine_OR_batch_default_native')
        native.q2_input='vec4';native.linear(object(),Array((2688,)),1856,2688,42,decode=True);assert calls[-1][0]=='q2_mv_vec4'
        ops.linear(object(),Array((2688,)),128,2688,8,decode=True);assert calls[-1][0]=='q8_mv'
        ops.linear(object(),Array((2688,)),128,2688,0,decode=True);assert calls[-1][0]=='packed_mv'
        passed('native_Q2_and_non_Q2_paths_keep_original_kernels')
        for invalid in (Array((2688,),ptr=8),Array((2688,),dtype=np.float16),Array((2688,),contiguous=False),Array((2687,))):
            rejects(lambda:ops.linear(object(),invalid,1856,2688,42,decode=True))
        rejects(lambda:ops.linear(object(),Array((2689,)),1856,2689,42,decode=True))
        rejects(lambda:ops.linear(object(),Array((6*2688-1,)),1856,2688,42,ids=Array((6,),np.int32),selected=6,per_input=True,decode=True))
        ops.q2_input='scalar';rejects(lambda:ops.linear(object(),Array((2688,)),1856,2688,42,decode=True));ops.q2_input='vec4'
        passed('vec4_alignment_dtype_contiguity_cols_and_complete_rows_guards_preserved')
        token_tensor=next(t for t in q2 if len(t.shape)==2);rows,cols=token_tensor.shape
        batch=BatchOps(eng);batch.dense_mode='tile24';batch.expert_mode='tile24';batch.linear(token_tensor.name,Array((7,cols)));assert calls[-1][0]=='q2_tile24'
        batch.dense_mode='batched_mv';batch.linear(token_tensor.name,Array((7,cols)));assert calls[-1][0]=='q2_mv_vec4'
        passed('BatchOps_tile24_and_batched_MV_fallback_remain_native')
        def scalar_forward(token,logits=True):
            eng.linear(token_tensor.name,Array((cols,)));assert calls[-1][0]=='q2_mv_vec4_or'
            return token
        eng.forward=scalar_forward;eng.synchronize=lambda:None;eng.position=0;eng.ctx=131072;eng.failed=False
        assert eng.prefill([1,2],chunk_size=0)==2 and not hasattr(eng,'_prefill_active') and ops.q2_decode=='or'
        passed('scalar_prefill_reuses_OR_forward_without_transient_phase_state')
        codes=np.arange(256,dtype=np.uint32)[:,None];shifts=np.asarray((0,2,4,6),np.uint32)[None,:]
        fields=(codes>>shifts)&3;native_coeff=(fields.astype(np.int32)-1).astype(np.float32)
        bits=np.bitwise_or(np.uint32(0x4b000000),fields);coeff=bits.view(np.float32)-np.float32(8388609.0)
        assert coeff.view(np.uint32).tobytes()==native_coeff.view(np.uint32).tobytes()
        assert np.all(coeff.view(np.uint32)[fields==1]==0);passed('CPU1024_coefficient_bits_all256_codes_including_code3_and_positive_zero')
        syntax={'feature_version':(3,6)} if sys.version_info>=(3,8) else {}
        for path in (ROOT/'kindi').glob('*.py'):ast.parse(path.read_text(encoding='utf8'),**syntax)
        passed('Python36_source_AST')
    finally:
        lightning.CudaOps=original_cudaops;sys.modules.pop('cupy',None)
        for key,value in saved.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    assert 'cupy' not in sys.modules and 'torch' not in sys.modules
    report={'complete':True,'passed':True,'checks':checks,'gpu_used':False,'actual_CUDA_compiler_called':False,'fake_compiler_calls':len(compiles),'real_model_header_only':True,'Q2_tensors_dispatched':len(q2),'coefficient_bits_checked':1024,'candidate_GPU_acceptance':False}
    if a.output:Path(a.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'passed':True,'checks':len(checks),'gpu_used':False,'Q2_tensors':len(q2)}))


if __name__=='__main__':main()
