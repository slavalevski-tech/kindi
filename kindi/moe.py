"""Reusable packed linear, routing and selected-expert CUDA operations."""
import os
import hashlib
import numpy as np


class CudaOps(object):
    def __init__(self):
        import cupy as cp
        self.cp = cp
        # Other adapters retain scalar Q2 loads; Lightning explicitly selects its profile.
        self.q2_input = 'scalar'
        self.profile=os.environ.get('KINDI_LIGHTNING_KERNELS','optimized')
        if self.profile not in ('baseline','optimized'):raise ValueError('unknown Lightning numerical profile')
        self.q2_decode=os.environ.get('KINDI_LIGHTNING_Q2_DECODE','native')
        if self.q2_decode not in ('native','or'):raise ValueError('Q2 decode mode must be native or or')
        if self.q2_decode=='or' and self.profile!='optimized':raise ValueError('OR Q2 decode requires optimized kernels')
        self.ssm_strategy=os.environ.get('KINDI_LIGHTNING_SSM_STRATEGY','legacy')
        if self.ssm_strategy not in ('legacy','shared','precomputed','auto'):
            raise ValueError('SSM strategy must be legacy, shared, precomputed or auto')
        if self.ssm_strategy!='legacy' and self.profile!='optimized':
            raise ValueError('Kepler SSM strategies require optimized Lightning kernels')
        self.ssm_parameters_scratch=None
        code = open(os.path.join(os.path.dirname(__file__), 'lightning.cu'), encoding='utf-8').read()
        code += '\n'+open(os.path.join(os.path.dirname(__file__),'lightning_batch.cu'),encoding='utf-8').read()
        code += '\n'+open(os.path.join(os.path.dirname(__file__),'lightning_kv.cu'),encoding='utf-8').read()
        self.code_hash=hashlib.sha256(code.encode('utf-8')).hexdigest()
        self.module = cp.RawModule(code=code, options=('--std=c++11',))
        self.k = {name:self.module.get_function(name) for name in (
            'packed_mv','q2_mv','q2_mv_vec4','q2_mv_vec4_or','q8_mv','head_q8_split','head_q8_split_mv','embedding','rms','relu2','residual','route','combine',
            'causal_conv','ssm_step','attention','embedding_batch','rms_batch','route_batch','expert_histogram',
            'expert_offsets','expert_scatter','packed_mm','grouped_mv','combine_batch','conv_chunk','ssm_chunk','attention_tiled',
            'unpack_dense','unpack_jobs','gather_jobs','scatter_jobs','grouped_mv4','q2_tile8','q2_tile16','q2_tile24','q2_tile32',
            'kv_store_fp16','kv_store_q8','attention_fp16','attention_q8','kv_decode_fp16','kv_decode_q8',
            'attention_warp','attention_fp16_warp','attention_q8_warp',
            'attention_tiles_fp32','attention_tiles_fp16','attention_tiles_q8','attention_merge_tiles',
            'attention_tiles_fp32_sub8_fixed','attention_tiles_fp16_sub8_fixed','attention_tiles_q8_sub8_fixed',
            'attention_tiles_fp32_quad','attention_tiles_fp16_quad','attention_tiles_q8_quad',
            'attention_tiles_fp32_gqa16_typed32','attention_tiles_fp16_gqa16_typed32','attention_tiles_q8_gqa16_typed32',
            'attention_tiles_fp32_gqa16_typed64','attention_tiles_fp16_gqa16_typed64','attention_tiles_q8_gqa16_typed64',
            'attention_tiles_fp16_gqa16_vector2_typed32',
            'v5_ssm_step_shared','v5_ssm_step_precomputed','v5_ssm_parameters',
            'v5_ssm_chunk_shared','v5_ssm_chunk_precomputed')}
        self.null = np.uint64(0)

    def configure_attention_bank4(self):
        """Explicit per-function request; never change context-wide bank state."""
        import ctypes
        driver=ctypes.WinDLL('nvcuda.dll') if os.name=='nt' else ctypes.CDLL('libcuda.so.1')
        setter=driver.cuFuncSetSharedMemConfig
        setter.argtypes=[ctypes.c_void_p,ctypes.c_int];setter.restype=ctypes.c_int
        rows=[]
        for name in ('attention_tiles_fp16_gqa16_vector2_typed32','attention_merge_tiles'):
            pointer=int(self.k[name].kernel.ptr)
            if pointer<=0:raise RuntimeError('Invalid CUDA function handle: '+name)
            status=int(setter(ctypes.c_void_p(pointer),1)) # CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE
            if status!=0:raise RuntimeError('Per-function FOUR_BYTE request failed: '+name+' status '+str(status))
            rows.append({'function':name,'requested_bytes':4,'enum':1,'return_code':status})
        return tuple(rows)

    def split_head_linear(self,codes,scales,x):
        cp=self.cp
        if (self.profile!='optimized' or not isinstance(codes,cp.ndarray) or codes.dtype!=cp.int8
                or codes.shape!=(131072,2688) or not codes.flags.c_contiguous
                or not isinstance(scales,cp.ndarray) or scales.dtype!=cp.float32
                or scales.shape!=(131072,84) or not scales.flags.c_contiguous
                or not isinstance(x,cp.ndarray) or x.dtype!=cp.float32
                or not x.flags.c_contiguous or x.size!=2688 or x.data.ptr%4):
            raise ValueError('split head requires the fixed contiguous Q8 layout and one FP32 input')
        y=cp.empty(131072,cp.float32)
        self.k['head_q8_split_mv']((32768,),(128,),
            (codes,scales,x,y,np.int32(131072),np.int32(2688)))
        return y

    def ssm_parameters(self,dt,dtbias,a,tokens,heads,stride):
        count=2*tokens*heads
        if (not isinstance(dt,self.cp.ndarray) or dt.dtype!=self.cp.float32
                or not dt.flags.c_contiguous or not 1<=tokens<=256 or heads<1 or stride<heads
                or dt.size<(tokens-1)*stride+heads):
            raise ValueError('invalid contiguous FP32 SSM parameter source')
        if self.ssm_parameters_scratch is None or self.ssm_parameters_scratch.size<count:
            self.ssm_parameters_scratch=self.cp.empty(count,self.cp.float32)
        self.k['v5_ssm_parameters'](((tokens*heads+127)//128,),(128,),
            (dt,dtbias,a,self.ssm_parameters_scratch,np.int32(tokens),np.int32(heads),np.int32(stride)))
        return self.ssm_parameters_scratch

    def linear(self, packed, x, rows, cols, kind, ids=None, selected=1, per_input=False,decode=False):
        if decode and kind==42 and self.q2_decode=='or' and (self.profile!='optimized' or self.q2_input!='vec4'):
            raise ValueError('OR Q2 decode requires optimized vec4 input')
        y = self.cp.empty((selected, rows), self.cp.float32)
        if self.profile=='optimized' and kind in (8,42):
            kernel='q2_mv' if kind==42 else 'q8_mv'
            if kind==42 and self.q2_input=='vec4':
                required=cols*(selected if per_input else 1)
                if (not isinstance(x,self.cp.ndarray) or x.dtype!=self.cp.float32
                        or not x.flags.c_contiguous or x.data.ptr%16 or cols%64
                        or x.size<required):
                    raise ValueError('vec4 Q2 input requires contiguous 16-byte-aligned FP32 storage, cols divisible by64, and a complete input row')
                kernel='q2_mv_vec4'
                if decode and self.q2_decode=='or':kernel='q2_mv_vec4_or'
            self.k[kernel](((rows+3)//4,selected),(128,),
                (packed,x,y,ids if ids is not None else self.null,np.int32(rows),np.int32(cols),np.int32(selected),np.int32(per_input)))
            return y[0] if selected==1 else y
        self.k['packed_mv'](((rows+3)//4, selected),(128,),
            (packed,x,y,ids if ids is not None else self.null,np.int32(rows),np.int32(cols),
             np.int32(kind),np.int32(selected),np.int32(per_input)))
        return y[0] if selected == 1 else y

    def norm(self, x, w, eps=1e-5, width=None, gate=None):
        width = int(width or x.size)
        if x.size % width or w.size != x.size:
            raise ValueError('normalization shape mismatch')
        y = self.cp.empty_like(x)
        self.k['rms']((x.size//width,),(256,),
            (x,w,y,np.int32(width),np.int32(x.size//width),np.float32(eps),gate if gate is not None else self.null))
        return y

    def activate(self, x):
        self.k['relu2'](((x.size+255)//256,),(256,),(x,np.int32(x.size)))
        return x

    def add(self, x, y):
        self.k['residual'](((x.size+255)//256,),(256,),(x,y,np.int32(x.size)))

    def router(self, logits, bias, topk, scale):
        if not 1 <= topk <= logits.size or logits.size != bias.size:
            raise ValueError('invalid routing dimensions')
        ids=self.cp.empty(topk,self.cp.int32)
        weights=self.cp.empty(topk,self.cp.float32)
        if self.profile=='optimized' and logits.size<=256:
            threads=1
            while threads<logits.size:threads*=2
            self.k['route_batch']((1,),(threads,),(logits,bias,ids,weights,np.int32(1),np.int32(logits.size),np.int32(topk),np.float32(scale)))
            return ids,weights
        self.k['route']((1,),(1,),(logits,bias,ids,weights,np.int32(logits.size),np.int32(topk),np.float32(scale)))
        return ids, weights

    def combine(self, expert, weights, shared):
        topk,width=expert.shape
        y=self.cp.empty(width,self.cp.float32)
        self.k['combine'](((width+255)//256,),(256,),
            (expert,weights,shared,y,np.int32(width),np.int32(topk)))
        return y
