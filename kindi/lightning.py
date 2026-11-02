"""Native Kindi CUDA adapter for Nota Nemotron Lightning, sequential correctness path.

NoPE attention, sigmoid corrected top-k, ReLU2 experts, FP32 Mamba2 states.
Weights stay packed on the selected GPU. No model computation falls back to CPU.
"""
import hashlib
import json
import os
import numpy as np
from .packed import PackedFile
from .moe import CudaOps


class ModelSpec(object):
    def __init__(self, gguf):
        kv=gguf.kv
        if kv.get('general.architecture')!='nemotron_h_moe':
            raise ValueError('Lightning adapter requires nemotron_h_moe')
        p='nemotron_h_moe.'
        def get(key):
            if p+key not in kv: raise ValueError('missing model metadata: '+p+key)
            return kv[p+key]
        self.width=int(get('embedding_length'))
        self.layers=int(get('block_count'))
        ff=get('feed_forward_length'); kvh=get('attention.head_count_kv')
        if not isinstance(ff,list) or not isinstance(kvh,list) or len(ff)!=self.layers or len(kvh)!=self.layers:
            raise ValueError('explicit layer schedule required')
        self.blocks=['moe' if ff[i] else ('attention' if kvh[i] else 'mamba') for i in range(self.layers)]
        self.qheads=int(get('attention.head_count'))
        self.kvheads=max(kvh)
        self.hd=int(get('attention.key_length'))
        self.eps=float(get('attention.layer_norm_rms_epsilon'))
        self.experts=int(get('expert_count'));self.topk=int(get('expert_used_count'))
        self.scale=float(get('expert_weights_scale'))
        self.routed_width=int(get('expert_feed_forward_length'));self.shared_width=int(get('expert_shared_feed_forward_length'))
        self.inner=int(get('ssm.inner_size'));self.heads=int(get('ssm.time_step_rank'))
        self.state_size=int(get('ssm.state_size'));self.groups=int(get('ssm.group_count'))
        self.conv=int(get('ssm.conv_kernel'))
        if min(self.width,self.layers,self.qheads,self.kvheads,self.hd,self.inner,self.heads,self.state_size,self.groups,self.routed_width,self.shared_width)<=0:
            raise ValueError('model dimensions must be positive')
        if not 1<=self.topk<=self.experts or not np.isfinite(self.scale) or not np.isfinite(self.eps) or self.eps<=0:
            raise ValueError('invalid expert or normalization metadata')
        if any(v not in (0,self.kvheads) for v in kvh) or any(v not in (0,self.routed_width) for v in ff):
            raise ValueError('heterogeneous attention/expert widths require another adapter')
        self.head_dim=self.inner//self.heads
        self.conv_dim=self.inner+2*self.groups*self.state_size
        self.vocab=len(kv['tokenizer.ggml.tokens'])
        # Conservative checkpoint contract: do not silently run a different Nemotron variant.
        if self.inner%self.heads or self.heads%self.groups or self.inner%self.groups or self.conv<2:
            raise ValueError('invalid Mamba dimensions')
        if self.hd != int(get('attention.value_length')) or self.hd>256 or self.qheads%self.kvheads:
            raise ValueError('unsupported attention dimensions')
        if not get('expert_weights_norm') or int(get('expert_group_count'))!=1 or int(get('expert_group_used_count'))!=1 or int(get('expert_shared_count'))!=1:
            raise ValueError('unsupported router policy')
        # Community GGUF advertises extrapolated 1M; original Nota checkpoint specifies 256K.
        self.max_context=min(int(get('context_length')),262144)


class Engine(object):
    backend='kindi_lightning'
    state_version=4
    def __init__(self, path, ctx=8192, log=print,kv_dtype=None,kv_read=None):
        self.file=PackedFile(path);self.spec=s=ModelSpec(self.file.gguf)
        self.ctx=int(ctx)
        if not 1<=self.ctx<=s.max_context:raise ValueError('invalid context capacity')
        self._validate_tensors()
        self.head_layout=os.environ.get('KINDI_LIGHTNING_HEAD_LAYOUT','native')
        self._validate_head_layout()
        self.q2_input=os.environ.get('KINDI_LIGHTNING_Q2_INPUT','scalar')
        if self.q2_input not in ('scalar','vec4'):raise ValueError('Q2 input mode must be scalar or vec4')
        from .kv import validate_gqa_config
        validate_gqa_config(os.environ.get('KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY','legacy'),
            kv_read or os.environ.get('KINDI_LIGHTNING_KV_READ','auto'),
            os.environ.get('KINDI_LIGHTNING_ATTENTION','tiled'),
            os.environ.get('KINDI_LIGHTNING_KERNELS','optimized'),(s.qheads,s.kvheads,s.hd),
            kv_dtype=kv_dtype or os.environ.get('KINDI_LIGHTNING_KV','fp32'),ctx=self.ctx)
        self.ops=CudaOps();self.cp=cp=self.ops.cp
        self.q2_decode=self.ops.q2_decode
        if self.q2_decode=='or' and (self.ops.profile!='optimized' or self.q2_input!='vec4'):
            raise ValueError('OR Q2 decode requires optimized vec4 input')
        self.ssm_strategy=self.ops.ssm_strategy
        if self.ssm_strategy!='legacy' and (s.heads,s.head_dim,s.state_size,s.groups)!=(64,64,128,8):
            raise ValueError('Kepler SSM specialization requires Lightning geometry 64/64/128/8')
        if self.q2_input=='vec4' and self.ops.profile!='optimized':raise ValueError('vec4 Q2 input requires optimized Lightning kernels')
        self.ops.q2_input=self.q2_input
        self.kv_dtype=kv_dtype or os.environ.get('KINDI_LIGHTNING_KV','fp32')
        from .kv import KVCacheOps
        self.kv=KVCacheOps(self.ops,self.kv_dtype,kv_read)
        if self.kv.tile_strategy!='legacy' and (s.qheads,s.kvheads,s.hd)!=(32,2,128):
            raise ValueError('Kepler tile specialization requires the pinned Lightning geometry 32/2/128')
        self.kv_read=self.kv.read_mode
        if self.kv_dtype!='fp32' and (self.ops.profile!='optimized' or s.hd>128):raise ValueError('compressed KV requires optimized attention with dim<=128')
        self.memory=self.memory_plan(self.ctx)
        free,total=cp.cuda.runtime.memGetInfo()
        if self.memory['allocation_peak_with_reserve']>free:
            raise MemoryError('memory plan requires %d bytes, free %d' % (self.memory['allocation_peak_with_reserve'],free))
        self.file.load_gpu(cp,reserve_bytes=self.memory['allocation_peak_with_reserve']-self.memory['weights_packed'])
        self.w=self.file.resident
        self.head_codes=None;self.head_scales=None
        if self.head_layout in ('split','split_lane4'):self._prepare_split_head()
        self.states={};self.histories={};self.keys={};self.values={}
        for i,kind in enumerate(s.blocks):
            if kind=='mamba':
                self.states[i]=cp.zeros((s.heads,s.head_dim,s.state_size),cp.float32)
                self.histories[i]=cp.zeros((s.conv_dim,s.conv-1),cp.float32)
            elif kind=='attention':
                self.keys[i]=self.kv.allocate(self.ctx,s.kvheads,s.hd)
                self.values[i]=cp.empty_like(self.keys[i])
        self.position=0;self.last_logits=None;self.tokens=[];self.failed=False
        self.source_sha256=self.file.source_hash()
        properties=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        self.numeric_profile={'kernels':self.ops.profile,'q2_input':self.q2_input,'q2_decode':self.q2_decode,'q2_prefill':self.ops.prefill_q2,'attention_kv_load':self.ops.attention_kv_load,'head_layout':self.head_layout,'dense_prefill':os.environ.get('KINDI_LIGHTNING_DENSE_PREFILL','batched_mv'),
            'expert_prefill':os.environ.get('KINDI_LIGHTNING_EXPERT_PREFILL','grouped_mv'),'state':'fp32','kv':self.kv_dtype,'kv_read':self.kv_read,'attention':self.kv.attention_mode,'attention_tile_strategy':self.kv.tile_strategy,'position':'nope',
            'dt':'softplus_no_extra_clamp','ssm_strategy':self.ssm_strategy,'state_version':self.state_version,'kernel_hash':self.ops.code_hash,
            'compute_capability':[properties['major'],properties['minor']],'cuda_runtime':cp.cuda.runtime.runtimeGetVersion()}
        source=hashlib.sha256()
        for name in ('lightning.py','lightning_batch.py','lightning_tokenizer.py','packed.py','moe.py','kv.py'):
            with open(os.path.join(os.path.dirname(__file__),name),'rb') as f:source.update(f.read())
        self.numeric_profile['adapter_hash']=source.hexdigest()
        self.numeric_profile['attention_workspace_bytes']=self.memory['attention_scratch']
        if self.kv.gqa_stage:
            self.numeric_profile.update(attention_gqa_stage_keys=self.kv.gqa_stage,
                attention_gqa_shared_bytes=self.kv.gqa_shared_bytes,attention_gqa_block_threads=512)
        if self.kv.tile_strategy=='gqa16_vec2_32_bank4':
            self.numeric_profile.update(attention_shared_bank_bytes_requested=4,
                attention_shared_bank_scope='per_function_tiles_and_merge',attention_strategy_context_limit=65536)
        self.fingerprint=hashlib.sha256((self.source_sha256+json.dumps(self.numeric_profile,sort_keys=True)).encode()).hexdigest()
        log('Lightning CUDA loaded: %.3f GiB weights; context %d; FP32 state, %s KV' % (self.memory['weights']/2**30,self.ctx,self.kv_dtype))

    def _validate_tensors(self):
        s=self.spec;t=self.file.tensors
        def require(name,shape,kind=None):
            if name not in t or t[name].shape!=tuple(shape):raise ValueError('missing/incorrect tensor: '+name)
            if kind is not None and t[name].kind!=kind:raise ValueError('wrong tensor format: '+name)
        require('token_embd.weight',(s.vocab,s.width),8)
        require('output.weight',(s.vocab,s.width),8)
        require('output_norm.weight',(s.width,),0)
        for i,kind in enumerate(s.blocks):
            p='blk.%d.'%i;require(p+'attn_norm.weight',(s.width,),0)
            if kind=='mamba':
                for name,shape in [('ssm_a',(s.heads,1)),('ssm_d',(s.heads,1)),
                    ('ssm_dt.bias',(s.heads,)),('ssm_conv1d.bias',(s.conv_dim,)),
                    ('ssm_conv1d.weight',(s.conv_dim,s.conv)),('ssm_norm.weight',(s.groups,s.inner//s.groups))]:require(p+name,shape,0)
                require(p+'ssm_in.weight',(s.inner+s.conv_dim+s.heads,s.width),42)
                require(p+'ssm_out.weight',(s.width,s.inner),42)
            elif kind=='moe':
                require(p+'ffn_gate_inp.weight',(s.experts,s.width),0)
                require(p+'exp_probs_b.bias',(s.experts,),0)
                for suffix in ('exps','shexp'):
                    up=t.get(p+'ffn_up_'+suffix+'.weight');down=t.get(p+'ffn_down_'+suffix+'.weight')
                    if up is None or down is None or up.kind!=42 or down.kind!=42:raise ValueError('missing expert matrices')
                    hidden=s.routed_width if suffix=='exps' else s.shared_width
                    expected=(s.experts,hidden,s.width) if suffix=='exps' else (hidden,s.width)
                    require(up.name,expected,42)
                    require(down.name,((s.experts,s.width,up.shape[-2]) if suffix=='exps' else (s.width,up.shape[-2])),42)
            else:
                for name,n in [('attn_q',s.qheads),('attn_k',s.kvheads),('attn_v',s.kvheads)]:require(p+name+'.weight',(n*s.hd,s.width),42)
                require(p+'attn_output.weight',(s.width,s.qheads*s.hd),42)

    def _validate_head_layout(self):
        if self.head_layout=='split_lane4' and not 1<=self.ctx<=65536:
            raise ValueError('K6000 lane4 research is limited to ctx<=65536')
        if self.head_layout not in ('native','split','split_lane4'):
            raise ValueError('head layout must be native, split or split_lane4')
        if self.head_layout in ('split','split_lane4'):
            t=self.file.tensors['output.weight']
            if (os.environ.get('KINDI_LIGHTNING_KERNELS','optimized')!='optimized'
                    or (self.spec.vocab,self.spec.width)!=(131072,2688)
                    or t.shape!=(131072,2688) or t.kind!=8 or t.nbytes!=374341632):
                raise ValueError('split head requires optimized Lightning Q8 output 131072x2688')

    def _prepare_split_head(self):
        # Startup only: the packed loader has returned, and no state/KV/workspace
        # is resident yet. self.w and file.resident are the SAME owner dictionary.
        if (self.head_layout not in ('split','split_lane4') or hasattr(self,'states') or self.w is not self.file.resident
                or self.head_codes is not None or self.head_scales is not None):
            raise RuntimeError('split head conversion is startup-only')
        cp=self.cp;packed=self.w['output.weight'];t=self.file.tensors['output.weight']
        start=packed.data.ptr;end=start+packed.nbytes
        if any(name!='output.weight' and a.data.ptr<end and a.data.ptr+a.nbytes>start
                for name,a in self.w.items()):
            raise ValueError('packed output aliases another resident tensor')
        free,total=cp.cuda.runtime.memGetInfo()
        if free<self.memory['head_split_bytes']+self.memory['reserve']:
            raise MemoryError('insufficient startup memory for split head conversion')
        if self.head_layout=='split_lane4':
            if (packed.dtype!=cp.uint8 or not packed.flags.c_contiguous or packed.nbytes!=374341632
                    or packed.data.ptr%2):raise ValueError('invalid native Q8 input for direct lane4 conversion')
            codes=cp.empty((131072,21,32,4),cp.int8)
            scales=cp.empty((131072,84),cp.float32)
            if codes.data.ptr%4 or scales.data.ptr%16:raise ValueError('unaligned direct lane4 output')
            self.ops.k['head_q8_lane4_split'](((t.elements//4+255)//256,),(256,),
                (packed,codes,scales,np.int32(131072),np.int32(2688)))
        else:
            codes=cp.empty(t.shape,cp.int8);scales=cp.empty((t.shape[0],t.shape[1]//32),cp.float32)
            self.ops.k['head_q8_split'](((t.elements+255)//256,),(256,),
                (packed,codes,scales,np.int32(t.elements//32)))
        cp.cuda.get_current_stream().synchronize()
        self.head_codes=codes;self.head_scales=scales
        packed_bytes=packed.nbytes
        del self.w['output.weight']
        del packed
        if self.head_layout=='split_lane4':
            self.head_loader_diagnostics={'layout':'split_lane4','conversion':'direct_native_q8_0',
                'native_packed_bytes':packed_bytes,'resident_codes_bytes':codes.nbytes,
                'resident_scales_bytes':scales.nbytes,'resident_head_bytes':codes.nbytes+scales.nbytes,
                'codes_shape':list(codes.shape),'scales_shape':list(scales.shape),
                'old_split_codes_allocated':False,'packed_output_retired':True,
                'packed_output_in_resident':'output.weight' in self.w,
                'resident_dictionary_shared':self.w is self.file.resident,
                'startup_conversion_synchronized':True,'input_workspace_bytes':10752,
                'input_workspace_lifetime':'per_call_stream_ordered','input_workspace_persistent':False}

        # Retire the native357MiB owner BEFORE allocating KV. An allocator may
        # keep freed pool capacity for reuse; no packed output view is retained
        # by Engine, PackedFile or CudaOps.
        cp.get_default_memory_pool().free_all_blocks()

    def memory_plan(self,ctx):
        s=self.spec
        packed_weights=sum(t.nbytes for t in self.file.tensors.values())
        split=getattr(self,'head_layout','native') in ('split','split_lane4')
        head_input=10752 if getattr(self,'head_layout','native')=='split_lane4' else 0
        head_packed=self.file.tensors['output.weight'].nbytes if split else 0
        head_split=131072*2688+131072*84*4 if split else 0
        weights=packed_weights-head_packed+head_split
        state=s.blocks.count('mamba')*(s.heads*s.head_dim*s.state_size+s.conv_dim*(s.conv-1))*4 if 'mamba' in s.blocks else 0
        dtype=getattr(self,'kv_dtype','fp32')
        if dtype=='q8_0' and s.hd%32:raise ValueError('Q8_0 KV requires head dimension divisible by 32')
        row_bytes=s.kvheads*((s.hd//32)*34 if dtype=='q8_0' else s.hd*(2 if dtype=='fp16' else 4))
        kv=s.blocks.count('attention')*2*ctx*row_bytes
        scratch=2*ctx*s.kvheads*s.hd*4 if dtype!='fp32' and getattr(self,'kv_read','auto')!='direct' else 0
        attention_scratch=self.kv.split_workspace_bytes if hasattr(self,'kv') and self.kv.attention_mode=='split' else 0
        # Prefill accepts at most256 tokens/chunk; one reusable layer-sized
        # parameter buffer holds delta and decay, never a copy per layer.
        ssm_scratch=2*min(ctx,256)*s.heads*4 if getattr(self,'ssm_strategy','legacy') in ('auto','precomputed') else 0
        reserve=512*1024**2
        steady=weights+state+kv+scratch+attention_scratch+ssm_scratch+head_input+reserve
        startup=packed_weights+head_split+reserve
        plan={'weights':weights,'weights_packed':packed_weights,'head_packed_retired_bytes':head_packed,
            'head_split_bytes':head_split,'head_steady_extra_bytes':head_split-head_packed,
            'head_startup_transient_bytes':head_split,'startup_with_reserve':startup,
            'allocation_peak_with_reserve':max(steady,startup),'state':state,'kv':kv,'kv_scratch':scratch,
            'attention_scratch':attention_scratch,'ssm_scratch':ssm_scratch,'reserve':reserve,'total_with_reserve':steady}
        if head_input:plan['head_input_workspace_bytes']=head_input
        if hasattr(self,'kv') and self.kv.gqa_stage:
            # Static per-CTA shared memory is resident on an SM, not global VRAM.
            plan['attention_static_shared_per_block']=self.kv.gqa_shared_bytes
            plan['attention_block_threads']=512
        return plan

    def linear(self,name,x,ids=None,per_input=False):
        t=self.file.tensors[name]
        n=t.shape[-2];k=t.shape[-1]
        if name=='output.weight' and getattr(self,'head_layout','native') in ('split','split_lane4'):
            if ids is not None or per_input:raise ValueError('split head requires one unselected input row')
            if self.head_layout=='split_lane4':
                return self.ops.split_head_lane4_linear(self.head_codes,self.head_scales,x)
            return self.ops.split_head_linear(self.head_codes,self.head_scales,x)
        selected=int(ids.size) if ids is not None else 1
        return self.ops.linear(self.w[name],x,n,k,t.kind,ids,selected,per_input,decode=True)

    def kv_ops(self):
        if not hasattr(self,'kv'):
            from .kv import KVCacheOps
            self.kv=KVCacheOps(self.ops,getattr(self,'kv_dtype','fp32'))
        return self.kv

    def store_kv(self,layer,position,key,value):
        s=self.spec;kv=self.kv_ops()
        kv.store(self.keys[layer],key,position,s.kvheads,s.hd)
        kv.store(self.values[layer],value,position,s.kvheads,s.hd)

    def block(self,i,x):
        s=self.spec;cp=self.cp;p='blk.%d.'%i;kind=s.blocks[i]
        z=self.ops.norm(x,self.w[p+'attn_norm.weight'],s.eps)
        if kind=='moe':
            logits=self.linear(p+'ffn_gate_inp.weight',z)
            ids,weights=self.ops.router(logits,self.w[p+'exp_probs_b.bias'],s.topk,s.scale)
            up=self.ops.activate(self.linear(p+'ffn_up_exps.weight',z,ids))
            down=self.linear(p+'ffn_down_exps.weight',up,ids,True)
            shared=self.linear(p+'ffn_down_shexp.weight',self.ops.activate(self.linear(p+'ffn_up_shexp.weight',z)))
            y=self.ops.combine(down,weights,shared)
        elif kind=='mamba':
            projected=self.linear(p+'ssm_in.weight',z)
            conv=cp.empty(s.conv_dim,cp.float32)
            self.ops.k['causal_conv'](((s.conv_dim+255)//256,),(256,),
                (projected[s.inner:s.inner+s.conv_dim],self.w[p+'ssm_conv1d.weight'],self.w[p+'ssm_conv1d.bias'],
                 self.histories[i],conv,np.int32(s.conv_dim),np.int32(s.conv)))
            y=cp.empty(s.inner,cp.float32)
            if self.ssm_strategy=='precomputed':
                parameters=self.ops.ssm_parameters(projected[s.inner+s.conv_dim:],self.w[p+'ssm_dt.bias'],self.w[p+'ssm_a'],1,s.heads,s.heads)
                self.ops.k['v5_ssm_step_precomputed']((s.inner,),(128,),
                    (conv,parameters,self.w[p+'ssm_d'],self.states[i],y,
                     np.int32(s.heads),np.int32(s.head_dim),np.int32(s.state_size),np.int32(s.groups)))
            else:
                kernel='ssm_step' if self.ssm_strategy=='legacy' else 'v5_ssm_step_shared'
                self.ops.k[kernel]((s.inner,),(128,),
                    (conv,projected[s.inner+s.conv_dim:],self.w[p+'ssm_dt.bias'],self.w[p+'ssm_a'],self.w[p+'ssm_d'],
                     self.states[i],y,np.int32(s.heads),np.int32(s.head_dim),np.int32(s.state_size),np.int32(s.groups)))
            y=self.ops.norm(y,self.w[p+'ssm_norm.weight'].reshape(-1),s.eps,s.inner//s.groups,projected[:s.inner])
            y=self.linear(p+'ssm_out.weight',y)
        else:
            q=self.linear(p+'attn_q.weight',z)
            k=self.linear(p+'attn_k.weight',z);v=self.linear(p+'attn_v.weight',z)
            self.store_kv(i,self.position,k,v)
            y=cp.empty(s.qheads*s.hd,cp.float32)
            if self.ops.profile=='optimized' and s.hd<=128:
                y=self.kv_ops().attention(q,self.keys[i],self.values[i],self.position,1,s.qheads,s.kvheads,s.hd)
            else:
                if self.kv_ops().mode!='fp32':raise ValueError('compressed KV requires optimized attention with dim<=128')
                self.ops.k['attention']((s.qheads,),(128 if s.hd<=128 else 256,),
                    (q,self.keys[i],self.values[i],y,np.int32(self.position+1),np.int32(s.qheads),np.int32(s.kvheads),np.int32(s.hd)))
            y=self.linear(p+'attn_output.weight',y)
        self.ops.add(x,y)
        return x

    def forward(self,token,logits=True,trace=None):
        if getattr(self,'failed',False):raise RuntimeError('CUDA context failed; restart this backend process')
        token=int(token);s=self.spec;cp=self.cp
        if not 0<=token<s.vocab:raise ValueError('token out of range')
        if self.position>=self.ctx:raise OverflowError('Lightning context full')
        x=cp.empty(s.width,cp.float32);t=self.file.tensors['token_embd.weight']
        self.ops.k['embedding'](((s.width+255)//256,),(256,),(self.w[t.name],x,np.int32(token),np.int32(s.width),np.int32(t.kind)))
        try:
            for i in range(s.layers):
                x=self.block(i,x)
                if trace is not None:trace(i,x,self)
            result=self.linear('output.weight',self.ops.norm(x,self.w['output_norm.weight'],s.eps)) if logits else None
            if logits:self.kv_ops().check()
            self.position+=1;self.tokens.append(token);self.last_logits=result
            return result
        except Exception:
            self.invalidate() # keep the original error if the CUDA context is no longer usable
            raise

    def prefill(self,tokens,progress=None,cancel=None,chunk_size=0):
        if getattr(self,'failed',False):raise RuntimeError('CUDA context failed; restart this backend process')
        tokens=list(tokens)
        if not tokens:raise ValueError('empty token sequence')
        if self.position+len(tokens)>self.ctx:raise OverflowError('Lightning context full')
        if chunk_size:
            if not 1<=chunk_size<=256:raise ValueError('chunk size must be in [1,256]')
            from .lightning_batch import BatchOps
            batch=BatchOps(self)
            for start in range(0,len(tokens),chunk_size):
                if cancel is not None and cancel():raise InterruptedError('prefill cancelled at chunk boundary')
                end=min(len(tokens),start+chunk_size)
                result=batch.forward(tokens[start:end],last_logits=(end==len(tokens)))
                # Bound the driver launch queue even in offline benchmarks and long-context tools.
                self.synchronize()
                if progress is not None:progress(end,len(tokens))
            return result
        result=None
        for i,token in enumerate(tokens):
            if cancel is not None and cancel():raise InterruptedError('prefill cancelled at token boundary')
            result=self.forward(token,logits=(i==len(tokens)-1))
            if cancel is not None or (i+1)%32==0 or i+1==len(tokens):self.synchronize()
            if progress is not None:progress(i+1,len(tokens))
        return result

    def reset(self):
        if getattr(self,'failed',False):raise RuntimeError('CUDA context failed; restart this backend process')
        for a in self.states.values():a.fill(0)
        for a in self.histories.values():a.fill(0)
        self.position=0;self.tokens=[];self.last_logits=None
        if hasattr(self,'kv'):self.kv.reset_error()

    def invalidate(self):
        try:
            self.reset()
            self.cp.cuda.Stream.null.synchronize()
        except Exception:
            self.failed=True;self.position=0;self.tokens=[];self.last_logits=None

    def synchronize(self):
        try:
            self.cp.cuda.Stream.null.synchronize();self.kv_ops().check()
        except Exception:
            self.invalidate();raise

    def snapshot(self):
        if getattr(self,'failed',False):raise RuntimeError('CUDA context failed; restart this backend process')
        self.kv_ops().check()
        n=self.position
        return {'version':self.state_version,'fingerprint':self.fingerprint,'tokens':list(self.tokens),'position':n,
            'states':{i:x.get() for i,x in self.states.items()},'histories':{i:x.get() for i,x in self.histories.items()},
            'keys':{i:x[:n].get() for i,x in self.keys.items()},'values':{i:x[:n].get() for i,x in self.values.items()},
            'logits':self.last_logits.get() if self.last_logits is not None else None}

    def restore(self,snapshot):
        if getattr(self,'failed',False):raise RuntimeError('CUDA context failed; restart this backend process')
        required=('version','fingerprint','position','tokens','logits','states','histories','keys','values')
        if not isinstance(snapshot,dict) or any(k not in snapshot for k in required):raise ValueError('incomplete snapshot')
        if snapshot['version']!=self.state_version or snapshot['fingerprint']!=self.fingerprint:
            raise ValueError('incompatible snapshot')
        n=snapshot['position']
        if not isinstance(n,(int,np.integer)) or isinstance(n,bool) or not isinstance(snapshot['tokens'],list):raise ValueError('invalid snapshot position/tokens')
        if n<0 or n>self.ctx or n!=len(snapshot['tokens']):raise ValueError('invalid snapshot position')
        if any(not isinstance(t,(int,np.integer)) or not 0<=t<self.spec.vocab for t in snapshot['tokens']):raise ValueError('invalid snapshot token')
        if snapshot['logits'] is not None and (not isinstance(snapshot['logits'],np.ndarray) or snapshot['logits'].shape!=(self.spec.vocab,) or snapshot['logits'].dtype!=np.float32):
            raise ValueError('invalid snapshot logits')
        for kind in ('states','histories','keys','values'):
            dst=getattr(self,kind)
            if not isinstance(snapshot[kind],dict) or set(dst)!=set(snapshot[kind]):raise ValueError('snapshot layer mismatch')
            for i,array in snapshot[kind].items():
                target=dst[i][:n] if kind in ('keys','values') else dst[i]
                if not isinstance(array,np.ndarray) or target.shape!=array.shape or array.dtype!=target.dtype:raise ValueError('invalid snapshot array')
        for kind in ('states','histories','keys','values'):
            for i,array in snapshot[kind].items():
                target=getattr(self,kind)[i][:n] if kind in ('keys','values') else getattr(self,kind)[i]
                target.set(array)
        self.position=n;self.tokens=list(snapshot['tokens'])
        self.last_logits=self.cp.asarray(snapshot['logits']) if snapshot['logits'] is not None else None
