"""Explicit KV storage contracts; arithmetic remains FP32 on every GPU."""
import os
import numpy as np

MODES=('fp32','fp16','q8_0')
GQA_VECTOR2='gqa16_vec2_32_bank4'
GQA_TILE_STAGES={'gqa16_32':32,'gqa16_64':64,GQA_VECTOR2:32}

def validate_gqa_config(strategy,read_mode,attention_mode,profile,geometry=None,kv_dtype=None,ctx=None):
    """Validate fixed-model GQA before CUDA/weight allocation in Engine."""
    stage=GQA_TILE_STAGES.get(strategy,0)
    if stage:
        if profile!='optimized' or attention_mode!='split':
            raise ValueError('GQA16 attention requires optimized split attention')
        if read_mode!='direct':raise ValueError('GQA16 attention requires explicit direct KV reads')
        if geometry is not None and tuple(geometry)!=(32,2,128):
            raise ValueError('GQA16 attention requires Lightning geometry 32/2/128')
        if strategy==GQA_VECTOR2:
            if kv_dtype!='fp16':raise ValueError('Vector2 bank4 requires FP16 KV')
            if ctx is not None and not 1<=ctx<=65536:
                raise ValueError('Vector2 bank4 requires context capacity <=65536')
    return stage

class KVCacheOps(object):
    def __init__(self,ops,mode,read_mode=None):
        if mode not in MODES:raise ValueError('KV format must be fp32, fp16 or q8_0')
        self.ops=ops;self.cp=ops.cp;self.mode=mode
        self.read_mode=read_mode or os.environ.get('KINDI_LIGHTNING_KV_READ','auto')
        if self.read_mode not in ('direct','expanded','auto'):raise ValueError('KV read mode must be direct, expanded or auto')
        self.attention_mode=os.environ.get('KINDI_LIGHTNING_ATTENTION','tiled')
        if self.attention_mode not in ('tiled','warp','split'):raise ValueError('attention mode must be tiled, warp or split')
        self.tile_strategy=os.environ.get('KINDI_LIGHTNING_ATTENTION_TILE_STRATEGY','legacy')
        if self.tile_strategy not in ('legacy','sub8','quad','auto','gqa16_32','gqa16_64',GQA_VECTOR2):
            raise ValueError('unknown attention tile strategy')
        if self.tile_strategy!='legacy' and (self.attention_mode!='split' or ops.profile!='optimized'):
            raise ValueError('Kepler attention tile strategies require optimized split attention')
        self.gqa_stage=validate_gqa_config(self.tile_strategy,self.read_mode,self.attention_mode,ops.profile,kv_dtype=self.mode)
        self.gqa_shared_bytes=(self.gqa_stage*128+16*128)*4 if self.gqa_stage else 0
        self.split_workspace_bytes=int(os.environ.get('KINDI_LIGHTNING_ATTENTION_WORKSPACE_MB','128'))*1024**2
        if not 1024**2<=self.split_workspace_bytes<=512*1024**2:raise ValueError('attention workspace must be 1..512 MiB')
        if self.ops.attention_kv_load in ('load2','load4') and (self.mode,self.read_mode,self.attention_mode,self.tile_strategy)!=('fp16','direct','split',GQA_VECTOR2):
            raise ValueError('FP16 vector loads require direct split gqa16_vec2_32_bank4')
        self.error=self.cp.zeros(1,self.cp.int32)
        self.capacity=0;self.scratch=None;self.split_scratch=None
        self.bank_requests=ops.configure_attention_bank4() if self.tile_strategy==GQA_VECTOR2 else ()

    def row_bytes(self,heads,dim):
        if self.mode=='q8_0':
            if dim%32:raise ValueError('Q8_0 KV requires head dimension divisible by 32')
            return heads*(dim//32)*34
        return heads*dim*(4 if self.mode=='fp32' else 2)

    def allocate(self,ctx,heads,dim):
        if min(ctx,heads,dim)<1:raise ValueError('invalid KV dimensions')
        if self.tile_strategy==GQA_VECTOR2 and ctx>65536:
            raise ValueError('Vector2 bank4 requires context capacity <=65536')
        self.capacity=max(ctx,self.capacity)
        self.row_bytes(heads,dim)
        shape=(ctx,heads,(dim//32)*34 if self.mode=='q8_0' else dim)
        dtype={'fp32':self.cp.float32,'fp16':self.cp.float16,'q8_0':self.cp.uint8}[self.mode]
        return self.cp.empty(shape,dtype)

    def store(self,dst,src,position,heads,dim):
        self.validate_storage(dst,heads,dim)
        if not isinstance(src,self.cp.ndarray):raise ValueError('KV source must be a GPU array')
        count=src.size;tokens=count//(heads*dim)
        if tokens<1 or count!=tokens*heads*dim or position<0 or position+tokens>dst.shape[0]:raise ValueError('invalid KV write range')
        if src.dtype!=self.cp.float32 or not src.flags.c_contiguous:raise ValueError('KV source must be contiguous FP32')
        target=dst[position:position+tokens]
        if self.mode=='fp32':target[:]=src.reshape(tokens,heads,dim)
        elif self.mode=='fp16':
            self.ops.k['kv_store_fp16'](((count+255)//256,),(256,),(src,target,np.int32(count),self.error))
        else:
            self.ops.k['kv_store_q8']((((count//32)+3)//4,),(128,),(src,target,np.int32(count//32),self.error))

    def attention(self,q,k,v,past,tokens,qheads,kvheads,dim):
        self.validate_storage(k,kvheads,dim);self.validate_storage(v,kvheads,dim)
        alignment=8 if self.ops.attention_kv_load=='load4' else 4
        if self.ops.attention_kv_load in ('load2','load4') and (k.data.ptr%alignment or v.data.ptr%alignment):
            raise ValueError('FP16 '+self.ops.attention_kv_load+' requires'+str(alignment)+'-byte-aligned K and V')
        if min(tokens,qheads,kvheads,dim)<1 or dim>128 or qheads%kvheads or past<0 or past+tokens>min(k.shape[0],v.shape[0]):raise ValueError('invalid attention dimensions or range')
        if self.tile_strategy==GQA_VECTOR2 and max(k.shape[0],v.shape[0],past+tokens)>65536:
            raise ValueError('Vector2 bank4 requires context capacity <=65536')
        if self.tile_strategy!='legacy' and (qheads,kvheads,dim)!=(32,2,128):
            raise ValueError('Kepler attention tiles require the pinned Lightning geometry 32/2/128')
        if not isinstance(q,self.cp.ndarray) or q.dtype!=self.cp.float32 or not q.flags.c_contiguous or q.size!=tokens*qheads*dim:raise ValueError('invalid attention query')
        out=self.cp.empty((tokens,qheads*dim),self.cp.float32)
        name={'fp32':'attention_tiled','fp16':'attention_fp16','q8_0':'attention_q8'}[self.mode]
        length=past+tokens
        expand_auto=(self.mode=='q8_0' and (tokens>=8 or length>=4096)) or (self.mode=='fp16' and self.attention_mode=='split' and tokens>=8 and length>=512)
        if self.mode!='fp32' and (self.read_mode=='expanded' or (self.read_mode=='auto' and expand_auto)):
            # One reusable layer-sized scratch, not an expanded copy of every layer's KV.
            self.capacity=max(self.capacity,k.shape[0])
            if self.scratch is None or self.scratch[0].shape[0]<self.capacity or self.scratch[0].shape[1:]!=(kvheads,dim):
                self.scratch=(self.cp.empty((self.capacity,kvheads,dim),self.cp.float32),self.cp.empty((self.capacity,kvheads,dim),self.cp.float32))
            count=length*kvheads*dim;decode=self.ops.k['kv_decode_fp16' if self.mode=='fp16' else 'kv_decode_q8']
            decode(((count+255)//256,),(256,),(k,self.scratch[0],np.int32(count)))
            decode(((count+255)//256,),(256,),(v,self.scratch[1],np.int32(count)))
            k,v=self.scratch;name='attention_tiled'
        if self.attention_mode=='split' and length>=512:
            if self.split_scratch is None:self.split_scratch=self.cp.empty(self.split_workspace_bytes//4,self.cp.float32)
            tiles=(length+127)//128;per_query=qheads*tiles*(dim+2)
            batch=min(tokens,self.split_scratch.size//per_query)
            if batch<1:raise MemoryError('attention tile workspace too small for this geometry')
            dtype='fp32' if name=='attention_tiled' else ('fp16' if self.mode=='fp16' else 'q8')
            query=q.reshape(tokens,qheads*dim)
            for start in range(0,tokens,batch):
                count=min(batch,tokens-start);position=past+start
                strategy=self.tile_strategy
                if strategy=='auto':
                    # Use actual subbatch size: at128K a128MiB workspace fits
                    # only7 queries. Pairing is useful when enough work is
                    # available, whereas sub8 avoids its partial-pair overhead.
                    strategy='quad' if ((dtype=='q8' and count>=2) or (dtype=='fp32' and count>=8)) else 'sub8'
                kernel='attention_tiles_'+dtype
                grid=(qheads,count,tiles)
                block=(128,)
                if self.gqa_stage:
                    kernel+=('_gqa16_vector2_typed32' if strategy==GQA_VECTOR2 else '_gqa16_typed'+str(self.gqa_stage))
                    grid=(kvheads,count,tiles);block=(512,)
                elif strategy=='quad':
                    kernel+='_quad';grid=(16,(count+1)//2,tiles)
                elif strategy=='sub8':kernel+='_sub8_fixed'
                self.ops.k[kernel](grid,block,
                    (query[start:start+count],k,v,self.split_scratch,np.int32(count),np.int32(position),np.int32(qheads),np.int32(kvheads),np.int32(dim),np.int32(tiles)))
                self.ops.k['attention_merge_tiles']((qheads,count),(128,),
                    (self.split_scratch,out[start:start+count],np.int32(count),np.int32(position),np.int32(qheads),np.int32(dim),np.int32(tiles)))
            return out
        if self.attention_mode in ('warp','split'):name='attention_warp' if name=='attention_tiled' else name+'_warp'
        self.ops.k[name]((qheads,tokens),(128,),(q,k,v,out,np.int32(tokens),np.int32(past),np.int32(qheads),np.int32(kvheads),np.int32(dim)))
        return out

    def validate_storage(self,array,heads,dim):
        if min(heads,dim)<1:raise ValueError('invalid KV geometry')
        self.row_bytes(heads,dim)
        tail=(heads,(dim//32)*34 if self.mode=='q8_0' else dim)
        dtype={'fp32':self.cp.float32,'fp16':self.cp.float16,'q8_0':self.cp.uint8}[self.mode]
        if not isinstance(array,self.cp.ndarray) or array.ndim!=3 or array.shape[1:]!=tail or array.dtype!=dtype or not array.flags.c_contiguous:raise ValueError('invalid KV storage layout')

    def check(self):
        if self.mode!='fp32' and int(self.error.get()[0]):raise FloatingPointError('KV storage rejected non-finite or out-of-range values')

    def reset_error(self):self.error.fill(0)
