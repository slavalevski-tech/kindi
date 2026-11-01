// Storage conversion only. Kepler performs attention arithmetic in FP32.
extern "C" __global__ void kv_store_fp16(const float* x,unsigned short* out,int n,int* error) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float value=x[i];
    if(!isfinite(value) || fabsf(value)>65504.f){atomicOr(error,1);value=0;}
    out[i]=__half_as_ushort(__float2half_rn(value));
}
extern "C" __global__ void kv_store_q8(const float* x,unsigned char* out,int blocks,int* error) {
    int lane=threadIdx.x&31,block=blockIdx.x*4+(threadIdx.x>>5);if(block>=blocks)return;
    float value=x[(size_t)block*32+lane];
    if(!isfinite(value)){atomicOr(error,1);value=0;}
    float maximum=fabsf(value);
    for(int shift=16;shift;shift/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,shift));
    maximum=__shfl_sync(0xffffffffu,maximum,0);
    float scale=maximum/127.f;
    if(scale>65504.f){if(lane==0)atomicOr(error,1);scale=0;}
    unsigned char* dst=out+(size_t)block*34;
    if(lane==0)*reinterpret_cast<unsigned short*>(dst)=__half_as_ushort(__float2half_rn(scale));
    // RNE, bounded to [-127,127]; scales use FP16, codes use the unrounded FP32 scale.
    int code=scale>0?__float2int_rn(value/scale):0;
    reinterpret_cast<signed char*>(dst+2)[lane]=(signed char)max(-127,min(127,code));
}
template<int TYPE> __device__ __forceinline__ float kv_read(const unsigned char* data,size_t index) {
    if(TYPE==0)return __ldg(reinterpret_cast<const float*>(data)+index);
    if(TYPE==1)return __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(data)+index)));
    const unsigned char* p=data+(index/32)*34;
    float scale=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(p))));
    return scale*(float)__ldg(reinterpret_cast<const signed char*>(p+2)+(index%32));
}
extern "C" __global__ void kv_decode_fp16(const unsigned char* src,float* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=kv_read<1>(src,i);
}
extern "C" __global__ void kv_decode_q8(const unsigned char* src,float* out,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=kv_read<8>(src,i);
}
template<int TYPE> __device__ void attention_storage(const float* q,const unsigned char* k,const unsigned char* v,float* out,
    int tokens,int past,int qheads,int kvheads,int dim) {
    int h=blockIdx.x,token=blockIdx.y,t=threadIdx.x,lane=t&31,warp=t>>5;
    if(token>=tokens)return;
    int kv=h/(qheads/kvheads),length=past+token+1;
    const float* query=q+((size_t)token*qheads+h)*dim;
    __shared__ float score[128],reduction[128],prob[128],alpha,beta,total,maxval;
    if(t==0){total=0;maxval=-INF;}__syncthreads();float acc=0;
    for(int base=0;base<length;base+=128) {
        int count=min(128,length-base);
        for(int j=warp;j<count;j+=4) {
            size_t key=((size_t)(base+j)*kvheads+kv)*dim;
            float dot=0;for(int c=lane;c<dim;c+=32)dot=fmaf(query[c],kv_read<TYPE>(k,key+c),dot);
            dot=warp_sum(dot);if(lane==0)score[j]=dot*rsqrtf((float)dim);
        }
        __syncthreads();reduction[t]=t<count?score[t]:-INF;__syncthreads();
        for(int d=64;d;d/=2){if(t<d)reduction[t]=fmaxf(reduction[t],reduction[t+d]);__syncthreads();}
        float tilemax=reduction[0];prob[t]=t<count?expf(score[t]-tilemax):0;__syncthreads();
        reduction[t]=prob[t];__syncthreads();
        for(int d=64;d;d/=2){if(t<d)reduction[t]+=reduction[t+d];__syncthreads();}
        if(t==0){float next=fmaxf(maxval,tilemax);alpha=expf(maxval-next);beta=expf(tilemax-next);total=total*alpha+reduction[0]*beta;maxval=next;}
        __syncthreads();
        if(t<dim){float sum=0;for(int j=0;j<count;j++)sum=fmaf(prob[j],kv_read<TYPE>(v,((size_t)(base+j)*kvheads+kv)*dim+t),sum);acc=acc*alpha+sum*beta;}
        __syncthreads();
    }
    if(t<dim)out[((size_t)token*qheads+h)*dim+t]=acc/total;
}
extern "C" __global__ void attention_fp16(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim) {attention_storage<1>(q,k,v,out,tokens,past,qheads,kvheads,dim);}
extern "C" __global__ void attention_q8(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim) {attention_storage<8>(q,k,v,out,tokens,past,qheads,kvheads,dim);}

// Same 128-element reduction tree and tile recurrence as attention_tiled.
// Shared loads plus warp shuffles replace fourteen block-wide reduction barriers.
template<int TYPE> __device__ void attention_storage_warp(const float* q,const unsigned char* k,const unsigned char* v,float* out,
    int tokens,int past,int qheads,int kvheads,int dim) {
    int h=blockIdx.x,token=blockIdx.y,t=threadIdx.x,lane=t&31,warp=t>>5;
    if(token>=tokens)return;
    int kv=h/(qheads/kvheads),length=past+token+1;
    const float* query=q+((size_t)token*qheads+h)*dim;
    __shared__ float score[128],prob[128],alpha,beta,total,maxval;
    if(t==0){total=0;maxval=-INF;}__syncthreads();float acc=0;
    float q0=lane<dim?query[lane]:0, q1=lane+32<dim?query[lane+32]:0;
    float q2=lane+64<dim?query[lane+64]:0, q3=lane+96<dim?query[lane+96]:0;
    for(int base=0;base<length;base+=128) {
        int count=min(128,length-base);
        for(int j=warp;j<128;j+=4) {
            float dot=0;
            if(j<count) {
                size_t key=((size_t)(base+j)*kvheads+kv)*dim;
                if(lane<dim)dot=fmaf(q0,kv_read<TYPE>(k,key+lane),dot);
                if(lane+32<dim)dot=fmaf(q1,kv_read<TYPE>(k,key+lane+32),dot);
                if(lane+64<dim)dot=fmaf(q2,kv_read<TYPE>(k,key+lane+64),dot);
                if(lane+96<dim)dot=fmaf(q3,kv_read<TYPE>(k,key+lane+96),dot);
                dot=warp_sum(dot);
            }
            if(lane==0)score[j]=j<count?dot*rsqrtf((float)dim):-INF;
        }
        __syncthreads();
        float maximum=fmaxf(fmaxf(score[lane],score[lane+64]),fmaxf(score[lane+32],score[lane+96]));
        for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
        float tilemax=__shfl_sync(0xffffffffu,maximum,0);
        prob[t]=t<count?expf(score[t]-tilemax):0;__syncthreads();
        if(warp==0) {
            float sum=prob[lane]+prob[lane+64];sum+=prob[lane+32]+prob[lane+96];sum=warp_sum(sum);
            if(lane==0){float next=fmaxf(maxval,tilemax);alpha=expf(maxval-next);beta=expf(tilemax-next);total=__fmaf_rn(total,alpha,__fmul_rn(sum,beta));maxval=next;}
        }
        __syncthreads();
        if(t<dim){float sum=0;for(int j=0;j<count;j++)sum=fmaf(prob[j],kv_read<TYPE>(v,((size_t)(base+j)*kvheads+kv)*dim+t),sum);acc=acc*alpha+sum*beta;}
        __syncthreads();
    }
    if(t<dim)out[((size_t)token*qheads+h)*dim+t]=acc/total;
}
extern "C" __global__ void attention_warp(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim) {attention_storage_warp<0>(q,k,v,out,tokens,past,qheads,kvheads,dim);}
extern "C" __global__ void attention_fp16_warp(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim) {attention_storage_warp<1>(q,k,v,out,tokens,past,qheads,kvheads,dim);}
extern "C" __global__ void attention_q8_warp(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim) {attention_storage_warp<8>(q,k,v,out,tokens,past,qheads,kvheads,dim);}

// Compute independent 128-key tiles in parallel; merge in the ORIGINAL tile order.
// Unlike a tree merge across tiles, this retains the online-softmax rounding path.
template<int TYPE> __device__ void attention_tiles(const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads,int kvheads,int dim,int tiles) {
    int h=blockIdx.x,token=blockIdx.y,tile=blockIdx.z,t=threadIdx.x,lane=t&31,warp=t>>5;
    int base=tile*128,length=past+token+1;if(token>=tokens || base>=length)return;
    int kv=h/(qheads/kvheads),count=min(128,length-base);
    const float* query=q+((size_t)token*qheads+h)*dim;
    __shared__ float score[128],prob[128];
    float q0=lane<dim?query[lane]:0,q1=lane+32<dim?query[lane+32]:0;
    float q2=lane+64<dim?query[lane+64]:0,q3=lane+96<dim?query[lane+96]:0;
    for(int j=warp;j<128;j+=4) {
        float dot=0;
        if(j<count) {
            size_t key=((size_t)(base+j)*kvheads+kv)*dim;
            if(lane<dim)dot=fmaf(q0,kv_read<TYPE>(k,key+lane),dot);
            if(lane+32<dim)dot=fmaf(q1,kv_read<TYPE>(k,key+lane+32),dot);
            if(lane+64<dim)dot=fmaf(q2,kv_read<TYPE>(k,key+lane+64),dot);
            if(lane+96<dim)dot=fmaf(q3,kv_read<TYPE>(k,key+lane+96),dot);
            dot=warp_sum(dot);
        }
        if(lane==0)score[j]=j<count?dot*rsqrtf((float)dim):-INF;
    }
    __syncthreads();
    float maximum=fmaxf(fmaxf(score[lane],score[lane+64]),fmaxf(score[lane+32],score[lane+96]));
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
    float tilemax=__shfl_sync(0xffffffffu,maximum,0);
    prob[t]=t<count?expf(score[t]-tilemax):0;__syncthreads();
    size_t offset=(((size_t)token*qheads+h)*tiles+tile)*(dim+2);
    if(warp==0) {
        float sum=prob[lane]+prob[lane+64];sum+=prob[lane+32]+prob[lane+96];sum=warp_sum(sum);
        if(lane==0){parts[offset+dim]=tilemax;parts[offset+dim+1]=sum;}
    }
    if(t<dim){float sum=0;for(int j=0;j<count;j++)sum=fmaf(prob[j],kv_read<TYPE>(v,((size_t)(base+j)*kvheads+kv)*dim+t),sum);parts[offset+t]=sum;}
}
extern "C" __global__ void attention_tiles_fp32(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim,int tiles) {attention_tiles<0>(q,k,v,out,tokens,past,qheads,kvheads,dim,tiles);}
extern "C" __global__ void attention_tiles_fp16(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim,int tiles) {attention_tiles<1>(q,k,v,out,tokens,past,qheads,kvheads,dim,tiles);}
extern "C" __global__ void attention_tiles_q8(const float* q,const unsigned char* k,const unsigned char* v,float* out,int tokens,int past,int qheads,int kvheads,int dim,int tiles) {attention_tiles<8>(q,k,v,out,tokens,past,qheads,kvheads,dim,tiles);}
extern "C" __global__ void attention_merge_tiles(const float* parts,float* out,int tokens,int past,int qheads,int dim,int tiles) {
    int h=blockIdx.x,token=blockIdx.y,t=threadIdx.x,length=past+token+1;
    __shared__ float alpha,beta,total,maxval;
    if(t==0){total=0;maxval=-INF;}__syncthreads();float acc=0;
    for(int tile=0;tile*128<length;tile++) {
        size_t offset=(((size_t)token*qheads+h)*tiles+tile)*(dim+2);
        if(t==0){float tilemax=parts[offset+dim],next=fmaxf(maxval,tilemax);alpha=expf(maxval-next);beta=expf(tilemax-next);
            total=__fmaf_rn(total,alpha,__fmul_rn(parts[offset+dim+1],beta));maxval=next;}
        __syncthreads();
        if(t<dim)acc=__fmaf_rn(acc,alpha,__fmul_rn(parts[offset+t],beta));
        __syncthreads();
    }
    if(t<dim)out[((size_t)token*qheads+h)*dim+t]=acc/total;
}

// V5 pinned derivative: attention_quad.cu SHA256 e6af54e443b9e5033dad07f895167d2f551b4d306f65b85e0b7c554400b92d78
// V5 R&D only: two neighboring queries x two heads in one KV group.
// Same per-output FMA/softmax/reduction/value order as canonical V4.
// Append to canonical CUDA; CUDA10.2/sm35 SIMT, no Engine integration.
template<int TYPE> __device__ void attention_tiles_quad(
    const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads,int kvheads,int dim,int tiles) {
    int group=qheads/kvheads,pairs=(group+1)/2,pair=blockIdx.x;
    int h=(pair/pairs)*group+2*(pair%pairs),hb=h+1;
    int a=blockIdx.y*2,b=a+1,tile=blockIdx.z;
    int t=threadIdx.x,lane=t&31,warp=t>>5;
    if(a>=tokens)return;
    bool has_head=hb<qheads && hb/group==h/group,has_query=b<tokens;
    int base=tile*128;
    int count_a=max(0,min(128,past+a+1-base));
    int count_b=has_head?count_a:0;
    int count_c=has_query?max(0,min(128,past+b+1-base)):0;
    int count_d=has_head?count_c:0;
    int count=max(count_a,count_c);
    if(count==0)return;
    int kv=pair/pairs;
    const float* qa=(true)?q+((size_t)a*qheads+h)*dim:q+((size_t)a*qheads+h)*dim;
    float qa0=lane+0<dim?qa[lane+0]:0;
    float qa1=lane+32<dim?qa[lane+32]:0;
    float qa2=lane+64<dim?qa[lane+64]:0;
    float qa3=lane+96<dim?qa[lane+96]:0;
    const float* qb=(has_head)?q+((size_t)a*qheads+hb)*dim:q+((size_t)a*qheads+h)*dim;
    float qb0=lane+0<dim?qb[lane+0]:0;
    float qb1=lane+32<dim?qb[lane+32]:0;
    float qb2=lane+64<dim?qb[lane+64]:0;
    float qb3=lane+96<dim?qb[lane+96]:0;
    const float* qc=(has_query)?q+((size_t)b*qheads+h)*dim:q+((size_t)a*qheads+h)*dim;
    float qc0=lane+0<dim?qc[lane+0]:0;
    float qc1=lane+32<dim?qc[lane+32]:0;
    float qc2=lane+64<dim?qc[lane+64]:0;
    float qc3=lane+96<dim?qc[lane+96]:0;
    const float* qd=(has_head && has_query)?q+((size_t)b*qheads+hb)*dim:q+((size_t)a*qheads+h)*dim;
    float qd0=lane+0<dim?qd[lane+0]:0;
    float qd1=lane+32<dim?qd[lane+32]:0;
    float qd2=lane+64<dim?qd[lane+64]:0;
    float qd3=lane+96<dim?qd[lane+96]:0;
    __shared__ float score_a[128],prob_a[128],score_b[128],prob_b[128],score_c[128],prob_c[128],score_d[128],prob_d[128];
    for(int j=warp;j<128;j+=4) {
        float dot_a=0,dot_b=0,dot_c=0,dot_d=0;
        bool valid_a=j<count_a,valid_b=j<count_b,valid_c=j<count_c,valid_d=j<count_d;
        if(j<count) {
            size_t key=((size_t)(base+j)*kvheads+kv)*dim;
            if(lane+0<dim) {
                float x=kv_read<TYPE>(k,key+lane+0);
                if(valid_a)dot_a=fmaf(qa0,x,dot_a);
                if(valid_b)dot_b=fmaf(qb0,x,dot_b);
                if(valid_c)dot_c=fmaf(qc0,x,dot_c);
                if(valid_d)dot_d=fmaf(qd0,x,dot_d);
            }
            if(lane+32<dim) {
                float x=kv_read<TYPE>(k,key+lane+32);
                if(valid_a)dot_a=fmaf(qa1,x,dot_a);
                if(valid_b)dot_b=fmaf(qb1,x,dot_b);
                if(valid_c)dot_c=fmaf(qc1,x,dot_c);
                if(valid_d)dot_d=fmaf(qd1,x,dot_d);
            }
            if(lane+64<dim) {
                float x=kv_read<TYPE>(k,key+lane+64);
                if(valid_a)dot_a=fmaf(qa2,x,dot_a);
                if(valid_b)dot_b=fmaf(qb2,x,dot_b);
                if(valid_c)dot_c=fmaf(qc2,x,dot_c);
                if(valid_d)dot_d=fmaf(qd2,x,dot_d);
            }
            if(lane+96<dim) {
                float x=kv_read<TYPE>(k,key+lane+96);
                if(valid_a)dot_a=fmaf(qa3,x,dot_a);
                if(valid_b)dot_b=fmaf(qb3,x,dot_b);
                if(valid_c)dot_c=fmaf(qc3,x,dot_c);
                if(valid_d)dot_d=fmaf(qd3,x,dot_d);
            }
            if(valid_a)dot_a=warp_sum(dot_a);
            if(valid_b)dot_b=warp_sum(dot_b);
            if(valid_c)dot_c=warp_sum(dot_c);
            if(valid_d)dot_d=warp_sum(dot_d);
        }
        if(lane==0) {
            score_a[j]=valid_a?dot_a*rsqrtf((float)dim):-INF;
            score_b[j]=valid_b?dot_b*rsqrtf((float)dim):-INF;
            score_c[j]=valid_c?dot_c*rsqrtf((float)dim):-INF;
            score_d[j]=valid_d?dot_d*rsqrtf((float)dim):-INF;
        }
    }
    __syncthreads();
    float ma=fmaxf(fmaxf(score_a[lane],score_a[lane+64]),fmaxf(score_a[lane+32],score_a[lane+96]));
    float mb=fmaxf(fmaxf(score_b[lane],score_b[lane+64]),fmaxf(score_b[lane+32],score_b[lane+96]));
    float mc=fmaxf(fmaxf(score_c[lane],score_c[lane+64]),fmaxf(score_c[lane+32],score_c[lane+96]));
    float md=fmaxf(fmaxf(score_d[lane],score_d[lane+64]),fmaxf(score_d[lane+32],score_d[lane+96]));
    for(int d=16;d;d/=2) {
        ma=fmaxf(ma,__shfl_down_sync(0xffffffffu,ma,d));
        mb=fmaxf(mb,__shfl_down_sync(0xffffffffu,mb,d));
        mc=fmaxf(mc,__shfl_down_sync(0xffffffffu,mc,d));
        md=fmaxf(md,__shfl_down_sync(0xffffffffu,md,d));
    }
    float max_a=__shfl_sync(0xffffffffu,ma,0);
    prob_a[t]=t<count_a?expf(score_a[t]-max_a):0;
    float max_b=__shfl_sync(0xffffffffu,mb,0);
    prob_b[t]=t<count_b?expf(score_b[t]-max_b):0;
    float max_c=__shfl_sync(0xffffffffu,mc,0);
    prob_c[t]=t<count_c?expf(score_c[t]-max_c):0;
    float max_d=__shfl_sync(0xffffffffu,md,0);
    prob_d[t]=t<count_d?expf(score_d[t]-max_d):0;
    __syncthreads();
    size_t offset_a=(((size_t)a*qheads+h)*tiles+tile)*(dim+2);
    size_t offset_b=(((size_t)a*qheads+hb)*tiles+tile)*(dim+2);
    size_t offset_c=(((size_t)b*qheads+h)*tiles+tile)*(dim+2);
    size_t offset_d=(((size_t)b*qheads+hb)*tiles+tile)*(dim+2);
    if(warp==0) {
        if(count_a>0) {
            float sum=prob_a[lane]+prob_a[lane+64];
            sum+=prob_a[lane+32]+prob_a[lane+96];sum=warp_sum(sum);
            if(lane==0){parts[offset_a+dim]=max_a;parts[offset_a+dim+1]=sum;}
        }
        if(count_b>0) {
            float sum=prob_b[lane]+prob_b[lane+64];
            sum+=prob_b[lane+32]+prob_b[lane+96];sum=warp_sum(sum);
            if(lane==0){parts[offset_b+dim]=max_b;parts[offset_b+dim+1]=sum;}
        }
        if(count_c>0) {
            float sum=prob_c[lane]+prob_c[lane+64];
            sum+=prob_c[lane+32]+prob_c[lane+96];sum=warp_sum(sum);
            if(lane==0){parts[offset_c+dim]=max_c;parts[offset_c+dim+1]=sum;}
        }
        if(count_d>0) {
            float sum=prob_d[lane]+prob_d[lane+64];
            sum+=prob_d[lane+32]+prob_d[lane+96];sum=warp_sum(sum);
            if(lane==0){parts[offset_d+dim]=max_d;parts[offset_d+dim+1]=sum;}
        }
    }
    if(t<dim) {
        float sum_a=0,sum_b=0,sum_c=0,sum_d=0;
        for(int j=0;j<count;j++) {
            float x=kv_read<TYPE>(v,((size_t)(base+j)*kvheads+kv)*dim+t);
            if(j<count_a)sum_a=fmaf(prob_a[j],x,sum_a);
            if(j<count_b)sum_b=fmaf(prob_b[j],x,sum_b);
            if(j<count_c)sum_c=fmaf(prob_c[j],x,sum_c);
            if(j<count_d)sum_d=fmaf(prob_d[j],x,sum_d);
        }
        if(count_a>0)parts[offset_a+t]=sum_a;
        if(count_b>0)parts[offset_b+t]=sum_b;
        if(count_c>0)parts[offset_c+t]=sum_c;
        if(count_d>0)parts[offset_d+t]=sum_d;
    }
}
extern "C" __global__ void attention_tiles_fp32_quad(
    const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads,int kvheads,int dim,int tiles) {
    attention_tiles_quad<0>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);
}
extern "C" __global__ void attention_tiles_fp16_quad(
    const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads,int kvheads,int dim,int tiles) {
    attention_tiles_quad<1>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);
}
extern "C" __global__ void attention_tiles_q8_quad(
    const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads,int kvheads,int dim,int tiles) {
    attention_tiles_quad<8>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);
}

// V5 pinned derivative: attention_sub8_fixed.cu SHA256 cc0e712a9e6d62d729cbcad726a457a272544e94411341b60839141067c7d0dd
// Fixed qheads32/kvheads2/dim128 only; host validation is mandatory.
// Runtime dim_arg intentionally retained only for canonical rsqrtf evaluation.
// Isolated V5 subwarp8 dot, independently derived from this fork's V4 tree.
// Concept references/provenance: ATTENTION-SUB8-ATTRIBUTION.txt.
// Append to canonical V4 CUDA. Single query/head/CTA; no pairing or Engine.
template<int TYPE> __device__ void attention_tiles_sub8_fixed(
    const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads_arg,int kvheads_arg,int dim_arg,int tiles) {
    const int qheads=32,kvheads=2,dim=128; // Host must validate this exact model shape.
    int h=blockIdx.x,token=blockIdx.y,tile=blockIdx.z,t=threadIdx.x,lane=t&31,warp=t>>5;
    int sub=t&7,keygroup=t>>3;
    unsigned mask=0xffu<<((t&31)&~7);
    int base=tile*128,length=past+token+1;if(token>=tokens || base>=length)return;
    int kv=h/(qheads/kvheads),count=min(128,length-base);
    const float* query=q+((size_t)token*qheads+h)*dim;
    __shared__ float score[128],prob[128];
    float q00=query[sub];
    float q01=query[sub+32];
    float q02=query[sub+64];
    float q03=query[sub+96];
    float q80=query[sub+8];
    float q81=query[sub+40];
    float q82=query[sub+72];
    float q83=query[sub+104];
    float q160=query[sub+16];
    float q161=query[sub+48];
    float q162=query[sub+80];
    float q163=query[sub+112];
    float q240=query[sub+24];
    float q241=query[sub+56];
    float q242=query[sub+88];
    float q243=query[sub+120];
    for(int j=keygroup;j<128;j+=16) {
        float dot=0;
        if(j<count) {
            size_t key=((size_t)(base+j)*kvheads+kv)*dim;
            float a0=0,a8=0,a16=0,a24=0;
            a0=fmaf(q00,kv_read<TYPE>(k,key+sub),a0);
            a0=fmaf(q01,kv_read<TYPE>(k,key+sub+32),a0);
            a0=fmaf(q02,kv_read<TYPE>(k,key+sub+64),a0);
            a0=fmaf(q03,kv_read<TYPE>(k,key+sub+96),a0);
            a8=fmaf(q80,kv_read<TYPE>(k,key+sub+8),a8);
            a8=fmaf(q81,kv_read<TYPE>(k,key+sub+40),a8);
            a8=fmaf(q82,kv_read<TYPE>(k,key+sub+72),a8);
            a8=fmaf(q83,kv_read<TYPE>(k,key+sub+104),a8);
            a16=fmaf(q160,kv_read<TYPE>(k,key+sub+16),a16);
            a16=fmaf(q161,kv_read<TYPE>(k,key+sub+48),a16);
            a16=fmaf(q162,kv_read<TYPE>(k,key+sub+80),a16);
            a16=fmaf(q163,kv_read<TYPE>(k,key+sub+112),a16);
            a24=fmaf(q240,kv_read<TYPE>(k,key+sub+24),a24);
            a24=fmaf(q241,kv_read<TYPE>(k,key+sub+56),a24);
            a24=fmaf(q242,kv_read<TYPE>(k,key+sub+88),a24);
            a24=fmaf(q243,kv_read<TYPE>(k,key+sub+120),a24);
            // V4 shuffle16 then8 tree for old lanes sub,sub+8,sub+16,sub+24.
            a0+=a16;a8+=a24;a0+=a8;
            dot=a0;
            for(int d=4;d;d>>=1)dot+=__shfl_down_sync(mask,dot,d,8);
        }
        if(sub==0)score[j]=j<count?dot*rsqrtf((float)dim_arg):-INF;
    }
    __syncthreads();
    float maximum=fmaxf(fmaxf(score[lane],score[lane+64]),fmaxf(score[lane+32],score[lane+96]));
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
    float tilemax=__shfl_sync(0xffffffffu,maximum,0);
    prob[t]=t<count?expf(score[t]-tilemax):0;__syncthreads();
    size_t offset=(((size_t)token*qheads+h)*tiles+tile)*(dim+2);
    if(warp==0) {
        float sum=prob[lane]+prob[lane+64];sum+=prob[lane+32]+prob[lane+96];sum=warp_sum(sum);
        if(lane==0){parts[offset+dim]=tilemax;parts[offset+dim+1]=sum;}
    }
    if(t<dim){float sum=0;for(int j=0;j<count;j++)sum=fmaf(prob[j],kv_read<TYPE>(v,((size_t)(base+j)*kvheads+kv)*dim+t),sum);parts[offset+t]=sum;}
}
extern "C" __global__ void attention_tiles_fp32_sub8_fixed(const float* q,const unsigned char* k,const unsigned char* v,float* parts,int tokens,int past,int qheads,int kvheads,int dim,int tiles) {attention_tiles_sub8_fixed<0>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);}
extern "C" __global__ void attention_tiles_fp16_sub8_fixed(const float* q,const unsigned char* k,const unsigned char* v,float* parts,int tokens,int past,int qheads,int kvheads,int dim,int tiles) {attention_tiles_sub8_fixed<1>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);}
extern "C" __global__ void attention_tiles_q8_sub8_fixed(const float* q,const unsigned char* k,const unsigned char* v,float* parts,int tokens,int past,int qheads,int kvheads,int dim,int tiles) {attention_tiles_sub8_fixed<8>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);}

// Frozen typed GQA16 operator source4afe7c7d (Intel111-case proof).
// Separate typed derivative of frozen1439/7338; no F32 evidence changes.
// TYPE only0(F32),1(F16),8(Q8_0); STAGE only32/64.
// Experimental fixed-model GQA16 sharing. Append to unchanged V4 CUDA.
// Host MUST enforce qheads32/kvheads2/dim128, grid(2,tokens,tiles), block512.
// Derives four-FMA/reduction/merge arithmetic from this fork's V4 and sub8.
// No third-party kernel copied. See PROTOCOL.txt for scope and attribution.

// XOR only permutes addresses within each 128-float shared row. It keeps
// four sub8 key groups in distinct banks (4-byte banks), with no padding.
__device__ __forceinline__ int gqa16_typed_index(int key, int column) {
    return key*128+(column^((key&3)*8));
}

template<int TYPE,int STAGE> __device__ __forceinline__ void attention_gqa16_typed(
    const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads_arg,int kvheads_arg,int dim_arg,int tiles) {
    const int kv=blockIdx.x,token=blockIdx.y,tile=blockIdx.z;
    const int t=threadIdx.x,lane=t&31,head=t>>5,sub=lane&7,group=lane>>3;
    const int base=tile*128,length=past+token+1;
    if(token>=tokens || base>=length || kv>=2 || blockDim.x!=512)return;
    const int count=min(128,length-base),h=kv*16+head;
    const float* query=q+((size_t)token*32+h)*128;
    const unsigned mask=0xffu<<(lane&~7);
    __shared__ float stage[STAGE*128];
    __shared__ float score_prob[16*128];
    const int headbase=head*128;

    float q00=query[sub],q01=query[sub+32],q02=query[sub+64],q03=query[sub+96];
    float q80=query[sub+8],q81=query[sub+40],q82=query[sub+72],q83=query[sub+104];
    float q160=query[sub+16],q161=query[sub+48],q162=query[sub+80],q163=query[sub+112];
    float q240=query[sub+24],q241=query[sub+56],q242=query[sub+88],q243=query[sub+120];
    for(int i=t;i<16*128;i+=512)score_prob[i]=-INF;
    if(STAGE==64)__syncthreads();

    #pragma unroll 1
    for(int phase=0;phase<128/STAGE;phase++) {
        for(int i=t;i<STAGE*128;i+=512) {
            const int key=i>>7,column=i&127,j=phase*STAGE+key;
            stage[gqa16_typed_index(key,column)]=j<count?
                kv_read<TYPE>(k,((size_t)(base+j)*2+kv)*128+column):0.f;
        }
        __syncthreads();
        #pragma unroll 1
        for(int key=group;key<STAGE;key+=4) {
            const int j=phase*STAGE+key;
            if(j<count) {
                float a0=0,a8=0,a16=0,a24=0;
                a0=fmaf(q00,stage[gqa16_typed_index(key,sub)],a0);
                a0=fmaf(q01,stage[gqa16_typed_index(key,sub+32)],a0);
                a0=fmaf(q02,stage[gqa16_typed_index(key,sub+64)],a0);
                a0=fmaf(q03,stage[gqa16_typed_index(key,sub+96)],a0);
                a8=fmaf(q80,stage[gqa16_typed_index(key,sub+8)],a8);
                a8=fmaf(q81,stage[gqa16_typed_index(key,sub+40)],a8);
                a8=fmaf(q82,stage[gqa16_typed_index(key,sub+72)],a8);
                a8=fmaf(q83,stage[gqa16_typed_index(key,sub+104)],a8);
                a16=fmaf(q160,stage[gqa16_typed_index(key,sub+16)],a16);
                a16=fmaf(q161,stage[gqa16_typed_index(key,sub+48)],a16);
                a16=fmaf(q162,stage[gqa16_typed_index(key,sub+80)],a16);
                a16=fmaf(q163,stage[gqa16_typed_index(key,sub+112)],a16);
                a24=fmaf(q240,stage[gqa16_typed_index(key,sub+24)],a24);
                a24=fmaf(q241,stage[gqa16_typed_index(key,sub+56)],a24);
                a24=fmaf(q242,stage[gqa16_typed_index(key,sub+88)],a24);
                a24=fmaf(q243,stage[gqa16_typed_index(key,sub+120)],a24);
                a0+=a16;a8+=a24;a0+=a8;
                for(int d=4;d;d>>=1)a0+=__shfl_down_sync(mask,a0,d,8);
                // dim_arg must remain runtime; host validation belongs outside
                // this kernel so the compiler cannot infer a constant rsqrt.
                if(sub==0)score_prob[headbase+j]=a0*rsqrtf((float)dim_arg);
            }
        }
        // Every head finished the phase before shared staging is overwritten.
        __syncthreads();
    }

    float s0=score_prob[headbase+lane],s1=score_prob[headbase+lane+32];
    float s2=score_prob[headbase+lane+64],s3=score_prob[headbase+lane+96];
    float maximum=fmaxf(fmaxf(s0,s2),fmaxf(s1,s3));
    for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
    const float tilemax=__shfl_sync(0xffffffffu,maximum,0);
    float p0=lane<count?expf(s0-tilemax):0.f;
    float p1=lane+32<count?expf(s1-tilemax):0.f;
    float p2=lane+64<count?expf(s2-tilemax):0.f;
    float p3=lane+96<count?expf(s3-tilemax):0.f;
    float total=p0+p2;total+=p1+p3;total=warp_sum(total);
    score_prob[headbase+lane]=p0;score_prob[headbase+lane+32]=p1;
    score_prob[headbase+lane+64]=p2;score_prob[headbase+lane+96]=p3;
    const size_t offset=(((size_t)token*32+h)*tiles+tile)*130;
    if(lane==0){parts[offset+128]=tilemax;parts[offset+129]=total;}
    __syncthreads();

    float sum0=0,sum1=0,sum2=0,sum3=0;
    #pragma unroll 1
    for(int phase=0;phase<128/STAGE;phase++) {
        for(int i=t;i<STAGE*128;i+=512) {
            const int key=i>>7,column=i&127,j=phase*STAGE+key;
            stage[gqa16_typed_index(key,column)]=j<count?
                kv_read<TYPE>(v,((size_t)(base+j)*2+kv)*128+column):0.f;
        }
        __syncthreads();
        const int valid=max(0,min(STAGE,count-phase*STAGE));
        #pragma unroll 1
        for(int key=0;key<valid;key++) {
            const float probability=score_prob[headbase+phase*STAGE+key];
            sum0=fmaf(probability,stage[gqa16_typed_index(key,lane)],sum0);
            sum1=fmaf(probability,stage[gqa16_typed_index(key,lane+32)],sum1);
            sum2=fmaf(probability,stage[gqa16_typed_index(key,lane+64)],sum2);
            sum3=fmaf(probability,stage[gqa16_typed_index(key,lane+96)],sum3);
        }
        __syncthreads();
    }
    parts[offset+lane]=sum0;parts[offset+lane+32]=sum1;
    parts[offset+lane+64]=sum2;parts[offset+lane+96]=sum3;
}

extern "C" __global__ void attention_tiles_fp32_gqa16_typed64(const float* q,const unsigned char* k,const unsigned char* v,float* parts,int tokens,int past,int qheads,int kvheads,int dim,int tiles){attention_gqa16_typed<0,64>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);}

extern "C" __global__ void attention_tiles_fp32_gqa16_typed32(const float* q,const unsigned char* k,const unsigned char* v,float* parts,int tokens,int past,int qheads,int kvheads,int dim,int tiles){attention_gqa16_typed<0,32>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);}

extern "C" __global__ void attention_tiles_fp16_gqa16_typed64(const float* q,const unsigned char* k,const unsigned char* v,float* parts,int tokens,int past,int qheads,int kvheads,int dim,int tiles){attention_gqa16_typed<1,64>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);}

extern "C" __global__ void attention_tiles_fp16_gqa16_typed32(const float* q,const unsigned char* k,const unsigned char* v,float* parts,int tokens,int past,int qheads,int kvheads,int dim,int tiles){attention_gqa16_typed<1,32>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);}

extern "C" __global__ void attention_tiles_q8_gqa16_typed64(const float* q,const unsigned char* k,const unsigned char* v,float* parts,int tokens,int past,int qheads,int kvheads,int dim,int tiles){attention_gqa16_typed<8,64>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);}

extern "C" __global__ void attention_tiles_q8_gqa16_typed32(const float* q,const unsigned char* k,const unsigned char* v,float* parts,int tokens,int past,int qheads,int kvheads,int dim,int tiles){attention_gqa16_typed<8,32>(q,k,v,parts,tokens,past,qheads,kvheads,dim,tiles);}

// V6 opt-in: exact frozen load-only vector2 source follows; FP16 dispatch only.
// Isolated V-only float2 remap of frozen typed GQA16/stage32 (4afe7c7d).
// Fixed32Q/2KV/d128; SAME grid(2,tokens,tiles), block512,24KiB shared.
// QK/max/sum and PRIVATE valid-count loop unchanged; per-output FMA order same.
// Only V shared reads and independent output-column ownership are remapped.
// Explicitly aligned stage; no bank-mode API calls in device or host code.
// Concept: NVIDIA Kepler8-byte shared banks; code/math reused from this fork.
// CPU/static preparation only; no performance or GPU-parity claim.

__device__ __forceinline__ int gqa16_vector2_index(int key,int column) {
    return key*128+(column^((key&3)*8));
}

template<int TYPE> __device__ __forceinline__ void attention_gqa16_vector2_typed(
    const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads_arg,int kvheads_arg,int dim_arg,int tiles) {
    const int kv=blockIdx.x,token=blockIdx.y,tile=blockIdx.z;
    const int t=threadIdx.x,lane=t&31,head=t>>5,sub=lane&7,group=lane>>3;
    const int base=tile*128,length=past+token+1;
    if(token>=tokens || base>=length || kv>=2 || blockDim.x!=512)return;
    const int count=min(128,length-base),h=kv*16+head;
    const float* query=q+((size_t)token*32+h)*128;
    const unsigned mask=0xffu<<(lane&~7);
    __shared__ __align__(16) float stage[32*128];
    __shared__ float score_prob[16*128];
    const int headbase=head*128;

    float q00=query[sub],q01=query[sub+32],q02=query[sub+64],q03=query[sub+96];
    float q80=query[sub+8],q81=query[sub+40],q82=query[sub+72],q83=query[sub+104];
    float q160=query[sub+16],q161=query[sub+48],q162=query[sub+80],q163=query[sub+112];
    float q240=query[sub+24],q241=query[sub+56],q242=query[sub+88],q243=query[sub+120];
    for(int i=t;i<16*128;i+=512)score_prob[i]=-INF;

    #pragma unroll 1
    for(int phase=0;phase<128/32;phase++) {
        for(int i=t;i<32*128;i+=512) {
            const int key=i>>7,column=i&127,j=phase*32+key;
            stage[gqa16_vector2_index(key,column)]=j<count?
                kv_read<TYPE>(k,((size_t)(base+j)*2+kv)*128+column):0.f;
        }
        __syncthreads();
        #pragma unroll 1
        for(int key=group;key<32;key+=4) {
            const int j=phase*32+key;
            if(j<count) {
                float a0=0,a8=0,a16=0,a24=0;
                a0=fmaf(q00,stage[gqa16_vector2_index(key,sub)],a0);
                a0=fmaf(q01,stage[gqa16_vector2_index(key,sub+32)],a0);
                a0=fmaf(q02,stage[gqa16_vector2_index(key,sub+64)],a0);
                a0=fmaf(q03,stage[gqa16_vector2_index(key,sub+96)],a0);
                a8=fmaf(q80,stage[gqa16_vector2_index(key,sub+8)],a8);
                a8=fmaf(q81,stage[gqa16_vector2_index(key,sub+40)],a8);
                a8=fmaf(q82,stage[gqa16_vector2_index(key,sub+72)],a8);
                a8=fmaf(q83,stage[gqa16_vector2_index(key,sub+104)],a8);
                a16=fmaf(q160,stage[gqa16_vector2_index(key,sub+16)],a16);
                a16=fmaf(q161,stage[gqa16_vector2_index(key,sub+48)],a16);
                a16=fmaf(q162,stage[gqa16_vector2_index(key,sub+80)],a16);
                a16=fmaf(q163,stage[gqa16_vector2_index(key,sub+112)],a16);
                a24=fmaf(q240,stage[gqa16_vector2_index(key,sub+24)],a24);
                a24=fmaf(q241,stage[gqa16_vector2_index(key,sub+56)],a24);
                a24=fmaf(q242,stage[gqa16_vector2_index(key,sub+88)],a24);
                a24=fmaf(q243,stage[gqa16_vector2_index(key,sub+120)],a24);
                a0+=a16;a8+=a24;a0+=a8;
                for(int d=4;d;d>>=1)a0+=__shfl_down_sync(mask,a0,d,8);
                // dim_arg must remain runtime; host validation belongs outside
                // this kernel so the compiler cannot infer a constant rsqrt.
                if(sub==0)score_prob[headbase+j]=a0*rsqrtf((float)dim_arg);
            }
        }
        // Every head finished the phase before shared staging is overwritten.
        __syncthreads();
    }

    float s0=score_prob[headbase+lane],s1=score_prob[headbase+lane+32];
    float s2=score_prob[headbase+lane+64],s3=score_prob[headbase+lane+96];
    float maximum=fmaxf(fmaxf(s0,s2),fmaxf(s1,s3));
    for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_down_sync(0xffffffffu,maximum,d));
    const float tilemax=__shfl_sync(0xffffffffu,maximum,0);
    float p0=lane<count?expf(s0-tilemax):0.f;
    float p1=lane+32<count?expf(s1-tilemax):0.f;
    float p2=lane+64<count?expf(s2-tilemax):0.f;
    float p3=lane+96<count?expf(s3-tilemax):0.f;
    float total=p0+p2;total+=p1+p3;total=warp_sum(total);
    score_prob[headbase+lane]=p0;score_prob[headbase+lane+32]=p1;
    score_prob[headbase+lane+64]=p2;score_prob[headbase+lane+96]=p3;
    const size_t offset=(((size_t)token*32+h)*tiles+tile)*130;
    if(lane==0){parts[offset+128]=tilemax;parts[offset+129]=total;}
    __syncthreads();

    float sum0=0,sum1=0,sum2=0,sum3=0;
    #pragma unroll 1
    for(int phase=0;phase<128/32;phase++) {
        for(int i=t;i<32*128;i+=512) {
            const int key=i>>7,column=i&127,j=phase*32+key;
            stage[gqa16_vector2_index(key,column)]=j<count?
                kv_read<TYPE>(v,((size_t)(base+j)*2+kv)*128+column):0.f;
        }
        __syncthreads();
        const int valid=max(0,min(32,count-phase*32));
        #pragma unroll 1
        for(int key=0;key<valid;key++) {
            const float probability=score_prob[headbase+phase*32+key];
            // Even XOR offset preserves each adjacent float pair and8B alignment.
            const float2 low=*reinterpret_cast<const float2*>(stage+gqa16_vector2_index(key,2*lane));
            const float2 high=*reinterpret_cast<const float2*>(stage+gqa16_vector2_index(key,2*lane+64));
            sum0=fmaf(probability,low.x,sum0);
            sum1=fmaf(probability,low.y,sum1);
            sum2=fmaf(probability,high.x,sum2);
            sum3=fmaf(probability,high.y,sum3);
        }
        __syncthreads();
    }
    parts[offset+2*lane]=sum0;parts[offset+2*lane+1]=sum1;
    parts[offset+2*lane+64]=sum2;parts[offset+2*lane+65]=sum3;
}

extern "C" __global__ void attention_tiles_fp32_gqa16_vector2_typed32(
    const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads_arg,int kvheads_arg,int dim_arg,int tiles) {
    attention_gqa16_vector2_typed<0>(q,k,v,parts,tokens,past,qheads_arg,kvheads_arg,dim_arg,tiles);
}

extern "C" __global__ void attention_tiles_fp16_gqa16_vector2_typed32(
    const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads_arg,int kvheads_arg,int dim_arg,int tiles) {
    attention_gqa16_vector2_typed<1>(q,k,v,parts,tokens,past,qheads_arg,kvheads_arg,dim_arg,tiles);
}

extern "C" __global__ void attention_tiles_q8_gqa16_vector2_typed32(
    const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads_arg,int kvheads_arg,int dim_arg,int tiles) {
    attention_gqa16_vector2_typed<8>(q,k,v,parts,tokens,past,qheads_arg,kvheads_arg,dim_arg,tiles);
}

