// AGPL-3.0: exact tested global four-halfword staging body, no arithmetic change.
__device__ __forceinline__ void v8_attention_stage4_body(
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
        for(int i=4*t;i<32*128;i+=2048) {
            const int key=i>>7,column=i&127,j=phase*32+key;
            float4 value=make_float4(0.f,0.f,0.f,0.f);
            if(j<count) {
                const size_t index=((size_t)(base+j)*2+kv)*128+column;
                const uint2 packed=__ldg(reinterpret_cast<const uint2*>(k)+index/4);
                value.x=__half2float(__ushort_as_half((unsigned short)packed.x));
                value.y=__half2float(__ushort_as_half((unsigned short)(packed.x>>16)));
                value.z=__half2float(__ushort_as_half((unsigned short)packed.y));
                value.w=__half2float(__ushort_as_half((unsigned short)(packed.y>>16)));
            }
            *reinterpret_cast<float4*>(stage+gqa16_vector2_index(key,column))=value;
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
        for(int i=4*t;i<32*128;i+=2048) {
            const int key=i>>7,column=i&127,j=phase*32+key;
            float4 value=make_float4(0.f,0.f,0.f,0.f);
            if(j<count) {
                const size_t index=((size_t)(base+j)*2+kv)*128+column;
                const uint2 packed=__ldg(reinterpret_cast<const uint2*>(v)+index/4);
                value.x=__half2float(__ushort_as_half((unsigned short)packed.x));
                value.y=__half2float(__ushort_as_half((unsigned short)(packed.x>>16)));
                value.z=__half2float(__ushort_as_half((unsigned short)packed.y));
                value.w=__half2float(__ushort_as_half((unsigned short)(packed.y>>16)));
            }
            *reinterpret_cast<float4*>(stage+gqa16_vector2_index(key,column))=value;
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

extern "C" __global__ void v8_attention_stage4(
    const float* q,const unsigned char* k,const unsigned char* v,float* parts,
    int tokens,int past,int qheads_arg,int kvheads_arg,int dim_arg,int tiles) {
    v8_attention_stage4_body(q,k,v,parts,tokens,past,qheads_arg,kvheads_arg,dim_arg,tiles);
}


// Experimental load-only FP16 pair staging; original math and tile geometry frozen.
