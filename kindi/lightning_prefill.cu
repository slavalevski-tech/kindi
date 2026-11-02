// AGPL-3.0 derivative of local q2_row_tile, same FP32 order; experimental only.
template<int ROWS_PER_WARP>
__device__ __forceinline__ void v8_q2_row_tile(const unsigned char* w,const float* x,float* y,
    int rows,int cols,int items,int topk,int per_assignment,const int* order,
    const int* job_expert,const int* job_start,const int* job_length,const int* job_count) {
    int job=blockIdx.y,t=threadIdx.x,lane=t&31,warp=t>>5;
    if(job>=(job_count?*job_count:(items+7)/8))return;
    int count=job_length?job_length[job]:min(8,items-job*8);
    int first=job_start?job_start[job]:job*8,expert=job_expert?job_expert[job]:0;
    int rowbase=blockIdx.x*(ROWS_PER_WARP*8)+warp;
    size_t rb=(size_t)(cols/64)*18;
    __shared__ float xs[8][4][32];
    __shared__ int assignment[8];
    if(t<8)assignment[t]=t<count?(order?order[first+t]:first+t):-1;
    __syncthreads();
    float sums[ROWS_PER_WARP][8]={0};
    for(int base=0;base<cols;base+=128) {
        int column=base+lane*4,aa=assignment[warp];
        if(aa>=0 && column<cols) {
            int input=per_assignment?aa:aa/topk;
            float4 value=__ldg(reinterpret_cast<const float4*>(x+(size_t)input*cols+column));
            xs[warp][0][lane]=value.x;xs[warp][1][lane]=value.y;
            xs[warp][2][lane]=value.z;xs[warp][3][lane]=value.w;
        }
        __syncthreads();
        if(column<cols) {
            float d[ROWS_PER_WARP],q0[ROWS_PER_WARP],q1[ROWS_PER_WARP];
            float q2[ROWS_PER_WARP],q3[ROWS_PER_WARP];
            #pragma unroll
            for(int r=0;r<ROWS_PER_WARP;r++) {
                int row=rowbase+r*8;unsigned q=0;d[r]=0.f;
                if(row<rows) {
                    const unsigned char* b=w+((size_t)expert*rows+row)*rb+(column/64)*18;
                    d[r]=__half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(b))));
                    q=__ldg(b+2+(column%64)/4);
                }
                q0[r]=q2_or_coeff<0>(q);q1[r]=q2_or_coeff<2>(q);
                q2[r]=q2_or_coeff<4>(q);q3[r]=q2_or_coeff<6>(q);
            }
            #pragma unroll
            for(int s=0;s<8;s++)if(s<count) {
                float x0=xs[s][0][lane],x1=xs[s][1][lane];
                float x2=xs[s][2][lane],x3=xs[s][3][lane];
                #pragma unroll
                for(int r=0;r<ROWS_PER_WARP;r++)if(rowbase+r*8<rows) {
                    float partial=fmaf(q0[r],x0,0.f);
                    partial=fmaf(q1[r],x1,partial);partial=fmaf(q2[r],x2,partial);
                    partial=fmaf(q3[r],x3,partial);
                    sums[r][s]=fmaf(d[r],partial,sums[r][s]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int s=0;s<8;s++)if(s<count) {
        #pragma unroll
        for(int r=0;r<ROWS_PER_WARP;r++) {
            float sum=warp_sum(sums[r][s]);int row=rowbase+r*8;
            if(lane==0 && row<rows)y[(size_t)assignment[s]*rows+row]=sum;
        }
    }
}


extern "C" __global__ void v8_or24(const unsigned char* w,const float* x,float* y,
    int rows,int cols,int items,int topk,int per_assignment,const int* order,
    const int* job_expert,const int* job_start,const int* job_length,const int* job_count) {
    v8_q2_row_tile<3>(w,x,y,rows,cols,items,topk,per_assignment,order,job_expert,job_start,job_length,job_count);
}

extern "C" __global__ void v8_or32(const unsigned char* w,const float* x,float* y,
    int rows,int cols,int items,int topk,int per_assignment,const int* order,
    const int* job_expert,const int* job_start,const int* job_length,const int* job_count) {
    v8_q2_row_tile<4>(w,x,y,rows,cols,items,topk,per_assignment,order,job_expert,job_start,job_length,job_count);
}

extern "C" __global__ __launch_bounds__(256,3) void v8_or24_lb3(const unsigned char* w,const float* x,float* y,
    int rows,int cols,int items,int topk,int per_assignment,const int* order,
    const int* job_expert,const int* job_start,const int* job_length,const int* job_count) {
    v8_q2_row_tile<3>(w,x,y,rows,cols,items,topk,per_assignment,order,job_expert,job_start,job_length,job_count);
}

extern "C" __global__ __launch_bounds__(256,4) void v8_or24_lb4(const unsigned char* w,const float* x,float* y,
    int rows,int cols,int items,int topk,int per_assignment,const int* order,
    const int* job_expert,const int* job_start,const int* job_length,const int* job_count) {
    v8_q2_row_tile<3>(w,x,y,rows,cols,items,topk,per_assignment,order,job_expert,job_start,job_length,job_count);
}
