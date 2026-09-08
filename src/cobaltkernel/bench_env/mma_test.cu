// mma_test.cu -- does the PTX mma.sync tensor-core family compile AND run on sm_120a?
//   bf16 m16n8k16   (correctness-checked against a scalar reference)
//   fp16 m16n8k16   (compile+run)
//   s8   m16n8k32   (compile+run)
//   e4m3 fp8 m16n8k32 (compile only guarded by __CUDA_ARCH__; reported)
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));return 1;} }while(0)

// A: 16x16 bf16 row-major, B: 16x8 bf16 col-major(=K-major), D: 16x8 fp32
__global__ void mma_bf16_k(const __nv_bfloat16* A, const __nv_bfloat16* B, float* D){
  int lane = threadIdx.x & 31;
  // m16n8k16 A-fragment: 4 x b32 (8 bf16), layout per PTX ISA
  int gr = lane >> 2, gc = lane & 3;
  __nv_bfloat16 a[8], b[4];
  // a0,a1 -> row gr,     cols 2*gc, 2*gc+1
  // a2,a3 -> row gr+8,   cols 2*gc, 2*gc+1
  // a4,a5 -> row gr,     cols 2*gc+8, 2*gc+9
  // a6,a7 -> row gr+8,   cols 2*gc+8, 2*gc+9
  a[0]=A[(gr  )*16 + 2*gc  ]; a[1]=A[(gr  )*16 + 2*gc+1];
  a[2]=A[(gr+8)*16 + 2*gc  ]; a[3]=A[(gr+8)*16 + 2*gc+1];
  a[4]=A[(gr  )*16 + 2*gc+8]; a[5]=A[(gr  )*16 + 2*gc+9];
  a[6]=A[(gr+8)*16 + 2*gc+8]; a[7]=A[(gr+8)*16 + 2*gc+9];
  // b0,b1 -> col gr, k = 2*gc, 2*gc+1 ; b2,b3 -> col gr, k = 2*gc+8, 2*gc+9
  b[0]=B[(gr)*16 + 2*gc  ]; b[1]=B[(gr)*16 + 2*gc+1];
  b[2]=B[(gr)*16 + 2*gc+8]; b[3]=B[(gr)*16 + 2*gc+9];
  unsigned const* A32 = reinterpret_cast<unsigned const*>(a);
  unsigned const* B32 = reinterpret_cast<unsigned const*>(b);
  float d[4] = {0,0,0,0};
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(A32[0]),"r"(A32[1]),"r"(A32[2]),"r"(A32[3]), "r"(B32[0]),"r"(B32[1]));
  // D fragment: d0,d1 -> row gr, col 2*gc(+1); d2,d3 -> row gr+8
  D[(gr  )*8 + 2*gc  ] = d[0]; D[(gr  )*8 + 2*gc+1] = d[1];
  D[(gr+8)*8 + 2*gc  ] = d[2]; D[(gr+8)*8 + 2*gc+1] = d[3];
}

__global__ void mma_f16_k(const __half* A, const __half* B, float* D){
  unsigned a[4]={0,0,0,0}, b[2]={0,0}; float d[4]={0,0,0,0};
  int lane=threadIdx.x&31;
  const unsigned* A32=reinterpret_cast<const unsigned*>(A);
  const unsigned* B32=reinterpret_cast<const unsigned*>(B);
  a[0]=A32[lane]; a[1]=A32[lane+32]; a[2]=A32[lane+64]; a[3]=A32[lane+96];
  b[0]=B32[lane]; b[1]=B32[lane+32];
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  if(lane<4) D[lane]=d[lane];
}

__global__ void mma_s8_k(const int* A, const int* B, int* D){
  int lane=threadIdx.x&31;
  unsigned a[4], b[2]; int d[4]={0,0,0,0};
  a[0]=A[lane]; a[1]=A[lane+32]; a[2]=A[lane+64]; a[3]=A[lane+96];
  b[0]=B[lane]; b[1]=B[lane+32];
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  if(lane<4) D[lane]=d[lane];
}

__global__ void mma_e4m3_k(const int* A, const int* B, float* D){
  int lane=threadIdx.x&31;
  unsigned a[4], b[2]; float d[4]={0,0,0,0};
  a[0]=A[lane]; a[1]=A[lane+32]; a[2]=A[lane+64]; a[3]=A[lane+96];
  b[0]=B[lane]; b[1]=B[lane+32];
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  if(lane<4) D[lane]=d[lane];
}

int main(){
  // ---- bf16 m16n8k16 with a real correctness check ----
  __nv_bfloat16 hA[256], hB[128]; float ref[128];
  for(int i=0;i<256;i++) hA[i] = __float2bfloat16(((i*37)%13)-6);
  for(int i=0;i<128;i++) hB[i] = __float2bfloat16(((i*17)%7)-3);
  for(int m=0;m<16;m++) for(int n=0;n<8;n++){
    float s=0; for(int k=0;k<16;k++) s += __bfloat162float(hA[m*16+k])*__bfloat162float(hB[n*16+k]);
    ref[m*8+n]=s;
  }
  __nv_bfloat16 *dA,*dB; float* dD;
  CK(cudaMalloc(&dA,sizeof(hA))); CK(cudaMalloc(&dB,sizeof(hB))); CK(cudaMalloc(&dD,128*4));
  CK(cudaMemcpy(dA,hA,sizeof(hA),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,hB,sizeof(hB),cudaMemcpyHostToDevice));
  mma_bf16_k<<<1,32>>>(dA,dB,dD);
  cudaError_t e=cudaDeviceSynchronize();
  if(e){ printf("mma.bf16.m16n8k16 : RUN FAILED (%s)\n", cudaGetErrorString(e)); }
  else {
    float got[128]; CK(cudaMemcpy(got,dD,128*4,cudaMemcpyDeviceToHost));
    double md=0; for(int i=0;i<128;i++){ double dd=fabs(got[i]-ref[i]); if(dd>md) md=dd; }
    printf("mma.sync.m16n8k16 bf16.bf16.f32 : COMPILES + RUNS, max|err| vs scalar ref = %g %s\n",
           md, md<1e-2 ? "(CORRECT)" : "(MISMATCH)");
  }
  // ---- the rest: compile + run, no numeric check ----
  __half* dHa; CK(cudaMalloc(&dHa, 4096)); CK(cudaMemset(dHa,0,4096));
  mma_f16_k<<<1,32>>>(dHa,dHa,dD);
  e=cudaDeviceSynchronize();
  printf("mma.sync.m16n8k16 f16.f16.f32   : %s\n", e?cudaGetErrorString(e):"COMPILES + RUNS");
  int* dI; CK(cudaMalloc(&dI,4096)); CK(cudaMemset(dI,0,4096)); int* dOi; CK(cudaMalloc(&dOi,64));
  mma_s8_k<<<1,32>>>(dI,dI,dOi);
  e=cudaDeviceSynchronize();
  printf("mma.sync.m16n8k32 s8.s8.s32     : %s\n", e?cudaGetErrorString(e):"COMPILES + RUNS");
  mma_e4m3_k<<<1,32>>>(dI,dI,dD);
  e=cudaDeviceSynchronize();
  printf("mma.sync.m16n8k32 e4m3(fp8).f32 : %s\n", e?cudaGetErrorString(e):"COMPILES + RUNS");
  return 0;
}
