// gemv4bit.cu -- attainable bandwidth for a memory-bound 4-bit dequant-GEMV,
// sized to a MedGemma-27B MLP matrix (N=21504 rows x K=5376), 4-bit packed
// nibbles + per-group(128) bf16 scale & 4-bit zero => 57.8 MB of weights.
//
// Two regimes are measured, because L2 on the 2g.48gb slice is 64 MiB and a
// single such matrix (57.8 MB) FITS IN L2 -- repeating one matrix measures L2,
// not HBM. The honest decode number is the multi-copy (L2-busting) one.
//
// Usage: ./gemv4bit [copies=16] [reps=20]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

#define N_ROWS 21504
#define K_DIM  5376
#define GROUP  128
#define ROW_BYTES (K_DIM/2)             // 2688
#define NGROUP (K_DIM/GROUP)            // 42

// one warp per row; uint4 (16B) per lane => 512B per warp per step, fully coalesced
template<bool DEQUANT>
__global__ void gemv4(const uint4* __restrict__ W,      // [N_ROWS][ROW_BYTES]
                      const __nv_bfloat16* __restrict__ S, // [N_ROWS][NGROUP] scales
                      const __nv_bfloat16* __restrict__ X, // [K_DIM]
                      float* __restrict__ Y,               // [N_ROWS]
                      int n_rows, size_t stride16)
{
  extern __shared__ __nv_bfloat16 xs[];
  for (int i = threadIdx.x; i < K_DIM; i += blockDim.x) xs[i] = X[i];
  __syncthreads();

  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n_rows) return;

  const uint4* wrow = W + (size_t)row * stride16;
  float acc = 0.f;
  // ROW_BYTES/16 = 168 uint4 per row; warp strides 32 uint4 = 512B
  for (int u = lane; u < ROW_BYTES/16; u += 32) {
    uint4 v = wrow[u];
    if (DEQUANT) {
      const unsigned* w32 = reinterpret_cast<const unsigned*>(&v);
      int k0 = u * 32;                                   // 16 bytes = 32 nibbles
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        unsigned p = w32[j];
        int kb = k0 + j*8;
        float sc = __bfloat162float(S[(size_t)row*NGROUP + (kb/GROUP)]);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
          int q = (int)((p >> (4*t)) & 0xF) - 8;         // symmetric zero-point
          acc = fmaf((float)q * sc, __bfloat162float(xs[kb + t]), acc);
        }
      }
    } else {
      acc += (float)(v.x ^ v.y ^ v.z ^ v.w);             // stream-only reference
    }
  }
  #pragma unroll
  for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
  if (lane == 0) Y[row] = acc;
}

int main(int argc,char**argv){
  int copies = argc>1? atoi(argv[1]) : 16;
  int reps   = argc>2? atoi(argv[2]) : 20;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  size_t wbytes = (size_t)N_ROWS*ROW_BYTES;               // 57.80 MB
  size_t sbytes = (size_t)N_ROWS*NGROUP*2;                // bf16 scales
  size_t per    = wbytes + sbytes;
  printf("device %s  SMs=%d  L2=%.1f MiB\n", p.name, p.multiProcessorCount, p.l2CacheSize/1048576.0);
  printf("matrix N=%d K=%d 4-bit g%d : weights %.2f MB + scales %.2f MB = %.2f MB/copy, %d copies (%.2f GB)\n",
         N_ROWS, K_DIM, GROUP, wbytes/1e6, sbytes/1e6, per/1e6, copies, copies*per/1e9);

  uint4 **W = (uint4**)malloc(copies*sizeof(void*));
  __nv_bfloat16 **S = (__nv_bfloat16**)malloc(copies*sizeof(void*));
  for(int c=0;c<copies;c++){
    CK(cudaMalloc(&W[c], wbytes)); CK(cudaMemset(W[c], 0x37, wbytes));
    CK(cudaMalloc(&S[c], sbytes)); CK(cudaMemset(S[c], 0x3c, sbytes));
  }
  __nv_bfloat16* X; CK(cudaMalloc(&X, K_DIM*2)); CK(cudaMemset(X, 0x3c, K_DIM*2));
  float* Y; CK(cudaMalloc(&Y, N_ROWS*4));

  int thr = 256, wpb = thr/32;
  int grid = (N_ROWS + wpb - 1)/wpb;
  size_t smem = K_DIM*sizeof(__nv_bfloat16);              // 10.5 KB
  size_t stride16 = ROW_BYTES/16;
  printf("launch: grid=%d blocks x %d thr (%d warps/blk = 1 row/warp), smem=%zu B\n", grid, thr, wpb, smem);

  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  const char* names[2] = {"stream-only (no dequant)","dequant-GEMV (4b->bf16 fma)"};
  for (int mode=0; mode<2; ++mode) {
    for (int use_copies : {1, copies}) {
      // warm
      for(int c=0;c<use_copies;c++){
        if(mode) gemv4<true><<<grid,thr,smem>>>(W[c],S[c],X,Y,N_ROWS,stride16);
        else     gemv4<false><<<grid,thr,smem>>>(W[c],S[c],X,Y,N_ROWS,stride16);
      }
      CK(cudaDeviceSynchronize());
      double best=0;
      for(int r=0;r<reps;r++){
        CK(cudaEventRecord(a));
        for(int c=0;c<use_copies;c++){
          if(mode) gemv4<true><<<grid,thr,smem>>>(W[c],S[c],X,Y,N_ROWS,stride16);
          else     gemv4<false><<<grid,thr,smem>>>(W[c],S[c],X,Y,N_ROWS,stride16);
        }
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms,a,b));
        double gbs = (double)use_copies*per/1e9/(ms/1e3);
        if(gbs>best) best=gbs;
      }
      printf("  %-28s %-22s : %7.1f GB/s   (%.3f ms / matrix)\n",
             names[mode],
             use_copies==1 ? "1 copy (L2-RESIDENT)" : "multi-copy (HBM)",
             best, per/1e9/best*1e3);
    }
  }
  return 0;
}
