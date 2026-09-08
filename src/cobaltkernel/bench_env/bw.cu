// bw.cu -- device memory read bandwidth ceiling for a MIG slice.
//   * vectorized read-reduce (float4 / 16B per thread per step) over N GB
//   * cudaMemcpy D2D sanity reference (counts 2*bytes: read+write)
// Usage: ./bw [gib=6] [reps=5]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__global__ void read_reduce(const float4* __restrict__ p, size_t n4, float* out){
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x*blockDim.x;
  float4 acc = make_float4(0,0,0,0);
  for(; i < n4; i += stride){
    float4 v = p[i];
    acc.x+=v.x; acc.y+=v.y; acc.z+=v.z; acc.w+=v.w;
  }
  float s = acc.x+acc.y+acc.z+acc.w;
  // keep the compiler honest without a real write in the common path
  if (s == 1.2345e30f) out[0] = s;
}

int main(int argc,char**argv){
  double gib = argc>1? atof(argv[1]) : 6.0;
  int reps  = argc>2? atoi(argv[2]) : 5;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  size_t bytes = (size_t)(gib*1073741824.0) & ~(size_t)255;
  size_t n4 = bytes/16;
  printf("device: %s  SMs=%d  buffer=%.2f GiB\n", p.name, p.multiProcessorCount, bytes/1073741824.0);

  float4 *d; float* out;
  CK(cudaMalloc(&d, bytes)); CK(cudaMalloc(&out, 4));
  CK(cudaMemset(d, 1, bytes));

  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

  // sweep a couple of grid sizes; report the best
  int blk = 256;
  int best_wpsm = 0; double best = 0;
  for (int wpsm : {4, 8, 16, 32}) {
    int grid = p.multiProcessorCount*wpsm;
    read_reduce<<<grid,blk>>>(d,n4,out); CK(cudaDeviceSynchronize());   // warm
    double bg = 0;
    for(int r=0;r<reps;r++){
      CK(cudaEventRecord(a));
      read_reduce<<<grid,blk>>>(d,n4,out);
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float ms; CK(cudaEventElapsedTime(&ms,a,b));
      double gbs = bytes/1e9/(ms/1e3);
      if(gbs>bg) bg=gbs;
    }
    printf("  read-reduce grid=%5d (%2d blk/SM) : %7.1f GB/s\n", grid, wpsm, bg);
    if(bg>best){best=bg;best_wpsm=wpsm;}
  }
  printf("BEST read bandwidth      : %.1f GB/s (%d blocks/SM)\n", best, best_wpsm);

  // D2D memcpy reference (half the buffer -> other half)
  size_t h = bytes/2;
  double bmc = 0;
  for(int r=0;r<reps;r++){
    CK(cudaEventRecord(a));
    CK(cudaMemcpy(d, (char*)d + h, h, cudaMemcpyDeviceToDevice));
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms,a,b));
    double gbs = 2.0*h/1e9/(ms/1e3);
    if(gbs>bmc) bmc=gbs;
  }
  printf("cudaMemcpy D2D (r+w)     : %.1f GB/s\n", bmc);
  cudaFree(d); cudaFree(out);
  return 0;
}
