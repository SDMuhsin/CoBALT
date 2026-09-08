// coop.cu -- can we run a PERSISTENT cooperative megakernel on a MIG slice?
//   * cudaLaunchCooperativeKernel + cg::grid_group::sync()
//   * co-resident blocks/SM for 256 threads with a given dynamic smem budget
//   * cost of one grid.sync() (loop of N syncs, timed)
//   * empty-kernel per-launch overhead for comparison
// Usage: ./coop [smem_kb=48] [threads=256] [iters=1000]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__global__ void sync_loop(int iters, int* flag){
  cg::grid_group g = cg::this_grid();
  extern __shared__ char sm[];
  if (threadIdx.x == 0) sm[0] = (char)blockIdx.x;   // touch smem so it is not elided
  for (int i=0;i<iters;i++) g.sync();
  if (blockIdx.x==0 && threadIdx.x==0) *flag = iters + (int)sm[0];
}
__global__ void nosync_loop(int iters, int* flag){
  extern __shared__ char sm[];
  if (threadIdx.x == 0) sm[0] = (char)blockIdx.x;
  int a = 0;
  for (int i=0;i<iters;i++) a = a*1664525 + 1013904223;
  if (blockIdx.x==0 && threadIdx.x==0 && a==7) *flag = a + (int)sm[0];
}
__global__ void empty_kernel(){}

int main(int argc,char**argv){
  int smem_kb = argc>1? atoi(argv[1]) : 48;
  int thr     = argc>2? atoi(argv[2]) : 256;
  int iters   = argc>3? atoi(argv[3]) : 1000;
  size_t smem = (size_t)smem_kb*1024;
  if (smem < 4) smem = 4;   // extern __shared__ needs a non-empty allocation

  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
  int coop=0; CK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0));
  printf("device: %s\nSMs=%d  cooperativeLaunch=%d  smemPerSM=%zu  optinPerBlock=%zu\n",
         p.name, p.multiProcessorCount, coop, p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin);
  printf("config: %d threads/block, %d KB dynamic smem, %d syncs\n", thr, smem_kb, iters);

  if (smem > p.sharedMemPerBlock)
    CK(cudaFuncSetAttribute(sync_loop, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  if (smem > p.sharedMemPerBlock)
    CK(cudaFuncSetAttribute(nosync_loop, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  int bpsm=0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, sync_loop, thr, smem));
  printf("occupancy blocks/SM      : %d  -> max co-resident blocks = %d (threads=%d)\n",
         bpsm, bpsm*p.multiProcessorCount, bpsm*p.multiProcessorCount*thr);

  int maxgrid=0;
  maxgrid = bpsm * p.multiProcessorCount;
  if (maxgrid == 0) { printf("FATAL: zero co-resident blocks at this smem budget\n"); return 1; }

  int* flag; CK(cudaMalloc(&flag,4));
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));

  for (int frac = 1; frac <= 2; ++frac) {
    int grid = (frac==1) ? maxgrid : p.multiProcessorCount;   // full occupancy, then 1 block/SM
    void* args[] = {(void*)&iters, (void*)&flag};
    // warm
    CK(cudaLaunchCooperativeKernel((void*)sync_loop, dim3(grid), dim3(thr), args, smem, 0));
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("COOPERATIVE LAUNCH FAILED grid=%d: %s\n", grid, cudaGetErrorString(e)); return 1; }
    double best_s = 1e30, best_ns = 1e30;
    for (int r=0;r<5;r++){
      CK(cudaEventRecord(a));
      CK(cudaLaunchCooperativeKernel((void*)sync_loop, dim3(grid), dim3(thr), args, smem, 0));
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float ms; CK(cudaEventElapsedTime(&ms,a,b)); if(ms<best_s) best_s=ms;
      CK(cudaEventRecord(a));
      nosync_loop<<<grid,thr,smem>>>(iters, flag);
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      CK(cudaEventElapsedTime(&ms,a,b)); if(ms<best_ns) best_ns=ms;
    }
    printf("grid=%4d (%2d blk/SM): coop kernel %8.3f ms | no-sync ref %7.3f ms | per grid.sync = %6.3f us\n",
           grid, grid/p.multiProcessorCount, best_s, best_ns, (best_s-best_ns)*1000.0/iters);
    printf("                      raw (sync-loop time / iters)                        = %6.3f us\n",
           best_s*1000.0/iters);
  }

  // empty-kernel launch overhead: back-to-back launches on a stream, then sync
  {
    int N=1000; double best=1e30;
    empty_kernel<<<1,32>>>(); CK(cudaDeviceSynchronize());
    for(int r=0;r<5;r++){
      CK(cudaEventRecord(a));
      for(int i=0;i<N;i++) empty_kernel<<<p.multiProcessorCount,thr>>>();
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float ms; CK(cudaEventElapsedTime(&ms,a,b)); if(ms<best) best=ms;
    }
    printf("empty kernel back-to-back launch (grid=%d,thr=%d): %.3f us / launch\n",
           p.multiProcessorCount, thr, best*1000.0/N);
    best=1e30;
    for(int r=0;r<200;r++){
      CK(cudaEventRecord(a));
      empty_kernel<<<p.multiProcessorCount,thr>>>();
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float ms; CK(cudaEventElapsedTime(&ms,a,b)); if(ms<best) best=ms;
    }
    printf("empty kernel launch+sync round trip                : %.3f us\n", best*1000.0);
  }
  return 0;
}
