// props.cu -- dump the device properties that constrain a persistent megakernel.
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do{ cudaError_t e=(x); if(e){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 1;} }while(0)

int main(){
  int n; CK(cudaGetDeviceCount(&n));
  for(int d=0; d<n; ++d){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,d));
    printf("=== device %d: %s ===\n", d, p.name);
    printf("compute capability      : %d.%d\n", p.major, p.minor);
    printf("multiProcessorCount     : %d\n", p.multiProcessorCount);
    printf("totalGlobalMem          : %.2f GiB\n", p.totalGlobalMem/1073741824.0);
    printf("l2CacheSize             : %d bytes (%.2f MiB)\n", p.l2CacheSize, p.l2CacheSize/1048576.0);
    printf("persistingL2CacheMaxSize: %d bytes\n", p.persistingL2CacheMaxSize);
    printf("sharedMemPerBlock       : %zu\n", p.sharedMemPerBlock);
    printf("sharedMemPerBlockOptin  : %zu\n", p.sharedMemPerBlockOptin);
    printf("sharedMemPerMultiproc   : %zu\n", p.sharedMemPerMultiprocessor);
    printf("regsPerBlock/SM         : %d / %d\n", p.regsPerBlock, p.regsPerMultiprocessor);
    printf("maxThreadsPerBlock      : %d\n", p.maxThreadsPerBlock);
    printf("maxThreadsPerMultiProc  : %d\n", p.maxThreadsPerMultiProcessor);
    printf("maxBlocksPerMultiProc   : %d\n", p.maxBlocksPerMultiProcessor);
    printf("warpSize                : %d\n", p.warpSize);
    printf("memoryBusWidth          : %d bit\n", p.memoryBusWidth);
    int mclk=0, gclk=0;
    cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, d);
    cudaDeviceGetAttribute(&gclk, cudaDevAttrClockRate, d);
    printf("memoryClockRate (attr)  : %d kHz\n", mclk);
    printf("clockRate (attr)        : %d kHz\n", gclk);
    printf("theoretical peak BW     : %.1f GB/s (full GPU, not slice)\n",
           2.0*mclk*1000.0*(p.memoryBusWidth/8.0)/1e9);
    printf("cooperativeLaunch       : %d\n", p.cooperativeLaunch);
    printf("clusterLaunch           : %d\n", p.clusterLaunch);
    printf("mpsEnabled              : %d\n", p.mpsEnabled);
    printf("globalL1CacheSupported  : %d\n", p.globalL1CacheSupported);
    printf("unifiedAddressing       : %d\n", p.unifiedAddressing);
    printf("concurrentKernels       : %d\n", p.concurrentKernels);
    printf("asyncEngineCount        : %d\n", p.asyncEngineCount);
    printf("isMultiGpuBoard         : %d\n", p.isMultiGpuBoard);
    printf("hostRegisterSupported   : %d\n", p.hostRegisterSupported);
    printf("memoryPoolsSupported    : %d\n", p.memoryPoolsSupported);
    printf("gpuDirectRDMASupported  : %d\n", p.gpuDirectRDMASupported);
    printf("hostNativeAtomicSupported: %d\n", p.hostNativeAtomicSupported);
    int v;
    CK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, d));
    printf("attr MaxSmemPerBlockOptin: %d\n", v);
    CK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxBlocksPerMultiprocessor, d));
    printf("attr MaxBlocksPerSM     : %d\n", v);
    CK(cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d));
    printf("attr MultiProcessorCount: %d\n", v);
    CK(cudaDeviceGetAttribute(&v, cudaDevAttrCooperativeLaunch, d));
    printf("attr CooperativeLaunch  : %d\n", v);
    CK(cudaDeviceGetAttribute(&v, cudaDevAttrClusterLaunch, d));
    printf("attr ClusterLaunch      : %d\n", v);
    int rt, dr; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&dr);
    printf("runtime / driver CUDA   : %d / %d\n", rt, dr);
  }
  return 0;
}
