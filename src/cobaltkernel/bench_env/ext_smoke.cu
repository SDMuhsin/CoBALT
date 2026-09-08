// ext_smoke.cu -- minimal torch C++/CUDA extension: proves cpp_extension.load()
// can JIT a sm_120a kernel (incl. a bf16 mma.sync) against torch 2.13+cu130.
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAException.h>

__global__ void axpy_kernel(const __nv_bfloat16* x, __nv_bfloat16* y, float a, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2bfloat16(a*__bfloat162float(x[i]) + __bfloat162float(y[i]));
}

// tiny tensor-core probe inside the extension: 16x16x16 bf16 mma, warp 0 only
__global__ void mma_probe_kernel(float* out){
  float d[4] = {0,0,0,0};
  unsigned a[4] = {0x3f803f80u,0x3f803f80u,0x3f803f80u,0x3f803f80u}; // bf16 1.0 x8
  unsigned b[2] = {0x3f803f80u,0x3f803f80u};
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  if (threadIdx.x == 0) out[0] = d[0];
}

void axpy(torch::Tensor x, torch::Tensor y, double a){
  TORCH_CHECK(x.is_cuda() && y.is_cuda() && x.scalar_type()==torch::kBFloat16);
  int n = x.numel();
  axpy_kernel<<<(n+255)/256, 256>>>(
      (const __nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)y.data_ptr(), (float)a, n);
  C10_CUDA_CHECK(cudaGetLastError());
}

torch::Tensor mma_probe(){
  auto out = torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  mma_probe_kernel<<<1,32>>>(out.data_ptr<float>());
  C10_CUDA_CHECK(cudaGetLastError());
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("axpy", &axpy, "bf16 axpy");
  m.def("mma_probe", &mma_probe, "bf16 m16n8k16 mma.sync probe");
}
