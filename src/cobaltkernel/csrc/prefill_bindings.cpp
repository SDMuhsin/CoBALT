// prefill_bindings.cpp -- torch extension glue for the Gemma3 PREFILL megakernel.
// Separate extension module (`cbk_prefill`) from the decode megakernel's bindings.cpp.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

#include "prefill_kernel.cuh"

namespace pf {
int pf_smem_bytes(int head_dim);
int pf_plan(int smem_bytes, int* blocks_out);
int pf_launch(PArgs a, int blocks, int smem_bytes, cudaStream_t stream);
int pf_smem_bytes_batch(int head_dim);
int pf_plan_batch(int smem_bytes, int* blocks_out);
int pf_launch_batch(PArgs a, int blocks, int smem_bytes, cudaStream_t stream);
}  // namespace pf

namespace {

template <typename T>
T* dp(const torch::Tensor& t) { return reinterpret_cast<T*>(t.data_ptr()); }

pf::MatDesc desc_of(const std::vector<int64_t>& v) {
  TORCH_CHECK(v.size() == 9, "MatDesc needs 9 int64 fields");
  pf::MatDesc d;
  d.data      = reinterpret_cast<const uint8_t*>(v[0]);
  d.row_off   = reinterpret_cast<const uint32_t*>(v[1]);
  d.scale     = reinterpret_cast<const __half*>(v[2]);
  d.zero      = reinterpret_cast<const uint8_t*>(v[3]);
  d.col_scale = reinterpret_cast<const __half*>(v[4]);
  d.K = v[5]; d.N = v[6]; d.G = v[7]; d.layout = v[8];
  return d;
}

class PrefillRunner {
 public:
  pf::PArgs a{};
  int blocks = 0, smem = 0, blocks_per_sm = 0;
  int bblocks = 0, bsmem = 0, bblocks_per_sm = 0;

  void configure(torch::Tensor layers, std::vector<int64_t> embed, torch::Tensor final_norm,
                 torch::Tensor inv_local, torch::Tensor inv_global, torch::Tensor rope_cs,
                 torch::Tensor rope_sn, torch::Tensor kcache, torch::Tensor vcache,
                 torch::Tensor tokens, torch::Tensor positions, torch::Tensor h, torch::Tensor xn, torch::Tensor qkvb,
                 torch::Tensor qb, torch::Tensor ao, torch::Tensor ob, torch::Tensor act,
                 torch::Tensor db, torch::Tensor hg, torch::Tensor logits,
                 torch::Tensor logit_rows, std::vector<int64_t> iv,
                 std::vector<double> fv) {
    TORCH_CHECK(iv.size() == 18, "ints must have 18 entries, got ", iv.size());
    TORCH_CHECK(fv.size() == 3, "floats must have 3 entries");
    a.layers = dp<pf::PLayerW>(layers);
    a.embed = desc_of(embed);
    a.final_norm = dp<__nv_bfloat16>(final_norm);
    a.inv_local = dp<float>(inv_local);
    a.inv_global = dp<float>(inv_global);
    a.rope_cs = dp<float>(rope_cs);
    a.rope_sn = dp<float>(rope_sn);
    a.kcache = dp<__nv_bfloat16>(kcache);
    a.vcache = dp<__nv_bfloat16>(vcache);
    a.tokens = dp<int>(tokens);
    a.positions = dp<int>(positions);
    a.h = dp<__nv_bfloat16>(h);   a.xn = dp<__nv_bfloat16>(xn);
    a.qkvb = dp<__nv_bfloat16>(qkvb); a.qb = dp<__nv_bfloat16>(qb);
    a.ao = dp<__nv_bfloat16>(ao); a.ob = dp<__nv_bfloat16>(ob);
    a.act = dp<__nv_bfloat16>(act); a.db = dp<__nv_bfloat16>(db);
    a.hg = dp<__nv_bfloat16>(hg);
    a.logits = dp<float>(logits);
    a.logit_rows = dp<int>(logit_rows);
    a.timings = nullptr;
    int i = 0;
    a.hidden = (int)iv[i++]; a.n_layers = (int)iv[i++]; a.n_heads = (int)iv[i++];
    a.n_kv = (int)iv[i++]; a.head_dim = (int)iv[i++]; a.inter = (int)iv[i++];
    a.vocab = (int)iv[i++]; a.M = (int)iv[i++]; a.R = (int)iv[i++];
    a.pos0 = (int)iv[i++]; a.kv_b = (int)iv[i++]; a.seq = (int)iv[i++];
    a.max_ctx = (int)iv[i++]; a.sliding_window = (int)iv[i++];
    a.nq_dim = (int)iv[i++]; a.nkv_dim = (int)iv[i++]; a.nqkv = (int)iv[i++];
    a.kv_group = (int)iv[i++];
    a.batch_mode = 0;
    a.eps = (float)fv[0]; a.attn_scale = (float)fv[1]; a.embed_scale = (float)fv[2];
    smem = pf::pf_smem_bytes(a.head_dim);
    int r = pf::pf_plan(smem, &blocks);
    TORCH_CHECK(r > 0, "pf_plan failed rc=", r);
    blocks_per_sm = r;
    bsmem = pf::pf_smem_bytes_batch(a.head_dim);
    int rb = pf::pf_plan_batch(bsmem, &bblocks);
    TORCH_CHECK(rb > 0, "pf_plan_batch failed rc=", rb);
    bblocks_per_sm = rb;
  }

  void set_run(int64_t M, int64_t R, int64_t pos0, int64_t seq) {
    a.M = (int)M; a.R = (int)R; a.pos0 = (int)pos0; a.seq = (int)seq;
  }

  int64_t num_blocks() const { return blocks; }
  int64_t num_blocks_per_sm() const { return blocks_per_sm; }
  int64_t smem_bytes() const { return smem; }
  int64_t num_blocks_batch() const { return bblocks; }
  int64_t num_blocks_per_sm_batch() const { return bblocks_per_sm; }
  int64_t smem_bytes_batch() const { return bsmem; }

  void run(c10::optional<torch::Tensor> timings) {
    pf::PArgs aa = a;
    aa.batch_mode = 0;
    aa.timings = timings.has_value() ? dp<long long>(*timings) : nullptr;
    cudaStream_t s = c10::cuda::getCurrentCUDAStream();
    int rc = pf::pf_launch(aa, blocks, smem, s);
    TORCH_CHECK(rc == 0, "pf_launch failed rc=", rc);
    C10_CUDA_CHECK(cudaGetLastError());
  }

  // One decode step for M INDEPENDENT sequences (row m = sequence m at positions[m]).
  void run_batch(c10::optional<torch::Tensor> timings) {
    pf::PArgs aa = a;
    aa.batch_mode = 1;
    aa.timings = timings.has_value() ? dp<long long>(*timings) : nullptr;
    cudaStream_t s = c10::cuda::getCurrentCUDAStream();
    int rc = pf::pf_launch_batch(aa, bblocks, bsmem, s);
    TORCH_CHECK(rc == 0, "pf_launch_batch failed rc=", rc);
    C10_CUDA_CHECK(cudaGetLastError());
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<PrefillRunner>(m, "PrefillRunner")
      .def(py::init<>())
      .def("configure", &PrefillRunner::configure)
      .def("set_run", &PrefillRunner::set_run)
      .def("run", &PrefillRunner::run, py::arg("timings") = py::none())
      .def("run_batch", &PrefillRunner::run_batch, py::arg("timings") = py::none())
      .def("num_blocks_batch", &PrefillRunner::num_blocks_batch)
      .def("num_blocks_per_sm_batch", &PrefillRunner::num_blocks_per_sm_batch)
      .def("smem_bytes_batch", &PrefillRunner::smem_bytes_batch)
      .def_readwrite("blocks", &PrefillRunner::blocks)
      .def("num_blocks", &PrefillRunner::num_blocks)
      .def("num_blocks_per_sm", &PrefillRunner::num_blocks_per_sm)
      .def("smem_bytes", &PrefillRunner::smem_bytes);
  m.attr("PLAYERW_INT64_FIELDS") = 44;
}
