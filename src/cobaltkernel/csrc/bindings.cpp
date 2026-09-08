// bindings.cpp -- torch extension glue for the Gemma3 decode megakernel.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

#include "megakernel.cuh"

namespace cbk {
int mega_plan(int M, int KG, int smem_bytes, int* blocks_out);
int mega_launch(Args a, int M, int KG, int blocks, int smem_bytes, cudaStream_t stream);
void mega_set_minb(int v);
}  // namespace cbk

namespace {

template <typename T>
T* dp(const torch::Tensor& t) {
  return reinterpret_cast<T*>(t.data_ptr());
}

class MegaRunner {
 public:
  cbk::Args a{};
  int M = 1, blocks = 0, smem = 0, blocks_per_sm = 0, kg = 2;
  int64_t prefetch_o = 1;

  void configure(torch::Tensor layers, std::vector<int64_t> embed, torch::Tensor final_norm,
                 torch::Tensor inv_local, torch::Tensor inv_global, torch::Tensor rope_cs,
                 torch::Tensor rope_sn, torch::Tensor kcache,
                 torch::Tensor vcache, torch::Tensor tokens, torch::Tensor positions,
                 torch::Tensor h, torch::Tensor h2, torch::Tensor qkv, torch::Tensor attn_out,
                 torch::Tensor obuf, torch::Tensor act, torch::Tensor dbuf,
                 torch::Tensor partials, torch::Tensor logits, torch::Tensor amax_val,
                 torch::Tensor amax_idx, torch::Tensor xbuf, torch::Tensor rsums,
                 std::vector<int64_t> iv, std::vector<double> fv,
                 int64_t smem_bytes, int64_t minb) {
    cbk::mega_set_minb((int)minb);
    TORCH_CHECK(iv.size() == 15, "ints must have 15 entries");
    TORCH_CHECK(fv.size() == 3, "floats must have 3 entries");
    a.layers = dp<cbk::LayerW>(layers);
    TORCH_CHECK(embed.size() == 9, "embed MatDesc must have 9 int64 fields");
    {
      cbk::MatDesc d;
      d.data      = reinterpret_cast<const uint8_t*>(embed[0]);
      d.row_off   = reinterpret_cast<const uint32_t*>(embed[1]);
      d.scale     = reinterpret_cast<const __half*>(embed[2]);
      d.zero      = reinterpret_cast<const uint8_t*>(embed[3]);
      d.col_scale = reinterpret_cast<const __half*>(embed[4]);
      d.K = embed[5]; d.N = embed[6]; d.G = embed[7]; d.layout = embed[8];
      a.embed = d;
    }
    a.final_norm = dp<__nv_bfloat16>(final_norm);
    a.inv_local = dp<float>(inv_local);
    a.inv_global = dp<float>(inv_global);
    a.rope_cs = dp<float>(rope_cs);
    a.rope_sn = dp<float>(rope_sn);
    a.kcache = dp<__nv_bfloat16>(kcache);
    a.vcache = dp<__nv_bfloat16>(vcache);
    a.tokens = dp<int>(tokens);
    a.positions = dp<int>(positions);
    a.h = dp<__nv_bfloat16>(h);
    a.h2 = dp<__nv_bfloat16>(h2);
    a.qkv = dp<__nv_bfloat16>(qkv);
    a.attn_out = dp<__nv_bfloat16>(attn_out);
    a.obuf = dp<__nv_bfloat16>(obuf);
    a.act = dp<__nv_bfloat16>(act);
    a.dbuf = dp<__nv_bfloat16>(dbuf);
    a.partials = dp<float>(partials);
    a.logits = dp<float>(logits);
    a.amax_val = dp<float>(amax_val);
    a.amax_idx = dp<int>(amax_idx);
    a.xbuf = dp<__nv_bfloat16>(xbuf);
    a.rsums = dp<float>(rsums);
    a.dbg_h = nullptr;
    a.timings = nullptr;
    int i = 0;
    a.hidden = (int)iv[i++]; a.n_layers = (int)iv[i++]; a.n_heads = (int)iv[i++];
    a.n_kv = (int)iv[i++]; a.head_dim = (int)iv[i++]; a.inter = (int)iv[i++];
    a.vocab = (int)iv[i++]; a.M = (int)iv[i++]; a.max_ctx = (int)iv[i++];
    a.sliding_window = (int)iv[i++]; a.nq_dim = (int)iv[i++]; a.nkv_dim = (int)iv[i++];
    a.nqkv = (int)iv[i++]; a.kv_group = (int)iv[i++]; a.xcap = (int)iv[i++];
    a.eps = (float)fv[0]; a.attn_scale = (float)fv[1]; a.embed_scale = (float)fv[2];
    a.split = 1;
    a.prefetch_o = (int)prefetch_o;
    M = a.M;
    smem = (int)smem_bytes;
    kg = a.kv_group;
    TORCH_CHECK(kg == 1 || kg == 2 || kg == 4, "kv_group must be 1, 2 or 4; got ", kg);
    int r = cbk::mega_plan(M, kg, smem, &blocks);
    TORCH_CHECK(r > 0, "mega_plan failed (M must be 1,2,4 or 8); rc=", r);
    blocks_per_sm = r;
  }

  int64_t num_blocks() const { return blocks; }
  int64_t num_blocks_per_sm() const { return blocks_per_sm; }

  void step(std::vector<int64_t> tok, std::vector<int64_t> pos, int64_t split,
            c10::optional<torch::Tensor> dbg_h, c10::optional<torch::Tensor> timings) {
    cbk::Args aa = a;
    TORCH_CHECK((int)tok.size() == M && (int)pos.size() == M, "tok/pos must be M long");
    for (int i = 0; i < M; ++i) { aa.tok_v[i] = (int)tok[i]; aa.pos_v[i] = (int)pos[i]; }
    aa.split = (int)split;
    aa.prefetch_o = (int)prefetch_o;
    aa.dbg_h = dbg_h.has_value() ? dp<__nv_bfloat16>(*dbg_h) : nullptr;
    aa.timings = timings.has_value() ? dp<long long>(*timings) : nullptr;
    cudaStream_t s = c10::cuda::getCurrentCUDAStream();
    int rc = cbk::mega_launch(aa, M, kg, blocks, smem, s);
    TORCH_CHECK(rc == 0, "mega_launch failed rc=", rc);
    C10_CUDA_CHECK(cudaGetLastError());
  }
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<MegaRunner>(m, "MegaRunner")
      .def(py::init<>())
      .def("configure", &MegaRunner::configure)
      .def("step", &MegaRunner::step, py::arg("tokens"), py::arg("positions"),
           py::arg("split"), py::arg("dbg_h") = py::none(),
           py::arg("timings") = py::none())
      .def_readwrite("prefetch_o", &MegaRunner::prefetch_o)
      .def_readwrite("blocks", &MegaRunner::blocks)
      .def("num_blocks", &MegaRunner::num_blocks)
      .def("num_blocks_per_sm", &MegaRunner::num_blocks_per_sm);
  m.attr("LAYERW_INT64_FIELDS") = 44;
}
