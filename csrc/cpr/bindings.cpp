// CPR-SINQ: Python Bindings using PyBind11
//
// This file creates the Python module interface for CPR kernels

#include <torch/extension.h>
#include "cpr.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CPR-SINQ: Column-Precision Reordering CUDA kernels";

    // Bit packing functions
    m.def("pack_6bit", &cpr::pack_6bit,
          "Pack int8 weights to 6-bit representation",
          py::arg("weights"));

    m.def("unpack_6bit", &cpr::unpack_6bit,
          "Unpack 6-bit weights to int8",
          py::arg("packed"), py::arg("n_cols"));

    m.def("pack_5bit", &cpr::pack_5bit,
          "Pack int8 weights to 5-bit representation",
          py::arg("weights"));

    m.def("unpack_5bit", &cpr::unpack_5bit,
          "Unpack 5-bit weights to int8",
          py::arg("packed"), py::arg("n_cols"));

    // Dequantization functions
    m.def("dequantize_6bit", &cpr::dequantize_6bit,
          "Dequantize 6-bit packed weights to FP16",
          py::arg("packed"), py::arg("scales"), py::arg("zeros"),
          py::arg("n_cols"), py::arg("tile_size"));

    m.def("dequantize_5bit", &cpr::dequantize_5bit,
          "Dequantize 5-bit packed weights to FP16",
          py::arg("packed"), py::arg("scales"), py::arg("zeros"),
          py::arg("n_cols"), py::arg("tile_size"));

    // Matrix multiplication
    m.def("cpr_matmul", &cpr::cpr_matmul,
          "CPR quantized matrix multiplication Y = X @ W^T",
          py::arg("X"),
          py::arg("W_high_packed"), py::arg("W_low_packed"),
          py::arg("scales_high"), py::arg("zeros_high"),
          py::arg("scales_low"), py::arg("zeros_low"),
          py::arg("col_indices"),
          py::arg("n_high_cols"), py::arg("n_low_cols"),
          py::arg("tile_size"));

    m.def("cpr_linear", &cpr::cpr_linear,
          "CPR quantized linear layer Y = X @ W^T + bias",
          py::arg("X"),
          py::arg("W_high_packed"), py::arg("W_low_packed"),
          py::arg("scales_high"), py::arg("zeros_high"),
          py::arg("scales_low"), py::arg("zeros_low"),
          py::arg("col_indices"),
          py::arg("bias"),
          py::arg("n_high_cols"), py::arg("n_low_cols"),
          py::arg("tile_size"));

    // Utility functions
    m.def("check_cuda", &cpr::check_cuda,
          "Print CUDA device information");

    m.def("get_optimal_tile_size", &cpr::get_optimal_tile_size,
          "Get optimal tile size for given dimensions",
          py::arg("n_rows"), py::arg("n_cols"));

    // Optimized tiled kernel
    m.def("cpr_matmul_tiled", &cpr::cpr_matmul_tiled,
          "Tiled CPR matmul with shared memory staging",
          py::arg("X"),
          py::arg("W_high_packed"), py::arg("W_low_packed"),
          py::arg("scales_high"), py::arg("zeros_high"),
          py::arg("scales_low"), py::arg("zeros_low"),
          py::arg("col_indices"),
          py::arg("n_high_cols"), py::arg("n_low_cols"),
          py::arg("tile_size"));

    // MMA (Tensor Core) kernels
    m.def("mma_matmul_simple", &cpr::mma_matmul_simple,
          "Simple MMA-based matmul for FP16 (baseline)",
          py::arg("A"), py::arg("B"));

    m.def("wmma_matmul_tiled", &cpr::wmma_matmul_tiled,
          "Tiled WMMA matmul with shared memory",
          py::arg("A"), py::arg("B"));

    m.def("wmma_matmul_pipelined", &cpr::wmma_matmul_pipelined,
          "Pipelined WMMA matmul with double buffering",
          py::arg("A"), py::arg("B"));

    m.def("wmma_matmul_optimized", &cpr::wmma_matmul_optimized,
          "Optimized WMMA with async copy and deeper pipeline",
          py::arg("A"), py::arg("B"));

    m.def("wmma_matmul_heavy", &cpr::wmma_matmul_heavy,
          "Heavy register blocking WMMA",
          py::arg("A"), py::arg("B"));

    m.def("wmma_matmul_v2", &cpr::wmma_matmul_v2,
          "WMMA v2 with vectorized loads and larger K tile",
          py::arg("A"), py::arg("B"));

    m.def("wmma_matmul_warp_special", &cpr::wmma_matmul_warp_special,
          "Warp-specialized WMMA with bank conflict avoidance",
          py::arg("A"), py::arg("B"));

    m.def("wmma_matmul_small_tile", &cpr::wmma_matmul_small_tile,
          "Small tile WMMA for higher occupancy",
          py::arg("A"), py::arg("B"));
}
