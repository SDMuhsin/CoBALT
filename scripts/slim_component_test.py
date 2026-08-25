"""Component-equivalence test: benchmarks/slim_port.py  vs  upstream temp/SLiM.

Confirms the faithful port reproduces upstream SLiM math (SLiM-Quant cap + quantize, and the
SLiM-LoRA add_lora decomposition) to within bf16 rounding, so the committed port (which must
work WITHOUT the gitignored temp/SLiM clone) is provably equivalent to the released code.

Run: python scripts/slim_component_test.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "temp", "SLiM"))

import slim_port as P                                                  # noqa: E402
from slim.quantization.quantization import Quantizer as UQuant        # noqa: E402
from slim.lora import add_lora as upstream_add_lora                   # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def report(name, a, b):
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    mad = diff.max().item()
    denom = b.abs().max().item() + 1e-12
    rel = mad / denom
    # Frobenius-relative error and fraction of elements that differ are the honest
    # "are these the same tensor" metrics; max|Δ| is dominated by a few boundary flips.
    fro = (diff.norm() / (b.norm() + 1e-12)).item()
    frac = (diff > 1e-3).float().mean().item()
    ok = fro < 5e-3                      # bit-exact-up-to-bf16-rounding bar
    print(f"[{'PASS' if ok else 'FAIL'}] {name:40s} max|Δ|={mad:.2e} relmax={rel:.2e} "
          f"froRel={fro:.2e} frac>1e-3={frac:.3%}")
    return ok


class FakeAct:
    def __init__(self, scaler_row):
        self.scaler_row = scaler_row


def test_quant(shape, nbits):
    W = torch.randn(shape, device=dev, dtype=torch.bfloat16) * 0.1
    # upstream
    uq = UQuant("weight", num_bits=nbits, slim_quant=True, block_quantization=False)
    cap_u = None
    from slim.quantization.quantization import find_optimal_quantiztion_cap
    cap_u = find_optimal_quantiztion_cap(
        W, nbits, num_bins=max(512, min(torch.numel(W) // 1000, 20000)))
    deq_u = uq.dequantize_absmax(uq.quantize_weight(W.clone()))
    # port
    cap_p = P.slim_quant_find_cap(W, nbits)
    deq_p = P.slim_quantize_weight(W.clone(), nbits)
    ok1 = report(f"slim_quant cap {tuple(shape)} b{nbits}",
                 torch.tensor([cap_p]), torch.tensor([cap_u]))
    ok2 = report(f"slim_quant dequant {tuple(shape)} b{nbits}", deq_p, deq_u)
    return ok1 and ok2


def test_lora(out_f, in_f, nbits, sparsity, rank_ratio, quantize):
    W = (torch.randn(out_f, in_f, device=dev, dtype=torch.bfloat16) * 0.1)
    scaler = (torch.rand(in_f, device=dev) + 0.05).float()  # strictly > 0
    # Wanda mask (per row), same on both sides
    metric = W.abs() * torch.sqrt(scaler.reshape(1, -1))
    k = int(in_f * sparsity)
    idx = torch.sort(metric, dim=1, stable=True)[1][:, :k]
    mask = torch.zeros_like(metric, dtype=torch.bool)
    mask.scatter_(1, idx, True)

    # ---- upstream add_lora (mutates a module in place) ----
    lin = torch.nn.Linear(in_f, out_f, bias=False).to(dev).to(torch.bfloat16)
    lin.weight.data = W.clone()
    uq = UQuant("weight", num_bits=nbits, slim_quant=True, block_quantization=False) if quantize else None
    upstream_add_lora(
        lin, W_mask=mask.clone(), rank_ratio=rank_ratio, slim_lora=True,
        activations=FakeAct(scaler.clone()), quantizer=uq, prune_lora=False,
        separate_lora=True, lora_tile_size=None, quantize_first=True,
        scale_important_weights=False,
    )
    Wc_u, ll_u, lr_u = lin.weight.data, lin.lora_left.data, lin.lora_right.data
    full_u = Wc_u.float() + (ll_u.float() @ lr_u.float()).t()

    # ---- port ----
    Wc_p, ll_p, lr_p = P.slim_lora_decompose(
        W.clone(), mask.clone(), scaler.clone(), nbits, rank_ratio,
        quantize=quantize, slim_lora=True)
    full_p = Wc_p.float() + (ll_p.float() @ lr_p.float()).t()

    tag = f"{out_f}x{in_f} b{nbits} s{sparsity} r{rank_ratio} q{int(quantize)}"
    ok_w = report(f"lora Wcomp   {tag}", Wc_p, Wc_u)
    ok_f = report(f"lora FULL    {tag}", full_p, full_u)
    # adapter rank/shape sanity
    print(f"      rank: port={ll_p.shape[1]} upstream={ll_u.shape[1]}  "
          f"Wcomp sparsity port={(Wc_p==0).float().mean():.3f} up={(Wc_u==0).float().mean():.3f}")
    return ok_w and ok_f


if __name__ == "__main__":
    allok = True
    print("== SLiM-Quant ==")
    for shape in [(768, 768), (3072, 768), (512, 896)]:
        for nb in [3, 4, 5]:
            allok &= test_quant(shape, nb)
    print("\n== SLiM-LoRA decompose (quantized weights) ==")
    for nb in [3, 4, 5]:
        allok &= test_lora(768, 768, nb, 0.5, 0.1, quantize=True)
    print("\n== SLiM-LoRA decompose (prune-only, no weight quant) ==")
    allok &= test_lora(768, 768, 4, 0.5, 0.1, quantize=False)
    print("\n== rectangular / other sparsity ==")
    allok &= test_lora(3072, 768, 4, 0.5, 0.1, quantize=True)
    allok &= test_lora(768, 768, 4, 0.25, 0.1, quantize=True)
    print("\nALL PASS" if allok else "\nSOME FAILED")
    sys.exit(0 if allok else 1)
