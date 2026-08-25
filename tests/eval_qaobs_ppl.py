"""
Full PPL evaluation of QAOBS (Quantization-Aware OBS).
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import (
    compute_activation_norms,
    create_sparsity_mask,
    compute_hessian_inverse,
)


class QAOBSQuantLinear(nn.Module):
    """QAOBS Quantized Linear layer."""

    def __init__(self, W_q, scales, zeros, mask, scale2, bias, meta):
        super().__init__()
        self.register_buffer('W_q', W_q)
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mask', mask)
        self.register_buffer('scale2', scale2)
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self.meta = meta

    def forward(self, x):
        input_dtype = x.dtype
        Q = self.W_q.float()
        z = self.zeros.float()
        s1 = self.scales.float()
        s2 = self.scale2.float()

        if len(s1.shape) == 3:
            K, N = Q.shape
            n_groups = s1.shape[1]
            group_size = N // n_groups
            Q_grouped = Q.view(K, n_groups, group_size)
            W_deq = (Q_grouped - z) * s1
            W_deq = W_deq.view(K, N)
        else:
            W_deq = (Q - z) * s1

        W_deq = W_deq * s2
        W_deq = W_deq * self.mask.float()
        W_deq = W_deq.to(input_dtype)

        out = x @ W_deq.T
        if self.bias is not None:
            out = out + self.bias
        return out


def quantize_and_dequant_full(W_norm, nbits, group_size, mu1, mu2, mask):
    """Quantize and return both Q and dequantized weights."""
    K, N = W_norm.shape
    min_max = [0, 2**nbits - 1]

    Q, scales, zeros, _ = quantize_rtn(W_norm * mask, min_max, group_size=group_size)

    # Combine scales with mu2
    if len(scales.shape) == 3:
        scales_combined = scales * mu2.unsqueeze(1)
    else:
        scales_combined = scales * mu2

    # Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2 * mu1 * mask

    return Q, scales_combined, zeros, W_deq


def qaobs_sparse_quantize(W, activations, sparsity, nbits, group_size=64, n_iter=3, device='cuda'):
    """
    QAOBS sparse quantization.
    """
    orig_dtype = W.dtype
    W = W.float().to(device)
    K, N = W.shape

    # Sinkhorn normalize
    W_norm_orig, mu1, mu2 = sinkhorn_log(W, order=16)

    # Compute importance
    if activations is not None:
        act_norms = compute_activation_norms(activations.to(device))
        importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2
    else:
        importance = W.abs() * mu1.unsqueeze(0) * mu2

    # Create mask
    mask = create_sparsity_mask(importance, sparsity)

    # Prepare for QAOBS iterations
    X = activations.float().to(device) if activations is not None else None
    if X is not None:
        if X.dim() == 3:
            X = X.view(-1, X.shape[-1])
        n_samples = min(X.shape[0], 256)
        X_sub = X[:n_samples]

        # Precompute target output and Hessian inverse
        target_out = X_sub @ W.T

        H = X_sub.T @ X_sub
        damping = max(0.01 * H.diag().mean().item(), 1e-3)
        H = H + damping * torch.eye(N, device=device, dtype=H.dtype)
        H_inv = torch.linalg.inv(H)

        # Start with simple pruned weights
        W_current = W * mask

        for it in range(n_iter):
            # Quantize current
            W_norm_current = W_current / (mu2 * mu1)
            Q, scales, zeros, W_deq = quantize_and_dequant_full(
                W_norm_current, nbits, group_size, mu1, mu2, mask
            )

            # Compute residual
            quant_out = X_sub @ W_deq.T
            residual = target_out - quant_out

            # Compute correction
            delta_W = torch.zeros_like(W)
            for i in range(K):
                r_i = residual[:, i]
                grad = X_sub.T @ r_i
                delta_w = H_inv @ grad
                delta_W[i] = delta_w * mask[i]

            # Apply damped correction
            W_current = W_current + delta_W * 0.5
            W_current = W_current.clamp(-W.abs().max() * 2, W.abs().max() * 2)

        # Final quantization
        W_norm_final = W_current / (mu2 * mu1)
        Q, scales, zeros, W_deq = quantize_and_dequant_full(
            W_norm_final, nbits, group_size, mu1, mu2, mask
        )
    else:
        # No activations - just simple sparse quantization
        W_sparse = W * mask
        W_norm_sparse = W_sparse / (mu2 * mu1)
        Q, scales, zeros, W_deq = quantize_and_dequant_full(
            W_norm_sparse, nbits, group_size, mu1, mu2, mask
        )

    meta = {
        'sparsity': sparsity,
        'nbits': nbits,
        'group_size': group_size,
        'n_iter': n_iter,
        'method': 'qaobs',
    }

    return Q.to(orig_dtype), scales.to(orig_dtype), zeros.to(orig_dtype), \
           mask.to(orig_dtype), mu1.to(orig_dtype), meta


def apply_qaobs_quantization(model, calibration_data, nbits, sparsity, n_iter=3, device='cuda'):
    """Apply QAOBS quantization to model."""
    from benchmarks.benchmark_suite import collect_activations, get_layer_paths

    layer_activations = collect_activations(model, calibration_data, device)
    layer_paths = get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(model.model.layers, desc=f"QAOBS {nbits}b/{int(sparsity*100)}%")):
        layer = layer.to(device)

        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue

            if not isinstance(linear, nn.Linear):
                continue

            W = linear.weight.data.clone()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            act_key = f'layer_{layer_idx}.{attr_path}'
            activations = layer_activations.get(act_key, None)
            if activations is not None:
                activations = activations.to(device)

            W_q, scales, zeros, mask, scale2, meta = qaobs_sparse_quantize(
                W, activations,
                sparsity=sparsity,
                nbits=nbits,
                n_iter=n_iter,
                device=device
            )

            new_layer = QAOBSQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()

        model.model.layers[layer_idx] = layer
        gc.collect()
        torch.cuda.empty_cache()

    return model


def evaluate_ppl(model, test_data, device='cuda', seq_len=2048):
    """Evaluate perplexity."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(test_data) - seq_len, seq_len), desc="Evaluating"):
            batch = test_data[i:i+seq_len].unsqueeze(0).to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            n_tokens = batch.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

            if (i // seq_len + 1) % 4 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


def main():
    print("="*60)
    print("QAOBS Full PPL Evaluation")
    print("="*60)

    device = 'cuda'
    model_name = "Qwen/Qwen2.5-0.5B"
    sparsity = 0.35
    nbits = 3

    print(f"\nConfig: {model_name}, {sparsity*100:.0f}% sparsity, {nbits}-bit")

    # Load data
    print("\nLoading tokenizer and data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_text = '\n\n'.join(dataset['text'])
    test_data = tokenizer.encode(test_text, return_tensors='pt')[0]

    calib_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    calib_text = '\n\n'.join(calib_dataset['text'][:1000])
    calibration_data = tokenizer.encode(calib_text, return_tensors='pt')[0][:8192]

    print(f"Test tokens: {len(test_data)}")

    # Test different n_iter values
    results = {}
    for n_iter in [1, 3]:
        print(f"\n{'='*60}")
        print(f"Testing QAOBS with n_iter={n_iter}")
        print("="*60)

        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        # Apply QAOBS
        print(f"\nApplying QAOBS (n_iter={n_iter})...")
        model = apply_qaobs_quantization(
            model, calibration_data, nbits, sparsity, n_iter=n_iter, device=device
        )

        # Evaluate
        print("\nEvaluating PPL...")
        ppl = evaluate_ppl(model, test_data, device)
        results[n_iter] = ppl

        print(f"\nQAOBS (n_iter={n_iter}) PPL: {ppl:.2f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Baseline SINQ-Sparse PPL: 28.10")
    for n_iter, ppl in results.items():
        improvement = (28.10 - ppl) / 28.10 * 100
        status = "BETTER" if ppl < 28.10 else "WORSE"
        print(f"QAOBS (n_iter={n_iter}) PPL: {ppl:.2f} ({improvement:+.2f}%) [{status}]")


if __name__ == '__main__':
    main()
