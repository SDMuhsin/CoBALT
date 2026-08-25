"""
Debug MWC PPL - identify where the quantization pipeline breaks
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import compute_hessian_inverse
from datasets import load_dataset


def quick_ppl(model, tokenizer, max_samples=10):
    """Quick PPL evaluation on a few samples."""
    model.eval()
    device = next(model.parameters()).device
    test_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, sample in enumerate(test_dataset):
            if i >= max_samples:
                break
            text = sample['text']
            if len(text.strip()) < 10:
                continue
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            if input_ids.shape[1] < 2:
                continue
            outputs = model(input_ids, labels=input_ids)
            total_loss += outputs.loss.item() * (input_ids.shape[1] - 1)
            total_tokens += input_ids.shape[1] - 1

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


class DebugQuantizedLinear(torch.nn.Module):
    """Simplified quantized linear for debugging."""

    def __init__(self, W, X_calib, nbits=4, group_size=64, sparsity=0.35, use_mwc=False):
        super().__init__()
        self.nbits = nbits
        self.group_size = group_size

        W = W.float()
        K, N = W.shape
        n_groups = N // group_size
        min_max = [0, 2**nbits - 1]
        n_prune = int(K * N * sparsity)

        # Sinkhorn
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        self.register_buffer('mu1', mu1)
        self.register_buffer('mu2', mu2)

        # Importance and mask
        X_calib = X_calib.float()
        act_norms = torch.norm(X_calib, dim=0)
        importance = W_norm.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)
        threshold = importance.view(-1).sort().values[n_prune]
        mask = (importance > threshold).float()
        pruned_mask = 1 - mask

        # Hessian
        n_samples = min(X_calib.shape[0], 256)
        X_sample = X_calib[:n_samples]
        H_inv = compute_hessian_inverse(X_sample, damping=None)
        H_inv_diag = H_inv.diag()

        # Apply compensation
        W_sparse = W_norm * mask
        W_comp = W_sparse.clone()

        for i in range(K):
            pruned_weights = W_norm[i] * pruned_mask[i]

            if use_mwc:
                weighted_pruned = pruned_weights * mu1 / H_inv_diag
                compensation = -H_inv @ weighted_pruned / mu1
            else:
                compensation = -H_inv @ (pruned_weights / H_inv_diag)

            W_comp[i] = W_sparse[i] + compensation * mask[i]

        # Quantize
        Q, scales, zeros, _ = quantize_rtn(W_comp, min_max, group_size=group_size)

        self.register_buffer('W_q', Q.to(torch.int8))
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mask', mask)

        self.K = K
        self.N = N
        self.n_groups = n_groups

        # Debug: store original weight for comparison
        self.register_buffer('W_orig', W.clone())

    def forward(self, x):
        input_dtype = x.dtype
        Q_g = self.W_q.view(self.K, self.n_groups, self.group_size).float()
        W_deq = (Q_g - self.zeros) * self.scales
        W_deq = W_deq.view(self.K, self.N)
        W_deq = W_deq * self.mu2.unsqueeze(1) * self.mu1.unsqueeze(0)
        W_deq = W_deq.to(input_dtype)
        return x @ W_deq.T

    def reconstruction_error(self):
        """Compute reconstruction error vs original weight."""
        Q_g = self.W_q.view(self.K, self.n_groups, self.group_size).float()
        W_deq = (Q_g - self.zeros) * self.scales
        W_deq = W_deq.view(self.K, self.N)
        W_deq = W_deq * self.mu2.unsqueeze(1) * self.mu1.unsqueeze(0)

        error = ((self.W_orig - W_deq) ** 2).mean().item()
        orig_norm = (self.W_orig ** 2).mean().item()
        return error, orig_norm, error / (orig_norm + 1e-8)


def main():
    print("="*70)
    print("DEBUG MWC PPL Pipeline")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "Qwen/Qwen2.5-0.5B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Baseline PPL
    print("\n--- Step 1: Baseline Model PPL ---")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    baseline_ppl = quick_ppl(model, tokenizer)
    print(f"Baseline PPL: {baseline_ppl:.2f}")

    # Step 2: Collect calibration data
    print("\n--- Step 2: Collecting calibration data ---")
    calib_data = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if name not in calib_data:
                calib_data[name] = []
            calib_data[name].append(input[0].detach())
        return hook

    # Only quantize first layer for debugging
    layer = model.model.layers[0]
    test_modules = [
        ('model.layers.0.mlp.gate_proj', layer.mlp.gate_proj),
    ]

    for name, module in test_modules:
        hooks.append(module.register_forward_hook(make_hook(name)))

    # Run calibration
    calib_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

    calib_dataset = calib_dataset.map(tokenize, batched=True, remove_columns=['text'])
    calib_dataset.set_format('torch')
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=4, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(calib_loader):
            if batch_idx >= 4:
                break
            input_ids = batch['input_ids'].to(device)
            model(input_ids)

    for hook in hooks:
        hook.remove()

    print(f"Collected calibration data for {len(calib_data)} modules")
    for name, data in calib_data.items():
        X = torch.cat(data, dim=0)
        print(f"  {name}: shape={X.shape}, mean={X.float().mean():.4f}, std={X.float().std():.4f}")

    # Step 3: Quantize single layer and check reconstruction
    print("\n--- Step 3: Quantize single layer ---")
    name, module = test_modules[0]
    X_calib = torch.cat(calib_data[name], dim=0).view(-1, module.in_features)
    print(f"Calibration data shape: {X_calib.shape}")

    # Check original weight
    W_orig = module.weight.data
    print(f"Original weight: shape={W_orig.shape}, mean={W_orig.float().mean():.6f}, std={W_orig.float().std():.6f}")

    # Create quantized layer
    quant_layer = DebugQuantizedLinear(
        W_orig, X_calib, nbits=4, sparsity=0.35, use_mwc=False
    )

    # Check reconstruction
    error, orig_norm, rel_error = quant_layer.reconstruction_error()
    print(f"Reconstruction: error={error:.6f}, orig_norm={orig_norm:.6f}, relative={rel_error:.4f}")

    # Check quantized weight values
    print(f"mu1: mean={quant_layer.mu1.mean():.4f}, std={quant_layer.mu1.std():.4f}")
    print(f"mu2: mean={quant_layer.mu2.mean():.4f}, std={quant_layer.mu2.std():.4f}")
    print(f"scales: mean={quant_layer.scales.mean():.6f}, std={quant_layer.scales.std():.6f}")
    print(f"zeros: mean={quant_layer.zeros.mean():.4f}, std={quant_layer.zeros.std():.4f}")
    print(f"W_q: min={quant_layer.W_q.min()}, max={quant_layer.W_q.max()}")

    # Step 4: Replace layer and test PPL
    print("\n--- Step 4: Replace single layer and test PPL ---")
    parent = model.model.layers[0].mlp
    setattr(parent, 'gate_proj', quant_layer)

    single_layer_ppl = quick_ppl(model, tokenizer)
    print(f"Single layer quantized PPL: {single_layer_ppl:.2f}")
    print(f"Degradation: {(single_layer_ppl - baseline_ppl) / baseline_ppl * 100:.1f}%")

    # Step 5: Test forward pass manually
    print("\n--- Step 5: Manual forward pass test ---")
    test_input = torch.randn(1, quant_layer.N, device=device, dtype=torch.float16)

    # Original output (approximately, using stored W_orig)
    orig_out = test_input @ quant_layer.W_orig.T.to(test_input.dtype)
    print(f"Original output: mean={orig_out.mean():.4f}, std={orig_out.std():.4f}")

    # Quantized output
    quant_out = quant_layer(test_input)
    print(f"Quantized output: mean={quant_out.mean():.4f}, std={quant_out.std():.4f}")

    # Error
    out_error = ((orig_out - quant_out) ** 2).mean().item()
    out_rel_error = out_error / ((orig_out ** 2).mean().item() + 1e-8)
    print(f"Output error: {out_error:.6f}, relative: {out_rel_error:.4f}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
