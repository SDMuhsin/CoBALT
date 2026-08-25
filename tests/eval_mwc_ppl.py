"""
MWC PPL Evaluation - Full model test
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import compute_hessian_inverse
from datasets import load_dataset


class MWCQuantizedLinear(torch.nn.Module):
    """Linear layer with MWC compensation."""

    def __init__(self, W, X_calib, nbits=4, group_size=64, sparsity=0.35, use_mwc=True):
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
                # MWC compensation
                weighted_pruned = pruned_weights * mu1 / H_inv_diag
                compensation = -H_inv @ weighted_pruned / mu1
            else:
                # Standard OBS compensation
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

    def forward(self, x):
        input_dtype = x.dtype
        Q_g = self.W_q.view(self.K, self.n_groups, self.group_size).float()
        W_deq = (Q_g - self.zeros) * self.scales
        W_deq = W_deq.view(self.K, self.N)
        W_deq = W_deq * self.mu2.unsqueeze(1) * self.mu1.unsqueeze(0)
        W_deq = W_deq.to(input_dtype)
        return x @ W_deq.T


def quantize_model(model, calib_loader, nbits=4, use_mwc=True):
    """Quantize model with MWC or standard compensation."""
    device = next(model.parameters()).device

    # Collect calibration data
    calib_data = {}
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            if name not in calib_data:
                calib_data[name] = []
            calib_data[name].append(input[0].detach())
        return hook

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run calibration
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(calib_loader):
            if batch_idx >= 4:
                break
            input_ids = batch['input_ids'].to(device)
            model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Quantize layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
            if name in calib_data:
                X_calib = torch.cat(calib_data[name], dim=0).view(-1, module.in_features)
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name)

                new_layer = MWCQuantizedLinear(
                    module.weight.data,
                    X_calib,
                    nbits=nbits,
                    use_mwc=use_mwc
                )
                setattr(parent, child_name, new_layer)

    return model


def evaluate_ppl(model, tokenizer, dataset, max_samples=50):
    """Evaluate perplexity on WikiText-2."""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            text = sample['text']
            if len(text.strip()) < 10:
                continue

            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=2048)
            input_ids = inputs['input_ids'].to(device)

            if input_ids.shape[1] < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            total_loss += loss.item() * (input_ids.shape[1] - 1)
            total_tokens += input_ids.shape[1] - 1

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def main():
    print("="*70)
    print("MWC PPL Evaluation")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load calibration data
    calib_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    def tokenize(examples):
        return tokenizer(examples['text'], truncation=True, max_length=2048, padding='max_length')

    calib_dataset = calib_dataset.map(tokenize, batched=True, remove_columns=['text'])
    calib_dataset.set_format('torch')
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=4, shuffle=False)

    # Load test data
    test_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Test Standard OBS
    print("\n--- Standard OBS Compensation ---")
    model_std = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    model_std = quantize_model(model_std, calib_loader, nbits=4, use_mwc=False)
    ppl_std = evaluate_ppl(model_std, tokenizer, test_dataset)
    print(f"Standard OBS PPL: {ppl_std:.2f}")
    del model_std
    torch.cuda.empty_cache()

    # Test MWC
    print("\n--- MWC Compensation ---")
    model_mwc = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    model_mwc = quantize_model(model_mwc, calib_loader, nbits=4, use_mwc=True)
    ppl_mwc = evaluate_ppl(model_mwc, tokenizer, test_dataset)
    print(f"MWC PPL: {ppl_mwc:.2f}")
    del model_mwc
    torch.cuda.empty_cache()

    # Results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Standard OBS: PPL = {ppl_std:.2f}")
    print(f"MWC:          PPL = {ppl_mwc:.2f}")
    improvement = (ppl_std - ppl_mwc) / ppl_std * 100
    print(f"Improvement:  {improvement:+.2f}%")


if __name__ == '__main__':
    main()
