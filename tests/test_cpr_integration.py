#!/usr/bin/env python3
"""
Test CPRLinear integration with SINQ framework.

Tests:
1. State dict save/load roundtrip
2. Model patching with CPR quantization
3. Full model save/load cycle
"""

import torch
import torch.nn as nn
import sys
import os
import tempfile
sys.path.insert(0, '/workspace/SINQ')

from sinq.cprlinear import CPRLinear, cpr_quant_config


def test_state_dict_roundtrip():
    """Test that CPRLinear state_dict save/load preserves weights."""
    print("\n" + "=" * 60)
    print("TEST: CPRLinear State Dict Roundtrip")
    print("=" * 60)

    # Create original linear and quantize
    in_features = 512
    out_features = 256

    linear = nn.Linear(in_features, out_features, bias=True).cuda().half()
    cpr1 = CPRLinear.from_linear(linear, high_frac=0.25).cuda()

    # Save state dict
    state_dict = cpr1.state_dict()

    # Create new instance and load
    cpr2 = CPRLinear.from_state_dict(state_dict, device="cuda")

    # Compare dequantized weights
    W1 = cpr1.dequantize()
    W2 = cpr2.dequantize()

    diff = (W1 - W2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Weight comparison after roundtrip:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    # Test forward pass
    x = torch.randn(32, in_features, dtype=torch.float16, device='cuda')
    with torch.no_grad():
        y1 = cpr1(x)
        y2 = cpr2(x)

    out_diff = (y1 - y2).abs().max().item()
    print(f"  Output difference: {out_diff:.6e}")

    if max_diff < 1e-6 and out_diff < 1e-6:
        print("PASS: State dict roundtrip preserves weights exactly")
        return True
    else:
        print("FAIL: State dict roundtrip introduces errors")
        return False


def test_file_save_load():
    """Test saving and loading CPRLinear to/from file."""
    print("\n" + "=" * 60)
    print("TEST: CPRLinear File Save/Load")
    print("=" * 60)

    in_features = 512
    out_features = 256

    linear = nn.Linear(in_features, out_features, bias=True).cuda().half()
    cpr1 = CPRLinear.from_linear(linear).cuda()

    # Get reference output
    x = torch.randn(32, in_features, dtype=torch.float16, device='cuda')
    with torch.no_grad():
        y1 = cpr1(x)

    # Save to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "cpr_layer.pt")
        torch.save(cpr1.state_dict(), filepath)

        # Load back
        state_dict = torch.load(filepath)
        cpr2 = CPRLinear.from_state_dict(state_dict, device="cuda")

        # Compare outputs
        with torch.no_grad():
            y2 = cpr2(x)

    out_diff = (y1 - y2).abs().max().item()
    print(f"Output difference after file save/load: {out_diff:.6e}")

    if out_diff < 1e-6:
        print("PASS: File save/load preserves weights exactly")
        return True
    else:
        print("FAIL: File save/load introduces errors")
        return False


def test_cpr_config():
    """Test CPR quantization config generation."""
    print("\n" + "=" * 60)
    print("TEST: CPR Quant Config")
    print("=" * 60)

    config = cpr_quant_config(high_frac=0.25, high_bits=6, low_bits=5)

    print(f"Config: {config}")

    expected_avg = 0.25 * 6 + 0.75 * 5
    if abs(config["avg_bits"] - expected_avg) < 0.001:
        print(f"PASS: Average bits correct ({config['avg_bits']:.3f} == {expected_avg:.3f})")
        return True
    else:
        print(f"FAIL: Average bits incorrect ({config['avg_bits']:.3f} != {expected_avg:.3f})")
        return False


def test_multi_layer_model():
    """Test CPRLinear in a multi-layer model."""
    print("\n" + "=" * 60)
    print("TEST: Multi-Layer Model with CPRLinear")
    print("=" * 60)

    # Create a simple MLP
    class SimpleMLP(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
            self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
            self.act = nn.GELU()

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    hidden_size = 256
    model = SimpleMLP(hidden_size).cuda().half()

    # Get reference output
    x = torch.randn(8, hidden_size, dtype=torch.float16, device='cuda')
    with torch.no_grad():
        y_ref = model(x)

    # Quantize with CPR
    model.fc1 = CPRLinear.from_linear(model.fc1).cuda()
    model.fc2 = CPRLinear.from_linear(model.fc2).cuda()

    # Get quantized output
    with torch.no_grad():
        y_quant = model(x)

    # Compare
    diff = (y_ref - y_quant).abs()
    max_diff = diff.max().item()
    rel_diff = (diff / (y_ref.abs() + 1e-6)).mean().item()

    print(f"Multi-layer model comparison:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean relative difference: {rel_diff:.6e}")

    # Save and load model state
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "mlp_cpr.pt")

        # Save all layer states
        state = {
            'fc1': model.fc1.state_dict(),
            'fc2': model.fc2.state_dict(),
        }
        torch.save(state, filepath)

        # Load back into new model
        model2 = SimpleMLP(hidden_size).cuda().half()
        loaded = torch.load(filepath)

        model2.fc1 = CPRLinear.from_state_dict(loaded['fc1'], device="cuda")
        model2.fc2 = CPRLinear.from_state_dict(loaded['fc2'], device="cuda")

        with torch.no_grad():
            y_loaded = model2(x)

    load_diff = (y_quant - y_loaded).abs().max().item()
    print(f"  Difference after load: {load_diff:.6e}")

    if rel_diff < 0.5 and load_diff < 1e-6:  # Quantization error expected
        print("PASS: Multi-layer model works correctly")
        return True
    else:
        print("FAIL: Multi-layer model has issues")
        return False


def main():
    print("=" * 60)
    print("CPRLinear Integration Test Suite")
    print("=" * 60)

    results = {}
    results['state_dict'] = test_state_dict_roundtrip()
    results['file_save'] = test_file_save_load()
    results['config'] = test_cpr_config()
    results['multi_layer'] = test_multi_layer_model()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        all_pass = all_pass and passed

    print("\n" + ("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"))
    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
