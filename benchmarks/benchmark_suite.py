#!/usr/bin/env python3
"""
SINQ Comprehensive Benchmark Suite

Unified benchmarking script for evaluating SINQ quantization techniques across
multiple models and precision levels.

Benchmark Matrix: 5 models x 5 techniques x 5 precisions = 125 configurations

Usage:
    python benchmarks/benchmark_suite.py --model qwen-0.5b --technique sinq --precision 4
    python benchmarks/benchmark_suite.py --model all --technique all --precision all
    python benchmarks/benchmark_suite.py --resume  # Resume from last checkpoint
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import gc
import csv
import math
import fcntl
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Import SINQ modules
from sinq.dual_shift import quantize_dual_scale_shift, quantize_rtn
from sinq.sinkhorn import sinkhorn_log
from sinq.sparse_quant import sparse_quantize_sinq, dequantize_sparse_sinq
from sinq.cpr_model import CPRLinearFused, CPRLinearMultiPrecision
# AWQ building blocks (SINQ authors' own AWQ implementation)
from sinq.awq import compute_awq_scale, rtn_fake_quant, tiled_fake_quant_rectangle

# =============================================================================
# Configuration
# =============================================================================

MODELS = {
    # Qwen family
    'qwen-0.5b': 'Qwen/Qwen2.5-0.5B',
    'qwen-1.5b': 'Qwen/Qwen2.5-1.5B',
    'qwen-3b': 'Qwen/Qwen2.5-3B',
    # Llama family
    'llama-7b': 'huggyllama/llama-7b',
    'tinyllama': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    # Recent Llama (camera-ready model #2). Ungated faithful mirror of
    # meta-llama/Llama-3.2-3B (no HF token in this env); model_type=llama so
    # get_layer_paths returns the standard 7 decoder linears.
    'llama-3.2-3b': 'unsloth/Llama-3.2-3B',
    # Recent code model (camera-ready model #4). model_type=starcoder2, GQA
    # 24 q / 2 KV heads (v=256-dim, gemma-like bottleneck), NON-gated MLP
    # (mlp.c_fc + mlp.c_proj) -> needs ARCH_LAYER_PATHS['starcoder2'].
    'starcoder2-3b': 'bigcode/starcoder2-3b',
    # Microsoft Phi family
    'phi-2': 'microsoft/phi-2',
    # Other architectures
    'pythia-1.4b': 'EleutherAI/pythia-1.4b',
    'opt-125m': 'facebook/opt-125m',
    'opt-350m': 'facebook/opt-350m',
    'opt-1.3b': 'facebook/opt-1.3b',
    'stablelm-2': 'stabilityai/stablelm-2-1_6b',
    'mistral-7b': 'mistralai/Mistral-7B-v0.1',
    # Google Gemma family
    'gemma-2b': 'google/gemma-2b',
    'gemma-2-2b': 'unsloth/gemma-2-2b',  # ungated mirror; held-out mechanism test (predict gain>1)
    'gemma-7b': 'unsloth/gemma-7b',      # ungated mirror; gemma-1 design at 7B (predict strong amplify)
    'opt-2.7b': 'facebook/opt-2.7b',     # larger OPT: does amplification appear at scale?
    # NON-gemma CoBALT-win hunt (collapse + balance-rescuable screen)
    'gpt2-xl': 'openai-community/gpt2-xl',
    'bloom-1b7': 'bigscience/bloom-1b7',
    'phi-1_5': 'microsoft/phi-1_5',
}

TECHNIQUES = ['fp16', 'sinq', 'cpr-sinq', 'sinq-sparse', 'cpr-sparse', 'wanda', 'sparsegpt', 'scab', 'prism',
              # Additional literature baselines (added 2026-06-23)
              'awq', 'gptq', 'spqr', 'wanda-awq', 'wanda-sinq',
              # JSQ joint prune+quant baseline (ICML 2024): faithful WnAn + weight-only variant
              'jsq', 'jsq-wo',
              # SLiM-LoRA joint prune+quant+low-rank baseline (ICML 2025)
              'slim',
              # Bias-correction ablation study (2026-06-24):
              # Study A — within-SINQ 2×2 component ablation (correction × extras)
              'abl-base', 'abl-corr', 'abl-extras', 'abl-full',
              # Study B — sparse-aware-statistics graft onto foreign quantizers
              'sparsegpt-corr', 'slim-corr', 'jsq-wo-corr']

# Techniques that prune weights (and therefore report a meaningful --sparsity).
# 'sparse' substring also catches sinq-sparse / cpr-sparse.
SPARSE_TECHNIQUES = ('wanda', 'sparsegpt', 'scab', 'prism', 'wanda-awq', 'wanda-sinq', 'jsq', 'jsq-wo', 'slim',
                     'abl-base', 'abl-corr', 'abl-extras', 'abl-full',
                     'sparsegpt-corr', 'slim-corr', 'jsq-wo-corr')


def uses_sparsity(technique: str) -> bool:
    """True if the technique prunes weights (so the --sparsity value is meaningful)."""
    return 'sparse' in technique or technique in SPARSE_TECHNIQUES


PRECISIONS = [3, 4, 5, 6, 8]

# Evaluation parameters
EVAL_CONFIG = {
    'seq_len': 2048,
    'n_test_samples': 16,
    'n_calibration_samples': 16,
    'calibration_seq_len': 512,
    'sparsity': 0.35,  # For sparse techniques
    'cpr_group_size': 128,
}

# Dataset configurations
DATASETS = {
    'wikitext2': {
        'name': 'wikitext',
        'subset': 'wikitext-2-raw-v1',
        'test_split': 'test',
        'train_split': 'train',
        'text_column': 'text',
        'type': 'perplexity',
    },
    'wikitext103': {
        'name': 'Salesforce/wikitext',
        'subset': 'wikitext-103-raw-v1',
        'test_split': 'test',
        'train_split': 'train',
        'text_column': 'text',
        'type': 'perplexity',
    },
    'c4': {
        'name': 'allenai/c4',
        'subset': 'en',
        'test_split': 'validation',
        'train_split': 'train',
        'text_column': 'text',
        'type': 'perplexity',
        'streaming': True,  # C4 is large, use streaming
    },
    'ptb': {
        # Penn Treebank. datasets 5.x dropped the ptb_text_only loading script, so we
        # read the auto-generated parquet conversion (refs/convert/parquet). Column is
        # 'sentence'; test split concatenated + windowed exactly like wikitext2/ptb in
        # get_test_data (standard GPTQ/SparseGPT PTB-PPL protocol).
        'name': 'ptb_text_only',
        'subset': None,
        'revision': 'refs/convert/parquet',
        'test_split': 'test',
        'train_split': 'train',
        'text_column': 'sentence',
        'type': 'perplexity',
    },
    'mmlu': {
        'name': 'cais/mmlu',
        'subset': 'all',
        'test_split': 'test',
        'validation_split': 'validation',
        'type': 'accuracy',
        'eval_func': 'mmlu',
        'n_few_shot': 5,
    },
    'hellaswag': {
        'name': 'Rowan/hellaswag',
        'test_split': 'validation',  # No test labels, use validation
        'type': 'accuracy',
        'eval_func': 'hellaswag',
        'n_few_shot': 0,  # Zero-shot by default
    },
    'arc_challenge': {
        'name': 'allenai/ai2_arc',
        'subset': 'ARC-Challenge',
        'test_split': 'test',
        'type': 'accuracy',
        'eval_func': 'arc',
        'n_few_shot': 0,
    },
    'winogrande': {
        'name': 'allenai/winogrande',
        'subset': 'winogrande_xl',
        'test_split': 'validation',  # Test has no labels
        'type': 'accuracy',
        'eval_func': 'winogrande',
        'n_few_shot': 0,
    },
    'boolq': {
        'name': 'google/boolq',
        'test_split': 'validation',
        'type': 'accuracy',
        'eval_func': 'boolq',
        'n_few_shot': 0,
    },
    'lambada': {
        'name': 'EleutherAI/lambada_openai',
        'subset': 'en',
        'test_split': 'test',
        'type': 'perplexity',
        'text_column': 'text',
    },
    'pile': {
        'name': 'monology/pile-uncopyrighted',
        'test_split': 'train',  # Only train split available, we sample from it
        'type': 'perplexity',
        'text_column': 'text',
        'streaming': True,
    },
}

DATASET_NAMES = list(DATASETS.keys())

RESULTS_DIR = Path(__file__).parent.parent / 'results'
TRACKER_PATH = Path(__file__).parent.parent / 'llmdocs' / 'BENCHMARK_TRACKER.md'

# Architecture-specific layer paths
# Different model architectures use different attribute names for linear layers
ARCH_LAYER_PATHS = {
    'default': [
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
    ],
    'phi': [
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.dense', 'mlp.fc1', 'mlp.fc2'
    ],
    'llama': [
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
    ],
    'gpt_neox': [  # Pythia uses GPT-NeoX architecture
        'attention.query_key_value', 'attention.dense',
        'mlp.dense_h_to_4h', 'mlp.dense_4h_to_h'
    ],
    'opt': [  # OPT architecture
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.out_proj', 'fc1', 'fc2'
    ],
    'gemma': [  # Gemma architecture (similar to Llama)
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
    ],
    'mistral': [  # Mistral architecture (similar to Llama)
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
    ],
    'starcoder2': [  # StarCoder2: NON-gated MLP (c_fc/c_proj, NOT gate/up/down).
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.o_proj', 'mlp.c_fc', 'mlp.c_proj'
    ],
}

def get_layer_paths(model) -> list:
    """Get the correct layer paths for the given model architecture."""
    model_type = getattr(model.config, 'model_type', '').lower()
    if 'phi' in model_type:
        return ARCH_LAYER_PATHS['phi']
    elif 'llama' in model_type:
        return ARCH_LAYER_PATHS['llama']
    elif 'gpt_neox' in model_type:
        return ARCH_LAYER_PATHS['gpt_neox']
    elif 'opt' in model_type:
        return ARCH_LAYER_PATHS['opt']
    elif 'gemma' in model_type:
        return ARCH_LAYER_PATHS['gemma']
    elif 'mistral' in model_type:
        return ARCH_LAYER_PATHS['mistral']
    elif 'starcoder2' in model_type:
        return ARCH_LAYER_PATHS['starcoder2']
    return ARCH_LAYER_PATHS['default']


def get_model_backbone(model):
    """Get the backbone/transformer module for different architectures."""
    model_type = getattr(model.config, 'model_type', '').lower()
    if 'gpt_neox' in model_type:
        return model.gpt_neox
    elif 'opt' in model_type:
        return model.model.decoder
    elif hasattr(model, 'model'):
        return model.model
    elif hasattr(model, 'transformer'):
        return model.transformer
    return model


def get_embed_tokens(model):
    """Get embedding layer for different architectures."""
    model_type = getattr(model.config, 'model_type', '').lower()
    if 'gpt_neox' in model_type:
        return model.gpt_neox.embed_in
    elif 'opt' in model_type:
        return model.model.decoder.embed_tokens
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return model.transformer.wte
    return None


def get_transformer_layers(model):
    """Get transformer layers for different architectures."""
    model_type = getattr(model.config, 'model_type', '').lower()
    if 'gpt_neox' in model_type:
        return model.gpt_neox.layers
    elif 'opt' in model_type:
        return model.model.decoder.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    return None


def get_final_layernorm(model):
    """Get final layer norm for different architectures."""
    model_type = getattr(model.config, 'model_type', '').lower()
    if 'gpt_neox' in model_type:
        return model.gpt_neox.final_layer_norm
    elif 'opt' in model_type:
        return model.model.decoder.final_layer_norm
    elif hasattr(model, 'model'):
        if hasattr(model.model, 'norm'):
            return model.model.norm
        elif hasattr(model.model, 'final_layernorm'):
            return model.model.final_layernorm
        elif hasattr(model.model, 'ln_f'):
            return model.model.ln_f
    return None


def move_embed_to_device(model, device):
    """Move embedding layer to device.

    Note: We move just the weight tensor to avoid recursion issues with
    weight-tied models (where embed and lm_head share weights).
    """
    model_type = getattr(model.config, 'model_type', '').lower()

    try:
        if 'gpt_neox' in model_type:
            model.gpt_neox.embed_in.weight.data = model.gpt_neox.embed_in.weight.data.to(device)
        elif 'opt' in model_type:
            model.model.decoder.embed_tokens.weight.data = model.model.decoder.embed_tokens.weight.data.to(device)
            if hasattr(model.model.decoder, 'embed_positions'):
                model.model.decoder.embed_positions.weight.data = model.model.decoder.embed_positions.weight.data.to(device)
            # OPT-350M-style models have a pre-layer embedding projection (word_embed_proj_dim !=
            # hidden_size). It must be on-device or the per-layer calibration forward dies before
            # reaching the decoder layers (leaving them un-calibrated).
            if getattr(model.model.decoder, 'project_in', None) is not None:
                model.model.decoder.project_in.weight.data = model.model.decoder.project_in.weight.data.to(device)
                if model.model.decoder.project_in.bias is not None:
                    model.model.decoder.project_in.bias.data = model.model.decoder.project_in.bias.data.to(device)
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens.weight.data = model.model.embed_tokens.weight.data.to(device)
    except Exception:
        # Fallback: try to move the whole module (works for most models)
        embed = get_embed_tokens(model)
        if embed is not None and hasattr(embed, 'weight'):
            embed.weight.data = embed.weight.data.to(device)


def move_final_layers_to_device(model, device):
    """Move final layers (norm + lm_head) to device.

    Note: We move just the weight tensors to avoid recursion issues with
    weight-tied models (where embed and lm_head share weights).
    """
    model_type = getattr(model.config, 'model_type', '').lower()

    try:
        # Move final layer norm
        if 'gpt_neox' in model_type:
            ln = model.gpt_neox.final_layer_norm
            ln.weight.data = ln.weight.data.to(device)
            if ln.bias is not None:
                ln.bias.data = ln.bias.data.to(device)
        elif 'opt' in model_type:
            if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm is not None:
                ln = model.model.decoder.final_layer_norm
                ln.weight.data = ln.weight.data.to(device)
                if ln.bias is not None:
                    ln.bias.data = ln.bias.data.to(device)
            # OPT-350M-style post-layer output projection (hidden_size -> word_embed_proj_dim).
            if getattr(model.model.decoder, 'project_out', None) is not None:
                model.model.decoder.project_out.weight.data = model.model.decoder.project_out.weight.data.to(device)
                if model.model.decoder.project_out.bias is not None:
                    model.model.decoder.project_out.bias.data = model.model.decoder.project_out.bias.data.to(device)
        elif hasattr(model, 'model'):
            if hasattr(model.model, 'norm') and model.model.norm is not None:
                ln = model.model.norm
                ln.weight.data = ln.weight.data.to(device)
                if hasattr(ln, 'bias') and ln.bias is not None:
                    ln.bias.data = ln.bias.data.to(device)
            elif hasattr(model.model, 'final_layernorm') and model.model.final_layernorm is not None:
                ln = model.model.final_layernorm
                ln.weight.data = ln.weight.data.to(device)
                if hasattr(ln, 'bias') and ln.bias is not None:
                    ln.bias.data = ln.bias.data.to(device)

        # Move lm_head - be careful with weight tying
        if hasattr(model, 'lm_head') and model.lm_head is not None:
            if hasattr(model.lm_head, 'weight'):
                model.lm_head.weight.data = model.lm_head.weight.data.to(device)
                if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
                    model.lm_head.bias.data = model.lm_head.bias.data.to(device)
        elif hasattr(model, 'embed_out') and model.embed_out is not None:
            if hasattr(model.embed_out, 'weight'):
                model.embed_out.weight.data = model.embed_out.weight.data.to(device)
    except Exception as e:
        # Fallback for unusual architectures
        pass


def set_transformer_layer(model, layer_idx, layer):
    """Set a transformer layer by index for different architectures."""
    model_type = getattr(model.config, 'model_type', '').lower()
    if 'gpt_neox' in model_type:
        model.gpt_neox.layers[layer_idx] = layer
    elif 'opt' in model_type:
        model.model.decoder.layers[layer_idx] = layer
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        model.model.layers[layer_idx] = layer
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        model.transformer.h[layer_idx] = layer

# =============================================================================
# Data Loading
# =============================================================================

def get_test_data(tokenizer, seq_len: int = 2048, n_samples: int = 16,
                  dataset_key: str = 'wikitext2') -> torch.Tensor:
    """Load test data for perplexity evaluation.

    Args:
        tokenizer: Tokenizer to use
        seq_len: Sequence length for each sample
        n_samples: Number of samples to load
        dataset_key: Dataset to use ('wikitext2', 'ptb', 'c4')

    Returns:
        Tensor of shape [n_samples, seq_len]
    """
    ds_config = DATASETS[dataset_key]

    if ds_config.get('streaming', False):
        # For C4/Pile, use streaming to avoid downloading entire dataset
        subset = ds_config.get('subset', None)
        dataset = load_dataset(
            ds_config['name'],
            subset,
            split=ds_config['test_split'],
            streaming=True,
            revision=ds_config.get('revision', None),
            trust_remote_code=True
        )
        # Collect enough text from streaming dataset
        texts = []
        total_tokens = 0
        target_tokens = seq_len * n_samples * 2  # Get extra to ensure enough samples
        for item in dataset:
            texts.append(item[ds_config['text_column']])
            total_tokens += len(item[ds_config['text_column']].split())
            if total_tokens > target_tokens:
                break
        text = '\n\n'.join(texts)
    else:
        subset = ds_config.get('subset', None)
        dataset = load_dataset(
            ds_config['name'],
            subset,
            split=ds_config['test_split'],
            revision=ds_config.get('revision', None),
            trust_remote_code=True
        )
        text_col = ds_config['text_column']
        text = '\n\n'.join([item for item in dataset[text_col] if item.strip()])

    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids[0]

    samples = []
    for i in range(0, min(len(input_ids) - seq_len, n_samples * seq_len), seq_len):
        samples.append(input_ids[i:i + seq_len])

    if len(samples) < n_samples:
        print(f"  Warning: Only {len(samples)} samples available (requested {n_samples})")

    return torch.stack(samples[:n_samples])


def get_calibration_data(tokenizer, n_samples: int = 16, seq_len: int = 512,
                         dataset_key: str = 'wikitext2') -> torch.Tensor:
    """Load training data for calibration.

    Args:
        tokenizer: Tokenizer to use
        n_samples: Number of calibration samples
        seq_len: Sequence length for each sample
        dataset_key: Dataset to use ('wikitext2', 'ptb', 'c4')

    Returns:
        Tensor of shape [n_samples, seq_len]
    """
    ds_config = DATASETS[dataset_key]

    if ds_config.get('streaming', False):
        # For C4, use streaming
        dataset = load_dataset(
            ds_config['name'],
            ds_config['subset'],
            split=ds_config['train_split'],
            streaming=True,
            trust_remote_code=True
        )
        texts = []
        total_tokens = 0
        target_tokens = seq_len * n_samples * 2
        for item in dataset:
            texts.append(item[ds_config['text_column']])
            total_tokens += len(item[ds_config['text_column']].split())
            if total_tokens > target_tokens:
                break
        text = '\n\n'.join(texts)
    else:
        dataset = load_dataset(
            ds_config['name'],
            ds_config['subset'],
            split=ds_config['train_split'],
            trust_remote_code=True
        )
        text_col = ds_config['text_column']
        text = '\n\n'.join([item for item in dataset[text_col] if item.strip()])

    encodings = tokenizer(
        text, return_tensors='pt',
        max_length=seq_len * n_samples * 2,
        truncation=True
    )
    input_ids = encodings.input_ids[0]

    samples = []
    for i in range(0, min(len(input_ids) - seq_len, n_samples * seq_len), seq_len):
        samples.append(input_ids[i:i + seq_len])

    return torch.stack(samples[:n_samples])


# =============================================================================
# MMLU Evaluation
# =============================================================================

MMLU_SUBJECTS = [
    'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
    'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
    'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics',
    'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic',
    'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
    'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
    'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
    'high_school_physics', 'high_school_psychology', 'high_school_statistics',
    'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
    'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management',
    'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
    'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
    'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
    'sociology', 'us_foreign_policy', 'virology', 'world_religions'
]

def format_mmlu_prompt(question: str, choices: List[str], few_shot_examples: List[dict] = None) -> str:
    """Format MMLU question with optional few-shot examples."""
    prompt = ""

    # Add few-shot examples if provided
    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Question: {ex['question']}\n"
            for i, choice in enumerate(ex['choices']):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += f"Answer: {chr(65 + ex['answer'])}\n\n"

    # Add the actual question
    prompt += f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "Answer:"

    return prompt


@torch.no_grad()
def evaluate_mmlu(model, tokenizer, device: str = 'cuda',
                  n_few_shot: int = 5, max_subjects: int = None,
                  verbose: bool = True) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on MMLU benchmark.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        device: Device to use
        n_few_shot: Number of few-shot examples (0 for zero-shot)
        max_subjects: Maximum number of subjects to evaluate (None for all)
        verbose: Print progress

    Returns:
        Tuple of (overall_accuracy, per_subject_accuracy_dict)
    """
    model.eval()

    subjects = MMLU_SUBJECTS[:max_subjects] if max_subjects else MMLU_SUBJECTS

    total_correct = 0
    total_count = 0
    per_subject_acc = {}

    # Token IDs for A, B, C, D
    answer_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in ['A', 'B', 'C', 'D']]

    for subject in tqdm(subjects, desc="MMLU", disable=not verbose):
        try:
            # Load subject data
            test_data = load_dataset('cais/mmlu', subject, split='test', trust_remote_code=True)

            # Get few-shot examples from dev set if needed
            few_shot_examples = []
            if n_few_shot > 0:
                dev_data = load_dataset('cais/mmlu', subject, split='dev', trust_remote_code=True)
                for i in range(min(n_few_shot, len(dev_data))):
                    few_shot_examples.append({
                        'question': dev_data[i]['question'],
                        'choices': dev_data[i]['choices'],
                        'answer': dev_data[i]['answer']
                    })

            subject_correct = 0
            subject_count = 0

            for item in test_data:
                question = item['question']
                choices = item['choices']
                correct_answer = item['answer']  # 0, 1, 2, or 3

                # Format prompt
                prompt = format_mmlu_prompt(question, choices, few_shot_examples)

                # Tokenize
                inputs = tokenizer(prompt, return_tensors='pt').to(device)

                # Get logits for next token
                outputs = model(**inputs)
                next_token_logits = outputs.logits[0, -1, :]

                # Get probabilities for answer tokens
                answer_logits = next_token_logits[answer_tokens]
                predicted_answer = answer_logits.argmax().item()

                if predicted_answer == correct_answer:
                    subject_correct += 1
                subject_count += 1

            subject_acc = subject_correct / subject_count if subject_count > 0 else 0
            per_subject_acc[subject] = subject_acc
            total_correct += subject_correct
            total_count += subject_count

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to evaluate {subject}: {e}")
            continue

    overall_acc = total_correct / total_count if total_count > 0 else 0

    return overall_acc, per_subject_acc


@torch.no_grad()
def evaluate_hellaswag(model, tokenizer, device: str = 'cuda',
                       max_samples: int = None, verbose: bool = True) -> float:
    """Evaluate model on HellaSwag benchmark.

    HellaSwag tests commonsense reasoning by asking the model to complete sentences.
    We use the likelihood-based evaluation: pick the completion with highest probability.
    """
    model.eval()

    dataset = load_dataset('Rowan/hellaswag', split='validation', trust_remote_code=True)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0

    for item in tqdm(dataset, desc="HellaSwag", disable=not verbose):
        ctx = item['ctx']
        endings = item['endings']
        label = int(item['label'])

        # Compute log-likelihood for each ending
        scores = []
        for ending in endings:
            full_text = ctx + " " + ending
            inputs = tokenizer(full_text, return_tensors='pt').to(device)

            # Get the ending tokens to compute likelihood
            ctx_tokens = tokenizer(ctx, return_tensors='pt')['input_ids'].shape[1]

            outputs = model(**inputs)
            logits = outputs.logits

            # Compute log probability of ending tokens
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
            target_ids = inputs['input_ids'][0, 1:]

            # Only score the ending tokens (after context)
            ending_log_prob = log_probs[ctx_tokens-1:, :].gather(1, target_ids[ctx_tokens-1:].unsqueeze(1)).sum()
            scores.append(ending_log_prob.item())

        predicted = scores.index(max(scores))
        if predicted == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


@torch.no_grad()
def evaluate_arc(model, tokenizer, device: str = 'cuda',
                 max_samples: int = None, verbose: bool = True) -> float:
    """Evaluate model on ARC-Challenge benchmark.

    ARC tests science knowledge with multiple-choice questions.
    """
    model.eval()

    dataset = load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test', trust_remote_code=True)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Token IDs for A, B, C, D, E
    answer_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in ['A', 'B', 'C', 'D', 'E']]

    correct = 0
    total = 0

    for item in tqdm(dataset, desc="ARC-Challenge", disable=not verbose):
        question = item['question']
        choices = item['choices']
        answer_key = item['answerKey']

        # Build prompt
        prompt = f"Question: {question}\n"
        choice_labels = choices['label']
        choice_texts = choices['text']
        for lbl, txt in zip(choice_labels, choice_texts):
            prompt += f"{lbl}. {txt}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

        # Get probabilities for answer tokens
        n_choices = len(choice_labels)
        choice_logits = next_token_logits[answer_tokens[:n_choices]]
        predicted_idx = choice_logits.argmax().item()
        predicted_label = choice_labels[predicted_idx]

        if predicted_label == answer_key:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


@torch.no_grad()
def evaluate_winogrande(model, tokenizer, device: str = 'cuda',
                        max_samples: int = None, verbose: bool = True) -> float:
    """Evaluate model on WinoGrande benchmark.

    WinoGrande tests commonsense reasoning with pronoun resolution.
    """
    model.eval()

    dataset = load_dataset('allenai/winogrande', 'winogrande_xl', split='validation', trust_remote_code=True)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0

    for item in tqdm(dataset, desc="WinoGrande", disable=not verbose):
        sentence = item['sentence']
        option1 = item['option1']
        option2 = item['option2']
        answer = item['answer']  # '1' or '2'

        # Replace _ with each option and compute likelihood
        scores = []
        for option in [option1, option2]:
            filled = sentence.replace('_', option)
            inputs = tokenizer(filled, return_tensors='pt').to(device)
            outputs = model(**inputs)

            # Compute average log probability
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
            target_ids = inputs['input_ids'][0, 1:]
            avg_log_prob = log_probs.gather(1, target_ids.unsqueeze(1)).mean()
            scores.append(avg_log_prob.item())

        predicted = '1' if scores[0] > scores[1] else '2'
        if predicted == answer:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


@torch.no_grad()
def evaluate_boolq(model, tokenizer, device: str = 'cuda',
                   max_samples: int = None, verbose: bool = True) -> float:
    """Evaluate model on BoolQ benchmark.

    BoolQ tests reading comprehension with yes/no questions.
    """
    model.eval()

    dataset = load_dataset('google/boolq', split='validation')
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    # Token IDs for yes/no
    yes_token = tokenizer.encode(" Yes", add_special_tokens=False)[-1]
    no_token = tokenizer.encode(" No", add_special_tokens=False)[-1]

    correct = 0
    total = 0

    for item in tqdm(dataset, desc="BoolQ", disable=not verbose):
        passage = item['passage']
        question = item['question']
        answer = item['answer']  # True or False

        # Build prompt
        prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

        # Compare yes vs no probability
        yes_logit = next_token_logits[yes_token].item()
        no_logit = next_token_logits[no_token].item()

        predicted = yes_logit > no_logit
        if predicted == answer:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


# =============================================================================
# Flip Rate Evaluation (Dutta et al. 2024)
# =============================================================================

@torch.no_grad()
def get_hellaswag_predictions(model, tokenizer, device: str = 'cuda',
                               max_samples: int = None) -> List[int]:
    """Get HellaSwag predictions for flip rate calculation."""
    model.eval()
    dataset = load_dataset('Rowan/hellaswag', split='validation', trust_remote_code=True)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    predictions = []
    for item in tqdm(dataset, desc="HellaSwag predictions"):
        ctx = item['ctx']
        endings = item['endings']

        scores = []
        for ending in endings:
            full_text = ctx + " " + ending
            inputs = tokenizer(full_text, return_tensors='pt').to(device)
            ctx_tokens = tokenizer(ctx, return_tensors='pt')['input_ids'].shape[1]
            outputs = model(**inputs)
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
            target_ids = inputs['input_ids'][0, 1:]
            ending_log_prob = log_probs[ctx_tokens-1:, :].gather(1, target_ids[ctx_tokens-1:].unsqueeze(1)).sum()
            scores.append(ending_log_prob.item())

        predictions.append(scores.index(max(scores)))

    return predictions


@torch.no_grad()
def get_mmlu_predictions(model, tokenizer, device: str = 'cuda',
                          max_subjects: int = None) -> List[int]:
    """Get MMLU predictions for flip rate calculation."""
    model.eval()
    subjects = MMLU_SUBJECTS[:max_subjects] if max_subjects else MMLU_SUBJECTS

    answer_tokens = [tokenizer.encode(f" {c}", add_special_tokens=False)[-1] for c in ['A', 'B', 'C', 'D']]
    predictions = []

    for subject in tqdm(subjects, desc="MMLU predictions"):
        try:
            test_data = load_dataset('cais/mmlu', subject, split='test', trust_remote_code=True)
            dev_data = load_dataset('cais/mmlu', subject, split='dev', trust_remote_code=True)

            few_shot_examples = []
            for i in range(min(5, len(dev_data))):
                few_shot_examples.append({
                    'question': dev_data[i]['question'],
                    'choices': dev_data[i]['choices'],
                    'answer': dev_data[i]['answer']
                })

            for item in test_data:
                prompt = format_mmlu_prompt(item['question'], item['choices'], few_shot_examples)
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
                outputs = model(**inputs)
                next_token_logits = outputs.logits[0, -1, :]
                answer_logits = next_token_logits[answer_tokens]
                predictions.append(answer_logits.argmax().item())
        except Exception:
            continue

    return predictions


@torch.no_grad()
def get_boolq_predictions(model, tokenizer, device: str = 'cuda',
                           max_samples: int = None) -> List[bool]:
    """Get BoolQ predictions for flip rate calculation."""
    model.eval()
    dataset = load_dataset('google/boolq', split='validation')
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    yes_token = tokenizer.encode(" Yes", add_special_tokens=False)[-1]
    no_token = tokenizer.encode(" No", add_special_tokens=False)[-1]
    predictions = []

    for item in tqdm(dataset, desc="BoolQ predictions"):
        prompt = f"Passage: {item['passage']}\nQuestion: {item['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(device)
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        predictions.append(next_token_logits[yes_token].item() > next_token_logits[no_token].item())

    return predictions


def compute_flip_rate(fp16_predictions: List, quant_predictions: List) -> float:
    """Compute flip rate between FP16 and quantized model predictions.

    Flip rate = percentage of samples where predictions differ.
    Lower is better (0% = perfect match with FP16).
    """
    assert len(fp16_predictions) == len(quant_predictions), "Prediction lists must have same length"

    flips = sum(1 for fp16, quant in zip(fp16_predictions, quant_predictions) if fp16 != quant)
    return (flips / len(fp16_predictions)) * 100 if fp16_predictions else 0.0


def run_flip_rate_benchmark(
    model_key: str,
    technique: str,
    precision: int,
    task: str = 'hellaswag',
    max_samples: int = None,
    device: str = 'cuda',
    verbose: bool = True,
    sparsity: float = None,  # Sparsity level (0.0-1.0), defaults to EVAL_CONFIG['sparsity']
) -> Dict[str, Any]:
    """Run flip rate benchmark comparing quantized model to FP16 baseline.

    Args:
        model_key: Model to benchmark
        technique: Quantization technique
        precision: Bit precision
        task: Task to evaluate ('hellaswag', 'mmlu', 'boolq')
        max_samples: Maximum samples to evaluate
        device: Device to use
        verbose: Print progress
        sparsity: Sparsity level for sparse techniques (0.0-1.0), defaults to EVAL_CONFIG['sparsity']

    Returns:
        Dictionary with flip rate results
    """
    model_name = MODELS[model_key]
    sparsity = sparsity if sparsity is not None else EVAL_CONFIG['sparsity']

    result = {
        'model': model_key,
        'technique': technique,
        'precision': precision,
        'task': task,
        'sparsity': sparsity if uses_sparsity(technique) else 0,
        'flip_rate': None,
        'error': None,
    }

    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Flip Rate Benchmark: {model_name}")
            print(f"Task: {task}")
            print(f"Technique: {technique} @ {precision}-bit")
            print('='*60)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Step 1: Get FP16 predictions
        if verbose:
            print("\n[1/3] Getting FP16 baseline predictions...")

        fp16_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map='cuda', trust_remote_code=True
        )
        fp16_model.eval()

        if task == 'hellaswag':
            fp16_preds = get_hellaswag_predictions(fp16_model, tokenizer, device, max_samples)
        elif task == 'mmlu':
            fp16_preds = get_mmlu_predictions(fp16_model, tokenizer, device, max_samples)
        elif task == 'boolq':
            fp16_preds = get_boolq_predictions(fp16_model, tokenizer, device, max_samples)
        else:
            raise ValueError(f"Unknown flip rate task: {task}")

        del fp16_model
        gc.collect()
        torch.cuda.empty_cache()

        # Step 2: Apply quantization and get predictions
        if verbose:
            print(f"\n[2/3] Applying {technique} @ {precision}-bit...")

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map='cpu',
            trust_remote_code=True, low_cpu_mem_usage=True
        )
        move_embed_to_device(model, device)

        calibration_data = get_calibration_data(
            tokenizer, n_samples=EVAL_CONFIG['n_calibration_samples'],
            seq_len=EVAL_CONFIG['calibration_seq_len'], dataset_key='wikitext2'
        )

        if technique == 'prism':
            model = apply_prism_quantization(model, calibration_data, precision, sparsity, device)
        elif technique == 'scab':
            model = apply_scab_quantization(model, calibration_data, precision, sparsity, device)
        elif technique == 'sinq-sparse':
            model = apply_sparse_quantization(model, calibration_data, precision, sparsity, device)
        elif technique == 'sinq':
            model = apply_sinq_quantization(model, calibration_data, precision, device)
        elif technique == 'sparsegpt':
            model = apply_sparsegpt_pruning(model, calibration_data, sparsity, precision, device)
        else:
            raise ValueError(f"Unsupported technique for flip rate: {technique}")

        move_final_layers_to_device(model, device)
        model.eval()

        if verbose:
            print(f"\n[3/3] Getting quantized model predictions...")

        if task == 'hellaswag':
            quant_preds = get_hellaswag_predictions(model, tokenizer, device, max_samples)
        elif task == 'mmlu':
            quant_preds = get_mmlu_predictions(model, tokenizer, device, max_samples)
        elif task == 'boolq':
            quant_preds = get_boolq_predictions(model, tokenizer, device, max_samples)

        # Step 3: Compute flip rate
        flip_rate = compute_flip_rate(fp16_preds, quant_preds)
        n_total = len(fp16_preds)
        n_flipped = int(flip_rate * n_total / 100)
        result['flip_rate'] = flip_rate
        result['n_total'] = n_total
        result['n_flipped'] = n_flipped

        if verbose:
            print(f"\n{'='*60}")
            print(f"Flip Rate: {flip_rate:.2f}% ({n_flipped}/{n_total} flipped)")
            print('='*60)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    return result


# =============================================================================
# Quantized Layer Wrappers
# =============================================================================

class SINQLinear(nn.Module):
    """Linear layer with SINQ quantization applied.

    Supports both per-row and per-group quantization scales.
    With group_size: scales has shape [K, n_groups, 1], zeros has shape [K, n_groups, 1]
    Without group_size: scales has shape [K, 1], zeros has shape [K, 1]
    """

    def __init__(self, W_q, scales, scale2, zeros, bias, nbits, group_size=64):
        super().__init__()
        self.register_buffer('W_q', W_q)
        self.register_buffer('scales', scales)
        self.register_buffer('scale2', scale2)
        self.register_buffer('zeros', zeros)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.nbits = nbits
        self.group_size = group_size
        self._W_cached = None

    def forward(self, x):
        if self._W_cached is None:
            # SINQ dequantization: W = ((Q - z) * scales) * scale2
            Q = self.W_q.float()
            z = self.zeros.float()
            s1 = self.scales.float()
            s2 = self.scale2.float()

            if len(s1.shape) == 3:
                # Group-wise quantization: s1 is [K, n_groups, 1], z is [K, n_groups, 1]
                K, N = Q.shape
                n_groups = s1.shape[1]
                group_size = N // n_groups

                # Reshape Q to [K, n_groups, group_size] for dequantization
                Q_grouped = Q.view(K, n_groups, group_size)
                W_deq = (Q_grouped - z) * s1  # [K, n_groups, group_size]
                W_deq = W_deq.view(K, N)  # [K, N]
            else:
                # Per-row quantization
                W_deq = (Q - z) * s1

            # Apply column-wise scale (s2 is [1, N])
            W_deq = W_deq * s2
            self._W_cached = W_deq.to(x.dtype)

        out = torch.matmul(x, self._W_cached.t())
        if self.bias is not None:
            out = out + self.bias
        return out


class SparseQuantLinear(nn.Module):
    """Linear layer with sparse quantization applied."""

    def __init__(self, W_q, scales, zeros, mask, scale2, bias, meta):
        super().__init__()
        # Low-memory eval path (PRISM_LOWMEM=1): store the mask as bool and the integer
        # codes as uint8, and recompute the dequantized weight per forward instead of caching
        # it. This cuts the resident footprint ~3x (fp16 W_q + fp16 mask + fp16 cache -> uint8
        # codes + bool mask + transient dequant), letting the full-matrix Sinkhorn techniques
        # fit a 24GB MIG slice on llama-7b. The dequant math is unchanged (dequantize_sparse_sinq
        # casts both to float), so PPL is bit-identical — verified on qwen-0.5b.
        self.lowmem = os.environ.get('PRISM_LOWMEM', '0') == '1'
        if self.lowmem:
            mask = (mask != 0)  # bool
            if float(W_q.abs().max()) <= 255:
                W_q = W_q.round().to(torch.uint8)
        self.register_buffer('W_q', W_q)
        self.register_buffer('scales', scales)
        self.register_buffer('zeros', zeros)
        self.register_buffer('mask', mask)
        self.register_buffer('scale2', scale2)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.meta = meta
        self._W_cached = None

    def forward(self, x):
        if self.lowmem:
            W = dequantize_sparse_sinq(
                self.W_q, self.scales, self.zeros, self.mask, self.scale2, self.meta
            ).to(x.dtype)
            out = torch.matmul(x, W.t())
            del W
            if self.bias is not None:
                out = out + self.bias
            return out
        if self._W_cached is None:
            self._W_cached = dequantize_sparse_sinq(
                self.W_q, self.scales, self.zeros, self.mask, self.scale2, self.meta
            ).to(x.dtype)
        out = torch.matmul(x, self._W_cached.t())
        if self.bias is not None:
            out = out + self.bias
        return out


class SparseLinear(nn.Module):
    """Linear layer with pruning applied (no quantization).

    Used for Wanda baseline which is pure pruning without quantization.
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer('weight', weight)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        out = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out


class WandaLayerWrapper:
    """Wrapper to collect activation statistics for Wanda pruning.

    Tracks per-column L2 norm squared of activations (scaler_row).
    Based on the original Wanda implementation.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.device = layer.weight.device
        self.rows = layer.weight.data.shape[0]  # out_features
        self.columns = layer.weight.data.shape[1]  # in_features
        self.scaler_row = torch.zeros((self.columns,), device=self.device)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """Accumulate activation statistics from a batch."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        batch_size = inp.shape[0]

        if len(inp.shape) == 3:
            # [batch, seq_len, hidden] -> [batch*seq_len, hidden]
            inp = inp.reshape(-1, inp.shape[-1])

        # Transpose: [hidden, batch*seq_len] for column-wise norms
        inp = inp.t()

        # Running mean update for L2 norm squared per column
        self.scaler_row *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size

        inp = inp.float()
        # L2 norm squared per column (input feature dimension)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


class SparseGPTQuantizer:
    """Quantizer for SparseGPT - computes per-channel scales for RTN quantization.

    Based on the original SparseGPT implementation.
    """

    def __init__(self, shape=1):
        self.maxq = torch.tensor(0)
        self.scale = torch.zeros(shape)
        self.zero = torch.zeros(shape)

    def configure(self, bits: int, perchannel: bool = True, sym: bool = False):
        """Configure quantization parameters."""
        self.bits = bits
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym

    def find_params(self, x: torch.Tensor, weight: bool = True, prune_mask: torch.Tensor = None):
        """Find quantization scale and zero point for the given tensor.

        prune_mask (bool, same shape as x, True=pruned) activates the sparse-aware-statistics
        correction (Study B): the per-channel min/max is estimated over surviving weights only,
        so to-be-pruned entries do not set the quant grid. Default (None) is byte-identical to
        the original SparseGPT quantizer."""
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
                if prune_mask is not None:
                    prune_mask = prune_mask.flatten(1)
            else:
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        if prune_mask is not None:
            # Min/max over survivors only: mask pruned entries with ±inf so they are ignored.
            x_lo = x.masked_fill(prune_mask, float('inf'))
            x_hi = x.masked_fill(prune_mask, float('-inf'))
            xmin = torch.minimum(x_lo.min(1)[0], torch.zeros(x.shape[0], device=dev))
            xmax = torch.maximum(x_hi.max(1)[0], torch.zeros(x.shape[0], device=dev))
        else:
            xmin = torch.minimum(x.min(1)[0], torch.zeros(x.shape[0], device=dev))
            xmax = torch.maximum(x.max(1)[0], torch.zeros(x.shape[0], device=dev))

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]

        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = torch.round(-xmin / self.scale)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)

    def ready(self) -> bool:
        return torch.all(self.scale != 0)


class SparseGPTLayerWrapper:
    """Wrapper to collect Hessian for SparseGPT pruning + quantization.

    Collects H = X^T X for OBS-based pruning.
    Based on the original SparseGPT implementation.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]  # out_features
        self.columns = W.shape[1]  # in_features
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """Accumulate Hessian from a batch of inputs."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = (2 / self.nsamples) ** 0.5 * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self,
        sparsity: float,
        blocksize: int = 128,
        percdamp: float = 0.01,
        sparse_aware_scale: bool = False,
        fixed_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Apply SparseGPT pruning + quantization.

        Args:
            sparsity: Target sparsity (0.35 = 35% weights pruned)
            blocksize: Block size for iterative pruning
            percdamp: Damping factor for Hessian inverse
            sparse_aware_scale: graft PRISM's sparse-aware-statistics correction onto
                SparseGPT's quantizer (Study B). Runs a two-pass scheme: pass 1 realizes the
                OBS prune mask under the standard (dense-W) grid; the grid is then recomputed
                over surviving weights only; pass 2 re-quantizes with that grid while holding
                the pass-1 mask fixed, so the ONLY change is the quant grid's support.

        Returns:
            Pruned (and optionally quantized) weight matrix
        """
        W = self.layer.weight.data.clone().float()

        # Configure quantizer if set (standard path computes the grid on dense W up-front;
        # the sparse-aware path defers the grid until the prune mask is known).
        if self.quantizer is not None and not sparse_aware_scale:
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H

        # Handle dead columns
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Compute Hessian inverse via Cholesky
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        def _sweep(Wsrc, fixed_mask=None):
            """One block-wise OBS prune+quant sweep. If fixed_mask (bool, True=pruned) is
            given, prune exactly those positions instead of recomputing the per-block OBS
            mask. Returns (quantized W, realized full prune mask)."""
            W = Wsrc.clone()
            full_mask = torch.zeros_like(W, dtype=torch.bool)
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if fixed_mask is None:
                    # Compute pruning mask for this block using OBS criterion
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
                else:
                    mask1 = fixed_mask[:, i1:i2]
                full_mask[:, i1:i2] = mask1

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    q = w.clone()
                    q[mask1[:, i]] = 0  # Prune

                    # Quantize if quantizer is configured
                    if self.quantizer is not None:
                        q = self._quantize(
                            q.unsqueeze(1),
                            self.quantizer.scale,
                            self.quantizer.zero,
                            self.quantizer.maxq
                        ).flatten()

                    Q1[:, i] = q

                    # OBS error compensation
                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                W[:, i1:i2] = Q1

                # Propagate error to remaining columns
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            return W, full_mask

        if sparse_aware_scale and self.quantizer is not None:
            # Pass 1: realize the prune mask under the standard dense-W grid.
            self.quantizer.find_params(W, weight=True)
            _, full_mask = _sweep(W)
            # Recompute the grid over survivors only, then re-quantize with the mask fixed.
            self.quantizer.scale = torch.zeros_like(self.quantizer.scale)  # invalidate ready()
            self.quantizer.find_params(W, weight=True, prune_mask=full_mask)
            Wq, _ = _sweep(W, fixed_mask=full_mask)
            return Wq.to(self.layer.weight.dtype)

        Wq, _ = _sweep(W, fixed_mask=fixed_mask)
        return Wq.to(self.layer.weight.dtype)

    def _quantize(self, x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: torch.Tensor) -> torch.Tensor:
        """RTN quantization."""
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    def free(self):
        """Free memory."""
        self.H = None
        torch.cuda.empty_cache()

# =============================================================================
# Perplexity Evaluation
# =============================================================================

def clear_sparse_cache(model):
    """Clear cached dequantized weights from quantized layers to free memory.

    This helps prevent OOM during evaluation by clearing any weight caches.
    Works with CPRLinearFused, CPRLinearMultiPrecision, and other quantized layers.
    """
    for module in model.modules():
        # Clear any _weight_dequant_cache from CPR layers
        if hasattr(module, '_weight_dequant_cache'):
            module._weight_dequant_cache = None
        # Clear any _W_cached from other quantized layers
        if hasattr(module, '_W_cached'):
            module._W_cached = None

@torch.no_grad()
def evaluate_perplexity(model, test_data, device: str = 'cuda') -> float:
    """Evaluate perplexity on test data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Clear any cached weights before evaluation to free memory
    clear_sparse_cache(model)
    gc.collect()
    torch.cuda.empty_cache()

    for i in range(len(test_data)):
        batch = test_data[i:i + 1].to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        n_tokens = batch.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

        # Periodically clear cache to prevent memory buildup
        if (i + 1) % 4 == 0:
            clear_sparse_cache(model)
            gc.collect()
            torch.cuda.empty_cache()

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    # Final cleanup
    clear_sparse_cache(model)
    gc.collect()
    torch.cuda.empty_cache()

    return perplexity

# =============================================================================
# Activation Collection
# =============================================================================

def collect_activations(model, calibration_data, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Collect activations for quantization calibration."""
    layer_activations = {}
    hooks = []
    activation_cache = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if name not in activation_cache:
                activation_cache[name] = []
            # Flatten to (tokens, in_features) so activations from calls with
            # differing sequence length / rank are concatenable along dim 0. Most
            # encoders always hand this Linear a (batch, seq, H) tensor with a
            # fixed seq, but deberta-v2/v3 disentangled attention feeds query_proj/
            # key_proj a SECOND, differently-shaped input — the (2*att_span, H)
            # relative-position embeddings (share_att_key) — so a naive dim-0 cat of
            # the raw (1,128,H) and (512,H) tensors mismatches. Downstream consumers
            # (mask_and_obs, compute_norm_scales) all reshape to (-1, H) anyway, so
            # this is numerically identical for the fixed-seq encoders (same row
            # order) and merely makes the deberta positional call concatenable.
            x = input[0].detach()
            activation_cache[name].append(x.reshape(-1, x.shape[-1]).cpu())
        return hook_fn

    # Register hooks on all linear layers
    layer_paths = get_layer_paths(model)
    for layer_idx, layer in enumerate(get_transformer_layers(model)):
        layer = layer.to(device)
        for attr_path in layer_paths:
            parts = attr_path.split('.')
            module = layer
            try:
                for p in parts:
                    module = getattr(module, p)
                if isinstance(module, nn.Linear):
                    name = f'layer_{layer_idx}.{attr_path}'
                    hooks.append(module.register_forward_hook(make_hook(name)))
            except AttributeError:
                continue

    # Run calibration forward passes
    model.eval()
    with torch.no_grad():
        for i in range(min(8, len(calibration_data))):
            batch = calibration_data[i:i+1].to(device)
            try:
                model(batch)
            except Exception:
                pass

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate activations
    for name, acts in activation_cache.items():
        layer_activations[name] = torch.cat(acts, dim=0)

    return layer_activations

# =============================================================================
# Quantization Techniques
# =============================================================================

def get_adaptive_nbits(W: torch.Tensor, target_nbits: int,
                        low_var_threshold: float = 0.001,
                        high_var_pct_threshold: float = 0.05) -> int:
    """
    Determine optimal bit precision based on weight distribution.

    Some models (e.g., Qwen2.5-3B) have layers with many near-zero rows that
    collapse to exactly zero at low bit widths (3-4 bit). This function detects
    such layers and returns a higher bit precision to avoid quantization collapse.

    Args:
        W: Weight matrix to analyze
        target_nbits: Desired bit precision
        low_var_threshold: Rows with std below this are considered "low variance"
        high_var_pct_threshold: If more than this fraction of rows are low-variance,
                               use higher precision

    Returns:
        Actual bit precision to use (>= target_nbits)
    """
    if target_nbits >= 5:
        # 5-bit and above don't have this issue
        return target_nbits

    row_stds = W.std(dim=1)
    low_var_pct = (row_stds < low_var_threshold).float().mean().item()

    if low_var_pct > high_var_pct_threshold:
        # Layer has many near-zero rows - use at least 5-bit to avoid collapse
        return max(target_nbits, 5)

    return target_nbits


def apply_sinq_quantization(model, calibration_data, nbits: int, device: str = 'cuda') -> nn.Module:
    """Apply SINQ quantization to model.

    Includes automatic detection of problematic layers (with many near-zero rows)
    that would collapse at low bit widths, and uses higher precision for those layers.
    """
    layer_paths = get_layer_paths(model)
    fallback_count = 0

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=f"SINQ {nbits}-bit")):
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

            W = linear.weight.data.clone().float()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            # Adaptive precision: detect problematic layers and use higher bits
            actual_nbits = get_adaptive_nbits(W, nbits)
            if actual_nbits != nbits:
                fallback_count += 1

            min_max = [0, 2**actual_nbits - 1]

            # Apply SINQ quantization
            W_q, scales, scale2, zeros = quantize_dual_scale_shift(
                W, min_max, method='sinq'
            )

            new_layer = SINQLinear(W_q, scales, scale2, zeros, bias, actual_nbits)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del W, linear
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    if fallback_count > 0:
        print(f"  Note: {fallback_count} layers used higher precision due to low-variance rows")

    return model


def apply_cpr_quantization(model, device: str = 'cuda', nbits: int = 8) -> nn.Module:
    """Apply CPR-SINQ quantization to model with configurable precision.

    Args:
        model: Model to quantize
        device: Device to use
        nbits: Bit precision for low-precision columns (3-8).
               High-precision columns always use INT8.
               When nbits=8, uses original CPRLinearFused (all INT8).
               When nbits<8, uses CPRLinearMultiPrecision (25% INT8, 75% at nbits).
    """
    group_size = EVAL_CONFIG['cpr_group_size']
    layer_paths = get_layer_paths(model)

    desc = f"CPR-SINQ {nbits}-bit" if nbits < 8 else "CPR-SINQ"

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=desc)):
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

            # Create CPR quantized layer
            if nbits == 8:
                # Use original INT8-only CPR
                cpr_layer = CPRLinearFused.from_linear(
                    linear,
                    group_size=group_size,
                    compute_dtype=torch.float16,
                )
            else:
                # Use multi-precision CPR (25% INT8 + 75% at nbits)
                cpr_layer = CPRLinearMultiPrecision.from_linear(
                    linear,
                    group_size=group_size,
                    low_bits=nbits,
                    high_prec_ratio=0.25,
                    compute_dtype=torch.float16,
                )
            setattr(parent, parts[-1], cpr_layer)

            del linear
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def apply_sparse_quantization(
    model,
    calibration_data,
    nbits: int,
    sparsity: float,
    device: str = 'cuda'
) -> nn.Module:
    """Apply SINQ-Sparse quantization to model."""
    # First collect activations
    layer_activations = collect_activations(model, calibration_data, device)
    layer_paths = get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=f"SINQ-Sparse {nbits}b/{int(sparsity*100)}%")):
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

            # Apply sparse quantization
            W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                W, activations,
                sparsity=sparsity,
                nbits=nbits,
                method='sinq_wanda' if activations is not None else 'sinq',
                device=device,
                use_compensation=True if activations is not None and sparsity > 0 else False,
                compensation_mode='batched_row_obs'
            )

            new_layer = SparseQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def apply_scab_quantization(
    model,
    calibration_data,
    nbits: int,
    sparsity: float,
    device: str = 'cuda'
) -> nn.Module:
    """Apply SCAB (Sinkhorn-Compensated Adaptive Bitwidth) quantization to model.

    SCAB improves upon SINQ-Sparse with:
    1. Inverse-μ importance: |w| * act_norms / (μ₁ × μ₂) - penalizes high-error weights
    2. Bit-adaptive MWC compensation: accounts for Sinkhorn's normalization factors
       with strength adapted to bit precision (50% at 3-bit, 100% at 4-bit+)
    """
    # First collect activations
    layer_activations = collect_activations(model, calibration_data, device)
    layer_paths = get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=f"SCAB {nbits}b/{int(sparsity*100)}%")):
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

            # Apply SCAB: inverse-μ importance + bit-adaptive MWC compensation
            W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                W, activations,
                sparsity=sparsity,
                nbits=nbits,
                method='sinq_wanda_inverse',  # Inverse-μ importance (key difference from SINQ-Sparse)
                device=device,
                use_compensation=True if activations is not None and sparsity > 0 else False,
                compensation_mode='bit_adaptive_mwc'  # SCAB's bit-adaptive μ-weighted compensation
            )

            new_layer = SparseQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def is_prenorm_architecture(model) -> bool:
    """Detect if model uses pre-norm architecture (LayerNorm before sublayer).

    Pre-norm models (OPT) have unnormalized activations between layers,
    making Sinkhorn μ factors unreliable. PRISM uses simplified importance
    weighting for these architectures.

    Returns:
        True if pre-norm (OPT), False if post-norm (Qwen, LLaMA)
    """
    model_type = type(model).__name__.lower()

    # Check model class name
    if 'opt' in model_type:
        return True

    # Check config if available
    if hasattr(model, 'config'):
        config = model.config
        # OPT uses do_layer_norm_before=True
        if hasattr(config, 'do_layer_norm_before') and config.do_layer_norm_before:
            return True
        # Some models use normalize_before
        if hasattr(config, 'normalize_before') and config.normalize_before:
            return True
        # Check model_type in config
        if hasattr(config, 'model_type') and 'opt' in config.model_type.lower():
            return True

    return False


def apply_prism_quantization(
    model,
    calibration_data,
    nbits: int,
    sparsity: float,
    device: str = 'cuda'
) -> nn.Module:
    """Apply PRISM (PRuning-Integrated Sparse Matrix) quantization to model.

    PRISM is the state-of-the-art joint pruning+quantization technique:
    1. Iterative importance refinement (adapts to sparse structure)
    2. OBS compensation (redistributes pruning error)
    3. Sparse-aware Sinkhorn normalization (computes μ only on non-zero weights)

    Architecture-aware (NEW):
    - Post-norm (Qwen, LLaMA): Full inverse-μ importance + sparse-aware Sinkhorn
    - Pre-norm (OPT): Simplified Wanda-style importance + standard Sinkhorn

    Empirical results:
    - Beats SparseGPT at sparsity ≤50%
    - 35% sparsity: 24.6% better than SparseGPT
    - 50% sparsity: 9.7% better than SparseGPT

    The sparse-aware normalization provides 3-10% additional improvement
    over standard Sinkhorn normalization (used in SCAB).
    """
    # Detect architecture type
    is_prenorm = is_prenorm_architecture(model)
    if is_prenorm:
        print(f"  [PRISM] Detected pre-norm architecture - using simplified importance weighting")

    # First collect activations
    layer_activations = collect_activations(model, calibration_data, device)
    layer_paths = get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=f"PRISM {nbits}b/{int(sparsity*100)}%")):
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

            # Apply PRISM: iterative refinement + OBS + sparse-aware Sinkhorn
            # Architecture-aware: use simplified importance for pre-norm (OPT)
            W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                W, activations,
                sparsity=sparsity,
                nbits=nbits,
                method='sinq_wanda_inverse',  # Inverse-μ importance (post-norm only)
                device=device,
                use_compensation=True if activations is not None and sparsity > 0 else False,
                compensation_mode='prism',  # PRISM's sparse-aware Sinkhorn
                is_prenorm=is_prenorm  # NEW: architecture-aware handling
            )

            new_layer = SparseQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


# Component map for the within-SINQ 2×2 ablation (Study A). Each cell runs through
# the IDENTICAL sparse_quantize_sinq pipeline as PRISM; only the listed toggles flip,
# so any PPL difference is attributable solely to those toggles.
#   abl-base   : Wanda imp,    no OBS, dense Sinkhorn       (correction OFF, extras OFF)
#   abl-corr   : Wanda imp,    no OBS, sparse-aware Sinkhorn(correction ON,  extras OFF)
#   abl-extras : inverse-μ imp, OBS,   dense Sinkhorn       (correction OFF, extras ON)  == prism-nocorr
#   abl-full   : inverse-μ imp, OBS,   sparse-aware Sinkhorn(correction ON,  extras ON)  == prism (sanity)
PRISM_ABLATION_MODES = {
    'abl-base':   ('abl_wanda_dense',  'sinq_wanda'),
    'abl-corr':   ('abl_wanda_corr',   'sinq_wanda'),
    'abl-extras': ('abl_prism_nocorr', 'sinq_wanda_inverse'),
    'abl-full':   ('abl_prism_full',   'sinq_wanda_inverse'),
}


def apply_prism_ablation_quantization(
    model,
    calibration_data,
    nbits: int,
    sparsity: float,
    device: str = 'cuda',
    compensation_mode: str = 'abl_prism_full',
    method: str = 'sinq_wanda_inverse',
) -> nn.Module:
    """Run a single cell of the PRISM 2×2 component ablation (Study A).

    Byte-identical to apply_prism_quantization except it forwards an explicit
    compensation_mode/method so the four ablation cells differ ONLY by the intended
    toggle (pruning metric / OBS / sparse-aware correction). See PRISM_ABLATION_MODES.
    """
    is_prenorm = is_prenorm_architecture(model)
    if is_prenorm:
        print(f"  [PRISM-ABL] Detected pre-norm architecture - using simplified importance weighting")

    layer_activations = collect_activations(model, calibration_data, device)
    layer_paths = get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model),
                                           desc=f"PRISM-ABL[{compensation_mode}] {nbits}b/{int(sparsity*100)}%")):
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

            W_q, scales, zeros, mask, scale2, meta = sparse_quantize_sinq(
                W, activations,
                sparsity=sparsity,
                nbits=nbits,
                method=method,
                device=device,
                use_compensation=True if activations is not None and sparsity > 0 else False,
                compensation_mode=compensation_mode,
                is_prenorm=is_prenorm
            )

            new_layer = SparseQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def apply_cpr_sparse_quantization(
    model,
    calibration_data,
    sparsity: float,
    device: str = 'cuda',
    nbits: int = 8
) -> nn.Module:
    """Apply CPR-Sparse (CPR + sparsity) quantization to model.

    This combines CPR quantization with unstructured sparsity.
    The approach:
    1. Apply sparsity mask to weights
    2. Quantize remaining weights with CPR at specified precision

    Args:
        nbits: Bit precision for low-precision columns (3-8).
               When nbits=8, uses original CPRLinearFused.
               When nbits<8, uses CPRLinearMultiPrecision.
    """
    # Collect activations for importance scoring
    layer_activations = collect_activations(model, calibration_data, device)
    group_size = EVAL_CONFIG['cpr_group_size']
    layer_paths = get_layer_paths(model)

    desc = f"CPR-Sparse {nbits}-bit {int(sparsity*100)}%" if nbits < 8 else f"CPR-Sparse {int(sparsity*100)}%"
    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=desc)):
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

            W = linear.weight.data.clone().float()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            # Get activations for importance
            act_key = f'layer_{layer_idx}.{attr_path}'
            activations = layer_activations.get(act_key, None)

            # Compute sparsity mask using Wanda-style importance
            if activations is not None and sparsity > 0:
                activations = activations.to(device).float()
                if activations.dim() == 3:
                    activations = activations.view(-1, activations.shape[-1])
                act_norms = torch.norm(activations, dim=0)  # [in_features]

                # Importance = |w| * activation_norm (per column)
                importance = W.abs() * act_norms.unsqueeze(0)

                # Per-row pruning
                K, N = W.shape
                n_prune_per_row = int(N * sparsity)
                mask = torch.ones_like(W)
                _, sorted_indices = importance.sort(dim=1)
                prune_indices = sorted_indices[:, :n_prune_per_row]
                mask.scatter_(1, prune_indices, 0)

                # Apply mask
                W = W * mask
                del activations
            else:
                mask = torch.ones_like(W)

            # Apply sparse weights to linear layer temporarily
            linear.weight.data = W.to(linear.weight.dtype)

            # Create CPR quantized layer from sparse weights
            if nbits == 8:
                cpr_layer = CPRLinearFused.from_linear(
                    linear,
                    group_size=group_size,
                    compute_dtype=torch.float16,
                )
            else:
                cpr_layer = CPRLinearMultiPrecision.from_linear(
                    linear,
                    group_size=group_size,
                    low_bits=nbits,
                    high_prec_ratio=0.25,
                    compute_dtype=torch.float16,
                )
            setattr(parent, parts[-1], cpr_layer)

            del W, linear
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def apply_wanda_pruning(
    model,
    calibration_data,
    sparsity: float,
    device: str = 'cuda'
) -> nn.Module:
    """Apply Wanda pruning to model (no quantization).

    Wanda (Weights AND Activations) prunes weights based on:
        importance = |W| * sqrt(activation_norm_squared)

    This is the pure Wanda baseline without any quantization.

    Args:
        model: Model to prune
        calibration_data: Calibration data for collecting activation statistics
        sparsity: Target sparsity ratio (0.35 = 35% weights pruned)
        device: Device to use

    Returns:
        Pruned model
    """
    layer_paths = get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=f"Wanda {int(sparsity*100)}%")):
        layer = layer.to(device)

        # Find all linear layers in this transformer block
        subset = {}
        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue

            if isinstance(linear, nn.Linear):
                subset[attr_path] = linear

        # Wrap layers to collect activation statistics
        wrapped_layers = {}
        for name, linear in subset.items():
            wrapped_layers[name] = WandaLayerWrapper(linear)

        # Register forward hooks to collect activations
        def make_add_batch_hook(name):
            def hook_fn(module, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return hook_fn

        handles = []
        for name, linear in subset.items():
            handles.append(linear.register_forward_hook(make_add_batch_hook(name)))

        # Run calibration forward passes
        with torch.no_grad():
            for i in range(min(8, len(calibration_data))):
                batch = calibration_data[i:i+1].to(device)
                try:
                    model(batch)
                except Exception:
                    pass

        # Remove hooks
        for h in handles:
            h.remove()

        # Prune each linear layer using Wanda importance metric
        for attr_path, linear in subset.items():
            W = linear.weight.data.clone().float()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            # Wanda importance: |W| * sqrt(scaler_row)
            # scaler_row is the running mean of squared L2 norms per column
            scaler_row = wrapped_layers[attr_path].scaler_row
            W_metric = torch.abs(W) * torch.sqrt(scaler_row.reshape(1, -1))

            # Per-row pruning (unstructured but balanced across rows)
            K, N = W.shape
            n_prune_per_row = int(N * sparsity)

            # Sort by importance within each row
            sort_res = torch.sort(W_metric, dim=1, stable=True)
            indices = sort_res[1][:, :n_prune_per_row]

            # Create mask and set pruned weights to zero
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            W_mask.scatter_(1, indices, True)
            W[W_mask] = 0

            # Replace with sparse linear layer
            parts = attr_path.split('.')
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)

            new_layer = SparseLinear(W.to(linear.weight.dtype), bias)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del W, linear
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def _apply_jsq(
    model,
    calibration_data,
    nbits: int,
    sparsity: float,
    device: str = 'cuda',
    weight_only: bool = False,
    rho: float = 2.1,
    clip_h: float = 0.01,
    alpha: float = 0.5,
    nsamples_calib: int = 16,
    sparse_aware_scale: bool = False,
) -> nn.Module:
    """JSQ (Guo et al., ICML 2024) joint sparsification + quantization.

    Per decoder layer (sequential via the standard per-layer harness, like Wanda/SparseGPT):
      1. Calibrate: hooks capture per-linear *clipped* input stats (activation editing,
         clip_h=0.01): Wanda L2-norm, mean activation, per-channel max.
      2. Prune: W_metric = |W|*sqrt(scaler_row) + rho*ss, where ss is the SAR
         quantization-range sensitivity (the prune<->quant 'bridge'). Per-row unstructured.
      3. Smooth (SmoothQuant alpha=0.5): migrate activation scale into weights. Skipped in
         weight_only mode (SmoothQuant only helps when activations are also quantized).
      4. Quantize: per-channel weight absmax to nbits; faithful mode also per-token activation
         absmax to nbits (WnAn, with q/k/v output also quantized as upstream). weight_only mode
         leaves activations in fp (apples-to-apples with PRISM/SparseGPT weight-only).

    Core math is a faithful port of temp/JSQ/jsq/* (component-equivalent; see jsq_port.py).
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from jsq_port import JSQStatWrapper, generate_ss, smooth_ln_fcs, WnAnLinear
    if _os.environ.get('JSQ_RHO') is not None:
        rho = float(_os.environ['JSQ_RHO'])
    if _os.environ.get('JSQ_CLIPH') is not None:
        clip_h = float(_os.environ['JSQ_CLIPH'])

    layer_paths = get_layer_paths(model)
    n_calib = min(nsamples_calib, len(calibration_data))
    # q/k/v outputs are activation-quantized too in faithful mode (simulates BMM-input quant)
    qkv_names = {'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'}

    mode = 'JSQ-WnAn' if not weight_only else 'JSQ-Wonly'
    desc = f"{mode} {int(sparsity*100)}% + {nbits}-bit"
    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=desc)):
        layer = layer.to(device)

        # ---- collect this block's linear layers ----
        subset = {}
        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue
            if isinstance(linear, nn.Linear):
                subset[attr_path] = linear

        # ---- 1. calibrate (hooks capture clipped-input stats) ----
        wrapped = {name: JSQStatWrapper(lin, clip_h=clip_h) for name, lin in subset.items()}

        def make_hook(name):
            def hook_fn(module, inp, out):
                wrapped[name].add_batch(inp[0])
            return hook_fn

        handles = [lin.register_forward_hook(make_hook(name)) for name, lin in subset.items()]
        _fwd_ok = 0
        with torch.no_grad():
            for i in range(n_calib):
                try:
                    model(calibration_data[i:i+1].to(device))
                    _fwd_ok += 1
                except Exception as _e:
                    if os.environ.get('JSQ_DEBUG'):
                        print(f"[jsq L{layer_idx}] fwd exc: {type(_e).__name__}: {str(_e)[:100]}")
        for h in handles:
            h.remove()
        if os.environ.get('JSQ_DEBUG'):
            _sr = wrapped.get('self_attn.q_proj')
            print(f"[jsq L{layer_idx}] fwd_ok={_fwd_ok}/{n_calib} "
                  f"q_proj.scaler_row.norm={_sr.scaler_row.norm():.4f} nsamples={_sr.nsamples}")

        # ---- 2. prune each linear with the joint metric ----
        for name, linear in subset.items():
            W = linear.weight.data.float()
            scaler_row = wrapped[name].scaler_row
            mean_act = wrapped[name].mean_activation()
            ss = generate_ss(mean_act, W)
            # Wanda salience I = ||X_j||_2 * |W| (paper Eq. 2: Frobenius norm SUMMED over
            # tokens). The released code uses sqrt(scaler_row) = RMS = ||X||_2/sqrt(N), which is
            # harmless for pure-Wanda ranking but makes I ~sqrt(N)x too small relative to the
            # range term A, so lambda*A swamps I and pruning collapses. Restoring the paper's
            # token-summed L2 norm rebalances S = I + rho*A. (JSQ_FROB=0 reverts to released RMS.)
            ntok = max(wrapped[name].nsamples, 1)
            frob = ntok if _os.environ.get('JSQ_FROB', '1') == '1' else 1.0
            wanda = torch.abs(W) * torch.sqrt(scaler_row.reshape(1, -1) * frob)
            W_metric = wanda + rho * ss

            N = W.shape[1]
            n_prune = int(N * sparsity)
            if n_prune > 0:
                idx = torch.sort(W_metric, dim=1, stable=True)[1][:, :n_prune]
                mask = torch.zeros_like(W_metric, dtype=torch.bool)
                mask.scatter_(1, idx, True)
                W[mask] = 0
                linear.weight.data = W.to(linear.weight.dtype)

        # ---- 3. SmoothQuant smoothing (BOTH variants). Smoothing migrates activation scale into
        #         weights; measured to help weight-only quant too (4b/50%: 105->70, 5b neutral), so
        #         it is NOT activation-quant-specific. "Turn off activation quant" = drop only the
        #         A-quant in step 4, keep smoothing. (JSQ_NO_SMOOTH=1 disables it for ablation.) ----
        if _os.environ.get('JSQ_NO_SMOOTH') != '1':
            try:
                # Gemma's RMSNorm applies a (1+weight) gain; SmoothQuant scale migration
                # must account for that or it overflows fp16 -> NaN (see smooth_ln_fcs).
                _uo = 'Gemma' in type(getattr(layer, 'input_layernorm', None)).__name__
                if hasattr(layer, 'input_layernorm') and hasattr(layer, 'self_attn'):
                    qkv = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
                    smooth_ln_fcs(layer.input_layernorm, qkv,
                                  wrapped['self_attn.q_proj'].act_max, alpha, unit_offset=_uo)
                if hasattr(layer, 'post_attention_layernorm') and hasattr(layer, 'mlp') \
                        and hasattr(layer.mlp, 'gate_proj'):
                    smooth_ln_fcs(layer.post_attention_layernorm,
                                  [layer.mlp.gate_proj, layer.mlp.up_proj],
                                  wrapped['mlp.gate_proj'].act_max, alpha, unit_offset=_uo)
            except (KeyError, AttributeError):
                pass  # architecture without the standard LN<->fc pairs: skip smoothing

        # ---- 4. quantize (replace linears with WnAnLinear) ----
        for name, linear in subset.items():
            parts = name.split('.')
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            new_linear = WnAnLinear.from_float(
                linear, nbits=nbits,
                act_quant=not weight_only,
                quantize_output=(not weight_only) and (name in qkv_names),
                sparse_aware_scale=sparse_aware_scale,
            ).to(device)
            setattr(parent, parts[-1], new_linear)
            del linear

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def apply_jsq_quantization(model, calibration_data, nbits: int, sparsity: float, device: str = 'cuda'):
    """Faithful JSQ: joint prune + WnAn (weights AND activations quantized to nbits)."""
    return _apply_jsq(model, calibration_data, nbits, sparsity, device, weight_only=False)


def apply_jsq_weightonly_quantization(model, calibration_data, nbits: int, sparsity: float,
                                      device: str = 'cuda', sparse_aware_scale: bool = False):
    """JSQ weight-only variant: joint prune + weight-only quant (no activation quant, no
    smoothing) for apples-to-apples comparison with PRISM/SparseGPT at matched bit-width.

    sparse_aware_scale=True grafts PRISM's sparse-aware-statistics correction onto JSQ's
    per-channel absmax RTN (Study B). Because the scale is a max, this is an exact no-op — a
    negative control showing the correction is variance/Sinkhorn-specific (see jsq_port.py)."""
    return _apply_jsq(model, calibration_data, nbits, sparsity, device, weight_only=True,
                      sparse_aware_scale=sparse_aware_scale)


def apply_slim_quantization(model, calibration_data, nbits: int, sparsity: float,
                            device: str = 'cuda', rank_ratio: float = 0.1,
                            quantize_weight: bool = True, slim_lora: bool = True,
                            sparse_aware_cap: bool = False) -> nn.Module:
    """SLiM-LoRA (Mozaffari et al., ICML 2025): the headline joint prune+quant+low-rank method.

    Per decoder layer (sequential via the standard per-layer harness, like Wanda/SparseGPT/JSQ):
      1. Calibrate: hooks capture per-linear Wanda activation norm (scaler_row), with the
         `shift_zeros` shift SLiM applies (shift_zero_metrics=True).
      2. Wanda prune: mask the smallest |W|*sqrt(scaler_row) per output row (unstructured).
      3. SLiM-Quant + saliency low-rank: `slim_lora_decompose` quantizes the survivors with
         per-matrix MSE-optimal-clip symmetric quant (SLiM-Quant) and fits a rank-`r` adapter to
         the activation-saliency-weighted compression error (SVD), re-quantizing W - LR. The
         fp16 adapter LR is added back at inference (SLiMLinear).

    Core math is a faithful, COMPONENT-BIT-EXACT port of temp/SLiM (see slim_port.py +
    scripts/slim_component_test.py). Unlike PRISM/SparseGPT/JSQ, SLiM's low-rank adapter adds
    ~rank_ratio extra params per matrix (NOT pure prune+quant) -- this overhead is kept explicit
    in SLiMLinear (sparse weight stored separately from the dense adapter), not folded away.

    Env knobs (ablation): SLIM_RANK (rank ratio, default 0.1), SLIM_NO_QUANT=1 (prune-only,
    paper Table 13), SLIM_NAIVE=1 (Naive-LoRA: plain error SVD instead of saliency-weighted),
    SLIM_NSAMPLES (calibration sample cap).
    """
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from slim_port import slim_lora_decompose, SLiMLinear, shift_zeros

    if _os.environ.get('SLIM_RANK') is not None:
        rank_ratio = float(_os.environ['SLIM_RANK'])
    if _os.environ.get('SLIM_NO_QUANT') == '1':
        quantize_weight = False
    if _os.environ.get('SLIM_NAIVE') == '1':
        slim_lora = False

    layer_paths = get_layer_paths(model)
    n_calib = len(calibration_data)
    if _os.environ.get('SLIM_NSAMPLES') is not None:
        n_calib = min(n_calib, int(_os.environ['SLIM_NSAMPLES']))

    adapter = 'SLiM-LoRA' if slim_lora else 'Naive-LoRA'
    qd = f"{nbits}-bit SLiM-Quant" if quantize_weight else "no-quant"
    desc = f"SLiM {int(sparsity*100)}% + {qd} ({adapter} r{rank_ratio})"
    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=desc)):
        layer = layer.to(device)

        # ---- collect this block's linear layers ----
        subset = {}
        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue
            if isinstance(linear, nn.Linear):
                subset[attr_path] = linear

        # ---- 1. calibrate: Wanda activation norm per linear ----
        wrapped = {name: WandaLayerWrapper(lin) for name, lin in subset.items()}

        def make_hook(name):
            def hook_fn(module, inp, out):
                wrapped[name].add_batch(inp[0].data, out.data)
            return hook_fn

        handles = [lin.register_forward_hook(make_hook(name)) for name, lin in subset.items()]
        with torch.no_grad():
            for i in range(n_calib):
                try:
                    model(calibration_data[i:i+1].to(device))
                except Exception:
                    pass  # forward dies at the first not-yet-moved CPU layer; stats already captured
        for h in handles:
            h.remove()

        # ---- 2+3. prune + SLiM-Quant + saliency low-rank, replace each linear ----
        for name, linear in subset.items():
            W = linear.weight.data
            scaler_row = wrapped[name].scaler_row.float()
            scaler_row = shift_zeros(scaler_row)  # SLiM shift_zero_metrics=True

            # Wanda mask (per output row, unstructured)
            W_metric = torch.abs(W) * torch.sqrt(scaler_row.reshape(1, -1))
            N = W.shape[1]
            n_prune = int(N * sparsity)
            mask = torch.zeros_like(W_metric, dtype=torch.bool)
            if n_prune > 0:
                idx = torch.sort(W_metric, dim=1, stable=True)[1][:, :n_prune]
                mask.scatter_(1, idx, True)

            Wc, ll, lr = slim_lora_decompose(
                W, mask, scaler_row, nbits, rank_ratio,
                quantize=quantize_weight, slim_lora=slim_lora,
                sparse_aware_cap=sparse_aware_cap)

            bias = linear.bias.data.clone() if linear.bias is not None else None
            parts = name.split('.')
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            new_linear = SLiMLinear(Wc.to(W.dtype), ll.to(W.dtype), lr.to(W.dtype), bias).to(device)
            setattr(parent, parts[-1], new_linear)
            del linear

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def apply_sparsegpt_pruning(
    model,
    calibration_data,
    sparsity: float,
    nbits: int,
    device: str = 'cuda',
    sparse_aware_scale: bool = False,
    sparsity_by_type: dict = None,
    mask_fn=None,
    percdamp: float = 0.01,
    blocksize: int = 128
) -> nn.Module:
    """Apply SparseGPT pruning + quantization to model.

    percdamp / blocksize (bit-MATCHED quality knobs, do NOT change bpw): the OBS
    Hessian Tikhonov damping (ridge = percdamp * mean(diag(H))) and the error-
    compensation block width. Exposed for hyperparameter grid search; defaults
    (0.01 / 128) reproduce the original hardcoded behaviour exactly.

    mask_fn (optional): callable(attr_path, W, H) -> bool tensor (True=pruned) that
    OVERRIDES SparseGPT's internal OBS mask with an externally supplied prune mask,
    while KEEPING SparseGPT's joint (quant-in-the-loop) sequential OBS recipe. Used
    for the mask-vs-recipe causal decomposition (feed CoBALT's balanced mask into the
    SparseGPT recipe). W and H are float [rows,cols] / [cols,cols] on `device`.

    SparseGPT uses OBS (Optimal Brain Surgeon) criterion for joint pruning and quantization:
        - Importance = W^2 / diag(H_inv)^2
        - Iterative block-wise pruning with error compensation
        - Optional per-channel quantization

    This is the joint sparse + quantization baseline.

    Args:
        model: Model to prune and quantize
        calibration_data: Calibration data for collecting Hessian
        sparsity: Target sparsity ratio (0.35 = 35% weights pruned)
        nbits: Quantization bit-width (0 = no quantization, pruning only)
        device: Device to use

    Returns:
        Pruned and quantized model
    """
    layer_paths = get_layer_paths(model)

    desc = f"SparseGPT {int(sparsity*100)}%"
    if nbits > 0:
        desc += f" + {nbits}-bit"

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=desc)):
        layer = layer.to(device)

        # Find all linear layers in this transformer block
        subset = {}
        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue

            if isinstance(linear, nn.Linear):
                subset[attr_path] = linear

        # Wrap layers to collect Hessian
        wrapped_layers = {}
        for name, linear in subset.items():
            wrapped_layers[name] = SparseGPTLayerWrapper(linear)
            # Configure quantizer if bits > 0
            if nbits > 0:
                wrapped_layers[name].quantizer = SparseGPTQuantizer()
                wrapped_layers[name].quantizer.configure(nbits, perchannel=True, sym=False)

        # Register forward hooks to collect Hessian
        def make_add_batch_hook(name):
            def hook_fn(module, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return hook_fn

        handles = []
        for name, linear in subset.items():
            handles.append(linear.register_forward_hook(make_add_batch_hook(name)))

        # Run calibration forward passes
        with torch.no_grad():
            for i in range(min(8, len(calibration_data))):
                batch = calibration_data[i:i+1].to(device)
                try:
                    model(batch)
                except Exception:
                    pass

        # Remove hooks
        for h in handles:
            h.remove()

        # Apply SparseGPT pruning + quantization to each linear layer
        for attr_path, linear in subset.items():
            bias = linear.bias.data.clone() if linear.bias is not None else None

            # Apply SparseGPT algorithm (per-type sparsity for v-dense, else uniform)
            sp = _effective_sparsity(attr_path, sparsity, sparsity_by_type)
            fm = None
            if mask_fn is not None:
                wl = wrapped_layers[attr_path]
                fm = mask_fn(attr_path, wl.layer.weight.data.clone().float(), wl.H.clone())
            W_pruned = wrapped_layers[attr_path].fasterprune(
                sparsity=sp,
                blocksize=blocksize,
                percdamp=percdamp,
                sparse_aware_scale=sparse_aware_scale,
                fixed_mask=fm
            )

            # Free Hessian memory
            wrapped_layers[attr_path].free()

            # Replace with sparse linear layer
            parts = attr_path.split('.')
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)

            new_layer = SparseLinear(W_pruned, bias)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del linear
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


# =============================================================================
# Additional literature baselines: AWQ, GPTQ, SpQR, Wanda+AWQ, Wanda+SINQ
# (added 2026-06-23 — see llmdocs/CONTEXT.md §8)
#
# All of these follow the repo's established methodology: the perplexity path
# does *fake-quant in float* (weights are quantized to N levels then stored
# dense), exactly like the existing SparseGPT / Wanda baselines, which also wrap
# their result in SparseLinear. This keeps every technique on an identical
# evaluation footing.
# =============================================================================

def _largest_divisor_leq(n: int, target: int) -> int:
    """Largest divisor of n that is <= target (and >= 1).

    Used to pick a group/tile size that evenly divides a layer's input dim,
    so the tiled AWQ / group-wise quant paths work for *any* model (not just
    those whose hidden sizes are multiples of 128).
    """
    target = min(target, n)
    for b in range(target, 0, -1):
        if n % b == 0:
            return b
    return 1


def _sinq_dequantize_dense(W_q, scales, scale2, zeros):
    """Dequantize SINQ dual-scale outputs to a dense weight matrix.

    Mirrors SINQLinear.forward exactly (per-row or group-wise scales):
        W = ((Q - z) * s1) * s2
    """
    Q = W_q.float()
    z = zeros.float()
    s1 = scales.float()
    s2 = scale2.float()
    if s1.dim() == 3:
        # Group-wise: s1/z are [K, n_groups, 1]
        K, N = Q.shape
        n_groups = s1.shape[1]
        group_size = N // n_groups
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq = ((Q_grouped - z) * s1).view(K, N)
    else:
        W_deq = (Q - z) * s1
    return W_deq * s2


def _wanda_scaler_row(acts: torch.Tensor, device: str) -> torch.Tensor:
    """Per-input-channel activation energy used by the Wanda importance metric.

    Returns mean-of-squares per column; ranking is identical to the running
    L2^2 accumulator in WandaLayerWrapper (so the resulting prune mask matches
    the existing `wanda` baseline).
    """
    X = acts.to(device).float()
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    return X.pow(2).mean(dim=0)  # [N]


def _wanda_row_mask(W: torch.Tensor, scaler_row: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Per-output-row Wanda prune mask (1 = keep, 0 = prune).

    Identical criterion to apply_wanda_pruning: importance = |W| * sqrt(scaler_row),
    prune the lowest `sparsity` fraction within each row.
    """
    W_metric = W.abs().float() * torch.sqrt(scaler_row.reshape(1, -1))
    K, N = W.shape
    n_prune = int(N * sparsity)
    mask = torch.ones_like(W_metric)
    if n_prune > 0:
        idx = torch.sort(W_metric, dim=1, stable=True)[1][:, :n_prune]
        mask.scatter_(1, idx, 0.0)
    return mask


def _groupwise_quant_dense(W: torch.Tensor, nbits: int, groupsize: int,
                           keep_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Group-wise asymmetric RTN quant along the input dim; returns dense dequant.

    If keep_mask is given (1 = normal, 0 = outlier), the per-group min/max are
    computed *excluding* outliers so a few large weights don't inflate the grid
    for the rest of the group — this is the key mechanism behind SpQR.
    """
    K, N = W.shape
    maxq = 2 ** nbits - 1
    gs = _largest_divisor_leq(N, groupsize)
    ng = N // gs
    Wg = W.view(K, ng, gs)

    if keep_mask is None:
        wmin = Wg.amin(dim=2, keepdim=True)
        wmax = Wg.amax(dim=2, keepdim=True)
    else:
        Mg = keep_mask.view(K, ng, gs) > 0
        big = torch.where(Mg, Wg, torch.full_like(Wg, float('inf')))
        small = torch.where(Mg, Wg, torch.full_like(Wg, float('-inf')))
        wmin = big.amin(dim=2, keepdim=True)
        wmax = small.amax(dim=2, keepdim=True)
        # Groups that are entirely outliers fall back to the full range
        wmin = torch.where(torch.isinf(wmin), Wg.amin(dim=2, keepdim=True), wmin)
        wmax = torch.where(torch.isinf(wmax), Wg.amax(dim=2, keepdim=True), wmax)

    wmin = torch.minimum(wmin, torch.zeros_like(wmin))
    wmax = torch.maximum(wmax, torch.zeros_like(wmax))
    scale = (wmax - wmin).clamp(min=1e-8) / maxq
    zero = torch.round(-wmin / scale)
    q = torch.clamp(torch.round(Wg / scale) + zero, 0, maxq)
    deq = (q - zero) * scale
    return deq.view(K, N)


class GPTQLayerWrapper:
    """Canonical GPTQ (Frantar et al., 2022) — ported from /workspace/CRB/gptq/gptq.py.

    Collects the input Hessian H = X^T X and applies group-wise OBS quantization
    with block-wise error compensation. Reuses SparseGPTQuantizer for find_params.
    Unlike SparseGPTLayerWrapper this re-estimates the quant grid every `groupsize`
    columns (group-wise scales), which is what published GPTQ-g128 numbers use.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        self.dev = layer.weight.device
        W = layer.weight.data.clone()
        self.rows, self.columns = W.shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(self, blocksize: int = 128, percdamp: float = 0.01, groupsize: int = 128) -> torch.Tensor:
        W = self.layer.weight.data.clone().float()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Q = torch.zeros_like(W)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                q = self._quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        return Q.to(self.layer.weight.dtype)

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        s, z, mq = self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
        q = torch.clamp(torch.round(x / s) + z, 0, mq)
        return s * (q - z)

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def apply_awq_quantization(model, calibration_data, nbits: int, device: str = 'cuda',
                           groupsize: int = 128) -> nn.Module:
    """Apply AWQ (Activation-aware Weight Quantization, Lin et al. 2023).

    Uses the SINQ authors' own AWQ implementation (sinq/awq.py): a grid search
    over per-input-channel scales that minimises layer-output reconstruction
    error, followed by group-wise RTN of the scaled weights.
    """
    layer_activations = collect_activations(model, calibration_data, device)
    layer_paths = get_layer_paths(model)
    min_max = [0, 2 ** nbits - 1]

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=f"AWQ {nbits}-bit")):
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

            W = linear.weight.data.clone().float()
            bias = linear.bias.data.clone() if linear.bias is not None else None
            N = W.shape[1]
            block = _largest_divisor_leq(N, groupsize)

            acts = layer_activations.get(f'layer_{layer_idx}.{attr_path}', None)
            if acts is not None:
                acts = acts.to(device).float()
                awq_scales = compute_awq_scale(W, acts, min_max, tile=block, method='awq')
            else:
                awq_scales = torch.ones(N, device=device)
            W_fq = tiled_fake_quant_rectangle(
                W, fakequant=rtn_fake_quant, min_max=min_max, block=block, scales=awq_scales
            )

            new_layer = SparseLinear(W_fq.to(linear.weight.dtype), bias).to(device)
            setattr(parent, parts[-1], new_layer)
            del W, linear
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def apply_gptq_quantization(model, calibration_data, nbits: int, device: str = 'cuda',
                            groupsize: int = 128) -> nn.Module:
    """Apply GPTQ (Frantar et al. 2022): Hessian-based OBS quantization, group-wise."""
    layer_paths = get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=f"GPTQ {nbits}-bit")):
        layer = layer.to(device)

        subset = {}
        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue
            if isinstance(linear, nn.Linear):
                subset[attr_path] = linear

        wrapped = {}
        for name, linear in subset.items():
            wrapped[name] = GPTQLayerWrapper(linear)
            wrapped[name].quantizer = SparseGPTQuantizer()
            wrapped[name].quantizer.configure(nbits, perchannel=True, sym=False)

        def make_hook(name):
            def hook_fn(module, inp, out):
                wrapped[name].add_batch(inp[0].data, out.data)
            return hook_fn

        handles = [linear.register_forward_hook(make_hook(name)) for name, linear in subset.items()]
        with torch.no_grad():
            for i in range(min(8, len(calibration_data))):
                try:
                    model(calibration_data[i:i+1].to(device))
                except Exception:
                    pass
        for h in handles:
            h.remove()

        for attr_path, linear in subset.items():
            bias = linear.bias.data.clone() if linear.bias is not None else None
            Q = wrapped[attr_path].fasterquant(blocksize=128, percdamp=0.01, groupsize=groupsize)
            wrapped[attr_path].free()

            parts = attr_path.split('.')
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            new_layer = SparseLinear(Q, bias).to(device)
            setattr(parent, parts[-1], new_layer)
            del linear
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def apply_spqr_quantization(model, calibration_data, nbits: int, device: str = 'cuda',
                            groupsize: int = 16, outlier_rate: float = 0.01) -> nn.Module:
    """Apply SpQR (Sparse-Quantized Representation, Dettmers et al. 2023).

    Core idea: quantize weights in *small* groups (β=16) and keep the most
    sensitive ~1% of weights in fp16 as a sparse outlier set, with the per-group
    grid recomputed *excluding* those outliers. This is what drives SpQR's
    near-lossless quality.

    Note: we report perplexity (fake-quant in float), so the secondary
    bi-level quantization of the per-group statistics (a storage optimisation
    that does not materially change PPL) is intentionally omitted. The labelled
    bit-width refers to the base grid; effective bits/param are higher once the
    fp16 outliers + group scales are counted.
    """
    layer_activations = collect_activations(model, calibration_data, device)
    layer_paths = get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model), desc=f"SpQR {nbits}-bit")):
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

            W = linear.weight.data.clone().float()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            acts = layer_activations.get(f'layer_{layer_idx}.{attr_path}', None)
            if acts is not None:
                col_imp = _wanda_scaler_row(acts, device).sqrt()  # ~ diag(H) sensitivity
            else:
                col_imp = torch.ones(W.shape[1], device=device)

            # Pass 1: base group quant -> sensitivity-weighted error per weight
            deq0 = _groupwise_quant_dense(W, nbits, groupsize)
            err = (W - deq0).abs() * col_imp.view(1, -1)

            # Outlier mask: most-sensitive `outlier_rate` fraction kept in fp16
            keep = torch.ones_like(W)
            k = int(W.numel() * outlier_rate)
            if k > 0:
                thr = torch.kthvalue(err.view(-1), W.numel() - k).values
                keep[err > thr] = 0.0

            # Pass 2: requantize with outliers excluded from group stats
            deq = _groupwise_quant_dense(W, nbits, groupsize, keep_mask=keep)
            W_final = torch.where(keep > 0, deq, W)

            new_layer = SparseLinear(W_final.to(linear.weight.dtype), bias).to(device)
            setattr(parent, parts[-1], new_layer)
            del W, linear
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def _effective_sparsity(attr_path, sparsity, sparsity_by_type):
    """Per-matrix sparsity: scalar `sparsity` unless a per-type dict is given (e.g. v-dense:
    {'v':0.0, others:0.7033}). Type = the linear's suffix ('self_attn.v_proj' -> 'v')."""
    if not sparsity_by_type:
        return sparsity
    suffix = attr_path.split('.')[-1]
    t = suffix[:-len('_proj')] if suffix.endswith('_proj') else suffix
    return sparsity_by_type.get(t, sparsity)


def apply_wanda_awq_quantization(model, calibration_data, nbits: int, sparsity: float,
                                 device: str = 'cuda', groupsize: int = 128,
                                 sparsity_by_type: dict = None) -> nn.Module:
    """Wanda pruning + AWQ quantization of the surviving weights.

    Composes the existing `wanda` per-row mask with AWQ: prune, AWQ-quantize the
    pruned matrix, then re-impose the zeros so structural sparsity is exact.
    `sparsity_by_type` (optional) enables per-type sparsity (e.g. v-dense) — held EQUAL
    across arms for the fairness comparison; default None = uniform `sparsity`.
    """
    layer_activations = collect_activations(model, calibration_data, device)
    layer_paths = get_layer_paths(model)
    min_max = [0, 2 ** nbits - 1]

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model),
                                           desc=f"Wanda+AWQ {nbits}b/{int(sparsity*100)}%")):
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

            W = linear.weight.data.clone().float()
            bias = linear.bias.data.clone() if linear.bias is not None else None
            N = W.shape[1]
            block = _largest_divisor_leq(N, groupsize)

            sp = _effective_sparsity(attr_path, sparsity, sparsity_by_type)
            acts = layer_activations.get(f'layer_{layer_idx}.{attr_path}', None)
            if acts is not None:
                scaler_row = _wanda_scaler_row(acts, device)
                mask = _wanda_row_mask(W, scaler_row, sp)
                W_pruned = W * mask
                acts_d = acts.to(device).float()
                awq_scales = compute_awq_scale(W_pruned, acts_d, min_max, tile=block, method='awq')
                W_fq = tiled_fake_quant_rectangle(
                    W_pruned, fakequant=rtn_fake_quant, min_max=min_max, block=block, scales=awq_scales
                ) * mask
            else:
                W_fq = tiled_fake_quant_rectangle(
                    W, fakequant=rtn_fake_quant, min_max=min_max, block=block,
                    scales=torch.ones(N, device=device)
                )

            new_layer = SparseLinear(W_fq.to(linear.weight.dtype), bias).to(device)
            setattr(parent, parts[-1], new_layer)
            del W, linear
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def apply_wanda_sinq_quantization(model, calibration_data, nbits: int, sparsity: float,
                                  device: str = 'cuda', sparsity_by_type: dict = None) -> nn.Module:
    """Wanda pruning + SINQ quantization of the surviving weights.

    The naive composition (off-the-shelf SINQ run on the pruned matrix, i.e.
    sparsity-*unaware* Sinkhorn) — this is the baseline PRISM's sparse-aware
    Sinkhorn improves upon. `sparsity_by_type` (optional) enables per-type sparsity
    (e.g. v-dense), held equal across arms; default None = uniform `sparsity`.
    """
    layer_activations = collect_activations(model, calibration_data, device)
    layer_paths = get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(get_transformer_layers(model),
                                           desc=f"Wanda+SINQ {nbits}b/{int(sparsity*100)}%")):
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

            W = linear.weight.data.clone().float()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            sp = _effective_sparsity(attr_path, sparsity, sparsity_by_type)
            acts = layer_activations.get(f'layer_{layer_idx}.{attr_path}', None)
            if acts is not None:
                scaler_row = _wanda_scaler_row(acts, device)
                mask = _wanda_row_mask(W, scaler_row, sp)
            else:
                mask = torch.ones_like(W)
            W_pruned = W * mask

            actual_nbits = get_adaptive_nbits(W_pruned, nbits)
            min_max = [0, 2 ** actual_nbits - 1]
            W_q, s1, s2, z = quantize_dual_scale_shift(W_pruned, min_max, method='sinq')
            W_deq = _sinq_dequantize_dense(W_q, s1, s2, z) * mask

            new_layer = SparseLinear(W_deq.to(linear.weight.dtype), bias).to(device)
            setattr(parent, parts[-1], new_layer)
            del W, linear
            torch.cuda.empty_cache()

        set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


# =============================================================================
# Main Benchmark Function
# =============================================================================

def run_benchmark(
    model_key: str,
    technique: str,
    precision: int,
    dataset_key: str = 'wikitext2',
    device: str = 'cuda',
    verbose: bool = True,
    mmlu_max_subjects: int = None,  # Limit MMLU subjects for faster testing
    sparsity: float = None,  # Sparsity level (0.0-1.0), defaults to EVAL_CONFIG['sparsity']
    downstream: bool = False,  # Run downstream-task suite after eval
    downstream_tasks: str = 'all',  # Comma-separated subset or 'all'
    downstream_limit: int = None,  # Subsample N examples per task
    downstream_gen_limit: int = None,  # Separate cap for generative tasks (math/humaneval)
    downstream_allow_codeexec: bool = False,  # Enable HumanEval execution
    downstream_csv_dir: str = None,  # Output dir for downstream CSVs
) -> Dict[str, Any]:
    """Run a single benchmark configuration.

    Args:
        model_key: Model to benchmark
        technique: Quantization technique to apply
        precision: Bit precision
        dataset_key: Dataset to evaluate on ('wikitext2', 'ptb', 'c4', 'mmlu')
        device: Device to use
        verbose: Print progress
        mmlu_max_subjects: Maximum MMLU subjects to evaluate (None for all 57)
        sparsity: Sparsity level for sparse techniques (0.0-1.0), defaults to EVAL_CONFIG['sparsity']

    Returns:
        Dictionary with benchmark results
    """
    model_name = MODELS[model_key]
    sparsity = sparsity if sparsity is not None else EVAL_CONFIG['sparsity']
    ds_config = DATASETS[dataset_key]
    eval_type = ds_config['type']

    result = {
        'model': model_key,
        'model_name': model_name,
        'technique': technique,
        'precision': precision,
        'dataset': dataset_key,
        'sparsity': sparsity if uses_sparsity(technique) else 0,
        'timestamp': datetime.now().isoformat(),
        'ppl': None,
        'accuracy': None,
        'error': None,
    }

    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            print(f"Technique: {technique}")
            print(f"Precision: {precision}-bit")
            print(f"Dataset: {dataset_key}")
            if uses_sparsity(technique):
                print(f"Sparsity: {sparsity*100:.0f}%")
            print('='*60)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load test data (for perplexity tasks)
        test_data = None
        if eval_type == 'perplexity':
            if verbose:
                print(f"Loading {dataset_key} test data...")
            test_data = get_test_data(
                tokenizer,
                seq_len=EVAL_CONFIG['seq_len'],
                n_samples=EVAL_CONFIG['n_test_samples'],
                dataset_key=dataset_key
            )

        # Load calibration data (always from wikitext2 for consistency)
        calibration_data = None
        if technique != 'fp16':
            if verbose:
                print("Loading calibration data...")
            calibration_data = get_calibration_data(
                tokenizer,
                n_samples=EVAL_CONFIG['n_calibration_samples'],
                seq_len=EVAL_CONFIG['calibration_seq_len'],
                dataset_key='wikitext2'  # Always calibrate on wikitext2
            )

        # Load model
        if verbose:
            print("Loading model...")

        if technique == 'fp16':
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='cuda',
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map='cpu',
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            move_embed_to_device(model, device)

        # Deterministic quantization: a few techniques (prism/scab/the ablation cells) inject
        # tiny randn noise into pruned regions to keep Sinkhorn well-defined. Seed before each
        # apply so an ablation's cell-to-cell differences come only from the toggles, not RNG.
        # (Noise is ~1e-8 scale, so this does not perturb published numbers meaningfully.)
        _seed = int(os.environ.get('PRISM_SEED', '0'))
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)

        # Apply quantization technique
        if technique == 'fp16':
            pass  # No quantization needed
        elif technique == 'sinq':
            model = apply_sinq_quantization(model, calibration_data, precision, device)
        elif technique == 'cpr-sinq':
            model = apply_cpr_quantization(model, device, nbits=precision)
        elif technique == 'sinq-sparse':
            model = apply_sparse_quantization(model, calibration_data, precision, sparsity, device)
        elif technique == 'cpr-sparse':
            model = apply_cpr_sparse_quantization(model, calibration_data, sparsity, device, nbits=precision)
        elif technique == 'wanda':
            model = apply_wanda_pruning(model, calibration_data, sparsity, device)
        elif technique == 'sparsegpt':
            model = apply_sparsegpt_pruning(model, calibration_data, sparsity, precision, device)
        elif technique == 'scab':
            model = apply_scab_quantization(model, calibration_data, precision, sparsity, device)
        elif technique == 'prism':
            model = apply_prism_quantization(model, calibration_data, precision, sparsity, device)
        elif technique == 'awq':
            model = apply_awq_quantization(model, calibration_data, precision, device)
        elif technique == 'gptq':
            model = apply_gptq_quantization(model, calibration_data, precision, device)
        elif technique == 'spqr':
            model = apply_spqr_quantization(model, calibration_data, precision, device)
        elif technique == 'wanda-awq':
            model = apply_wanda_awq_quantization(model, calibration_data, precision, sparsity, device)
        elif technique == 'wanda-sinq':
            model = apply_wanda_sinq_quantization(model, calibration_data, precision, sparsity, device)
        elif technique == 'jsq':
            model = apply_jsq_quantization(model, calibration_data, precision, sparsity, device)
        elif technique == 'jsq-wo':
            model = apply_jsq_weightonly_quantization(model, calibration_data, precision, sparsity, device)
        elif technique == 'slim':
            model = apply_slim_quantization(model, calibration_data, precision, sparsity, device)
        elif technique in PRISM_ABLATION_MODES:
            comp_mode, imp_method = PRISM_ABLATION_MODES[technique]
            model = apply_prism_ablation_quantization(
                model, calibration_data, precision, sparsity, device,
                compensation_mode=comp_mode, method=imp_method)
        elif technique == 'sparsegpt-corr':
            model = apply_sparsegpt_pruning(model, calibration_data, sparsity, precision, device,
                                            sparse_aware_scale=True)
        elif technique == 'slim-corr':
            model = apply_slim_quantization(model, calibration_data, precision, sparsity, device,
                                            sparse_aware_cap=True)
        elif technique == 'jsq-wo-corr':
            model = apply_jsq_weightonly_quantization(model, calibration_data, precision, sparsity, device,
                                                      sparse_aware_scale=True)
        else:
            raise ValueError(f"Unknown technique: {technique}")

        # Move remaining components to device
        if technique != 'fp16':
            move_final_layers_to_device(model, device)

        model.eval()

        # Evaluate based on dataset type
        if eval_type == 'perplexity':
            if verbose:
                print(f"Evaluating perplexity on {dataset_key}...")
            ppl = evaluate_perplexity(model, test_data, device)
            result['ppl'] = ppl
            if verbose:
                print(f"\nPerplexity: {ppl:.2f}")
        elif eval_type == 'accuracy':
            eval_func = ds_config.get('eval_func', 'mmlu')
            if verbose:
                print(f"Evaluating {dataset_key} accuracy...")

            if eval_func == 'mmlu':
                acc, per_subject = evaluate_mmlu(
                    model, tokenizer, device,
                    n_few_shot=ds_config.get('n_few_shot', 5),
                    max_subjects=mmlu_max_subjects,
                    verbose=verbose
                )
                result['mmlu_per_subject'] = per_subject
            elif eval_func == 'hellaswag':
                acc = evaluate_hellaswag(
                    model, tokenizer, device,
                    max_samples=mmlu_max_subjects,  # Reuse this param for sample limit
                    verbose=verbose
                )
            elif eval_func == 'arc':
                acc = evaluate_arc(
                    model, tokenizer, device,
                    max_samples=mmlu_max_subjects,
                    verbose=verbose
                )
            elif eval_func == 'winogrande':
                acc = evaluate_winogrande(
                    model, tokenizer, device,
                    max_samples=mmlu_max_subjects,
                    verbose=verbose
                )
            elif eval_func == 'boolq':
                acc = evaluate_boolq(
                    model, tokenizer, device,
                    max_samples=mmlu_max_subjects,
                    verbose=verbose
                )
            else:
                raise ValueError(f"Unknown eval function: {eval_func}")

            result['accuracy'] = acc
            if verbose:
                print(f"\n{dataset_key.upper()} Accuracy: {acc*100:.2f}%")

        # Downstream-task suite (purely additive; only runs when --downstream is passed)
        if downstream:
            try:
                import os as _os, sys as _sys
                _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
                from downstream import run_downstream_suite
                model.seqlen = EVAL_CONFIG['seq_len']  # CRB evals read model.seqlen
                ds_cfg = {
                    'timestamp': result.get('timestamp'),
                    'model': model_key,            # the --model key e.g. 'qwen-0.5b'
                    'model_name': model_name,      # HF id
                    'technique': technique,
                    'precision': precision,
                    'sparsity': result.get('sparsity'),
                    'dataset': dataset_key,
                }
                ds_tasks = 'all' if downstream_tasks in (None, 'all') else [t.strip() for t in downstream_tasks.split(',')]
                run_downstream_suite(model, tokenizer, device, tasks=ds_tasks,
                                     limit=downstream_limit, gen_limit=downstream_gen_limit,
                                     allow_codeexec=downstream_allow_codeexec,
                                     config=ds_cfg, results_dir=downstream_csv_dir,
                                     seqlen=EVAL_CONFIG['seq_len'], verbose=verbose)
            except Exception as e:
                print(f"[downstream] suite failed: {e}")

        # Cleanup
        del model
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    return result


def save_result(result: Dict[str, Any], results_dir: Path):
    """Save a single result to disk with file locking for concurrent access."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save individual result (include sparsity in filename for uniqueness)
    sparsity_str = f"_sp{int(result.get('sparsity', 0) * 100)}" if result.get('sparsity') else ""
    filename = f"{result['model']}_{result['technique']}_{result['precision']}bit{sparsity_str}.json"
    filepath = results_dir / filename
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=2)

    # Update consolidated results with file locking
    consolidated_path = results_dir / 'all_results.json'
    lock_path = results_dir / 'all_results.json.lock'

    # Use a separate lock file for atomic read-modify-write
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            # Read existing results
            if consolidated_path.exists():
                try:
                    with open(consolidated_path, 'r') as f:
                        all_results = json.load(f)
                except json.JSONDecodeError:
                    # File corrupted, start fresh
                    all_results = []
            else:
                all_results = []

            # Update or append result (include sparsity in key)
            key = (result['model'], result['technique'], result['precision'], result.get('sparsity', 0))
            updated = False
            for i, r in enumerate(all_results):
                r_key = (r['model'], r['technique'], r['precision'], r.get('sparsity', 0))
                if r_key == key:
                    all_results[i] = result
                    updated = True
                    break
            if not updated:
                all_results.append(result)

            # Write atomically via temp file
            temp_path = consolidated_path.with_suffix('.json.tmp')
            with open(temp_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            temp_path.rename(consolidated_path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# CSV column definitions for comprehensive result logging
CSV_COLUMNS = [
    # Input parameters
    'timestamp',
    'model',
    'model_name',
    'technique',
    'precision',
    'sparsity',
    'dataset',
    'eval_type',
    # Hyperparameters
    'seq_len',
    'n_test_samples',
    'n_calibration_samples',
    'calibration_seq_len',
    'group_size',
    'cpr_group_size',
    # Result metrics
    'ppl',
    'accuracy',
    'flip_rate',
    'flip_task',
    'n_flipped',
    'n_total',
    # Additional info
    'error',
    'duration_seconds',
    # MMLU breakdown (if applicable)
    'mmlu_per_subject',
]


def save_result_to_csv(
    result: Dict[str, Any],
    csv_path: Path,
    eval_config: Dict[str, Any] = None
) -> None:
    """Thread-safe CSV result saving with file locking.

    Uses fcntl.flock() to ensure atomic writes when multiple processes
    write to the same CSV file concurrently.

    Args:
        result: Dictionary containing benchmark results
        csv_path: Path to the CSV file
        eval_config: Evaluation configuration dict (optional, uses EVAL_CONFIG if None)
    """
    if eval_config is None:
        eval_config = EVAL_CONFIG

    # Prepare the row data
    row = {
        # Input parameters
        'timestamp': result.get('timestamp', datetime.now().isoformat()),
        'model': result.get('model', ''),
        'model_name': result.get('model_name', ''),
        'technique': result.get('technique', ''),
        'precision': result.get('precision', ''),
        'sparsity': result.get('sparsity', ''),
        'dataset': result.get('dataset', 'wikitext2'),
        'eval_type': result.get('eval_type', ''),
        # Hyperparameters from config
        'seq_len': eval_config.get('seq_len', ''),
        'n_test_samples': eval_config.get('n_test_samples', ''),
        'n_calibration_samples': eval_config.get('n_calibration_samples', ''),
        'calibration_seq_len': eval_config.get('calibration_seq_len', ''),
        'group_size': eval_config.get('group_size', ''),
        'cpr_group_size': eval_config.get('cpr_group_size', ''),
        # Result metrics
        'ppl': result.get('ppl', ''),
        'accuracy': result.get('accuracy', ''),
        'flip_rate': result.get('flip_rate', ''),
        'flip_task': result.get('flip_task', ''),
        'n_flipped': result.get('n_flipped', ''),
        'n_total': result.get('n_total', ''),
        # Additional info
        'error': result.get('error', ''),
        'duration_seconds': result.get('duration_seconds', ''),
        # MMLU breakdown (serialize as JSON string if present)
        'mmlu_per_subject': json.dumps(result.get('mmlu_per_subject')) if result.get('mmlu_per_subject') else '',
    }

    # Ensure parent directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Thread-safe write with file locking
    # Use 'a+' mode to allow both reading (to check header) and appending
    with open(csv_path, 'a+', newline='') as f:
        # Acquire exclusive lock (blocking)
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Check if file has content (header already written)
            f.seek(0)
            first_line = f.readline()
            has_header = first_line.strip().startswith('timestamp')

            # Seek to end for appending
            f.seek(0, 2)

            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction='ignore')

            # Write header if file is empty or doesn't have one
            if not has_header:
                writer.writeheader()

            # Write the result row
            writer.writerow(row)
            f.flush()  # Ensure data is written before releasing lock
            os.fsync(f.fileno())  # Force write to disk
        finally:
            # Release lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def update_tracker(result: Dict[str, Any], tracker_path: Path):
    """Update the benchmark tracker markdown file with the result."""
    if not tracker_path.exists():
        return

    # Read tracker
    with open(tracker_path, 'r') as f:
        content = f.read()

    # Map technique names to tracker format
    technique_map = {
        'fp16': 'FP16',
        'sinq': 'SINQ',
        'cpr-sinq': 'CPR-SINQ',
        'sinq-sparse': 'SINQ-Sparse',
        'cpr-sparse': 'CPR-Sparse',
        'wanda': 'Wanda',
        'sparsegpt': 'SparseGPT',
        'scab': 'SCAB',
        'prism': 'PRISM',
    }

    # Map model names to tracker format
    model_map = {
        'qwen-0.5b': 'Qwen2.5-0.5B',
        'qwen-1.5b': 'Qwen2.5-1.5B',
        'qwen-3b': 'Qwen2.5-3B',
        'llama-7b': 'LLaMA-7B',  # Changed from Llama-3.2-1B
        'phi-2': 'Phi-2',
    }

    model_name = model_map.get(result['model'], result['model'])
    technique_name = technique_map.get(result['technique'], result['technique'])
    precision = result['precision']
    ppl = result['ppl']

    # Update progress table (change [ ] to [x])
    # This is a simple heuristic - look for pattern matching the configuration

    # Update results table
    if ppl is not None:
        # Find the results table row and update the cell
        # Format: | Model | Technique | 3-bit | 4-bit | 5-bit | 6-bit | 8-bit |
        pass  # TODO: Implement proper markdown parsing/updating

    # For now, just append to changelog
    changelog_entry = f"| {datetime.now().strftime('%Y-%m-%d %H:%M')} | {model_name} {technique_name} {precision}-bit: PPL={ppl:.2f if ppl else 'ERR'} |"

    # Write back
    # Note: Full implementation would properly update tables
    # For now, we rely on manual tracker updates or separate aggregation


def run_full_benchmark(
    models: List[str] = None,
    techniques: List[str] = None,
    precisions: List[int] = None,
    resume: bool = False,
    device: str = 'cuda',
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Run the full benchmark matrix."""
    if models is None:
        models = list(MODELS.keys())
    if techniques is None:
        techniques = TECHNIQUES
    if precisions is None:
        precisions = PRECISIONS

    results = []

    # Handle FP16 and Wanda separately (precision doesn't apply)
    fp16_count = len(models) if 'fp16' in techniques else 0
    wanda_count = len(models) if 'wanda' in techniques else 0
    quant_techniques = [t for t in techniques if t not in ('fp16', 'wanda')]
    # All quantization techniques now support all precisions (including CPR)
    total_configs = fp16_count + wanda_count + len(models) * len(quant_techniques) * len(precisions)

    print(f"\n{'='*60}")
    print(f"SINQ Comprehensive Benchmark Suite")
    print(f"{'='*60}")
    print(f"Models: {len(models)}")
    print(f"Techniques: {len(techniques)}")
    print(f"Precisions: {len(precisions)}")
    print(f"Total Configurations: {total_configs}")
    print(f"{'='*60}\n")

    completed = 0

    # Check for existing results if resuming
    completed_configs = set()
    if resume and (RESULTS_DIR / 'all_results.json').exists():
        with open(RESULTS_DIR / 'all_results.json', 'r') as f:
            existing = json.load(f)
            for r in existing:
                if r['ppl'] is not None:
                    completed_configs.add((r['model'], r['technique'], r['precision']))

    for model_key in models:
        # FP16 baseline (no precision parameter)
        if 'fp16' in techniques:
            config = (model_key, 'fp16', 0)
            if resume and config in completed_configs:
                print(f"Skipping {model_key} FP16 (already completed)")
            else:
                result = run_benchmark(model_key, 'fp16', 0, device, verbose)
                results.append(result)
                save_result(result, RESULTS_DIR)
                completed += 1
                print(f"Progress: {completed}/{total_configs}")

        # Wanda baseline (pruning only, no precision parameter)
        if 'wanda' in techniques:
            config = (model_key, 'wanda', 0)
            if resume and config in completed_configs:
                print(f"Skipping {model_key} Wanda (already completed)")
            else:
                result = run_benchmark(model_key, 'wanda', 0, device, verbose)
                results.append(result)
                save_result(result, RESULTS_DIR)
                completed += 1
                print(f"Progress: {completed}/{total_configs}")

        # All quantization techniques (including CPR which now supports 3-8 bit)
        for technique in quant_techniques:
            for precision in precisions:
                config = (model_key, technique, precision)
                if resume and config in completed_configs:
                    print(f"Skipping {model_key} {technique} {precision}-bit (already completed)")
                    continue

                result = run_benchmark(model_key, technique, precision, device, verbose)
                results.append(result)
                save_result(result, RESULTS_DIR)
                completed += 1
                print(f"Progress: {completed}/{total_configs}")

    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary of benchmark results."""
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print('='*80)

    # Group by model and dataset
    by_model = {}
    for r in results:
        model = r['model']
        dataset = r.get('dataset', 'wikitext2')
        key = f"{model} ({dataset})"
        if key not in by_model:
            by_model[key] = []
        by_model[key].append(r)

    for model_key, model_results in by_model.items():
        print(f"\n{model_key.upper()}")
        print('-'*70)

        # Check if this is perplexity or accuracy evaluation
        has_accuracy = any(r.get('accuracy') is not None for r in model_results)

        if has_accuracy:
            print(f"{'Technique':<15} {'Precision':<10} {'Accuracy':<12} {'Status'}")
            print('-'*70)

            # Get FP16 baseline for comparison
            fp16_acc = None
            for r in model_results:
                if r['technique'] == 'fp16' and r.get('accuracy') is not None:
                    fp16_acc = r['accuracy']
                    break

            for r in sorted(model_results, key=lambda x: (x['technique'], x['precision'])):
                technique = r['technique']
                precision = f"{r['precision']}-bit" if r['precision'] > 0 else "N/A"
                acc = f"{r['accuracy']*100:.2f}%" if r.get('accuracy') is not None else "ERROR"

                status = ""
                if r.get('accuracy') is not None and fp16_acc is not None:
                    diff = (r['accuracy'] - fp16_acc) * 100
                    status = f"({diff:+.2f}% vs FP16)"

                if r.get('error'):
                    status = f"Error: {r['error'][:30]}"

                print(f"{technique:<15} {precision:<10} {acc:<12} {status}")
        else:
            print(f"{'Technique':<15} {'Precision':<10} {'PPL':<10} {'Status'}")
            print('-'*70)

            # Get FP16 baseline for comparison
            fp16_ppl = None
            for r in model_results:
                if r['technique'] == 'fp16' and r.get('ppl') is not None:
                    fp16_ppl = r['ppl']
                    break

            for r in sorted(model_results, key=lambda x: (x['technique'], x['precision'])):
                technique = r['technique']
                precision = f"{r['precision']}-bit" if r['precision'] > 0 else "N/A"
                ppl = f"{r['ppl']:.2f}" if r.get('ppl') is not None else "ERROR"

                status = ""
                if r.get('ppl') is not None and fp16_ppl is not None:
                    ratio = r['ppl'] / fp16_ppl
                    status = f"({ratio:.1%} of FP16)"

                if r.get('error'):
                    status = f"Error: {r['error'][:30]}"

                print(f"{technique:<15} {precision:<10} {ppl:<10} {status}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SINQ Comprehensive Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single configuration
  python benchmarks/benchmark_suite.py --model qwen-0.5b --technique sinq --precision 4

  # Run on different datasets
  python benchmarks/benchmark_suite.py --model qwen-0.5b --technique scab --precision 4 --dataset ptb
  python benchmarks/benchmark_suite.py --model qwen-0.5b --technique scab --precision 4 --dataset c4
  python benchmarks/benchmark_suite.py --model qwen-0.5b --technique scab --precision 4 --dataset mmlu

  # Run all models with SINQ 4-bit
  python benchmarks/benchmark_suite.py --model all --technique sinq --precision 4

  # Run full benchmark matrix
  python benchmarks/benchmark_suite.py --model all --technique all --precision all

  # Resume interrupted benchmark
  python benchmarks/benchmark_suite.py --model all --technique all --precision all --resume

Available Models:
  qwen-0.5b, qwen-1.5b, qwen-3b, llama-7b, tinyllama, phi-2,
  pythia-1.4b, opt-1.3b, gemma-2b, mistral-7b

Available Techniques:
  fp16, sinq, cpr-sinq, sinq-sparse, cpr-sparse, wanda, sparsegpt, scab, prism

Available Datasets:
  PPL: wikitext2 (default), wikitext103, c4, lambada, pile
  Accuracy: mmlu, hellaswag, arc_challenge, winogrande, boolq
  Flip Rate: --flip-rate --flip-task [hellaswag|mmlu|boolq]

Available Precisions:
  3, 4, 5, 6, 8 (all techniques support all precisions; CPR uses 25% INT8 + 75% at specified bits)

Sparsity (--sparsity):
  0.0-1.0 (default: 0.35 = 35%%)
  Applies to: wanda, sparsegpt, sinq-sparse, cpr-sparse, scab, prism, wanda-awq, wanda-sinq
  (quant-only awq/gptq/spqr ignore --sparsity)
        """
    )

    parser.add_argument('--model', type=str, default='qwen-0.5b',
                       help='Model to benchmark (or "all")')
    parser.add_argument('--technique', type=str, default='sinq',
                       help='Quantization technique (or "all")')
    parser.add_argument('--precision', type=str, default='4',
                       help='Bit precision (or "all")')
    parser.add_argument('--sparsity', type=float, default=0.35,
                       help='Sparsity level for sparse techniques (0.0-1.0, default: 0.35 = 35%%)')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                       help='Dataset to evaluate on: wikitext2 (default), ptb, c4, mmlu')
    parser.add_argument('--mmlu-subjects', type=int, default=None,
                       help='Limit MMLU/accuracy tasks to first N subjects/samples')
    parser.add_argument('--flip-rate', action='store_true',
                       help='Run flip rate evaluation instead of standard benchmark')
    parser.add_argument('--flip-task', type=str, default='hellaswag',
                       choices=['hellaswag', 'mmlu', 'boolq'],
                       help='Task for flip rate evaluation')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--csv', type=str, default='benchmark_results.csv',
                       help='CSV filename for results (thread-safe, default: benchmark_results.csv)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Disable CSV output')
    parser.add_argument('--downstream', action='store_true',
                       help='Run downstream-task evaluation suite after PPL/accuracy eval')
    parser.add_argument('--downstream-tasks', type=str, default='all',
                       help='Comma-separated downstream task subset, or "all" (default: all)')
    parser.add_argument('--downstream-limit', type=int, default=None,
                       help='Subsample N examples per downstream task (default: all examples)')
    parser.add_argument('--downstream-gen-limit', type=int, default=None,
                       help='Separate sample cap for generative downstream tasks '
                            '(math/humaneval); falls back to --downstream-limit when unset')
    parser.add_argument('--downstream-allow-codeexec', action='store_true',
                       help='Enable HumanEval code execution in the downstream suite')
    parser.add_argument('--downstream-csv-dir', type=str, default=None,
                       help='Output dir for downstream CSVs (default: RESULTS_DIR/downstream)')
    parser.add_argument('--hf-token', type=str, default=None,
                       help='HuggingFace access token for gated models/datasets')

    args = parser.parse_args()

    # Login to HuggingFace if token provided
    if args.hf_token:
        try:
            from huggingface_hub import login
            login(token=args.hf_token, add_to_git_credential=False)
            if not args.quiet:
                print("Logged in to HuggingFace Hub")
        except Exception as e:
            print(f"Warning: HuggingFace login failed: {e}")

    # Parse arguments
    if args.model == 'all':
        models = list(MODELS.keys())
    else:
        models = [args.model]

    if args.technique == 'all':
        techniques = TECHNIQUES
    else:
        techniques = [args.technique]

    if args.precision == 'all':
        precisions = PRECISIONS
    else:
        precisions = [int(args.precision)]

    # Validate models
    for m in models:
        if m not in MODELS:
            print(f"Unknown model: {m}")
            print(f"Available: {list(MODELS.keys())}")
            sys.exit(1)

    # Validate techniques
    for t in techniques:
        if t not in TECHNIQUES:
            print(f"Unknown technique: {t}")
            print(f"Available: {TECHNIQUES}")
            sys.exit(1)

    # Validate dataset
    dataset = args.dataset
    if dataset not in DATASETS:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {list(DATASETS.keys())}")
        sys.exit(1)

    # Validate sparsity
    sparsity = args.sparsity
    if not (0.0 <= sparsity <= 1.0):
        print(f"Invalid sparsity: {sparsity}")
        print("Sparsity must be between 0.0 and 1.0 (e.g., 0.35 for 35%)")
        sys.exit(1)

    # Run benchmark(s)
    results = []

    # CSV path setup
    csv_path = RESULTS_DIR / args.csv if not args.no_csv else None

    # Handle flip rate evaluation separately
    if args.flip_rate:
        for model in models:
            for technique in techniques:
                if technique == 'fp16':
                    continue  # Skip FP16 - it's the baseline
                for precision in precisions:
                    if technique == 'wanda' and precision != precisions[0]:
                        continue  # Wanda doesn't use precision

                    start_time = time.time()
                    result = run_flip_rate_benchmark(
                        model_key=model,
                        technique=technique,
                        precision=precision,
                        task=args.flip_task,
                        max_samples=args.mmlu_subjects,
                        device=args.device,
                        verbose=not args.quiet,
                        sparsity=sparsity
                    )
                    result['duration_seconds'] = time.time() - start_time
                    result['eval_type'] = 'flip_rate'
                    result['flip_task'] = args.flip_task
                    results.append(result)

                    # Save flip rate results (JSON)
                    flip_results_dir = RESULTS_DIR / 'flip_rates'
                    flip_results_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{result['model']}_{result['technique']}_{result['precision']}bit_{args.flip_task}_flip.json"
                    with open(flip_results_dir / filename, 'w') as f:
                        json.dump(result, f, indent=2)

                    # Save to CSV (thread-safe)
                    if csv_path:
                        save_result_to_csv(result, csv_path)

        # Print flip rate summary
        print(f"\n{'='*60}")
        print("FLIP RATE RESULTS")
        print('='*60)
        for r in results:
            if r.get('flip_rate') is not None:
                print(f"{r['model']} | {r['technique']} {r['precision']}-bit | {args.flip_task}: {r['flip_rate']:.2f}%")
            elif r.get('error'):
                print(f"{r['model']} | {r['technique']} {r['precision']}-bit | ERROR: {r['error'][:50]}")
        print(f"\nResults saved to: {RESULTS_DIR / 'flip_rates'}")
        if csv_path:
            print(f"CSV results: {csv_path}")
        return

    # For single model/technique/precision, run directly with dataset support
    if len(models) == 1 and len(techniques) == 1 and len(precisions) == 1:
        start_time = time.time()
        result = run_benchmark(
            model_key=models[0],
            technique=techniques[0],
            precision=precisions[0],
            dataset_key=dataset,
            device=args.device,
            verbose=not args.quiet,
            mmlu_max_subjects=args.mmlu_subjects,
            sparsity=sparsity,
            downstream=args.downstream,
            downstream_tasks=args.downstream_tasks,
            downstream_limit=args.downstream_limit,
            downstream_gen_limit=args.downstream_gen_limit,
            downstream_allow_codeexec=args.downstream_allow_codeexec,
            downstream_csv_dir=args.downstream_csv_dir
        )
        result['duration_seconds'] = time.time() - start_time
        result['eval_type'] = DATASETS[dataset]['type']
        results.append(result)
        save_result(result, RESULTS_DIR)
        if csv_path:
            save_result_to_csv(result, csv_path)
    else:
        # For matrix benchmarks, run all combinations
        for model in models:
            for technique in techniques:
                for precision in precisions:
                    # Skip invalid combinations
                    if technique in ('fp16', 'wanda') and precision != precisions[0]:
                        continue  # FP16/Wanda don't use precision

                    start_time = time.time()
                    result = run_benchmark(
                        model_key=model,
                        technique=technique,
                        precision=precision,
                        dataset_key=dataset,
                        device=args.device,
                        verbose=not args.quiet,
                        mmlu_max_subjects=args.mmlu_subjects,
                        sparsity=sparsity,
                        downstream=args.downstream,
                        downstream_tasks=args.downstream_tasks,
                        downstream_limit=args.downstream_limit,
                        downstream_gen_limit=args.downstream_gen_limit,
                        downstream_allow_codeexec=args.downstream_allow_codeexec,
                        downstream_csv_dir=args.downstream_csv_dir
                    )
                    result['duration_seconds'] = time.time() - start_time
                    result['eval_type'] = DATASETS[dataset]['type']
                    results.append(result)
                    save_result(result, RESULTS_DIR)
                    if csv_path:
                        save_result_to_csv(result, csv_path)

    # Print summary
    print_summary(results)

    print(f"\nResults saved to: {RESULTS_DIR}")
    if csv_path:
        print(f"CSV results: {csv_path}")


if __name__ == '__main__':
    main()
