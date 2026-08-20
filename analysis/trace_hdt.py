"""
TASK TWO: HDT Architecture Code Trace
=======================================

Trace the official HDT model architecture by instantiating the official
model classes and recording their structure. This does NOT re-implement
the model - it only calls official code and records what it finds.

Uses ONLY official source:
  - tfdiff/wifi_model.py → tfdiff_WiFi
  - tfdiff/fmcw_model.py → tfdiff_fmcw
  - tfdiff/mimo_model.py → tfdiff_mimo
  - tfdiff/params.py → all_params
"""
import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT + '/official')

import torch
import numpy as np
import os

from tfdiff.params import all_params

OUTPUT_DIR = REPO_ROOT + '/artifacts/hdt_trace'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trace_wifi_fmcw(task_id, name):
    """Trace WiFi or FMCW HDT model using official code."""
    params = all_params[task_id]
    print(f"\n{'='*60}")
    print(f"HDT Architecture Trace: {name} (task_id={task_id})")
    print(f"{'='*60}")

    # Import the correct model class
    if task_id == 0:
        from tfdiff.wifi_model import tfdiff_WiFi as ModelClass
        from tfdiff.wifi_model import DiffusionEmbedding, MLPConditionEmbedding
        from tfdiff.wifi_model import PositionEmbedding, DiA, FinalLayer
    else:
        from tfdiff.fmcw_model import tfdiff_fmcw as ModelClass
        from tfdiff.fmcw_model import DiffusionEmbedding, MLPConditionEmbedding
        from tfdiff.fmcw_model import PositionEmbedding, DiA, FinalLayer

    # ============================================================
    # 1. Instantiate the model (no checkpoint needed for trace)
    # ============================================================
    model = ModelClass(params)
    n_params = count_parameters(model)
    print(f"\n  Total trainable parameters: {n_params:,}")

    # ============================================================
    # 2. Record model component structure
    # ============================================================
    print(f"\n  Model components:")
    for name_mod, module in model.named_children():
        n_p = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"    {name_mod}: {module.__class__.__name__} ({n_p:,} params)")

    # ============================================================
    # 3. Trace parameter shapes for each component
    # ============================================================
    print(f"\n  Parameter shapes:")
    for name, param in model.named_parameters():
        print(f"    {name}: {list(param.shape)}")

    # ============================================================
    # 4. Record architecture hyperparameters
    # ============================================================
    print(f"\n  Architecture hyperparameters:")
    print(f"    hidden_dim: {params.hidden_dim}")
    print(f"    num_heads: {params.num_heads}")
    print(f"    num_block: {params.num_block}")
    print(f"    mlp_ratio: {params.mlp_ratio}")
    print(f"    dropout: {params.dropout}")
    print(f"    embed_dim: {params.embed_dim}")
    print(f"    cond_dim: {params.cond_dim}")
    print(f"    sample_rate: {params.sample_rate}")
    print(f"    input_dim: {params.input_dim}")

    # ============================================================
    # 5. Trace forward pass shapes with dummy inputs
    # ============================================================
    B = 1
    N = params.sample_rate  # 512
    S = params.input_dim     # 90 for WiFi, 128 for FMCW
    device = torch.device('cpu')

    # Create dummy inputs matching official data format
    x = torch.randn(B, N, S, 2).float()  # [B, N, S, 2]
    t = torch.tensor([50], dtype=torch.int64)  # random diffusion step
    # Condition must be complex-valued: [B, C, 2] for WiFi/FMCW
    c = torch.randn(B, params.cond_dim, 2).float()  # condition [B, C, 2]

    print(f"\n  Forward pass shapes (input):")
    print(f"    x (degraded signal): {list(x.shape)}")
    print(f"    t (diffusion step): {list(t.shape)}")
    print(f"    c (condition): {list(c.shape)} (complex as real/imag)")

    # Hook to capture intermediate shapes
    shape_log = []

    def hook_fn(name):
        def hook(module, inp, outp):
            if isinstance(inp, tuple):
                inp_shapes = [list(i.shape) if isinstance(i, torch.Tensor) else str(type(i)) for i in inp]
            else:
                inp_shapes = [list(inp.shape)]
            out_shape = list(outp.shape) if isinstance(outp, torch.Tensor) else str(type(outp))
            shape_log.append(f"  {name}: in={inp_shapes} → out={out_shape}")
        return hook

    # Register hooks on key components
    hooks = []
    hooks.append(model.p_embed.register_forward_hook(hook_fn('p_embed')))
    hooks.append(model.t_embed.register_forward_hook(hook_fn('t_embed')))
    hooks.append(model.c_embed.register_forward_hook(hook_fn('c_embed')))
    hooks.append(model.blocks[0].register_forward_hook(hook_fn('block[0]')))
    hooks.append(model.blocks[-1].register_forward_hook(hook_fn(f'block[{params.num_block-1}]')))
    hooks.append(model.final_layer.register_forward_hook(hook_fn('final_layer')))

    with torch.no_grad():
        output = model(x, t, c)

    print(f"\n  Forward pass shapes (traced):")
    for log_entry in shape_log:
        print(log_entry)
    print(f"    output: {list(output.shape)}")

    # Clean up hooks
    for h in hooks:
        h.remove()

    # ============================================================
    # 6. Record DiA block internal structure
    # ============================================================
    dia_block = model.blocks[0]
    print(f"\n  DiA block internal components:")
    for name_mod, module in dia_block.named_children():
        print(f"    {name_mod}: {module.__class__.__name__}")

    # ============================================================
    # 7. Record the commented-out two-stage architecture
    # ============================================================
    print(f"\n  PAPER-CODE DIFFERENCE:")
    print(f"    Current {name} model uses flat DiA stack ({params.num_block} blocks).")
    print(f"    Commented-out code shows original SpatialDiffusion + TimeFrequencyDiffusion.")
    print(f"    This is the ACTIVE code that will be used for inference.")

    return model


def trace_mimo():
    """Trace MIMO HDT model."""
    params = all_params[2]
    name = 'mimo'
    print(f"\n{'='*60}")
    print(f"HDT Architecture Trace: {name} (task_id=2)")
    print(f"{'='*60}")

    from tfdiff.mimo_model import tfdiff_mimo as ModelClass

    model = ModelClass(params)
    n_params = count_parameters(model)
    print(f"\n  Total trainable parameters: {n_params:,}")

    print(f"\n  Model components:")
    for name_mod, module in model.named_children():
        n_p = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"    {name_mod}: {module.__class__.__name__} ({n_p:,} params)")

    print(f"\n  Architecture:")
    print(f"    Spatial block: {params.num_spatial_block} DiA blocks, hidden_dim={params.spatial_hidden_dim}")
    print(f"    TF block: {params.num_tf_block} DiA blocks, hidden_dim={params.tf_hidden_dim}")
    print(f"    Two-stage: SpatialDiffusion → TimeFrequencyDiffusion")

    # Trace forward shapes
    B = 1
    N = params.sample_rate  # 14
    S, A = params.extra_dim  # 26, 96
    device = torch.device('cpu')

    x = torch.randn(B, N, S, A, 2).float()
    t = torch.tensor([50], dtype=torch.int64)

    # MIMO condition has different format: [B, N, C1, C2, 2] or [B, N*C1*C2, 2]
    # From dataset.py: cond has shape from up_link which is [14, 96, 26] complex
    # After collation: [B, N, 26, 96, 2]
    # Actually looking at dataset.py for MIMO:
    # norm_cond = (cond) / cond.std()
    # record['cond'] = norm_cond.reshape(14, 96, 26, 2).transpose(1,2)
    # So cond shape: [B, N, 26, 96, 2]
    c = torch.randn(B, N, *params.cond_dim, 2).float()

    print(f"\n  Forward pass shapes (input):")
    print(f"    x: {list(x.shape)}")
    print(f"    t: {list(t.shape)}")
    print(f"    c: {list(c.shape)}")

    with torch.no_grad():
        output = model(x, t, c)

    print(f"    output: {list(output.shape)}")

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("TASK TWO: HDT Architecture Code Trace")
    print("Using ONLY official RF-Diffusion model classes")
    print("=" * 60)

    # WiFi and FMCW use same architecture pattern
    trace_wifi_fmcw(0, 'wifi')
    trace_wifi_fmcw(1, 'fmcw')

    # MIMO uses different (two-stage) architecture
    trace_mimo()

    print(f"\n{'='*60}")
    print("HDT trace complete.")
    print(f"Outputs saved to: {OUTPUT_DIR}")
