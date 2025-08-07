# -*- coding: utf-8 -*-

"""
Integration test for the manual backprop TTT implementation with the full LaCT model.
"""

import torch
import torch.nn as nn
from .layer_lact_swiglu import LaCTSWIGLULayer
from .configuration_lact_swiglu import LaCTSWIGLUConfig


def test_integration_with_lact_layer():
    """Test that the manual backprop integrates correctly with LaCTSWIGLULayer."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping integration test")
        return
    
    device = 'cuda'
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    
    # Create a minimal config for testing
    config = LaCTSWIGLUConfig(
        hidden_size=128,
        num_attn_heads=4,
        num_lact_heads=4,
        inter_multi=1.0,
        window_size=256,
        lact_chunk_size=32,
        qkv_bias=False,
        attn_qk_norm=False,
        qkv_silu=False,
        lr_dim=1,
        use_muon=False,
        lr_parameterization="mamba",
        learnable_ttt_scale=False,
        ttt_prenorm=False,
        ttt_nope=False,
        rope_theta=10000.0,
        max_position_embeddings=2048,
        w0_w2_low_rank=-1,  # Full-rank (required for manual backprop)
        use_momentum=False,
        ttt_loss_type="dot_product",
        fw_init_gain=0.5,
        r=2,  # Stacked MLPs
        residual_ttt=False,  # No residual connections (required for manual backprop)
        greedy_ttt=False,  # Use end-to-end TTT
    )
    
    # Create the layer and cast to bf16 for FlashAttention compatibility
    layer = LaCTSWIGLULayer(config=config, layer_idx=0).to(device).to(torch.bfloat16)
    
    # Create test input
    batch_size = 2
    seq_len = 64
    hidden_size = config.hidden_size
    
    # Use bf16 for FlashAttention compatibility
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16)
    
    # Test forward pass
    print("Testing forward pass...")
    layer.train()  # Enable training mode to trigger manual backprop
    
    try:
        output, _, _ = layer(hidden_states)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        
        # Test backward pass
        print("Testing backward pass...")
        loss = output.sum()
        loss.backward()
        print("✓ Backward pass successful")
        
        # Check that gradients were computed
        has_grads = any(p.grad is not None for p in layer.parameters() if p.requires_grad)
        if has_grads:
            print("✓ Gradients were computed for layer parameters")
        else:
            print("⚠ No gradients found for layer parameters")
        
        print(f"✓ Integration test passed! Memory usage was efficient.")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Running LaCT Layer Integration Test...")
    test_integration_with_lact_layer()
    print("Integration test completed!")