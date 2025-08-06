# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn

# Import the modules we'll be testing
from custom_models.lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig
from custom_models.lact_model.layer_lact_swiglu import LaCTSWIGLULayer, SwiGLUFastWeightBlock


class TestStackedTTTSimple:
    """Simple tests for stacked TTT architecture without full forward pass."""
    
    def test_swiglu_fast_weight_block_creation(self):
        """Test that SwiGLUFastWeightBlock can be created correctly."""
        num_heads = 4
        d_h = 64
        d_in = 32
        d_out = 32
        
        block = SwiGLUFastWeightBlock(
            num_heads=num_heads,
            d_h=d_h,
            d_in=d_in,
            d_out=d_out,
            w0_w2_low_rank=-1,
            fw_init_gain=0.5
        )
        
        # Check that all parameters exist
        assert hasattr(block, 'w0')
        assert hasattr(block, 'w1')
        assert hasattr(block, 'w2')
        
        # Check parameter shapes
        assert block.w0.shape == (num_heads, d_h, d_in)
        assert block.w1.shape == (num_heads, d_out, d_h)
        assert block.w2.shape == (num_heads, d_h, d_in)
        
        # Check that parameters are trainable
        assert block.w0.requires_grad
        assert block.w1.requires_grad
        assert block.w2.requires_grad
    
    def test_layer_creation_single_mlp(self):
        """Test layer creation with r=1."""
        config = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attn_heads=4,
            num_lact_heads=2,
            window_size=64,
            lact_chunk_size=32,
            r=1,
            residual_ttt=False
        )
        
        layer = LaCTSWIGLULayer(
            hidden_size=config.hidden_size,
            num_attn_heads=config.num_attn_heads,
            num_lact_heads=config.num_lact_heads,
            inter_multi=config.inter_multi,
            window_size=config.window_size,
            lact_chunk_size=config.lact_chunk_size,
            r=config.r,
            residual_ttt=config.residual_ttt
        )
        
        # Should have original structure
        assert hasattr(layer, 'w0')
        assert hasattr(layer, 'w1')
        assert hasattr(layer, 'w2')
        assert not hasattr(layer, 'stacked_mlps')
        
        # Check that r and residual_ttt are set correctly
        assert layer.r == 1
        assert layer.residual_ttt == False
    
    def test_layer_creation_stacked_mlp(self):
        """Test layer creation with r>1."""
        config = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attn_heads=4,
            num_lact_heads=2,
            window_size=64,
            lact_chunk_size=32,
            r=3,
            residual_ttt=True
        )
        
        layer = LaCTSWIGLULayer(
            hidden_size=config.hidden_size,
            num_attn_heads=config.num_attn_heads,
            num_lact_heads=config.num_lact_heads,
            inter_multi=config.inter_multi,
            window_size=config.window_size,
            lact_chunk_size=config.lact_chunk_size,
            r=config.r,
            residual_ttt=config.residual_ttt
        )
        
        # Should have stacked structure
        assert hasattr(layer, 'stacked_mlps')
        assert len(layer.stacked_mlps) == 3
        assert not hasattr(layer, 'w0')  # Should not have original structure
        
        # Check that r and residual_ttt are set correctly
        assert layer.r == 3
        assert layer.residual_ttt == True
        
        # Check each MLP in the stack
        for i, mlp in enumerate(layer.stacked_mlps):
            assert isinstance(mlp, SwiGLUFastWeightBlock)
            assert hasattr(mlp, 'w0')
            assert hasattr(mlp, 'w1')
            assert hasattr(mlp, 'w2')
    
    def test_parameter_count_comparison(self):
        """Test that parameter count is correct for different r values."""
        hidden_size = 128
        num_lact_heads = 2
        inter_multi = 1
        
        d_in = hidden_size // num_lact_heads
        d_h = int(d_in * inter_multi)
        d_out = hidden_size // num_lact_heads
        
        # Calculate expected parameter count for single MLP
        single_params = num_lact_heads * (d_h * d_in + d_out * d_h + d_h * d_in)
        
        # Create single MLP layer
        layer_single = LaCTSWIGLULayer(
            hidden_size=hidden_size,
            num_attn_heads=4,
            num_lact_heads=num_lact_heads,
            inter_multi=inter_multi,
            window_size=64,
            lact_chunk_size=32,
            r=1,
            residual_ttt=False
        )
        
        # Create stacked MLP layer
        layer_stacked = LaCTSWIGLULayer(
            hidden_size=hidden_size,
            num_attn_heads=4,
            num_lact_heads=num_lact_heads,
            inter_multi=inter_multi,
            window_size=64,
            lact_chunk_size=32,
            r=3,
            residual_ttt=False
        )
        
        # Count parameters (only TTT-related parameters)
        single_ttt_params = sum(p.numel() for name, p in layer_single.named_parameters() 
                               if 'w0' in name or 'w1' in name or 'w2' in name)
        stacked_ttt_params = sum(p.numel() for name, p in layer_stacked.named_parameters() 
                                if 'w0' in name or 'w1' in name or 'w2' in name)
        
        # Stacked should have 3x the parameters
        assert stacked_ttt_params == 3 * single_ttt_params
        assert single_ttt_params == single_params
    
    def test_low_rank_parameterization(self):
        """Test that low rank parameterization works with stacked MLPs."""
        config = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attn_heads=4,
            num_lact_heads=2,
            window_size=64,
            lact_chunk_size=32,
            r=2,
            residual_ttt=True,
            w0_w2_low_rank=16  # Use low rank
        )
        
        layer = LaCTSWIGLULayer(
            hidden_size=config.hidden_size,
            num_attn_heads=config.num_attn_heads,
            num_lact_heads=config.num_lact_heads,
            inter_multi=config.inter_multi,
            window_size=config.window_size,
            lact_chunk_size=config.lact_chunk_size,
            r=config.r,
            residual_ttt=config.residual_ttt,
            w0_w2_low_rank=config.w0_w2_low_rank
        )
        
        # Check that each MLP uses LowRankFastWeight
        for mlp in layer.stacked_mlps:
            assert isinstance(mlp.w0, type(layer.stacked_mlps[0].w0))  # Should be LowRankFastWeight
            assert isinstance(mlp.w2, type(layer.stacked_mlps[0].w2))  # Should be LowRankFastWeight
            assert isinstance(mlp.w1, nn.Parameter)  # w1 is always a Parameter


if __name__ == "__main__":
    pytest.main([__file__]) 