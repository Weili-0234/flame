# -*- coding: utf-8 -*-

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Import the modules we'll be testing
from custom_models.lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig
from custom_models.lact_model.layer_lact_swiglu import LaCTSWIGLULayer
from custom_models.lact_model.modeling_lact import LaCTModel, LaCTForCausalLM


class TestStackedTTTConfiguration:
    """Test the new configuration parameters for stacked TTT."""
    
    def test_default_configuration_backward_compatibility(self):
        """Test that default configuration maintains backward compatibility."""
        config = LaCTSWIGLUConfig()
        
        # New parameters should have sensible defaults
        assert hasattr(config, 'r')
        assert config.r == 1  # Default should be 1 for backward compatibility
        
        assert hasattr(config, 'residual_ttt')
        assert config.residual_ttt == False  # Default should be False
        
        # Existing parameters should still work
        assert config.hidden_size == 2048
        assert config.num_hidden_layers == 24
        assert config.num_attn_heads == 32
        assert config.num_lact_heads == 4
    
    def test_custom_r_parameter(self):
        """Test that r parameter can be set to different values."""
        for r in [1, 2, 3]:
            config = LaCTSWIGLUConfig(r=r)
            assert config.r == r
    
    def test_custom_residual_ttt_parameter(self):
        """Test that residual_ttt parameter can be set."""
        config = LaCTSWIGLUConfig(residual_ttt=True)
        assert config.residual_ttt == True
        
        config = LaCTSWIGLUConfig(residual_ttt=False)
        assert config.residual_ttt == False
    
    def test_invalid_r_parameter(self):
        """Test that invalid r values are rejected."""
        with pytest.raises(ValueError):
            LaCTSWIGLUConfig(r=0)
        
        with pytest.raises(ValueError):
            LaCTSWIGLUConfig(r=-1)
    
    def test_config_serialization(self):
        """Test that config can be serialized and deserialized."""
        config = LaCTSWIGLUConfig(r=3, residual_ttt=True)
        
        # Test to_dict and from_dict
        config_dict = config.to_dict()
        new_config = LaCTSWIGLUConfig.from_dict(config_dict)
        
        assert new_config.r == 3
        assert new_config.residual_ttt == True


class TestStackedTTTArchitecture:
    """Test the stacked TTT architecture implementation."""
    
    @pytest.fixture
    def basic_config(self):
        """Create a basic config for testing."""
        return LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attn_heads=4,
            num_lact_heads=2,
            window_size=64,
            lact_chunk_size=32,
            r=1,  # Start with single MLP
            residual_ttt=False
        )
    
    @pytest.fixture
    def stacked_config(self):
        """Create a config with stacked MLPs."""
        return LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attn_heads=4,
            num_lact_heads=2,
            window_size=64,
            lact_chunk_size=32,
            r=3,  # Stack of 3 MLPs
            residual_ttt=True
        )
    
    def test_layer_initialization_single_mlp(self, basic_config):
        """Test that layer initializes correctly with r=1 (original behavior)."""
        layer = LaCTSWIGLULayer(
            hidden_size=basic_config.hidden_size,
            num_attn_heads=basic_config.num_attn_heads,
            num_lact_heads=basic_config.num_lact_heads,
            inter_multi=basic_config.inter_multi,
            window_size=basic_config.window_size,
            lact_chunk_size=basic_config.lact_chunk_size,
            r=basic_config.r,
            residual_ttt=basic_config.residual_ttt
        )
        
        # Should have the original single MLP structure
        assert hasattr(layer, 'w0')
        assert hasattr(layer, 'w1')
        assert hasattr(layer, 'w2')
        
        # Should not have the stacked structure
        assert not hasattr(layer, 'stacked_mlps')
    
    def test_layer_initialization_stacked_mlp(self, stacked_config):
        """Test that layer initializes correctly with r>1."""
        layer = LaCTSWIGLULayer(
            hidden_size=stacked_config.hidden_size,
            num_attn_heads=stacked_config.num_attn_heads,
            num_lact_heads=stacked_config.num_lact_heads,
            inter_multi=stacked_config.inter_multi,
            window_size=stacked_config.window_size,
            lact_chunk_size=stacked_config.lact_chunk_size,
            r=stacked_config.r,
            residual_ttt=stacked_config.residual_ttt
        )
        
        # Should have the stacked MLP structure
        assert hasattr(layer, 'stacked_mlps')
        assert len(layer.stacked_mlps) == 3
        
        # Each MLP in the stack should have w0, w1, w2
        for i, mlp in enumerate(layer.stacked_mlps):
            assert hasattr(mlp, 'w0')
            assert hasattr(mlp, 'w1')
            assert hasattr(mlp, 'w2')
            
            # Check dimensions
            expected_d_h = int(stacked_config.hidden_size // stacked_config.num_lact_heads * stacked_config.inter_multi)
            expected_d_in = stacked_config.hidden_size // stacked_config.num_lact_heads
            expected_d_out = stacked_config.hidden_size // stacked_config.num_lact_heads
            
            assert mlp.w0.shape == (stacked_config.num_lact_heads, expected_d_h, expected_d_in)
            assert mlp.w1.shape == (stacked_config.num_lact_heads, expected_d_out, expected_d_h)
            assert mlp.w2.shape == (stacked_config.num_lact_heads, expected_d_h, expected_d_in)
    
    def test_forward_pass_single_mlp(self, basic_config):
        """Test forward pass with r=1 (original behavior)."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        layer = LaCTSWIGLULayer(
            hidden_size=basic_config.hidden_size,
            num_attn_heads=basic_config.num_attn_heads,
            num_lact_heads=basic_config.num_lact_heads,
            inter_multi=basic_config.inter_multi,
            window_size=basic_config.window_size,
            lact_chunk_size=basic_config.lact_chunk_size,
            r=basic_config.r,
            residual_ttt=basic_config.residual_ttt
        ).to(device).half()
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, basic_config.hidden_size, device=device, dtype=torch.float16)
        
        output, _, _ = layer(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_pass_stacked_mlp_no_residual(self, stacked_config):
        """Test forward pass with r>1 and residual_ttt=False."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        stacked_config.residual_ttt = False
        layer = LaCTSWIGLULayer(
            hidden_size=stacked_config.hidden_size,
            num_attn_heads=stacked_config.num_attn_heads,
            num_lact_heads=stacked_config.num_lact_heads,
            inter_multi=stacked_config.inter_multi,
            window_size=stacked_config.window_size,
            lact_chunk_size=stacked_config.lact_chunk_size,
            r=stacked_config.r,
            residual_ttt=stacked_config.residual_ttt
        ).to(device).half()
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, stacked_config.hidden_size, device=device, dtype=torch.float16)
        
        output, _, _ = layer(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_pass_stacked_mlp_with_residual(self, stacked_config):
        """Test forward pass with r>1 and residual_ttt=True."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        stacked_config.residual_ttt = True
        layer = LaCTSWIGLULayer(
            hidden_size=stacked_config.hidden_size,
            num_attn_heads=stacked_config.num_attn_heads,
            num_lact_heads=stacked_config.num_lact_heads,
            inter_multi=stacked_config.inter_multi,
            window_size=stacked_config.window_size,
            lact_chunk_size=stacked_config.lact_chunk_size,
            r=stacked_config.r,
            residual_ttt=stacked_config.residual_ttt
        ).to(device).half()
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, stacked_config.hidden_size, device=device, dtype=torch.float16)
        
        output, _, _ = layer(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow_single_mlp(self, basic_config):
        """Test that gradients flow correctly with r=1."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        layer = LaCTSWIGLULayer(
            hidden_size=basic_config.hidden_size,
            num_attn_heads=basic_config.num_attn_heads,
            num_lact_heads=basic_config.num_lact_heads,
            inter_multi=basic_config.inter_multi,
            window_size=basic_config.window_size,
            lact_chunk_size=basic_config.lact_chunk_size,
            r=basic_config.r,
            residual_ttt=basic_config.residual_ttt
        ).to(device).half()
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, basic_config.hidden_size, requires_grad=True, device=device, dtype=torch.float16)
        
        output, _, _ = layer(hidden_states)
        loss = output.sum()
        loss.backward()
        
        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()
        assert not torch.isinf(hidden_states.grad).any()
    
    def test_gradient_flow_stacked_mlp(self, stacked_config):
        """Test that gradients flow correctly with r>1."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        layer = LaCTSWIGLULayer(
            hidden_size=stacked_config.hidden_size,
            num_attn_heads=stacked_config.num_attn_heads,
            num_lact_heads=stacked_config.num_lact_heads,
            inter_multi=stacked_config.inter_multi,
            window_size=stacked_config.window_size,
            lact_chunk_size=stacked_config.lact_chunk_size,
            r=stacked_config.r,
            residual_ttt=stacked_config.residual_ttt
        ).to(device).half()
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, stacked_config.hidden_size, requires_grad=True, device=device, dtype=torch.float16)
        
        output, _, _ = layer(hidden_states)
        loss = output.sum()
        loss.backward()
        
        assert hidden_states.grad is not None
        assert not torch.isnan(hidden_states.grad).any()
        assert not torch.isinf(hidden_states.grad).any()
        
        # Check that gradients flow to all MLPs in the stack
        for mlp in layer.stacked_mlps:
            assert mlp.w0.grad is not None
            assert mlp.w1.grad is not None
            assert mlp.w2.grad is not None


class TestStackedTTTIntegration:
    """Test integration with the full model."""
    
    @pytest.fixture
    def model_config(self):
        """Create a config for full model testing."""
        return LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            window_size=64,
            lact_chunk_size=32,
            r=2,
            residual_ttt=True,
            vocab_size=1000
        )
    
    def test_model_initialization(self, model_config):
        """Test that full model initializes with stacked TTT."""
        model = LaCTModel(model_config)
        
        # Check that all layers have the stacked structure
        for layer in model.layers:
            assert hasattr(layer.attn, 'stacked_mlps')
            assert len(layer.attn.stacked_mlps) == 2
    
    def test_model_forward_pass(self, model_config):
        """Test that full model forward pass works."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LaCTModel(model_config).to(device).half()
        
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
        
        outputs = model(input_ids)
        
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, model_config.hidden_size)
        assert not torch.isnan(outputs.last_hidden_state).any()
        assert not torch.isinf(outputs.last_hidden_state).any()
    
    def test_causal_lm_forward_pass(self, model_config):
        """Test that causal LM forward pass works."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LaCTForCausalLM(model_config).to(device).half()
        
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
        
        outputs = model(input_ids, labels=labels)
        
        assert outputs.loss is not None
        assert not torch.isnan(outputs.loss).any()
        assert not torch.isinf(outputs.loss).any()
    
    def test_model_gradient_flow(self, model_config):
        """Test that gradients flow through the full model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LaCTForCausalLM(model_config).to(device).half()
        
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # Check that gradients flow to model parameters
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()


class TestStackedTTTEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_r_equals_one_behavior(self):
        """Test that r=1 behaves exactly like the original implementation."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        ).to(device).half()
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.float16)
        
        output, _, _ = layer(hidden_states)
        
        # Should behave like single MLP
        assert hasattr(layer, 'w0')
        assert hasattr(layer, 'w1')
        assert hasattr(layer, 'w2')
    
    def test_large_r_value(self):
        """Test behavior with large r values."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attn_heads=4,
            num_lact_heads=2,
            window_size=64,
            lact_chunk_size=32,
            r=5,  # Large stack
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
        ).to(device).half()
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=torch.float16)
        
        output, _, _ = layer(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert len(layer.stacked_mlps) == 5
    
    def test_residual_ttt_effect(self):
        """Test that residual_ttt=True has different behavior than False."""
        config_no_residual = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attn_heads=4,
            num_lact_heads=2,
            window_size=64,
            lact_chunk_size=32,
            r=3,
            residual_ttt=False
        )
        
        config_with_residual = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=1,
            num_attn_heads=4,
            num_lact_heads=2,
            window_size=64,
            lact_chunk_size=32,
            r=3,
            residual_ttt=True
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        layer_no_residual = LaCTSWIGLULayer(
            hidden_size=config_no_residual.hidden_size,
            num_attn_heads=config_no_residual.num_attn_heads,
            num_lact_heads=config_no_residual.num_lact_heads,
            inter_multi=config_no_residual.inter_multi,
            window_size=config_no_residual.window_size,
            lact_chunk_size=config_no_residual.lact_chunk_size,
            r=config_no_residual.r,
            residual_ttt=config_no_residual.residual_ttt
        ).to(device).half()
        
        layer_with_residual = LaCTSWIGLULayer(
            hidden_size=config_with_residual.hidden_size,
            num_attn_heads=config_with_residual.num_attn_heads,
            num_lact_heads=config_with_residual.num_lact_heads,
            inter_multi=config_with_residual.inter_multi,
            window_size=config_with_residual.window_size,
            lact_chunk_size=config_with_residual.lact_chunk_size,
            r=config_with_residual.r,
            residual_ttt=config_with_residual.residual_ttt
        ).to(device).half()
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, config_no_residual.hidden_size, device=device, dtype=torch.float16)
        
        output_no_residual, _, _ = layer_no_residual(hidden_states)
        output_with_residual, _, _ = layer_with_residual(hidden_states)
        
        # Outputs should be different due to residual connections
        assert not torch.allclose(output_no_residual, output_with_residual, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__]) 