import torch
import pytest
import warnings
from typing import Dict, Any

from custom_models.lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig
from custom_models.lact_model.modeling_lact import LaCTForCausalLM


class TestTTTRefactoring:
    """Test suite for TTT refactoring to validate end-to-end vs greedy approaches."""
    
    @pytest.fixture
    def base_config(self) -> LaCTSWIGLUConfig:
        """Base configuration for testing."""
        return LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            lact_chunk_size=32,
            r=2,  # Stacked TTT
            residual_ttt=True,
            greedy_ttt=False,  # New end-to-end approach
        )
    
    @pytest.fixture
    def greedy_config(self) -> LaCTSWIGLUConfig:
        """Configuration for legacy greedy approach."""
        config = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            lact_chunk_size=32,
            r=2,  # Stacked TTT
            residual_ttt=False,  # Should be disabled for greedy
            greedy_ttt=True,  # Legacy approach
        )
        return config
    
    def test_config_compatibility(self, base_config: LaCTSWIGLUConfig):
        """Test that configuration properly handles greedy_ttt flag."""
        # Test that greedy_ttt=True automatically disables residual_ttt
        config = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            r=2,
            residual_ttt=True,
            greedy_ttt=True,
        )
        assert config.greedy_ttt is True
        assert config.residual_ttt is False  # Should be automatically disabled
    
    def test_config_validation(self):
        """Test configuration validation for incompatible settings."""
        with pytest.warns(UserWarning, match="greedy_ttt.*residual_ttt"):
            config = LaCTSWIGLUConfig(
                hidden_size=128,
                num_hidden_layers=2,
                num_attn_heads=4,
                num_lact_heads=2,
                r=2,
                residual_ttt=True,
                greedy_ttt=True,
            )
    
    def test_model_initialization(self, base_config: LaCTSWIGLUConfig):
        """Test that models initialize correctly with new configuration."""
        model = LaCTForCausalLM(base_config)
        assert model is not None
        assert hasattr(model.config, 'greedy_ttt')
        assert model.config.greedy_ttt is False
    
    def test_model_initialization_greedy(self, greedy_config: LaCTSWIGLUConfig):
        """Test that models initialize correctly with greedy configuration."""
        model = LaCTForCausalLM(greedy_config)
        assert model is not None
        assert model.config.greedy_ttt is True
        assert model.config.residual_ttt is False
    
    def test_forward_pass_compatibility(self, base_config: LaCTSWIGLUConfig):
        """Test that forward pass works with new configuration."""
        model = LaCTForCausalLM(base_config)
        if torch.cuda.is_available():
            model = model.cuda().half()  # Use fp16 for FlashAttention compatibility
        model.eval()  # Set to eval mode to avoid fused cross-entropy
        batch_size, seq_len = 2, 64
        
        # Create dummy input
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # Forward pass should work
        # Note: For end-to-end TTT, we need gradients even in eval mode
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        assert outputs.logits.shape == (batch_size, seq_len, base_config.vocab_size)
        assert outputs.loss is None  # No labels provided
    
    def test_forward_pass_greedy(self, greedy_config: LaCTSWIGLUConfig):
        """Test that forward pass works with greedy configuration."""
        model = LaCTForCausalLM(greedy_config)
        if torch.cuda.is_available():
            model = model.cuda().half()  # Use fp16 for FlashAttention compatibility
        model.eval()  # Set to eval mode to avoid fused cross-entropy
        batch_size, seq_len = 2, 64
        
        input_ids = torch.randint(0, greedy_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        assert outputs.logits.shape == (batch_size, seq_len, greedy_config.vocab_size)
    
    def test_training_mode_gradients(self, base_config: LaCTSWIGLUConfig):
        """Test that gradients flow correctly in training mode."""
        model = LaCTForCausalLM(base_config)
        if torch.cuda.is_available():
            model = model.cuda().half()
        model.train()
        
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            labels = labels.cuda()
        
        # Forward pass with labels
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True
        )
        
        # Should have loss
        assert outputs.loss is not None
        assert outputs.loss.requires_grad
        
        # Backward pass
        outputs.loss.backward()
        
        # Check that gradients exist for key parameters
        assert model.model.embeddings.weight.grad is not None
        assert model.lm_head.weight.grad is not None
        
        # Check that TTT layer parameters have gradients
        for layer in model.model.layers:
            assert layer.attn.qkv.weight.grad is not None
            assert layer.attn.o_proj.weight.grad is not None
    
    def test_end_to_end_vs_greedy_behavior(self):
        """Test that end-to-end and greedy approaches produce different results."""
        # Create two models with same base config but different TTT approaches
        base_params = {
            'hidden_size': 128,
            'num_hidden_layers': 2,
            'num_attn_heads': 4,
            'num_lact_heads': 2,
            'lact_chunk_size': 32,
            'r': 2,
            'residual_ttt': True,
        }
        
        end_to_end_config = LaCTSWIGLUConfig(**base_params, greedy_ttt=False)
        greedy_config = LaCTSWIGLUConfig(**base_params, greedy_ttt=True)
        
        model_e2e = LaCTForCausalLM(end_to_end_config)
        model_greedy = LaCTForCausalLM(greedy_config)
        
        if torch.cuda.is_available():
            model_e2e = model_e2e.cuda().half()
            model_greedy = model_greedy.cuda().half()
        
        model_e2e.eval()
        model_greedy.eval()
        
        # Set same random seed for fair comparison
        torch.manual_seed(42)
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, end_to_end_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # Get outputs
        outputs_e2e = model_e2e(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        torch.manual_seed(42)  # Reset seed
        outputs_greedy = model_greedy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # The outputs should be different due to different TTT approaches
        # (Note: This is a heuristic test - in practice they might be similar)
        logits_diff = torch.abs(outputs_e2e.logits - outputs_greedy.logits).mean()
        assert logits_diff > 1e-6, "End-to-end and greedy approaches should produce different results"
    
    def test_stacked_ttt_with_residual(self, base_config: LaCTSWIGLUConfig):
        """Test that stacked TTT with residual connections works correctly."""
        model = LaCTForCausalLM(base_config)
        if torch.cuda.is_available():
            model = model.cuda().half()
        model.eval()
        batch_size, seq_len = 2, 64
        
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # Forward pass should work without errors
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        assert outputs.logits.shape == (batch_size, seq_len, base_config.vocab_size)
    
    def test_chunk_size_handling(self, base_config: LaCTSWIGLUConfig):
        """Test that different chunk sizes are handled correctly."""
        # Test with chunk size larger than sequence length
        config = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            lact_chunk_size=128,  # Larger than seq_len
            r=2,
            greedy_ttt=False,
        )
        
        model = LaCTForCausalLM(config)
        if torch.cuda.is_available():
            model = model.cuda().half()
        model.eval()
        batch_size, seq_len = 2, 64
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # Should handle gracefully
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_momentum_and_muon_compatibility(self, base_config: LaCTSWIGLUConfig):
        """Test that momentum and muon features work with new implementation."""
        config = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            lact_chunk_size=32,
            r=2,
            use_momentum=True,
            use_muon=True,
            greedy_ttt=False,
        )
        
        model = LaCTForCausalLM(config)
        if torch.cuda.is_available():
            model = model.cuda().half()
        model.eval()
        batch_size, seq_len = 2, 64
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # Should work without errors
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    
    def test_low_rank_parameterization(self, base_config: LaCTSWIGLUConfig):
        """Test that low rank parameterization works with new implementation."""
        config = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            lact_chunk_size=32,
            r=2,
            w0_w2_low_rank=16,  # Enable low rank
            greedy_ttt=False,
        )
        
        model = LaCTForCausalLM(config)
        if torch.cuda.is_available():
            model = model.cuda().half()
        model.eval()
        batch_size, seq_len = 2, 64
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # Should work without errors
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_muon_in_end_to_end_ttt(self):
        """Test that Muon optimization works correctly in end-to-end TTT."""
        config = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            lact_chunk_size=32,
            r=2,
            use_muon=True,
            use_momentum=True,
            greedy_ttt=False,
        )
        
        model = LaCTForCausalLM(config)
        
        # Move to GPU and set precision
        if torch.cuda.is_available():
            model = model.cuda().half()
        
        model.eval()
        
        # Create input
        batch_size, seq_len = 2, 512
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # Forward pass
        outputs = model(input_ids)
        
        # Check that outputs are valid
        assert outputs.logits is not None
        assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
        
        # Verify that Muon is properly applied by checking that gradients are orthogonalized
        # We can't directly test the orthogonalization, but we can verify the function exists
        # and the model runs without errors
        
        # Test with stacked MLPs and Muon
        config_stacked = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            lact_chunk_size=32,
            r=3,  # Use stacked MLPs
            residual_ttt=True,
            use_muon=True,
            use_momentum=True,
            greedy_ttt=False,
        )
        
        model_stacked = LaCTForCausalLM(config_stacked)
        
        # Move to GPU and set precision
        if torch.cuda.is_available():
            model_stacked = model_stacked.cuda().half()
        
        model_stacked.eval()
        
        # Forward pass with stacked MLPs
        outputs_stacked = model_stacked(input_ids)
        
        # Check that outputs are valid
        assert outputs_stacked.logits is not None
        assert outputs_stacked.logits.shape == (batch_size, seq_len, config.vocab_size)
        
        # Test with low-rank parameterization and Muon
        config_low_rank = LaCTSWIGLUConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attn_heads=4,
            num_lact_heads=2,
            lact_chunk_size=32,
            r=2,
            w0_w2_low_rank=32,
            use_muon=True,
            use_momentum=True,
            greedy_ttt=False,
        )
        
        model_low_rank = LaCTForCausalLM(config_low_rank)
        
        # Move to GPU and set precision
        if torch.cuda.is_available():
            model_low_rank = model_low_rank.cuda().half()
        
        model_low_rank.eval()
        
        # Forward pass with low-rank weights
        outputs_low_rank = model_low_rank(input_ids)
        
        # Check that outputs are valid
        assert outputs_low_rank.logits is not None
        assert outputs_low_rank.logits.shape == (batch_size, seq_len, config.vocab_size)


if __name__ == "__main__":
    pytest.main([__file__]) 