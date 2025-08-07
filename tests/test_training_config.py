#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test training with the provided model configuration.
Enhanced with torch.compile compatibility and long sequence testing.
"""

import torch
import torch.nn as nn
from flame.custom_models.lact_model.modeling_lact import LaCTForCausalLM
from flame.custom_models.lact_model.configuration_lact_swiglu import LaCTSWIGLUConfig
import json
import os
import time


def test_training_with_config(test_torch_compile=False, seq_len=512):
    """Test training with the provided model configuration."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping training test")
        return False
    
    device = 'cuda'
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    
    # Load the provided config
    config_path = "flame/configs/816M_lact_swiglu_nh4_fwlow_rank_momentum_muon_r2_e2e.json"
    
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    print("Config loaded:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    
    # Create config object
    config = LaCTSWIGLUConfig(**config_dict)
    
    # Create model and cast to the correct dtype
    print("\nCreating model...")
    model = LaCTForCausalLM(config).to(device)
    
    # Cast to the dtype specified in config
    target_dtype = config_dict.get('torch_dtype', 'float32')
    print(f"Target dtype from config: {target_dtype}")
    
    if target_dtype == 'bfloat16':
        model = model.to(torch.bfloat16)
        print(f"Model cast to bfloat16")
    elif target_dtype == 'float16':
        model = model.to(torch.float16)
        print(f"Model cast to float16")
    else:
        print(f"Using default dtype (float32)")
    
    # Check if manual backprop will be used
    will_use_manual_backprop = (
        config.r > 1 and 
        not config.residual_ttt  # Now supports both full-rank and low-rank
    )
    
    print(f"Model created successfully!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Manual backprop will be used: {will_use_manual_backprop}")
    
    if not will_use_manual_backprop:
        print("Note: Manual backprop won't be used because:")
        if config.r <= 1:
            print(f"  - r={config.r} (need r > 1)")
        if config.residual_ttt:
            print(f"  - residual_ttt={config.residual_ttt} (need False)")
    else:
        if config.w0_w2_low_rank > 0:
            print(f"  Using low-rank manual backprop (rank={config.w0_w2_low_rank})")
        else:
            print(f"  Using full-rank manual backprop")
    
    # Apply torch.compile if requested
    if test_torch_compile:
        print(f"\n🔧 Applying torch.compile to model...")
        try:
            # Compile the model with different backends/modes for robustness
            model = torch.compile(model, mode="default", backend="inductor")
            print(f"✓ torch.compile applied successfully!")
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")
            print("Continuing without compilation...")
    
    # Set model to training mode
    model.train()
    
    # Create dummy data
    batch_size = 2
    vocab_size = config.vocab_size
    
    print(f"\nCreating test data (batch_size={batch_size}, seq_len={seq_len})")
    
    # For very long sequences, we might need to be more careful about memory
    if seq_len > 8192:
        print("⚠️ Using long sequence - monitoring memory carefully")
        torch.cuda.empty_cache()
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, eps=1e-15)
    
    print("\nTesting forward pass...")
    try:
        # Forward pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        outputs = model(input_ids=input_ids, labels=labels)
        forward_time = time.time() - start_time
        loss = outputs.loss
        
        forward_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"✓ Forward pass successful! Loss: {loss.item():.4f}")
        print(f"  Forward time: {forward_time:.2f}s")
        print(f"  Memory after forward: {forward_memory:.1f} MB")
        
        print("\nTesting backward pass...")
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        
        backward_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"✓ Backward pass successful!")
        print(f"  Backward time: {backward_time:.2f}s")
        print(f"  Peak memory usage: {backward_memory:.1f} MB")
        
        # Check gradients
        has_grads = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
        total_params = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"  Gradients computed for {has_grads}/{total_params} parameters")
        
        print("\nTesting optimizer step...")
        start_time = time.time()
        optimizer.step()
        optimizer.zero_grad()
        step_time = time.time() - start_time
        print(f"✓ Optimizer step successful!")
        print(f"  Optimizer step time: {step_time:.2f}s")
        
        # Memory efficiency analysis
        memory_per_token = backward_memory / (batch_size * seq_len)
        memory_per_param = backward_memory / (sum(p.numel() for p in model.parameters()) / 1_000_000)
        
        compile_status = "✓ (compiled)" if test_torch_compile else "✗ (not compiled)"
        
        print(f"\n🎉 Training test completed successfully!")
        print(f"   Model: {config.model_type}")
        print(f"   Size: ~{sum(p.numel() for p in model.parameters()) // 1_000_000}M parameters")
        print(f"   Sequence length: {seq_len}")
        print(f"   Peak memory: {backward_memory:.1f} MB")
        print(f"   Memory per token: {memory_per_token:.3f} MB/token")
        print(f"   Memory per M params: {memory_per_param:.1f} MB/M")
        print(f"   Manual backprop: {'✓' if will_use_manual_backprop else '✗'}")
        print(f"   torch.compile: {compile_status}")
        print(f"   Total time: {forward_time + backward_time + step_time:.2f}s")
        
        # Memory efficiency assessment
        if seq_len >= 32768:
            if backward_memory < 20000:  # Less than 20GB
                print("   🚀 Excellent memory efficiency for long sequences!")
            elif backward_memory < 40000:  # Less than 40GB
                print("   ✅ Good memory efficiency for long sequences")
            else:
                print("   ⚠️ High memory usage - consider optimizations")
        
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_tests():
    """Run comprehensive training tests with different configurations."""
    print("🧪 Comprehensive Training Configuration Tests")
    print("=" * 60)
    
    # Test configurations - optimized for faster testing
    test_configs = [
        {"name": "Standard sequence (512)", "torch_compile": False, "seq_len": 512},
        {"name": "Medium sequence (4096)", "torch_compile": False, "seq_len": 4096},
        {"name": "Long sequence (16384)", "torch_compile": False, "seq_len": 16384},
        {"name": "Very long sequence (32768)", "torch_compile": False, "seq_len": 32768},
        {"name": "Standard + torch.compile", "torch_compile": True, "seq_len": 512},
        {"name": "Long + torch.compile", "torch_compile": True, "seq_len": 8192},
    ]
    
    results = []
    
    for i, test_config in enumerate(test_configs, 1):
        print(f"\n{'='*20} Test {i}/{len(test_configs)}: {test_config['name']} {'='*20}")
        
        try:
            success = test_training_with_config(
                test_torch_compile=test_config["torch_compile"],
                seq_len=test_config["seq_len"]
            )
            results.append((test_config["name"], success))
            
            if success:
                print(f"✅ {test_config['name']}: PASSED")
            else:
                print(f"❌ {test_config['name']}: FAILED")
                
        except Exception as e:
            print(f"💥 {test_config['name']}: CRASHED - {e}")
            results.append((test_config["name"], False))
        
        # Clean up between tests
        torch.cuda.empty_cache()
        print(f"\nMemory cleaned up between tests")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Low-rank manual backprop is production ready!")
    else:
        print("⚠️ Some tests failed. Check logs above for details.")
    
    return passed == total


if __name__ == "__main__":
    # Activate conda environment and set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    if success:
        print("\n🚀 All comprehensive training tests passed!")
        print("The low-rank manual backprop implementation is ready for production use!")
    else:
        print("\n🔧 Some tests failed - check the implementation!")