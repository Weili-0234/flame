#!/usr/bin/env python3
"""
Fix FLA (Flash Linear Attention) compatibility with torch.compile.

This script patches the problematic CUDA utility functions to be compatible
with torch._dynamo by allowing them in the computation graph.
"""

import torch
import torch.compiler
from fla.utils import get_multiprocessor_count

def patch_fla_for_compile():
    """Patch FLA functions to be compatible with torch.compile."""
    
    # Allow the problematic function in the computation graph
    torch.compiler.allow_in_graph(get_multiprocessor_count)
    
    print("✅ FLA functions patched for torch.compile compatibility")

if __name__ == "__main__":
    patch_fla_for_compile()