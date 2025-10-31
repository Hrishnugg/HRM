#!/usr/bin/env python3
"""Verify GPU, CUDA, and FlashAttention installation."""

import sys


def check_torch():
    """Check PyTorch installation and CUDA availability."""
    print("=" * 60)
    print("PyTorch & CUDA Check")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Compute capability: {props.major}.{props.minor}")
                print(f"    Total memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Multi-processors: {props.multi_processor_count}")
                
                # Check for Blackwell (sm_100)
                if props.major == 10 and props.minor == 0:
                    print(f"    ✓ Blackwell architecture detected (sm_100)")
                else:
                    print(f"    ⚠ Expected sm_100 (Blackwell), got sm_{props.major}{props.minor}")
        else:
            print("✗ CUDA not available!")
            return False
            
    except ImportError:
        print("✗ PyTorch not installed!")
        return False
    
    return True


def check_dtypes():
    """Check support for BFloat16 and Float16."""
    print("\n" + "=" * 60)
    print("Data Type Support")
    print("=" * 60)
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("✗ CUDA not available, skipping dtype checks")
            return False
        
        device = torch.device("cuda:0")
        
        # Test BFloat16
        try:
            x = torch.randn(4, 4, dtype=torch.bfloat16, device=device)
            y = x @ x.T
            print(f"✓ BFloat16 supported")
        except Exception as e:
            print(f"✗ BFloat16 error: {e}")
        
        # Test Float16
        try:
            x = torch.randn(4, 4, dtype=torch.float16, device=device)
            y = x @ x.T
            print(f"✓ Float16 supported")
        except Exception as e:
            print(f"✗ Float16 error: {e}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error checking dtypes: {e}")
        return False


def check_flash_attention():
    """Check FlashAttention installation and functionality."""
    print("\n" + "=" * 60)
    print("FlashAttention Check")
    print("=" * 60)
    
    try:
        import flash_attn
        from flash_attn import flash_attn_func
        print(f"✓ FlashAttention installed: {flash_attn.__version__}")
        
        # Try a minimal forward pass
        import torch
        if not torch.cuda.is_available():
            print("✗ CUDA not available, skipping FA forward test")
            return False
        
        device = torch.device("cuda:0")
        batch_size, seqlen, num_heads, head_dim = 2, 64, 8, 64
        
        # Create test tensors (FlashAttention expects float16 or bfloat16)
        q = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(batch_size, seqlen, num_heads, head_dim, dtype=torch.bfloat16, device=device)
        
        # Test forward pass
        try:
            out = flash_attn_func(q, k, v, causal=False)
            print(f"✓ FlashAttention forward pass successful")
            print(f"  Input shape: {q.shape}")
            print(f"  Output shape: {out.shape}")
            
            # Test causal variant
            out_causal = flash_attn_func(q, k, v, causal=True)
            print(f"✓ FlashAttention causal mode works")
            
            return True
            
        except Exception as e:
            print(f"✗ FlashAttention forward error: {e}")
            return False
            
    except ImportError:
        print("⚠ FlashAttention not installed (optional)")
        print("  The model will fall back to PyTorch SDPA")
        return True  # Not a fatal error
    except Exception as e:
        print(f"✗ FlashAttention error: {e}")
        return False


def check_sdpa():
    """Check PyTorch Scaled Dot Product Attention (SDPA)."""
    print("\n" + "=" * 60)
    print("PyTorch SDPA Check")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn.functional as F
        
        if not torch.cuda.is_available():
            print("✗ CUDA not available, skipping SDPA test")
            return False
        
        device = torch.device("cuda:0")
        batch_size, num_heads, seqlen, head_dim = 2, 8, 64, 64
        
        # Create test tensors
        q = torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(batch_size, num_heads, seqlen, head_dim, dtype=torch.bfloat16, device=device)
        
        # Test SDPA
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        print(f"✓ PyTorch SDPA works")
        print(f"  Input shape: {q.shape}")
        print(f"  Output shape: {out.shape}")
        
        # Test causal variant
        out_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"✓ PyTorch SDPA causal mode works")
        
        return True
        
    except Exception as e:
        print(f"✗ SDPA error: {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("HRM-v2 Environment Verification")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("PyTorch & CUDA", check_torch()))
    results.append(("Data Types", check_dtypes()))
    results.append(("PyTorch SDPA", check_sdpa()))
    results.append(("FlashAttention", check_flash_attention()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    all_critical_passed = results[0][1] and results[1][1] and results[2][1]
    
    if all_critical_passed:
        print("\n✓ Environment is ready for HRM-v2!")
        return 0
    else:
        print("\n✗ Some critical checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

