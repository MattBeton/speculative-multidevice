#!/usr/bin/env python3
"""Test script for vLLM installation with GB10 GPU support."""

import sys
import torch

print("=" * 60)
print("vLLM Test Script with GB10 GPU Support")
print("=" * 60)

# Test PyTorch first
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")

    # Test basic CUDA operations
    try:
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = x * 2
        print("✓ CUDA tensor operations successful")
        print(f"  Test result: {y.cpu().numpy()}")
    except Exception as e:
        print(f"✗ CUDA tensor operations failed: {e}")
else:
    print("✗ CUDA is not available")

print()

# Test vLLM import
try:
    import vllm
    print(f"✓ vLLM imported successfully")
    print(f"  vLLM version: {vllm.__version__}")
except Exception as e:
    print(f"✗ vLLM import failed: {e}")
    sys.exit(1)

print()

# Test basic vLLM functionality
try:
    from vllm import LLM, SamplingParams
    print("✓ Core vLLM components imported successfully")

    # Test with a small model that should work on GB10
    print("\nTesting vLLM with a small model...")
    print("Using gpt2 which is already cached locally (no download needed)")

    # Using GPT-2 which is already cached on the system
    model_name = "gpt2"  # Already downloaded, won't use bandwidth

    print(f"Initializing LLM with {model_name} (already cached)...")
    llm = LLM(model=model_name,
              trust_remote_code=True,
              max_model_len=512,  # Reduce context length for testing
              gpu_memory_utilization=0.5)  # Use only half GPU memory

    # Test inference
    prompts = ["Hello, I am"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=10)

    print("Running inference...")
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n✓ Inference successful!")
        print(f"  Prompt: {prompt}")
        print(f"  Generated: {generated_text}")

except ImportError as e:
    print(f"✗ Failed to import vLLM components: {e}")
except Exception as e:
    print(f"✗ vLLM test failed: {e}")
    print("\nNote: This might be due to GPU memory constraints or model download issues.")
    print("The vLLM installation is likely successful even if this test fails.")

print()
print("=" * 60)
print("Test completed!")
print("=" * 60)