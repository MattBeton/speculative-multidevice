#!/usr/bin/env python3
"""
Working Speculative Decoding Implementation using Transformers

Since vLLM has compatibility issues with ARM Blackwell architecture,
this implementation uses transformers directly with CUDA support.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np
from typing import List, Tuple, Optional

print("=" * 60)
print("Speculative Decoding with Transformers (ARM/Blackwell Compatible)")
print("=" * 60)

# Check CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print()

class SpeculativeDecoder:
    """
    Implements speculative decoding for faster generation.

    Uses a small draft model to generate candidates and a larger
    target model to verify them in batch.
    """

    def __init__(
        self,
        draft_model_name: str = "gpt2",  # Small, fast model
        target_model_name: str = "gpt2-medium",  # Larger, more accurate model
        device: str = None,
        gamma: int = 5  # Number of speculative tokens
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma

        print(f"Loading models on {self.device}...")

        # Load draft model (small, fast)
        print(f"Loading draft model: {draft_model_name}")
        self.draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.draft_model.eval()

        # Load target model (larger, accurate)
        print(f"Loading target model: {target_model_name}")
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.target_model.eval()

        # Ensure both models use the same tokenizer
        self.tokenizer = self.target_tokenizer

        print("Models loaded successfully!\n")

    def get_logits_batch(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """Get logits for a batch of token sequences."""
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            return outputs.logits

    def sample_from_logits(self, logits: torch.Tensor, temperature: float = 0.8) -> int:
        """Sample a token from logits distribution."""
        if temperature == 0:
            return torch.argmax(logits, dim=-1).item()

        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def speculative_generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.8,
        verbose: bool = True
    ) -> Tuple[str, dict]:
        """
        Generate text using speculative decoding.

        Returns generated text and statistics.
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_ids = input_ids.clone()

        stats = {
            "total_tokens": 0,
            "accepted_tokens": 0,
            "draft_time": 0,
            "verify_time": 0,
            "total_time": 0
        }

        start_time = time.time()

        while generated_ids.shape[1] < input_ids.shape[1] + max_tokens:
            # Step 1: Generate γ draft tokens with small model
            draft_start = time.time()
            draft_tokens = []
            draft_input = generated_ids.clone()

            for _ in range(self.gamma):
                draft_logits = self.get_logits_batch(self.draft_model, draft_input)
                next_token = self.sample_from_logits(
                    draft_logits[0, -1, :],
                    temperature=temperature
                )
                draft_tokens.append(next_token)
                draft_input = torch.cat([
                    draft_input,
                    torch.tensor([[next_token]], device=self.device)
                ], dim=1)

            stats["draft_time"] += time.time() - draft_start

            # Step 2: Verify all draft tokens with target model in one batch
            verify_start = time.time()

            # Create input with all draft tokens
            verify_input = torch.cat([
                generated_ids,
                torch.tensor([draft_tokens], device=self.device)
            ], dim=1)

            # Get target model logits for all positions
            target_logits = self.get_logits_batch(self.target_model, verify_input)

            # Step 3: Accept/reject draft tokens
            accepted_tokens = []
            for i, draft_token in enumerate(draft_tokens):
                position = generated_ids.shape[1] + i

                # Get probability distributions
                target_probs = torch.softmax(
                    target_logits[0, position - 1, :] / temperature,
                    dim=-1
                )

                # For simplicity, accept if target model agrees
                # (in production, use probability ratio for acceptance)
                target_token = torch.argmax(target_probs).item()

                if target_token == draft_token:
                    accepted_tokens.append(draft_token)
                    stats["accepted_tokens"] += 1
                else:
                    # Reject and use target model's token
                    accepted_tokens.append(target_token)
                    break  # Stop at first rejection

            stats["verify_time"] += time.time() - verify_start
            stats["total_tokens"] += len(accepted_tokens)

            # Add accepted tokens to generation
            generated_ids = torch.cat([
                generated_ids,
                torch.tensor([accepted_tokens], device=self.device)
            ], dim=1)

            # Check for EOS token
            if self.tokenizer.eos_token_id in accepted_tokens:
                break

        stats["total_time"] = time.time() - start_time

        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )

        if verbose:
            print(f"\nGeneration Statistics:")
            print(f"  Total tokens generated: {stats['total_tokens']}")
            print(f"  Accepted draft tokens: {stats['accepted_tokens']}")
            print(f"  Acceptance rate: {stats['accepted_tokens']/max(1, stats['total_tokens']):.2%}")
            print(f"  Draft time: {stats['draft_time']:.2f}s")
            print(f"  Verify time: {stats['verify_time']:.2f}s")
            print(f"  Total time: {stats['total_time']:.2f}s")
            print(f"  Tokens/second: {stats['total_tokens']/stats['total_time']:.2f}")

        return generated_text, stats


def main():
    """Main demonstration of speculative decoding."""

    print("Initializing Speculative Decoder...")
    print("Note: First run will download models if not cached locally.\n")

    try:
        # Initialize decoder with small models for testing
        # You can replace with larger models like:
        # draft: "microsoft/phi-2", target: "mistralai/Mistral-7B-v0.1"
        decoder = SpeculativeDecoder(
            draft_model_name="gpt2",
            target_model_name="gpt2-medium",
            gamma=5
        )

        # Test prompts
        prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a distant galaxy",
            "The key to successful machine learning is"
        ]

        for prompt in prompts:
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt}")
            print(f"{'='*60}")

            generated_text, stats = decoder.speculative_generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.8,
                verbose=True
            )

            print(f"\nGenerated text:")
            print(generated_text)

        print("\n" + "="*60)
        print("✓ Speculative decoding is working successfully!")
        print("You can now:")
        print("1. Load larger models for better quality")
        print("2. Adjust gamma for different speed/quality tradeoffs")
        print("3. Implement more sophisticated acceptance criteria")
        print("4. Process batches of prompts in parallel")
        print("="*60)

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have enough GPU memory for both models")
        print("2. Try smaller models if you encounter OOM errors")
        print("3. Check that transformers library is properly installed")


if __name__ == "__main__":
    main()