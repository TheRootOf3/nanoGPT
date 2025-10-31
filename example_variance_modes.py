"""
Example script demonstrating different variance modes for probabilistic embeddings.
"""

import torch
from model import GPT, GPTConfig


def demo_variance_mode(variance_mode, config_base):
    """Demonstrate a specific variance mode"""
    print(f"\n{'=' * 60}")
    print(f"VARIANCE MODE: {variance_mode.upper()}")
    print("=" * 60)

    config = GPTConfig(
        **{
            **config_base.__dict__,
            "use_probabilistic_embeddings": True,
            "variance_mode": variance_mode,
        }
    )

    model = GPT(config)

    # Get parameter counts
    total_params = model.get_num_params()

    # Count embedding-specific parameters
    mu_params = model.transformer.wte.mu.weight.numel()

    if variance_mode == "scalar":
        var_params = model.transformer.wte.logvar.weight.numel()
        var_description = f"{config.vocab_size} scalars (one per embedding)"
    elif variance_mode == "shared":
        var_params = model.transformer.wte.logvar.numel()
        var_description = f"{config.n_embd} elements (one vector shared by all)"
    else:  # 'full'
        var_params = model.transformer.wte.logvar.weight.numel()
        var_description = (
            f"{config.vocab_size} × {config.n_embd} (one vector per embedding)"
        )

    print(f"Total parameters: {total_params:,}")
    print(f"  Token mean (mu): {mu_params:,}")
    print(f"  Token variance: {var_params:,} ({var_description})")
    print(
        f"  Position embeddings: {model.transformer.wpe.weight.numel():,} (always static)"
    )

    # Test forward pass
    batch_size, seq_len = 4, 64
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets)
    print("\nForward pass:")
    print(f"  Input shape: {idx.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss (CE + beta*KL): {loss.item():.4f}")

    return model, total_params


def compare_all_modes():
    """Compare all variance modes"""
    print("=" * 60)
    print("COMPARING ALL VARIANCE MODES")
    print("=" * 60)

    config_base = GPTConfig(
        block_size=256,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=False,
    )

    # Static baseline
    config_static = GPTConfig(
        **{**config_base.__dict__, "use_probabilistic_embeddings": False}
    )
    model_static = GPT(config_static)
    static_params = model_static.get_num_params()

    print(f"\nBaseline (static embeddings): {static_params:,} parameters")

    # Test each variance mode
    results = {}
    for mode in ["scalar", "shared", "full"]:
        model, params = demo_variance_mode(mode, config_base)
        results[mode] = {
            "model": model,
            "params": params,
            "overhead": params - static_params,
        }

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)

    print(f"\n{'Mode':<12} {'Total Params':>15} {'vs Static':>15} {'Overhead':>12}")
    print("-" * 60)
    print(f"{'static':<12} {static_params:>15,} {'-':>15} {'-':>12}")

    for mode in ["scalar", "shared", "full"]:
        r = results[mode]
        ratio = r["params"] / static_params
        print(f"{mode:<12} {r['params']:>15,} {ratio:>14.2f}x {r['overhead']:>11,}")

    # Detail breakdown
    print("\n" + "=" * 60)
    print("PARAMETER BREAKDOWN")
    print("=" * 60)

    vocab_size = config_base.vocab_size
    n_embd = config_base.n_embd

    print(f"\nVocabulary size: {vocab_size:,}")
    print(f"Embedding dimension: {n_embd:,}")
    print(f"Base token embedding params: {vocab_size * n_embd:,}")

    print("\nVariance parameters:")
    print(f"  scalar: {vocab_size:,} (one scalar per token)")
    print(f"  shared: {n_embd:,} (one vector for all tokens)")
    print(f"  full:   {vocab_size * n_embd:,} (one vector per token)")

    print("\nNote: Token embedding means (mu) are tied with lm_head")
    print("Only variance parameters add to the total parameter count")


def test_generation():
    """Test generation with different modes"""
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    config_base = GPTConfig(
        block_size=128,
        vocab_size=1000,  # Smaller for faster demo
        n_layer=3,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
    )

    prompt = torch.randint(0, config_base.vocab_size, (1, 10))

    for mode in ["scalar", "shared", "full"]:
        print(f"\n{mode.upper()} mode:")
        config = GPTConfig(
            **{
                **config_base.__dict__,
                "use_probabilistic_embeddings": True,
                "variance_mode": mode,
            }
        )
        model = GPT(config)
        model.eval()

        with torch.no_grad():
            output = model.generate(prompt, max_new_tokens=5, temperature=1.0)
            print(f"  Generated shape: {output.shape}")
            print(f"  Total params: {model.get_num_params():,}")


if __name__ == "__main__":
    # Compare all modes
    compare_all_modes()

    # Test generation
    test_generation()

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
1. SCALAR MODE (recommended for most cases):
   - Minimal parameter overhead (only vocab_size extra params ~50K)
   - Each embedding can learn its own uncertainty level
   - Good balance between flexibility and efficiency
   - Weight tying keeps base model size nearly unchanged
   
2. SHARED MODE (most parameter-efficient):
   - Absolute minimum overhead (only n_embd extra params ~384)
   - All embeddings share the same variance pattern across dimensions
   - Best when you want minimal parameter increase
   - May be too restrictive if embeddings have different uncertainty levels
   
3. FULL MODE (most flexible):
   - Adds one full embedding table for variance
   - Each embedding dimension can have independent variance
   - Still uses weight tying for means (mu tied with lm_head)
   - Maximum flexibility, ~1.65x model size vs 2.29x without weight tying
    """)
    print("=" * 60)
