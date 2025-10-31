"""
Example script demonstrating the use of static vs probabilistic embeddings in GPT.
"""

import torch
from model import GPT, GPTConfig


def demo_static_embeddings():
    """Demonstrate standard GPT with static embeddings"""
    print("=" * 60)
    print("STATIC EMBEDDINGS MODEL")
    print("=" * 60)

    # Create config with static embeddings (default)
    config = GPTConfig(
        block_size=256,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=False,
        use_probabilistic_embeddings=False,  # Static embeddings
    )

    model = GPT(config)
    print(f"Total parameters: {model.get_num_params():,}")

    # Create dummy input
    batch_size, seq_len = 4, 64
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, loss = model(idx, targets)
    print(f"Input shape: {idx.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Generate
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            idx[:1, :10], max_new_tokens=20, temperature=0.8, top_k=40
        )
        print(f"Generated sequence shape: {generated.shape}")

    return model


def demo_probabilistic_embeddings():
    """Demonstrate GPT with VAE-style probabilistic embeddings"""
    print("\n" + "=" * 60)
    print("PROBABILISTIC EMBEDDINGS MODEL (VAE-style)")
    print("=" * 60)

    # Create config with probabilistic embeddings
    config = GPTConfig(
        block_size=256,
        vocab_size=50304,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=False,
        use_probabilistic_embeddings=True,  # Probabilistic embeddings!
        variance_mode="scalar",  # Options: 'scalar', 'shared', 'full'
    )

    model = GPT(config)
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Variance mode: {config.variance_mode}")
    print(
        "Note: Probabilistic embeddings only apply to tokens; positions always use static embeddings"
    )

    # Create dummy input
    batch_size, seq_len = 4, 64
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass - includes KL divergence in loss
    logits, loss = model(idx, targets)
    print(f"Input shape: {idx.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss (CE + beta*KL): {loss.item():.4f}")

    # Generate with sampling (stochastic)
    print("\nGeneration with stochastic sampling:")
    model.eval()
    with torch.no_grad():
        generated1 = model.generate(
            idx[:1, :10], max_new_tokens=20, temperature=0.8, top_k=40
        )
        generated2 = model.generate(
            idx[:1, :10], max_new_tokens=20, temperature=0.8, top_k=40
        )
        print(f"Generated sequence 1 shape: {generated1.shape}")
        print(f"Generated sequence 2 shape: {generated2.shape}")
        print(
            f"Sequences differ (due to stochastic embeddings): {not torch.equal(generated1, generated2)}"
        )

    # Switch to deterministic mode (use means only)
    print("\nSwitching to deterministic mode (mean embeddings):")
    model.use_mean_embeddings()

    # Test with temperature=1.0 and top_k (still has sampling randomness in token selection)
    with torch.no_grad():
        generated3 = model.generate(
            idx[:1, :10], max_new_tokens=20, temperature=1.0, top_k=None
        )
        generated4 = model.generate(
            idx[:1, :10], max_new_tokens=20, temperature=1.0, top_k=None
        )
        print(f"Generated sequence 3 shape: {generated3.shape}")
        print(f"Generated sequence 4 shape: {generated4.shape}")
        # Note: sequences may still differ due to multinomial sampling in generation
        # The embeddings themselves are now deterministic (using means)
        print(
            "Note: Sequences may still differ due to token sampling, but embeddings are deterministic"
        )

    return model


def compare_parameter_counts():
    """Compare parameter counts between the two approaches"""
    print("\n" + "=" * 60)
    print("PARAMETER COUNT COMPARISON")
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

    config_static = GPTConfig(
        **{**config_base.__dict__, "use_probabilistic_embeddings": False}
    )
    config_prob = GPTConfig(
        **{**config_base.__dict__, "use_probabilistic_embeddings": True}
    )

    model_static = GPT(config_static)
    model_prob = GPT(config_prob)

    static_params = model_static.get_num_params()
    prob_params = model_prob.get_num_params()

    print(f"Static embeddings:        {static_params:,} parameters")
    print(f"Probabilistic embeddings: {prob_params:,} parameters")
    print(f"Difference:               {prob_params - static_params:,} parameters")
    print(f"Ratio:                    {prob_params / static_params:.2f}x")

    # Break down the difference
    vocab_emb_diff = config_base.vocab_size * config_base.n_embd
    print("\nBreakdown of additional parameters:")
    print(f"Token embeddings (variance only): ~{vocab_emb_diff:,} (mode-dependent)")
    print("  - scalar mode: vocab_size params")
    print("  - shared mode: embedding_dim params")
    print("  - full mode: vocab_size × embedding_dim params")
    print("Note: Token means (mu) are tied with lm_head (no extra parameters)")
    print("Position embeddings: Always static (no additional parameters)")


if __name__ == "__main__":
    # Demonstrate static embeddings
    model_static = demo_static_embeddings()

    # Demonstrate probabilistic embeddings
    model_prob = demo_probabilistic_embeddings()

    # Compare parameter counts
    compare_parameter_counts()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("- Static embeddings: Standard approach, deterministic")
    print("- Probabilistic embeddings: VAE-style with learned mean and variance")
    print("  * Adds KL divergence regularization to the loss")
    print("  * Three variance modes available:")
    print("    - 'scalar': One variance value per embedding (minimal overhead)")
    print(
        "    - 'shared': One variance vector shared by all embeddings (smallest overhead)"
    )
    print(
        "    - 'full': One variance vector per embedding (most flexible, doubles params)"
    )
    print("  * Can use stochastic sampling or deterministic means")
    print("  * May help with regularization and representation learning")
    print("  * See example_variance_modes.py for detailed comparison")
    print("=" * 60)
