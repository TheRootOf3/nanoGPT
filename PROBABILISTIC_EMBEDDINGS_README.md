# Probabilistic Embeddings in GPT

This implementation adds support for **probabilistic embeddings** (VAE-style) alongside the standard static embeddings in the GPT model.

## Overview

The model now supports two types of embeddings:

1. **Static Embeddings** (default): Standard learnable embedding tables, as in the original GPT
2. **Probabilistic Embeddings**: VAE-style embeddings where the model learns mean (μ) and log variance (log σ²) for each token/position, then samples embeddings from a Gaussian distribution

## Key Features

### Static Embeddings
- **What**: Traditional approach where each token/position has a fixed embedding vector
- **When to use**: Default choice for most applications, faster training, deterministic
- **Parameter count**: Standard (vocab_size + block_size) × embedding_dim

### Probabilistic Embeddings
- **What**: Each token/position has a learned distribution N(μ, σ²) from which embeddings are sampled
- **When to use**: 
  - When you want stronger regularization
  - For representation learning research
  - When exploring uncertainty in embeddings
  - For applications requiring stochastic behavior
- **Parameter count**: 2 × (vocab_size + block_size) × embedding_dim (double due to mean + variance)
- **Loss**: Combined cross-entropy + β×KL divergence
  - KL divergence regularizes the embedding distributions toward N(0, 1)
  - β (default: 0.01) controls the strength of regularization

## Usage

### Creating a Model with Static Embeddings (Default)

```python
from model import GPT, GPTConfig

config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True,
    use_probabilistic_embeddings=False  # Default
)

model = GPT(config)
```

### Creating a Model with Probabilistic Embeddings

```python
from model import GPT, GPTConfig

config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True,
    use_probabilistic_embeddings=True  # Enable probabilistic embeddings
)

model = GPT(config)
```

### Training

Training is identical for both versions:

```python
# Forward pass
logits, loss = model(input_ids, targets)

# For static embeddings: loss = cross_entropy
# For probabilistic embeddings: loss = cross_entropy + beta * KL_divergence

# Backward pass
loss.backward()
optimizer.step()
```

### Inference with Probabilistic Embeddings

#### Stochastic Mode (Default)
Embeddings are sampled from the learned distributions:

```python
model.eval()
with torch.no_grad():
    output = model.generate(prompt, max_new_tokens=100)
    # Each generation will be different due to stochastic embeddings
```

#### Deterministic Mode (Use Means)
Use only the mean vectors (no sampling):

```python
model.eval()
model.use_mean_embeddings()  # Switch to deterministic mode

with torch.no_grad():
    output = model.generate(prompt, max_new_tokens=100)
    # Embeddings are now deterministic (though token sampling still has randomness)
```

## Implementation Details

### ProbabilisticEmbedding Class

```python
class ProbabilisticEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.mu = nn.Embedding(num_embeddings, embedding_dim)      # Mean
        self.logvar = nn.Embedding(num_embeddings, embedding_dim)  # Log variance
        
    def forward(self, idx):
        mu = self.mu(idx)
        logvar = self.logvar(idx)
        
        # Reparameterization trick: z = μ + σ * ε, where ε ~ N(0, 1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # KL divergence: KL(N(μ, σ²) || N(0, 1))
        kl_loss = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        
        return z, kl_loss
```

### Loss Computation

For probabilistic embeddings:
```
Total Loss = CE Loss + β × (KL_token + KL_position)
```

Where:
- `CE Loss`: Standard cross-entropy loss
- `KL_token`: KL divergence for token embeddings
- `KL_position`: KL divergence for position embeddings
- `β`: Weight factor (default: 0.01)

You can adjust β in the forward method:
```python
# In model.py, line ~257
beta = 0.01  # Adjust this value
loss = ce_loss + beta * kl_loss
```

## Parameter Count Comparison

Example for a small model (vocab_size=50304, block_size=256, n_embd=384):

| Component | Static | Probabilistic | Difference |
|-----------|--------|---------------|------------|
| Token embeddings | 19.3M | 38.6M | +19.3M |
| Position embeddings | 98K | 196K | +98K |
| **Total model** | **29.9M** | **68.6M** | **+38.6M (2.29×)** |

The increase is roughly 2× for the embedding parameters (exactly 2× for embeddings, slightly less for total model).

## Example Script

Run the provided example to see both versions in action:

```bash
python3 example_embeddings.py
```

This will:
1. Create and test a model with static embeddings
2. Create and test a model with probabilistic embeddings
3. Demonstrate stochastic vs deterministic generation
4. Compare parameter counts

## Benefits and Trade-offs

### Probabilistic Embeddings

**Benefits:**
- Stronger regularization through KL divergence
- Models uncertainty in embeddings
- May improve generalization
- Useful for representation learning research
- Stochastic behavior can help exploration

**Trade-offs:**
- 2× embedding parameters
- Slightly slower training (sampling + KL computation)
- Additional hyperparameter (β) to tune
- More complex to interpret
- No weight tying with output layer

### Static Embeddings

**Benefits:**
- Fewer parameters
- Faster training
- Simpler to understand and debug
- Deterministic behavior
- Weight tying possible with output layer

**Trade-offs:**
- Less regularization
- No built-in uncertainty modeling
- May overfit more easily

## Advanced Usage

### Adjusting β (KL Weight)

The β parameter controls the trade-off between reconstruction (cross-entropy) and regularization (KL divergence):

- **β = 0**: No regularization, behaves similarly to static embeddings
- **β = 0.001-0.01**: Light regularization (default range)
- **β = 0.1-1.0**: Strong regularization, may hurt reconstruction

To change β, edit `model.py` around line 257:
```python
beta = 0.01  # Your desired value
loss = ce_loss + beta * kl_loss
```

### Analyzing Learned Distributions

You can inspect the learned distributions:

```python
model = GPT(config)
# After training...

# Get mean and variance for token 100
token_id = 100
mu = model.transformer.wte.mu.weight[token_id]
logvar = model.transformer.wte.logvar.weight[token_id]
std = torch.exp(0.5 * logvar)

print(f"Token {token_id}:")
print(f"Mean: {mu[:5]}...")  # First 5 dimensions
print(f"Std: {std[:5]}...")   # First 5 dimensions
```

## References

This implementation is inspired by:
- **VAE (Variational Autoencoder)**: Kingma & Welling, 2013
- **β-VAE**: Higgins et al., 2017
- GPT architecture: Radford et al., 2018-2019

## Citation

If you use probabilistic embeddings in your research, consider citing the original VAE paper:

```bibtex
@article{kingma2013auto,
  title={Auto-encoding variational bayes},
  author={Kingma, Diederik P and Welling, Max},
  journal={arXiv preprint arXiv:1312.6114},
  year={2013}
}
```
