# Quick Reference: Static vs Probabilistic Embeddings

## At a Glance

| Feature | Static Embeddings | Probabilistic Embeddings |
|---------|------------------|-------------------------|
| **Enable** | `use_probabilistic_embeddings=False` | `use_probabilistic_embeddings=True` |
| **Parameters** | vocab_size × n_embd + block_size × n_embd | 2 × (vocab_size × n_embd + block_size × n_embd) |
| **Loss** | Cross-entropy only | Cross-entropy + β×KL divergence |
| **Deterministic** | Yes | No (stochastic sampling) |
| **Weight Tying** | Supported | Not supported |
| **Training Speed** | Faster | Slightly slower |
| **Regularization** | Standard dropout | Dropout + KL divergence |

## Code Snippets

### Static (Default)
```python
config = GPTConfig(use_probabilistic_embeddings=False)
model = GPT(config)

# Training
logits, loss = model(input_ids, targets)  # loss = CE

# Inference
output = model.generate(prompt, max_new_tokens=100)
```

### Probabilistic
```python
config = GPTConfig(use_probabilistic_embeddings=True)
model = GPT(config)

# Training
logits, loss = model(input_ids, targets)  # loss = CE + β×KL

# Inference (stochastic)
output = model.generate(prompt, max_new_tokens=100)

# Inference (deterministic)
model.use_mean_embeddings()
output = model.generate(prompt, max_new_tokens=100)
```

## Key Points

1. **Probabilistic embeddings double the embedding parameters** (mean + variance)
2. **KL divergence adds regularization** by pulling distributions toward N(0, 1)
3. **β parameter (default 0.01)** controls regularization strength
4. **Use `model.use_mean_embeddings()`** for deterministic inference
5. **No weight tying** for probabilistic embeddings (separate token embeddings and output layer)

## When to Use Which?

**Use Static Embeddings when:**
- You want standard, deterministic behavior
- Parameter efficiency is important
- You're fine-tuning from pretrained weights
- Simplicity is preferred

**Use Probabilistic Embeddings when:**
- You want stronger regularization
- You're researching representation learning
- You're interested in embedding uncertainty
- You want to explore stochastic models
- Parameter count is not a constraint

## Tuning β

- **β = 0.0**: No regularization (like static embeddings)
- **β = 0.001-0.01**: Light regularization ← **Start here**
- **β = 0.01-0.1**: Medium regularization
- **β = 0.1-1.0**: Strong regularization (may hurt performance)

Edit in `model.py` line ~257:
```python
beta = 0.01  # Adjust this value
loss = ce_loss + beta * kl_loss
```
