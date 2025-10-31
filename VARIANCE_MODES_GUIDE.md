# Variance Modes for Probabilistic Embeddings

## Overview

When using probabilistic embeddings (`use_probabilistic_embeddings=True`), you can choose between three variance parameterization modes via the `variance_mode` parameter.

## Variance Modes

### 1. SCALAR MODE (Recommended) ⭐
```python
config = GPTConfig(
    use_probabilistic_embeddings=True,
    variance_mode='scalar'
)
```

**How it works:**
- Each embedding learns ONE scalar variance value
- This scalar is broadcast across all embedding dimensions
- Example: Token 42 has variance σ² that applies to all 384 dimensions

**Parameters:**
- Variance params: `vocab_size` scalars
- For vocab_size=50,304: **50,304 extra parameters**

**Use when:**
- You want minimal parameter overhead
- Each token should have its own uncertainty level
- You don't need per-dimension variance control

**Pros:**
- ✅ Minimal overhead (~0.1% of embedding params)
- ✅ Each embedding has independent uncertainty
- ✅ Simple and interpretable

**Cons:**
- ❌ All dimensions of an embedding share the same variance

---

### 2. SHARED MODE (Most Efficient) 🏆
```python
config = GPTConfig(
    use_probabilistic_embeddings=True,
    variance_mode='shared'
)
```

**How it works:**
- ALL embeddings share ONE variance vector
- Different dimensions can have different variances
- Example: Dimension 0 has σ₀², dimension 1 has σ₁², etc., but all tokens share these

**Parameters:**
- Variance params: `embedding_dim` values
- For embedding_dim=384: **384 extra parameters**

**Use when:**
- Parameter efficiency is critical
- You believe all embeddings have similar uncertainty patterns
- You want variance to vary by dimension, not by token

**Pros:**
- ✅ Absolute minimum overhead (~0.002% of embedding params)
- ✅ Dimensions can have different variances
- ✅ Extremely parameter-efficient

**Cons:**
- ❌ All embeddings must share the same variance pattern
- ❌ Cannot model per-token uncertainty differences

---

### 3. FULL MODE (Maximum Flexibility)
```python
config = GPTConfig(
    use_probabilistic_embeddings=True,
    variance_mode='full'
)
```

**How it works:**
- Each embedding learns a FULL variance vector
- Complete independence: each dimension of each embedding has its own variance
- Example: Token 42, dimension 0 has its own σ²₄₂,₀

**Parameters:**
- Variance params: `vocab_size × embedding_dim`
- For vocab_size=50,304, embedding_dim=384: **19,316,736 extra parameters**

**Use when:**
- You need maximum flexibility
- Parameter count is not a concern
- Research requires full per-dimension variance control

**Pros:**
- ✅ Complete flexibility
- ✅ Each dimension of each embedding independent
- ✅ Can model complex variance patterns

**Cons:**
- ❌ Doubles embedding parameters (100% overhead)
- ❌ Highest computational cost
- ❌ May be overkill for most applications

---

## Comparison Table

| Mode | Variance Params | Overhead | Total Model Size* | Flexibility |
|------|----------------|----------|-------------------|-------------|
| scalar | vocab_size | 50K | 30.0M (1.00x) | Medium |
| shared | embedding_dim | 384 | 29.9M (1.00x) | Low |
| full | vocab_size × emb_dim | 19.3M | 49.3M (1.65x) | High |

*Based on vocab_size=50,304, embedding_dim=384, 6-layer model

**Note:** Token embedding means (mu) are tied with lm_head, so only variance parameters add overhead!

## Parameter Breakdown Example

For a model with:
- `vocab_size = 50,304`
- `embedding_dim = 384`
- `n_layer = 6`

### Static Embeddings (Baseline)
- Total: 29.9M parameters
- Token embeddings: 19.3M (tied with lm_head)

### Probabilistic Embeddings (with Weight Tying)
| Mode | Mean Params | Var Params | Total | vs Static |
|------|-------------|------------|-------|-----------|
| scalar | 19.3M (tied) | 50K | 30.0M | +1.00x |
| shared | 19.3M (tied) | 384 | 29.9M | +1.00x |
| full | 19.3M (tied) | 19.3M | 49.3M | +1.65x |

**Key insight:** Token embedding means (mu) are tied with lm_head, so only variance parameters add overhead!

## Usage Example

```python
from model import GPT, GPTConfig

# Choose your variance mode
config = GPTConfig(
    block_size=1024,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    use_probabilistic_embeddings=True,
    variance_mode='scalar'  # or 'shared' or 'full'
)

model = GPT(config)

# Training is identical regardless of mode
logits, loss = model(input_ids, targets)
loss.backward()
```

## Recommendations

### Choose SCALAR if:
- ✅ You want a good balance
- ✅ Nearly zero parameter overhead (~50K params)
- ✅ You want per-token uncertainty modeling
- ✅ **This is the recommended default**

### Choose SHARED if:
- ✅ Parameter efficiency is absolutely critical
- ✅ You believe uncertainty is dimension-specific, not token-specific
- ✅ You have a very large vocabulary
- ✅ You want the absolute minimum overhead (~384 params)

### Choose FULL if:
- ✅ You're doing research on embedding uncertainty
- ✅ You need maximum modeling flexibility
- ✅ You want per-dimension variance control
- ✅ ~1.65x model size is acceptable

## Implementation Details

### Scalar Mode
```python
# Shape: (num_embeddings, 1)
logvar = self.logvar(idx)  # Returns (batch, seq, 1)
logvar = logvar.expand_as(mu)  # Broadcast to (batch, seq, emb_dim)
```

### Shared Mode
```python
# Shape: (embedding_dim,)
logvar = self.logvar.expand_as(mu)  # Broadcast to match mu shape
```

### Full Mode
```python
# Shape: (num_embeddings, embedding_dim)
logvar = self.logvar(idx)  # Returns (batch, seq, emb_dim)
```

## See Also

- Run `python3 example_variance_modes.py` for detailed comparison
- See `PROBABILISTIC_EMBEDDINGS_README.md` for general probabilistic embeddings documentation
- See `EMBEDDINGS_QUICK_REFERENCE.md` for static vs probabilistic comparison
