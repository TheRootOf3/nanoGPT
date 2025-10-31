# Training with Probabilistic Embeddings

This guide explains how to use the training script with probabilistic embeddings.

## New Configuration Parameters

The training script now supports two additional parameters:

### `use_probabilistic_embeddings`
- **Type:** boolean
- **Default:** `False`
- **Description:** Enable VAE-style probabilistic embeddings for token embeddings
- **Usage:** Set to `True` to enable probabilistic embeddings

### `variance_mode`
- **Type:** string
- **Default:** `'scalar'`
- **Options:** `'scalar'`, `'shared'`, `'full'`
- **Description:** Controls how variance is parameterized for probabilistic embeddings
  - `'scalar'`: One scalar variance per embedding (minimal overhead ~50K params)
  - `'shared'`: One variance vector shared by all embeddings (smallest overhead ~384 params)
  - `'full'`: One variance vector per embedding (maximum flexibility, ~19M extra params)

## Usage Examples

### Command Line

Train with default (static) embeddings:
```bash
python train.py
```

Train with probabilistic embeddings (scalar mode):
```bash
python train.py --use_probabilistic_embeddings=True
```

Train with probabilistic embeddings (shared mode for maximum efficiency):
```bash
python train.py --use_probabilistic_embeddings=True --variance_mode=shared
```

Train with probabilistic embeddings (full mode for maximum flexibility):
```bash
python train.py --use_probabilistic_embeddings=True --variance_mode=full
```

### Configuration File

Create a config file (e.g., `config/train_probabilistic.py`):

```python
# Model settings
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
dropout = 0.0
bias = False

# Enable probabilistic embeddings
use_probabilistic_embeddings = True
variance_mode = 'scalar'  # or 'shared' or 'full'

# Other training settings...
batch_size = 12
max_iters = 600000
learning_rate = 6e-4
```

Then run:
```bash
python train.py config/train_probabilistic.py
```

## Checkpoint Compatibility

### Creating Checkpoints
Checkpoints created with probabilistic embeddings will store the `use_probabilistic_embeddings` and `variance_mode` settings in `model_args`.

### Resuming Training
When resuming from a checkpoint (`--init_from=resume`), the script automatically:
- Restores the `use_probabilistic_embeddings` setting from the checkpoint
- Restores the `variance_mode` setting from the checkpoint
- Ensures model architecture matches the checkpoint

**Important:** You cannot resume training with different embedding settings than the checkpoint was created with.

### Fine-tuning from GPT-2
When initializing from GPT-2 weights (`--init_from=gpt2*`), probabilistic embeddings are **not** used by default (GPT-2 uses static embeddings). The new parameters will use their default values.

## Parameter Overhead

For a GPT-2 sized model (vocab_size=50304, n_embd=768):

| Mode | Additional Parameters | Total Model Size | vs Static |
|------|----------------------|------------------|-----------|
| static (baseline) | 0 | 124M | 1.00x |
| scalar | ~50K | 124M | ~1.00x |
| shared | ~768 | 124M | ~1.00x |
| full | ~38M | 162M | ~1.31x |

**Note:** Token embedding means (mu) are tied with lm_head, so only variance parameters add overhead.

## Training Considerations

### Loss Function
When using probabilistic embeddings, the loss includes:
- Cross-entropy loss (standard language modeling objective)
- KL divergence loss (regularizes embeddings toward N(0,1))
- Total loss = CE + β × KL (β = 0.01 by default)

### Training Speed
- **Scalar mode**: Minimal impact (~1-2% slower)
- **Shared mode**: Negligible impact
- **Full mode**: Slightly slower (~5-10%) due to larger embedding tables

### Recommendations

**For most use cases:**
```bash
python train.py --use_probabilistic_embeddings=True --variance_mode=scalar
```
- Minimal overhead (~50K params)
- Good uncertainty modeling
- Recommended default

**For maximum parameter efficiency:**
```bash
python train.py --use_probabilistic_embeddings=True --variance_mode=shared
```
- Absolute minimum overhead (~768 params)
- Best for very large vocabularies
- Shared uncertainty pattern

**For research/maximum flexibility:**
```bash
python train.py --use_probabilistic_embeddings=True --variance_mode=full
```
- Maximum modeling flexibility
- Per-dimension variance control
- ~31% more parameters

## Example: Small Test Run

Quick test with probabilistic embeddings:
```bash
python train.py \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=128 \
    --block_size=256 \
    --use_probabilistic_embeddings=True \
    --variance_mode=scalar \
    --batch_size=4 \
    --max_iters=100 \
    --compile=False
```

Or use the provided test config:
```bash
python train.py config/train_probabilistic_test.py
```

## Monitoring Training

The training script logs are the same whether using static or probabilistic embeddings. The loss values are comparable, though probabilistic embeddings may show slightly higher initial loss due to the KL term.

Use wandb logging to track both components:
```bash
python train.py --wandb_log=True --use_probabilistic_embeddings=True
```

## Troubleshooting

**Q: Can I change variance_mode when resuming training?**
A: No, the variance_mode is part of the model architecture and must match the checkpoint.

**Q: Why is my model size different than expected?**
A: Make sure you're using `model.get_num_params()` which excludes position embeddings. The total parameter count includes position embeddings.

**Q: Does this work with DDP/distributed training?**
A: Yes, probabilistic embeddings work seamlessly with DDP and distributed training.

**Q: How do I verify which mode my checkpoint uses?**
```python
import torch
ckpt = torch.load('out/ckpt.pt')
print(ckpt['model_args'].get('use_probabilistic_embeddings', False))
print(ckpt['model_args'].get('variance_mode', 'scalar'))
```

## See Also

- `example_embeddings.py` - Demo of static vs probabilistic embeddings
- `example_variance_modes.py` - Detailed comparison of variance modes
- `VARIANCE_MODES_GUIDE.md` - Complete reference for variance modes
- `PROBABILISTIC_EMBEDDINGS_README.md` - Technical details on probabilistic embeddings
