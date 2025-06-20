import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from llmfoundry.models import MPTForCausalLM, MPTConfig


class MPTGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config: GPTConfig = config

        self.model = MPTForCausalLM(
            MPTConfig(
                n_heads=config.n_head,
                n_layers=config.n_layer,
                d_model=config.n_embd,
                max_seq_len=config.block_size,
                vocab_size=config.vocab_size,
                no_bias=not config.bias,
                attn_config={
                    "attn_impl": "torch",
                    # "attn_impl": "flash",
                    # "attn_type": "selective_multihead_attention",
                },
            )
        )

    def forward(self, idx, targets=None, output_attentions=False):
        output = self.model(idx, labels=targets, output_attentions=output_attentions)

        if output_attentions:
            return output.logits, output.loss, output.attentions

        return output.logits, output.loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        # flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        # flops_promised = 149.7e12  # A40 GPU bfloat16 peak flops is 149.7 TFLOPS
        flops_promised = 112e12  # V100 GPU float16 peak flops is 112 TFLOPS

        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.model.transformer.wpe.weight.numel()
        return n_params


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )
    # attention_layer: nn.Module = SplitCausalSelfAttention
    attention_layer: nn.Module = None
    output_attentions: bool = (
        False  # whether to return attention weights in the forward pass.
        # Note that this will change the attention implementation to eager instead of flash.
    )
