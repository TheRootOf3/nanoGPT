from typing import Optional

import torch
import torch.nn.functional as F


def biased_hsic(gram1, gram2):
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC) between two attention maps.
    """

    h = (
        torch.eye(gram1.shape[0])
        - torch.ones(gram1.shape[0], gram1.shape[0]) / gram1.shape[0]
    )  # centering matrix

    # compute the HSIC value
    hsic_value = torch.trace(gram1 @ h @ gram2 @ h) / ((gram1.shape[0] - 1) ** 2)
    return hsic_value


def unbiased_hsic(gram1, gram2):
    gram1_copy = gram1.clone()
    gram2_copy = gram2.clone()

    gram1_copy.fill_diagonal_(0)  # zero out the diagonal
    gram2_copy.fill_diagonal_(0)  # zero out the diagonal
    n = gram1_copy.shape[0]

    first_term = torch.trace(gram1_copy @ gram2_copy)
    second_term = (gram1_copy.sum() * gram2_copy.sum()) / ((n - 1) * (n - 2))
    third_term = (gram1_copy @ gram2_copy).sum() * 2 / (n - 2)

    return (first_term + second_term - third_term) / (n * (n - 3))


def cka(
    attn1: torch.Tensor | list[torch.Tensor],
    attn2: torch.Tensor | list[torch.Tensor],
    hsic_fn: callable = unbiased_hsic,
):
    """
    Computes the Centered Kernel Alignment (CKA) between two attention maps.
    """
    if len(attn1.shape) == 2:
        attn1 = [attn1]
        attn2 = [attn2]
    else:
        assert len(attn1) == len(
            attn2
        ), "Both attention map lists/tensors must have the same length."

    hsic_kl = hsic_kk = hsic_ll = 0
    for a1, a2 in zip(attn1, attn2):
        gram1 = a1 @ a1.T  # compute the Gram matrix
        gram2 = a2 @ a2.T  # compute the Gram matrix

        hsic_kl += hsic_fn(gram1, gram2) / len(attn1)
        hsic_kk += hsic_fn(gram1, gram1) / len(attn1)
        hsic_ll += hsic_fn(gram2, gram2) / len(attn1)

    return (hsic_kl / torch.sqrt(hsic_kk * hsic_ll)).item()


def cosine_similarity(attn1, attn2):
    attn1 = attn1.reshape(attn1.shape[0], -1)
    attn2 = attn2.reshape(attn2.shape[0], -1)

    return F.cosine_similarity(attn1, attn2).mean().item()


def aggregate_head_pairwise_metric_batch(
    attn_weights: torch.Tensor,
    metric_fn: callable,
    aggregate_fn: Optional[callable] = None,
) -> float:
    """
    Computes a pairwise metric (e.g., CKA, HSIC) for all head pairs in the attention weights.

    Parameters:
        attn_weights (torch.Tensor): Attention weights of shape (batch_size, n_heads, seq_len, seq_len).
        metric_fn (callable): Function to compute the pairwise metric.
        aggregate_fn (callable, optional): Function to aggregate the results across head pairs. If not provided, return the metric results.

    Returns:
        float: The aggregated metric value.
    """
    n_heads = attn_weights.shape[1]
    metrics = []

    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            metric_value = metric_fn(attn_weights[:, i], attn_weights[:, j])
            metrics.append(metric_value)

    if aggregate_fn is None:
        return metrics
    else:
        return aggregate_fn(torch.tensor(metrics)).item()
