from typing import Optional

import torch
import torch.nn.functional as F


def biased_hsic(gram1, gram2):
    """
    Computes the Hilbert-Schmidt Independence Criterion (HSIC) between two attention maps.
    This implementation is biased, meaning it does not center the Gram matrices.

    Args:
        gram1 (torch.Tensor): The Gram matrix of the first attention map.
        gram2 (torch.Tensor): The Gram matrix of the second attention map.

    Returns:
        torch.Tensor: The HSIC value between the two Gram matrices.
    """
    h = (
        torch.eye(gram1.shape[0])
        - torch.ones(gram1.shape[0], gram1.shape[0]) / gram1.shape[0]
    ).to(
        gram1.device
    )  # centering matrix

    # compute the HSIC value
    hsic_value = torch.trace(gram1 @ h @ gram2 @ h) / ((gram1.shape[0] - 1) ** 2)
    return hsic_value


def unbiased_hsic(gram1, gram2):
    """
    Computes the unbiased Hilbert-Schmidt Independence Criterion (HSIC) between two attention maps.
    This implementation centers the Gram matrices before computing the HSIC value.

    Args:
        gram1 (torch.Tensor): The Gram matrix of the first attention map.
        gram2 (torch.Tensor): The Gram matrix of the second attention map.
    Returns:
        torch.Tensor: The unbiased HSIC value between the two Gram matrices.
    """
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
    Computes the Centered Kernel Alignment (CKA) between two attention maps or lists of attention maps.
    CKA is a measure of similarity between two sets of representations, often used to compare attention heads.

    Args:
        attn1 (torch.Tensor | list[torch.Tensor]): The first attention map or a list of attention maps.
        attn2 (torch.Tensor | list[torch.Tensor]): The second attention map or a list of attention maps.
        hsic_fn (callable): The function to compute the Hilbert-Schmidt Independence Criterion (HSIC).
                            Defaults to unbiased_hsic.

    Returns:
        float: The CKA value between the two attention maps or lists of attention maps.
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


def cosine_similarity(attn1: torch.Tensor, attn2: torch.Tensor) -> float:
    """
    Computes the cosine similarity between two attention maps or lists of attention maps.

    Args:
        attn1 (torch.Tensor): The first attention map or a list of attention maps.
        attn2 (torch.Tensor): The second attention map or a list of attention maps.

    Returns:
        float: The cosine similarity value between the two attention maps or lists of attention maps.
    """
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

    Args:
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


def compute_pairwise_symmetric_similarity_matrix(
    attn_weights: torch.Tensor,
    metric_fn: callable,
) -> torch.Tensor:
    """
    Computes a pairwise similarity matrix for all head pairs in the attention weights.
    It assumes that the metric function is symmetric, meaning that the order of inputs does not matter.
    NOTE: The diagonal of the matrix is filled with ones, representing self-similarity.

    Args:
        attn_weights (torch.Tensor): Attention weights of shape (batch_size, n_heads, seq_len, seq_len).
        metric_fn (callable): Function to compute the pairwise metric. It must be symmetric.

    Returns:
        torch.Tensor: A symmetric matrix of shape (n_heads, n_heads) containing the pairwise metric values.
    """
    n_heads = attn_weights.shape[1]
    similarity_matrix = torch.zeros((n_heads, n_heads), dtype=torch.float32)

    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            metric_value = metric_fn(attn_weights[:, i], attn_weights[:, j])
            similarity_matrix[i, j] = metric_value
            similarity_matrix[j, i] = metric_value  # Symmetric matrix

    # Fill the diagonal with ones (self-similarity)
    for i in range(n_heads):
        similarity_matrix[i, i] = 1.0

    return similarity_matrix


def compute_aggr_pairwise_similarity(
    sim_matrix: torch.Tensor,
    aggregate_fn: callable,
) -> float:
    """
    Computes the aggregated pairwise similarity from a symmetric matrix of shape (n_heads, n_heads).

    Args:
        sim_matrix (torch.Tensor): A symmetric matrix of shape (n_heads, n_heads) containing pairwise similarity values.
        aggregate_fn (callable, optional): Function to aggregate the results across head pairs. If not provided, the mean of the upper triangular part is returned.

    Returns:
        float: The aggregated similarity value.
    """
    n = sim_matrix.shape[0]
    upper_tri_indices = torch.triu_indices(n, n, offset=1)
    upper_tri_values = sim_matrix[upper_tri_indices[0], upper_tri_indices[1]]

    return aggregate_fn(upper_tri_values).item()


def compute_mean_per_head_redundancy(
    sim_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Computes mean similarity values for a specific head across all other heads.

    Args:
        sim_matrix (torch.Tensor): A symmetric matrix of shape (n_heads, n_heads) containing pairwise similarity values.

    Returns:
        torch.Tensor: A tensor of shape (n_heads,) containing the similarity values for each head.
    """
    n = sim_matrix.shape[0]
    per_head_similarities = torch.zeros(n, dtype=torch.float32)
    for i in range(n):
        # subtract 1 to remove self-similarity, then divide by (n-1) to normalize
        per_head_similarities[i] = (sim_matrix[i, :].sum() - 1) / (n - 1)

    return per_head_similarities


def compute_max_per_head_redundancy(
    sim_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Computes max similarity values for a specific head across all other heads.

    Args:
        sim_matrix (torch.Tensor): A symmetric matrix of shape (n_heads, n_heads) containing pairwise similarity values.

    Returns:
        torch.Tensor: A tensor of shape (n_heads,) containing the similarity values for each head.
    """
    n = sim_matrix.shape[0]
    off_diagonal_sim_matrix = sim_matrix - torch.eye(
        n, dtype=torch.float32
    )  # zero out the diagonal
    per_head_similarities = torch.max(off_diagonal_sim_matrix, dim=1).values

    return per_head_similarities
