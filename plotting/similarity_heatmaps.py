from matplotlib import pyplot as plt
import numpy as np
import torch
import wandb


def get_similarity_heatmap(
    S: torch.Tensor,
    step: int,
    name: str,
    out_dir: str,
    plot_title: str,
) -> plt.Figure:
    """
    Generates a heatmap for the similarity matrix S.

    Args:
        S (torch.Tensor): Similarity matrix of shape (n_heads, n_heads).
        step (int): Current training step.
        name (str): Name for the plot, used in saving the file.
        out_dir (str): Output directory where the plot will be saved.
        plot_title (str): Title for the plot.

    Returns:
        plt.Figure: The generated heatmap figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    im = ax.imshow(S, vmin=0, vmax=1, cmap="viridis")
    ax.set_title(plot_title)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Head index")

    # Add ticks between 0 and S.shape[0], centered
    tick_positions = np.arange(S.shape[0])
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_positions)
    ax.set_yticklabels(tick_positions)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Add value annotations
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            value = S[i, j].item()
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if value < 0.5 else "black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    plt.savefig(
        f"{out_dir}/plots/{name.replace('/', '_')}_step_{step}.png",
        bbox_inches="tight",
        dpi=300,
    )
    return fig
    # Log the figure


def get_per_head_model_heatmap(
    per_layer_head_similarities: list[torch.Tensor],
    step: int,
    name: str,
    out_dir: str,
    plot_title: str,
) -> plt.Figure:
    """
    Generates a full model head redundancy heatmap.

    Args:
        per_layer_head_similarities (list[torch.Tensor]): List of tensors, each representing head similarities for a layer.
        step (int): Current training step.
        name (str): Name for the plot, used in saving the file.
        out_dir (str): Output directory where the plot will be saved.
        plot_title (str): Title for the plot.

    Returns:
        plt.Figure: The generated heatmap figure.
    """
    max_n_heads = max([len(s) for s in per_layer_head_similarities])
    # Ensure all tensors have the same number of heads by right padding with zeros
    per_layer_head_similarities = [
        torch.cat(
            [
                s,
                torch.zeros(max_n_heads - len(s)),
            ]
        )
        for s in per_layer_head_similarities
    ]

    S = torch.vstack(per_layer_head_similarities[::-1])

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    im = ax.imshow(S, vmin=0, vmax=1, cmap="viridis")
    ax.set_title(plot_title, fontsize=18)
    ax.set_xlabel("Head index", fontsize=14)
    ax.set_ylabel("Layer index", fontsize=14)

    # Set y-ticks to reflect the correct layer indices (reversed)
    num_layers = S.shape[0]
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(num_layers - 1, -1, -1))  # Reversed layer indices

    # Set x-ticks to reflect the correct head indices
    num_heads = S.shape[1]
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels(range(num_heads))
    ax.tick_params(axis="both", which="major", labelsize=14)
    # Add value annotations
    for i in range(S.shape[0]):  # iterate over layers
        for j in range(S.shape[1]):
            value = S[i, j].item()
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=14,
                color="white" if value < 0.5 else "black",
            )

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)

    plt.savefig(
        f"{out_dir}/plots/{name.replace('/', '_')}_step_{step}.png",
        bbox_inches="tight",
        dpi=300,
    )

    return fig
