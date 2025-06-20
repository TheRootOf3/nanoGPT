from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


def get_attn_map(
    attn_map: np.ndarray,
    token_labels: Optional[list[str]] = None,
    log_scale: bool = False,
) -> plt.Figure:
    """
    Visualizes an attention map with token labels.

    Parameters:
        attn_map (np.ndarray): The attention map to visualize. It is expected to be a 2D array.
        token_labels (list[str]): A list of token labels corresponding to the attention map.
        log_scale (bool, optional): If True, applies a logarithmic scale to the attention map values. Defaults to False.

    Returns:
        plt.Figure: A matplotlib figure containing the attention map visualization.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # mask for plotting the bad color for padding tokens (upper triangular part)
    attn_map = np.ma.masked_where(attn_map == 0, attn_map)

    cmap = plt.cm.coolwarm
    cmap.set_bad(color="grey")
    ax.matshow(np.log(attn_map) if log_scale else attn_map, cmap=cmap)

    ax.xaxis.set_label_position("bottom")  # Move x-axis label to the bottom
    ax.xaxis.tick_bottom()  # Ensure ticks are on the bottom
    if token_labels is not None:
        _ = ax.set_xticks(
            range(len(token_labels)),
            token_labels,
            fontsize=9,
            rotation=90,
        )
        _ = ax.set_yticks(
            range(len(token_labels)),
            token_labels,
            fontsize=9,
        )
    ax.set_ylabel("Query token", fontsize=12)
    ax.set_xlabel("Key token", fontsize=12)
    return fig
