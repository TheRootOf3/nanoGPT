import math


def head_max_similarity_schedule(
    max_iters: int, initial_threshold: float = 0.3, final_threshold: float = 0.9
) -> callable:
    """
    Inverse cosine schedule for the maximum similarity threshold between attention heads.
    This schedule is used to control the maximum allowed similarity between attention heads during training.
    The threshold starts at `initial_threshold` and gradually increases to `final_threshold`
    as training progresses, following a cosine decay pattern.

    Args:
        max_iters (int): Total number of iterations for the training.
        initial_threshold (float): Initial maximum similarity threshold at the start of training.
        final_threshold (float): Final maximum similarity threshold at the end of training.

    Returns:
        callable: A function that takes the current iteration number and returns the current maximum similarity threshold.
    """

    def schedule(iter_num: int):
        """
        Calculate the current maximum similarity threshold based on the iteration number.

        Args:
            iter_num (int): Current iteration number.

        Returns:
            float: Current maximum similarity threshold.
        """
        # Max similarity threshold means that we will not add any new heads until there exists pair of heads
        # with similarity value lower than the threshold. This promotes the model to learn different heads and
        # gradually allows for more similarity as the training progresses.

        t = iter_num / max_iters
        return (
            initial_threshold
            + (final_threshold - initial_threshold) * (1 - math.cos(math.pi * t)) / 2
        )

    return schedule
