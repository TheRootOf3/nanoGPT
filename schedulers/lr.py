import math


def wsd_schedule(
    n_iterations,
    final_lr_factor=0.0,
    fract_warmup=0.1,
    init_div_factor=100,
    fract_decay=0.1,
    decay_type="linear",
):
    """Warmup, hold, and decay schedule.
    From https://github.com/epfml/schedules-and-scaling/blob/main/src/optim/utils.py

    Args:
        n_iterations (int): Total number of iterations for the training.
        final_lr_factor (float): Final learning rate factor at the end of training.
        fract_warmup (float): Fraction of iterations for warmup phase.
        init_div_factor (int): Initial division factor for warmup phase.
        fract_decay (float): Fraction of iterations for decay phase.
        decay_type (str): Type of decay to apply during the decay phase. Options are:
            'linear', 'exp', 'cosine', 'miror_cosine', 'square', 'sqrt'.

    Returns:
        callable: A function that takes the current iteration number and returns the current learning rate factor.
    """
    n_anneal_steps = int(fract_decay * n_iterations)
    n_hold = n_iterations - n_anneal_steps

    def schedule(step):
        """
        Calculate the current learning rate factor based on the iteration number.

        Args:
            step (int): Current iteration number.

        Returns:
            float: Current learning rate factor.
        """

        if step < n_iterations * fract_warmup:
            return (step / (n_iterations * fract_warmup)) + (
                1 - step / (n_iterations * fract_warmup)
            ) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < n_iterations:
            if decay_type == "linear":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
            elif decay_type == "exp":
                return final_lr_factor ** ((step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                return (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "miror_cosine":
                cosine_value = (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
                linear_value = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
                return linear_value * 2 - cosine_value
            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - ((step - n_hold) / n_anneal_steps) ** 2
                )

            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - math.sqrt((step - n_hold) / n_anneal_steps)
                )

            else:
                raise ValueError(
                    f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp']"
                )

        else:
            return final_lr_factor

    return schedule
