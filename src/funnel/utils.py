"""Utility functions for the funnel package."""
import time

import numpy as np

from .logger import logger

LNPRI_LNL_COL_NAMES = ["log_prior", "log_likelihood"]


# make a decorator to time functions
def timefunct(func):
    """Decorator to time functions.

    Args:
        func (function): Function to time.

    Returns:
        function: Decorated function.
    """

    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        if runtime > 0.01:
            logger.debug(f"Runtime {func.__name__}: " f"{end - start:.2f}s")
        return result

    return timed


@timefunct
def get_post_mask(post: np.ndarray, idx: int, tol=1e-5) -> np.array:
    """Get the posterior mask after excluding the idx-th row.

    Args:
        post (np.ndarray): Posterior samples.
        idx (int): Index of the reference sample.
        tol (float, optional): Tolerance for the distance between samples. Defaults to 1e-5.

    Returns:
        np.array: Boolean mask for the posterior samples.
    """
    ref_samp = post[idx]
    distances = np.linalg.norm(post - ref_samp, axis=1)
    mask = distances > tol
    num_close = sum(~mask)

    if num_close > 3:
        logger.warning(
            f"Found {num_close} posterior samples " f"close to ref_samp (tol={tol})."
        )
    return mask
