"""Core functions for Fourier Integration evidence approximation."""
import os
import sys

import numpy as np
import pandas as pd
from tqdm.auto import trange
from typing import Tuple


from .logger import logger
from .utils import get_post_mask
import numba


@numba.jit(parallel=True)
def fi_ln_evidence(
    posterior_samples: np.ndarray,
    ref_samp: np.array,
    r: float,
    ref_lnpri: float,
    ref_lnl: float,
):
    """
    Returns the approx log-evidence of some posterior samples (using a reference parameter value).
    The approximation is based on the 'density estimation' method described in
    [Rotiroti et al., 2018](https://link.springer.com/article/10.1007/s11222-022-10131-0).

    :param posterior_samples:np.ndarray: Array of posterior samples [n_samples, n_dim]
    :param ref_samp:np.array: A reference parameter value [n_dim] (Not present in the posterior)
    :param r:float: A scaling factor
    :param ref_lnpri:float: The log of the reference prior
    :param ref_lnl:float: The log of the reference likelihood
    :return: The log of the approximated log-evidence
    """
    # approximating the normalised posterior probability at reference sample
    diff_from_ref = posterior_samples - ref_samp
    sin_diff = np.sin(r * diff_from_ref)
    integrand = sin_diff / diff_from_ref
    prod_res = np.nanprod(integrand, axis=1)
    sum_res = np.abs(np.nansum(prod_res))
    n_samp, n_dim = posterior_samples.shape
    const = 1 / (n_samp * np.power(np.pi, n_dim))
    approx_ln_post = np.log(sum_res * const)
    # using bayes theorem to get the approximated log-evidence
    return ref_lnpri + ref_lnl - approx_ln_post


def get_fi_lnz_list(
    posterior_samples: pd.DataFrame,
    r_vals: np.array = [],
    num_ref_proportion: float = 0.05,
    weight_samples_by_lnl: bool = False,
    select_rightmost_cluster: bool = False,
    cache_fn="",
) -> Tuple[np.array, np.array, pd.DataFrame]:
    if os.path.exists(cache_fn):
        data = np.load(cache_fn)
        return data["lnzs"], data["r_vals"], data["samp"]

    num_ref_params = np.ceil(num_ref_proportion * len(posterior_samples))

    if len(r_vals) == 0:
        r_vals = np.geomspace(1e-3, 1e10, 2000)

    if num_ref_params > len(posterior_samples):
        num_ref_params = len(posterior_samples)

    # unpacking posterior data
    ln_pri = posterior_samples["log_prior"].values
    ln_lnl = posterior_samples["log_likelihood"].values
    ln_ker = np.array(ln_pri) + np.array(ln_lnl) # posterior kernel value
    post = posterior_samples[
        posterior_samples.columns.drop(["log_prior", "log_likelihood"])
    ].values

    logger.info(
        f"Calculating FI LnZ with {num_ref_params} reference points "
        f"and a posterior of size:{post.shape}"
    )
    param_str = "\n".join(sorted(posterior_samples.columns.values))
    logger.info(f"Posterior columns:\n{param_str}")

    # randomly select reference points
    ref_idx = np.random.choice(len(post), num_ref_params, replace=False)
    if weight_samples_by_lnl:
        p = np.exp(ln_ker - np.nanmax(ln_ker))
        p /= np.nansum(p)
        # ref_idx = np.random.choice(len(post), num_ref_params, replace=False, p=p)
        # get the reference points with the highest posterior kernel values - we are converting the kernel to a density, not just the likelihood
        ref_idx = np.argsort(ln_ker)[-num_ref_params:]

    lnzs = np.zeros((num_ref_params, len(r_vals)))
    median_lnzs = np.zeros(num_ref_params)
    med_ = 0

    with trange(num_ref_params, desc="FI LnZ", postfix=f"FI LnZ: {med_}") as pbar:
        for i in pbar:
            refi = ref_idx[i]
            med_ = np.nanmedian(median_lnzs[:i]) if i > 0 else 0

            post_mask = get_post_mask(post, refi)
            fi_kwargs = dict(
                posterior_samples=post[post_mask],
                ref_samp=post[refi],
                ref_lnpri=ln_pri[refi],
                ref_lnl=ln_lnl[refi],
            )
            lnzs[i] = np.array([fi_ln_evidence(**fi_kwargs, r=ri) for ri in r_vals])
            median_lnzs[i] = np.nanmedian(lnzs[i])

            pbar.set_postfix_str(f"FI LnZ: {med_:.2f}")
            pbar.update()

    # find the left bound of the rightmost cluster
    left_bound = np.sort(ln_ker)[::-1][num_ref_params - 1]

    rightmost_var, pca_rotation = np.zeros(len(r_vals)), np.zeros(len(r_vals))
    for ri in r_vals:
        # the specific evidence sample for each R[i]
        lnzri = lnzs[:, ri]

        # estimate the variance of the rightmost cluster in the evidence vs. kernel plot
        rightmost_lnz = lnzri[np.argmax(ln_ker)]
        rightmost_var[ri] = np.var(lnzri[ln_ker >= left_bound])

        # estimate the angular bias
        pca_data = np.column_stack((ln_ker, lnzri))
        pca = PCA().fit(pca_data)
        pca_rotation[ri] = min(np.pi - np.arccos(pca.components_[0, 0]), np.abs(np.arccos(pca.components_[0, 0]) - 0.5 * np.pi), np.arccos(pca.components_[0, 0]))

    samp = post[ref_idx] # the selected reference points
    if cache_fn:
        np.savez(cache_fn, lnzs=lnzs, r_vals=r_vals, samp=samp)

    return lnzs, r_vals, samp, rightmost_var, pca_rotation, rightmost_lnz
