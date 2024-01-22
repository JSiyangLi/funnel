import h5py
import pandas as pd
from collections import namedtuple
import numpy as np


R_VALS = np.geomspace(1e1, 1e5, 100)
LVK_data = namedtuple("LVK_data", "posterior, lnz, lnz_err, lnBF")


def load_lvk_data(fpath):
    with h5py.File(fpath, "r") as f:
        sampling_params = list(f["C01:IMRPhenomXPHM/priors/analytic"].keys())
        # remove mass_1, mass_2 (as we already have chirp_mass, mass_ratio)
        sampling_params = [p for p in sampling_params if p not in ["mass_1", "mass_2"]]
        sampler_data = f["C01:IMRPhenomXPHM/meta_data/sampler"]

        lnz, lnz_err = (
            sampler_data["ln_evidence"][0],
            sampler_data["ln_evidence_error"][0],
        )
        lnBF = sampler_data["ln_bayes_factor"][0]
        post = f["C01:IMRPhenomXPHM"]["posterior_samples"][()]
        post = pd.DataFrame({name: post[name][:] for name in post.dtype.names})
        post = post[sampling_params + ["log_likelihood", "log_prior"]]
        post = post.loc[:, post.nunique() > 1]

    return LVK_data(post, lnz, lnz_err, lnBF)


def compute_fi_and_save(
    result, sampling_parms, cache_fn, num_ref_params=10, frac_post=0.1
):

    pass
