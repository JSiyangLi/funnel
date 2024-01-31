from bilby.gw.result import CBCResult
from funnel.fi_core import get_fi_lnz_list
from funnel.plotting import plot_fi_evidence_results
import matplotlib.pyplot as plt
import numpy as np

import h5py

if __name__ == '__main__':
    mcmc_res = CBCResult.from_hdf5("gw150914_ss.hdf5")
    ns_res = CBCResult.from_hdf5("gw150914_ns.hdf5")
    print(f"MCMC: {mcmc_res.log_evidence:.3f} +/- {mcmc_res.log_evidence_err:.3}")
    print(f"NS: {ns_res.log_evidence:.3f} +/- {ns_res.log_evidence_err:.3}")

    keys = ns_res.search_parameter_keys + ['log_prior', 'log_likelihood']

    posterior_samples = mcmc_res.posterior[keys]

    fi_results = get_fi_lnz_list(posterior_samples, r_vals=np.geomspace(1e0, 1e9, 1000), num_ref_params=1,
                                 weight_samples_by_lnl=True)

    lnzs, r_vals, samp = fi_results
    lnz_noise = ns_res.log_noise_evidence
    fi_lnz = lnzs[0] + lnz_noise
    ns_lnz = ns_res.log_bayes_factor + lnz_noise
    ns_lnz_err = ns_res.log_evidence_err
    ss_lnz = mcmc_res.log_bayes_factor + lnz_noise
    ss_lnz_err = mcmc_res.log_evidence_err

    data = dict(
        fi_lnz=fi_lnz,
        ns_lnz=ns_lnz,
        ss_lnz=ss_lnz,
        ns_lnz_err=ns_lnz_err,
        ss_lnz_err=ss_lnz_err,
        r_vals=r_vals,
    )

    # write data to HDF5 file
    with h5py.File("gw150914_fi.hdf5", "w") as f:
        for k, v in data.items():
            f.create_dataset(k, data=v)
