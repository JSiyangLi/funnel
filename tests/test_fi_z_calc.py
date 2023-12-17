"""All tests for the funnel package."""
import os
import shutil

import bilby
import matplotlib.pyplot as plt
import numpy as np
import pytest

from funnel.plotting import plot_fi_evidence_results

CLEAN = False

np.random.seed(42)

NLIVE = 200


@pytest.fixture()
def tmp_path():
    """Return the path to the temporary directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


@pytest.fixture()
def bilby_result(tmp_path):
    """Create a bilby result object with a Gaussian likelihood."""

    def model(time, m, c):
        return time * m + c

    outdir, label = tmp_path, "line"
    if os.path.exists(outdir) and CLEAN:
        shutil.rmtree(outdir)

    res_fn = os.path.join(outdir, f"{label}_result.json")
    if os.path.exists(res_fn):
        return bilby.result.read_in_result(res_fn)

    os.makedirs(outdir, exist_ok=True)

    # Now we define the injection parameters which we make simulated data with
    injection_parameters = dict(m=0.5, c=0.2)

    sampling_frequency = 10
    time_duration = 10
    time = np.arange(0, time_duration, 1 / sampling_frequency)
    n = len(time)
    sigma = np.random.normal(1, 0.01, n)
    data = model(time, **injection_parameters) + np.random.normal(0, sigma, n)

    likelihood = bilby.likelihood.GaussianLikelihood(time, data, model, sigma)

    priors = bilby.core.prior.PriorDict(
        dict(
            m=bilby.core.prior.Uniform(0, 5, "m"),
            c=bilby.core.prior.Uniform(-2, 2, "c"),
        )
    )

    # We quickly plot the data to check it looks sensible
    if not os.path.exists(outdir):
        fig, ax = plt.subplots()
        ax.plot(time, data, "o", label="data")
        ax.plot(time, model(time, **injection_parameters), "--r", label="signal")
        ax.set_xlabel("time")
        ax.set_ylabel("y")
        ax.legend()
        plt.savefig(f"{outdir}/data.png")

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=2000,
        outdir=outdir,
        label=label,
        injection_parameters=injection_parameters,
    )
    corner_fig = f"{outdir}/corner.png"
    if not os.path.exists(corner_fig):
        fig = result.plot_corner()
        fig.suptitle(
            "Nested Sampling LnZ = "
            f"{result.log_evidence:.2f}+/-{result.log_evidence_err:.2f}",
            y=1.1,
        )
        fig.savefig(f"{outdir}/corner.png")

    return result



def test_fi_integration_plot(bilby_result, tmp_path):
    """Test the fi_integration_plot function."""
    lnz, lnzerr = bilby_result.log_evidence, bilby_result.log_evidence_err
    # new_lnz = get_fi_lnz_no_r(
    #     posterior_samples=bilby_result.posterior,
    #     ref_samp=bilby_result.posterior.median(),
    # )
    label = "test"
    fig = plot_fi_evidence_results(
        posterior_samples=bilby_result.posterior,
        sampling_lnz=[lnz - lnzerr, lnz + lnzerr],
        num_ref_params=10,
        r_vals=np.geomspace(10, 1e5, 3),
    )
    fig.suptitle(label)
    fig.savefig(f"{tmp_path}/fi_evidence.png")
