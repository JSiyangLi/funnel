import numpy as np
from tqdm import trange, tqdm
from scipy.stats import multivariate_normal, cauchy, norm
import matplotlib.pyplot as plt
from funnel.fi_core import fi_ln_evidence
import os
from bilby.core.prior import PriorDict, Normal

np.random.seed(0)

OUTDIR = 'out_simulation'
os.makedirs(OUTDIR, exist_ok=True)

def true_lnZ(p, v):
    return np.log(np.power((v / (1 + v)), (p / 2)))



def log_like(theta, v):
    return -np.sum(np.power(theta, 2) / (2 * v))

def log_prior(theta):
    return np.sum(norm.logpdf(theta, loc=0, scale=1))


def simulate_posterior_samples(p, v, nsamples, inflation=1):
    mean = np.zeros(p)
    std = np.sqrt(v / (v + 1))
    return np.array([np.random.normal(mean, std) for _ in range(nsamples)]).reshape(-1, p)


def fi_lnZ(p, v, r, nsamples, inflation=1):
    post_samples = simulate_posterior_samples(p, v, nsamples, inflation)
    assert post_samples.shape == (nsamples, p)
    # ref_sample = np.mean(post_samples, axis=0)
    ref_sample = np.zeros(p)
    return fi_ln_evidence(
        posterior_samples=post_samples,
        ref_samp=ref_sample,
        ref_lnpri=log_prior(ref_sample),
        ref_lnl=log_like(ref_sample, v),
        r=r,
    )


def plot_simulation_hist(fi_lnzs, p, v, n, n_simulation):
    plt.figure()
    label = f"FI LnZ ({np.mean(fi_lnzs):.2f} +- {np.std(fi_lnzs):.2f})"
    plt.hist(fi_lnzs, bins=30, density=True, label=label)
    tru_lnz = true_lnZ(p, v)
    plt.axvline(x=true_lnZ(p, v), color="r", label=f"True LnZ ({tru_lnz:.2f})")
    plt.xlabel("LnZ")
    plt.title(f"p={p}, v={v}, nsamp={n}, n_sim={n_simulation}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"hist_p={p}_v={v}_nsamp={n}.png"))


def plot_lnz_vs_r_curve(p, v, nsamples):
    rs = np.geomspace(1, 500, 500)
    fi_lnzs = np.array([fi_lnZ(p, v,  r=r, nsamples=nsamples) for r in tqdm(rs, desc=f"FI LnZ(R) p={p}, v={v}")])
    plt.figure()
    label = f"FI LnZ ({np.mean(fi_lnzs):.2f} +- {np.std(fi_lnzs):.2f})"
    plt.plot(rs, fi_lnzs, label=label)
    tru_lnz = true_lnZ(p, v)
    plt.axhline(y=tru_lnz, color="r", label=f"True LnZ ({tru_lnz:.2f})")
    plt.xlabel("r")
    plt.title(f"p={p}, v={v}, nsamp={nsamples}")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"lnz_vs_r_p={p}_v={v}_nsamp={nsamples}.png"))


def run_simulations(p, v=0.1, nsamples=int(1e4), n_simulation=300):
    # ref_samples = np.random.multivariate_normal(np.zeros(p), np.identity(p), size=n_simulation)
    simulation_results = np.array([
        fi_lnZ(p, v, r=400, nsamples=nsamples)
        for _ in trange(n_simulation, desc=f"p={p}, nsamp={nsamples}")
    ])
    plot_simulation_hist(simulation_results, p, v, nsamples, n_simulation)
    plot_lnz_vs_r_curve(p, v, nsamples)


if __name__ == '__main__':
    run_simulations(1)
    run_simulations(20)
    run_simulations(100)
