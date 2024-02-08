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


def lnl_const(p, v):
    return (-p / 2) * np.log(2 * np.pi * v)

def true_lnZ(p, v):
    return (p / 2) * np.log(v / (1 + v)) + lnl_const(p, v)

def log_like(theta, p, v):
    return -np.sum(np.power(theta, 2) / (2 * v)) + lnl_const(p, v)


def log_prior(theta):
    return np.sum(norm.logpdf(theta, loc=0, scale=1))


def simulate_posterior_samples(p, v, nsamples, inflation=1):
    mean = np.zeros(p)
    std = np.sqrt(v / (v + 1))
    return np.array([np.random.normal(mean, std) for _ in range(nsamples)]).reshape(-1, p)


def fi_lnZ(p, v, r, nsamples, inflation=1):
    post_samples = simulate_posterior_samples(p=p, v=v, nsamples=nsamples)
    assert post_samples.shape == (nsamples, p)
    # ref_sample = np.mean(post_samples, axis=0)

    ref_sample = np.zeros(p)
    return fi_ln_evidence(
        posterior_samples=post_samples,
        ref_samp=ref_sample,
        ref_lnpri=log_prior(ref_sample),
        ref_lnl=log_like(ref_sample,  p=p, v=v),
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


def plot_lnz_vs_r_curve(p, v, nsamples, c="C0"):
    rs = np.geomspace(10, 1000, 50)
    tru_lnz = true_lnZ(p, v)
    print(f"True LnZ: {tru_lnz:.2f}")
    pbar = tqdm(total=len(rs), desc=f"FI LnZ(R) p={p}, v={v}")
    fi_lnzs = np.zeros(len(rs))
    # post the median last 5 values of fi_lnZ
    for i, r in enumerate(rs):
        fi_lnz = fi_lnZ(p=p, v=v, r=r, nsamples=nsamples)
        pbar.set_postfix({"FI LnZ": f"{fi_lnz:.2f}"})
        pbar.update()
        fi_lnzs[i] = fi_lnz

    plt.axhline(y=tru_lnz, color=c, alpha=0.4, zorder=-1)
    plt.xlabel("R")
    plt.title(f"p={p}, v={v}, nsamp={nsamples}")
    # label = f"P={p:002d}, med err={np.median(tru_lnz-fi_lnzs[rs>100]):.2f}"
    plt.plot(rs, fi_lnzs, color=c)
    plt.xscale("log")

    med_err = np.nanmedian(tru_lnz - fi_lnzs[rs > 100])
    const = np.log(np.power(2*np.pi * v, -p/2))
    txt1 = f"Median Error: {med_err:.2f}"
    txt2 = f"Const: {const:.2f}"
    print(txt1)
    print(txt2)


    # add text to plot top left corner
    plt.text(0.05, 0.95, txt1, ha='left', va='top', transform=plt.gca().transAxes)
    plt.text(0.05, 0.90, txt2, ha='left', va='top', transform=plt.gca().transAxes)


    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTDIR, f"lnz_vs_r_p={p}_v={v}_nsamp={nsamples}.png"))


def run_simulations(p, v=1, nsamples=int(1e4), n_simulation=100, c="C0"):
    # ref_samples = np.random.multivariate_normal(np.zeros(p), np.identity(p), size=n_simulation)
    # simulation_results = np.array([
    #     fi_lnZ(p, v, r=400, nsamples=nsamples)
    #     for _ in trange(n_simulation, desc=f"p={p}, nsamp={nsamples}")
    # ])
    # plot_simulation_hist(simulation_results, p, v, nsamples, n_simulation)
    plot_lnz_vs_r_curve(p, v, nsamples, c)


def main():
    plt.figure(figsize=(10, 5))
    run_simulations(1, c="C0")
    run_simulations(5, c="C1")
    run_simulations(10, c="C2")
    run_simulations(25, c="C3")
    run_simulations(50, c="C4")
    run_simulations(100, c="C5")
    # add legend to outside righthand side of plot
    plt.ylabel(r"True LnZ - FI LnZ")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"lnz_vs_r.png"))




def main2():
    dims = np.geomspace(1, 100, 30).astype(int)
    rs = np.geomspace(500, 1000, 100)
    v = 1
    nsamples = int(1e4)

    errors = []
    uncs = []

    for d in tqdm(dims, desc='Dims'):
        fi_lnzs = np.array([fi_lnZ(d, v, r=r, nsamples=nsamples) for r in rs])
        tru_lnz = true_lnZ(d, v)
        error = tru_lnz - fi_lnzs
        uncs.append(np.std(error))
        errors.append(np.median(error))
    # save dim, error, unc
    np.savetxt('errors.csv', np.array([dims, errors, uncs]).T, delimiter=',', header='dim,error,unc', comments='')

    # load dim, error, unc
    data = np.loadtxt('errors.csv', delimiter=',', skiprows=1)
    dims = data[:, 0]
    errors = data[:, 1]
    uncs = data[:, 2]

    # plot dim vs error and then shade in uncertain region
    plt.figure()
    plt.fill_between(dims, np.array(errors) - np.array(uncs), np.array(errors) + np.array(uncs), alpha=0.3,
                     color="tab:orange")
    plt.plot(dims, errors, label='Median Error', color="tab:orange")
    plt.xlabel('Dimension')
    plt.ylabel('True LnZ - FI LnZ')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig("sim_error.png")

if __name__ == '__main__':
    p = 1
    v = 1
    run_simulations(p, v=v, c="C1")
    true_ = true_lnZ(p, v)
    plt.ylabel('Error')
    plt.savefig("sim_error_curve.png")


