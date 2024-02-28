import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from tqdm import tqdm



import corner

# def lnl_const(p, v):
#     return (p / 2) * np.log(2 * np.pi * v)
#
# def lnl_const2(p, v):
#     return (p/2 ) * np.log(4 * np.pi * v * p)
#
#
# ps = np.geomspace(1, 100, 100)
#
# # Load the data
# data = pd.read_csv("simulation_study.csv")
# # Plot the error vs dimension
# plt.scatter(data['dim'], data['error'], label='Median error')
# plt.loglog(ps, lnl_const(ps, v=1), label='constant')
# # plt.loglog(ps, lnl_const2(ps, v=1), label='hack', color='r')
# plt.xlabel('Dimension')
# plt.ylabel('True LnZ - FI LnZ')
# plt.legend()
# plt.show()



def lnl_const(p, v):
    return -(p / 2) * np.log(2 * np.pi * v)

def true_lnZ(p, v):
    return (p / 2) * np.log(v / (1 + v)) + lnl_const(p, v)

def log_like(theta, p, v):
    return -np.sum(np.power(theta, 2) / (2 * v)) + lnl_const(p, v)


def log_prior(theta):
    return np.sum(norm.logpdf(theta, loc=0, scale=1))


def simulate_posterior_samples(p, v, nsamples, inflation=1):
    std = np.sqrt(v / (v + 1))
    post_pdf = norm(0, std)
    # sample from the posterior 'NSAMPLES' times, and repeat each sample 'P' times
    return np.array([
        post_pdf.rvs(size=p)
        for _ in range(nsamples)
    ]).reshape(-1, p)

def fi_ln_post(
        posterior_samples: np.ndarray,
        ref_samp: np.array,
        r: float,
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
    diff_from_ref = -posterior_samples + ref_samp
    sin_diff = np.sin(r * diff_from_ref)
    integrand = sin_diff / diff_from_ref
    prod_res = np.nanprod(integrand, axis=1)
    sum_res = np.abs(np.nansum(prod_res))
    n_samp, n_dim = posterior_samples.shape
    const = 1 / (n_samp * np.power(np.pi, n_dim))
    approx_ln_post = np.log(sum_res * const)
    # using bayes theorem to get the approximated log-evidence
    return approx_ln_post


v = 1
true_post =  norm(0, np.sqrt(v/(v+1)))


rs = np.geomspace(50, 100000, 5000)


def plot_fi_error(p, v, color='C0'):
    post_samples = simulate_posterior_samples(p=p, v=v, nsamples=10000)
    fi_posts = np.exp([fi_ln_post(post_samples, ref_samp=np.zeros(p), r=r) for r in tqdm(rs)])
    tru_pdf_at_0 = true_post.pdf(0) ** p
    # plot error vs rs
    plt.plot(rs, tru_pdf_at_0 - fi_posts, label=f'p{p:002d}, v={v}', color=color)
    plt.axhline(np.nanmedian(tru_pdf_at_0 - fi_posts), color='r', label='Median error')
# plt.xscale('log')


plt.figure()
for p in [2]:
    plot_fi_error(p, v, color=f'C{p-1}')

plt.xlabel('R')
plt.ylabel('Error')



# draw a line through 0
plt.axhline(y=0, color='k', linestyle='--')
plt.legend()
plt.show()



# p = 10
# corner.corner(simulate_posterior_samples(p=p, v=1, nsamples=10000), truths=np.zeros(p))
# plt.show()