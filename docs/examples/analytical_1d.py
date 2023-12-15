import numpy as np
import emcee
import pandas as pd
from scipy.stats import multivariate_normal, cauchy, norm
import matplotlib.pyplot as plt
from bilby.core.prior import Normal
from funnel.fi_core import get_fi_lnz_list
from funnel.plotting import plot_fi_evidence_results
from funnel.r_estimator import estimate_best_r, plot_two_part_model_samples_and_data
from scipy.stats import multivariate_normal, cauchy, norm

# Set seed for reproducibility
np.random.seed(42)

inflation = 1
n = int(1e3)
v = 0.01
p = 1
R = 40
Rt = 2000
Re = 360
tau = np.exp(-7.25)
eta = np.exp(-7.75)
target_mean = np.zeros(p)
target_var = v / (v + 1) * np.identity(p)
log_true_c = (p / 2) * (np.log(v) - np.log(1 + v))

PRIOR = Normal(0, 1)


def log_prior(theta):
    return norm.logpdf(theta, 0, 1)


# Define the joint density of data and parameter
def log_likelihood(theta: np.ndarray, v=0.01):
    return -np.power(theta, 2) / (2 * v)


def generate_posterior(v=0.01):
    posterior_samples = np.random.normal(scale=np.sqrt(v / (v + 1)), size=n)
    ln_pri = log_prior(posterior_samples)
    ln_lnl = log_likelihood(posterior_samples, v)
    return pd.DataFrame(dict(
        x=posterior_samples,
        log_prior=ln_pri,
        log_likelihood=ln_lnl,
    ))


# Initialize arrays for results
epanechnikov_results = np.zeros(300)
triangle_results = np.zeros(300)
doubleexp_results = np.zeros(300)
norm_results = np.zeros(300)
simulation_results = np.zeros(300)

posterior_samples = generate_posterior(v)

results = get_fi_lnz_list(posterior_samples, r_vals=np.geomspace(1e-2, 1e5, 2000), num_ref_params=10, )
lnzs, r_vals, samp = results
plt_kwgs = dict(lnzs=lnzs, r_vals=r_vals, sampling_lnz=[log_true_c], )

x = np.log(r_vals)
change_point_posterior = estimate_best_r(x, lnzs[0], n_steps=500)

fig = plot_fi_evidence_results(**plt_kwgs)
ax = fig.axes[0]
ax.axhline(np.median(change_point_posterior.posterior["lnz"]), color="tab:orange", linestyle="dashed",
           label="Estimated LnZ")
fig.tight_layout()
fig.savefig("GFIcase4.png")

fig = plot_two_part_model_samples_and_data(change_point_posterior.posterior, x, lnzs[0])
fig.savefig("changept.png")

#
#
# # Evaluate the posterior density
# for i in range(300):
#     # FI
#     target_sample = np.random.normal(scale=np.sqrt(target_var.ravel()), size=n)
#     a = np.sin(R * target_sample) / target_sample
#     post_dens = np.sum(a) / (n * np.pi)
#     lpriorlike = log_likelihood(np.array([0]), v)
#     simulation_results[i] = lpriorlike - np.log(post_dens)
#
#
# # Calculate statistics
# print(f"Mean of simulation results: {np.mean(simulation_results)}")
# print(f"Standard deviation of simulation results: {np.std(simulation_results) / np.sqrt(300)}")
#
# # Additional code for plotting (requires matplotlib)
# plt.figure(figsize=(12, 8))
# plt.subplot(321)
# plt.hist(simulation_results, bins=30, density=True)
# plt.axvline(log_true_c, color='red', linestyle='dashed', linewidth=2)
# plt.xlabel("Estimates of marginal likelihood")
# plt.title("The Fourier Integral Estimates")
#
# # Repeat the above plotting code for other result arrays (norm_results, doubleexp_results, triangle_results, epanechnikov_results)
#
# # Save the figures to a PDF file
# plt.savefig("GFIcase4.png")
# plt.show()
