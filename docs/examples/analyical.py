import numpy as np
import emcee
from scipy.stats import multivariate_normal, cauchy, norm
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

inflation = 1
n = int(1e3)
v = 0.01
p = 1
extraction_index = 42

R = 40
tau = np.exp(-7.25)
eta = np.exp(-7.75)
target_mean = np.zeros(p)
target_var = v / (v + 1) * np.identity(p)


def true_lnZ(p, v):
    return (p / 2) * (np.log(v) - np.log(1 + v))


def log_likelihood(theta, v):
    like = -np.sum(theta ** 2) / (2 * v)
    prior = np.sum(norm.logpdf(theta, loc=0, scale=1))
    return like + prior

def fi_lnZ(p,v):
    posterior_samples = np.random.multivariate_normal(np.zeros(p), target_var * inflation, size=n)

# Initialize arrays for results
iter = 300
simulation_results = np.zeros(iter)
simR_results = np.zeros(iter)
simR_lpriorlike = np.zeros(iter)




# Evaluate the posterior density
for i in range(iter):

    target_sample = np.random.multivariate_normal(np.zeros(p), target_var * inflation, size=n)
    a = np.abs(np.prod(np.sin(R * target_sample) / target_sample, axis=1))
    post_dens = np.sum(a) / (n * np.pi ** p)
    lpriorlike = log_likelihood(np.zeros(p), v)
    simulation_results[i] = lpriorlike - np.log(post_dens)

    if i == extraction_index:
        test_sample = target_sample

    print(f"iteration {i}")

# Calculate statistics
print(f"Mean of simulation results: {np.mean(simulation_results)}")
print(f"Standard deviation of simulation results: {np.std(simulation_results) / np.sqrt(iter)}")

# plot histogram of FI LnZ values and true value as vertical line
plt.hist(simulation_results, bins=30, density=True)
plt.axvline(x=log_true_c, color="r")
plt.show()
