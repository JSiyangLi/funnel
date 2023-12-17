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
Rt = 2000
Re = 360
tau = np.exp(-7.25)
eta = np.exp(-7.75)
target_mean = np.zeros(p)
target_var = v / (v + 1) * np.identity(p)
log_true_c = (p / 2) * (np.log(v) - np.log(1 + v))


# Define the joint density of data and parameter
def log_likelihood(theta, v):
    like = -np.sum(theta ** 2) / (2 * v)
    prior = np.sum(norm.logpdf(theta, loc=0, scale=1))
    return like + prior


# Initialize arrays for results
epanechnikov_results = np.zeros(iter)
triangle_results = np.zeros(iter)
doubleexp_results = np.zeros(iter)
norm_results = np.zeros(iter)
simulation_results = np.zeros(iter)
simR_results = np.zeros(iter)





# Evaluate the posterior density
for i in range(iter):
    if p == 1:
        # FI
        target_sample = np.random.normal(scale=np.sqrt(target_var.ravel()), size=n)
        a = np.sin(R * target_sample) / target_sample
        post_dens = np.sum(a) / (n * np.pi)
        lpriorlike = log_likelihood(np.array([0]), v)
        simulation_results[i] = lpriorlike - np.log(post_dens)

        # normal-FI
        post_dens = np.mean(norm.pdf(target_sample, loc=0, scale=4 * tau))
        norm_results[i] = lpriorlike - np.log(post_dens)

        # double-exponential-FI
        post_dens = np.mean(cauchy.pdf(target_sample, loc=0, scale=eta))
        doubleexp_results[i] = lpriorlike - np.log(post_dens)

        # triangle-kernel-FI
        a = (1 / (Rt * target_sample ** 2)) * (1 - np.cos(Rt * target_sample))
        post_dens = np.sum(a) / (n * np.pi)
        triangle_results[i] = lpriorlike - np.log(post_dens)

        # Epanechnikov-kernel-FI
        a = (-2 / Re) * (1 / target_sample ** 2) * np.cos(Re * (-target_sample)) + (2 / Re ** 2) * (
                1 / -target_sample ** 3) * np.sin(Re * (-target_sample))
        post_dens = np.sum(a) / (n * np.pi)
        epanechnikov_results[i] = lpriorlike - np.log(post_dens)

    else:
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

# Estimating LnZ using the same sample but alternating reference points

# Assuming 'iter', 'p', 'R', 'test_sample', 'n', 'pi', 'v', 'g', and 'ltrue.c' are defined

test_evaluation_index = np.random.choice(np.arange(1, iter + 1), iter, replace=True)
for j in range(iter):
    ref_index = test_evaluation_index[j]
    ref = test_sample[ref_index - 1]  # Adjusting for 1-based indexing in R
    evaluation_sample = np.delete(test_sample, ref_index - 1, axis=0)

    if p == 1:
        a = np.sin(R * (evaluation_sample - ref)) / (evaluation_sample - ref)
        post_dens = np.abs(np.sum(a) / (n * np.pi))
    else:
        a = np.abs(np.prod(np.sin(R * (evaluation_sample - ref)) / (evaluation_sample - ref), axis=1))
        post_dens = np.abs(np.sum(a) / (n * np.pi**p))

    lpriorlike = g(ref, v=v)
    simR_results[j] = lpriorlike - np.log(post_dens)

    print(f"iteration{j + 1}")

na_mean = np.mean(np.isnan(simR_results))
result_mean = np.mean(simR_results)
result_sd = np.std(simR_results) / np.sqrt(iter)

square_diff = (simulation_results - np.repeat(ltrue_c, iter))**2
mean_square_diff = np.mean(square_diff) / iter

print(f"Mean of simR_results: {result_mean}")
print(f"Standard deviation of simR_results / sqrt(iter): {result_sd}")

# Additional code for plotting (requires matplotlib)
plt.figure(figsize=(12, 8))
plt.subplot(321)
plt.hist(simulation_results, bins=30, density=True)
plt.axvline(log_true_c, color='red', linestyle='dashed', linewidth=2)
plt.xlabel("Estimates of marginal likelihood")
plt.title("The Fourier Integral Estimates")

plt.figure(figsize=(12, 8))
plt.subplot(321)
plt.hist(simR_results, bins=30, density=True)
plt.axvline(log_true_c, color='red', linestyle='dashed', linewidth=2)
plt.xlabel("Estimates of marginal likelihood")
plt.title("The Fourier Integral Estimates")

# Repeat the above plotting code for other result arrays (norm_results, doubleexp_results, triangle_results, epanechnikov_results)

# Save the figures to a PDF file
plt.savefig("GFIcase4.png")
plt.show()
