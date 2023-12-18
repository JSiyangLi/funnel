import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def model(x, m0, c0, var0, var1, turning_point, **kwargs):
    noise0 = np.random.normal(0, np.sqrt(var0), len(x))
    noise1 = np.random.normal(0, np.sqrt(var1), len(x))
    part0 = m0 * x + c0
    part1 = c0 + m0 * turning_point
    return np.where(x < turning_point, part0 + noise0, part1 + noise1)


def get_prior(x, y):
    priors = bilby.prior.PriorDict({
        'm0': bilby.prior.TruncatedGaussian(mu=-0.5, sigma=-3, maximum=0, minimum=-100, name='m0'),
        'c0': bilby.prior.Normal(mu=max(y.ravel()), sigma=100,name= 'c0'),
        'var0': bilby.prior.LogUniform(1e-5, 1, 'var0'),
        'var1': bilby.prior.LogUniform(1e-5, 20, 'var1'),
        'turning_point': bilby.prior.Uniform(min(x), max(x), 'turning_point'),
        'sigma': bilby.core.prior.LogUniform(1e-4, 10, "sigma")
    })
    return priors


class GaussianLikelihoodIgnoreNan(bilby.likelihood.GaussianLikelihood):
    def __init__(self, x, y, func, sigma=None, **kwargs):
        super(GaussianLikelihoodIgnoreNan, self).__init__(x=x, y=y, func=func, **kwargs)
        self.sigma = sigma

    def log_likelihood(self):
        return np.nansum(- (self.residual / self.sigma) ** 2 / 2 - np.log(2 * np.pi * self.sigma ** 2) / 2)


def __post_processing(df, x, y):
    """Post processing function to add r and lnz to the posterior"""
    df['lnz'] = df['turning_point'] * df['m0'] + df['c0']
    df['r'] = np.exp(x[np.argmax(y[:, np.newaxis] <= df['lnz'].values, axis=0)])
    return df


def estimate_best_r(r_vals, lnz_vals, n_steps=100):
    likelihood = GaussianLikelihoodIgnoreNan(r_vals, lnz_vals, model)

    priors = get_prior(r_vals, lnz_vals)

    # test that we're not getting nans
    for i in range(10):
        sample = priors.sample(1)
        likelihood.parameters.update(sample)
        assert np.isfinite(likelihood.log_likelihood()), "Likelihood --> nan!!"

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler='emcee',
        nsteps=n_steps,
        outdir='outdir',
        label='example',
        clean=False,
        walkers=10,
    )

    result.posterior = __post_processing(result.posterior, r_vals, lnz_vals)
    return result


def plot_two_part_model_samples_and_data(samples_df: pd.DataFrame, x, y):
    fig = plt.figure()
    plt.scatter(x, y, marker='o', color='k', s=0.1, zorder=10)
    if 'lnz' not in samples_df:
        samples_df = __post_processing(samples_df, x, y)
    samples_df = samples_df.sort_values('lnz')
    samples_df = samples_df.sample(100)
    samps = samples_df.to_dict('records')
    for s in samps:
        plt.plot(x, model(x, **s), color='C0', alpha=0.1)
        plt.axhline(s['lnz'], color='C1', alpha=0.3, zorder=-100)
        plt.axvline(np.log(s['r']), color='C2', alpha=0.3, zorder=-100)

    lnz_str = get_median_and_error_str(samples_df['lnz'])
    r_str = get_median_and_error_str(samples_df['r'])
    plt.plot([], [], color="k", label="Data")
    plt.plot([], [], color="C0", label="2-part model")
    plt.plot([], [], color="C1", label="LnZ: " + lnz_str)
    plt.plot([], [], color="C2", label="R: " + r_str)
    plt.legend(loc='upper right', fontsize=12)

    plt.xlim(min(x), max(x))
    plt.ylim(min(y), max(y));
    plt.xlabel(r"$\ln R$")
    plt.ylabel(r"$\ln\mathcal{Z}$")
    return fig


def get_median_and_error_str(s: np.array):
    med = np.median(s)
    upper = np.percentile(s, 84) - med
    lower = med - np.percentile(s, 16)
    fmt = "{:.2f}".format
    s = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    s = s.format(fmt(med), fmt(lower), fmt(upper))
    return s

# Loss function (can be altered to other sensible function)
def loss(rotation_bias, cluster_variance):
    return np.exp(2 * rotation_bias) + cluster_variance
