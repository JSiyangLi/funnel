import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp
from jax.numpy import prod, sum, sin, nansum, nanprod, power, log, exp, sqrt
from jax import jit
import os
import jax
from jax import config

config.update("jax_enable_x64", True)

OUTDIR = 'out_simulation'
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

PI = jnp.pi
R_DEFAULT = 100
V_DEFAULT =1


@jit
def ln_fi_post(θstar, θ, r=R_DEFAULT):
    n, d = θ.shape
    assert θstar.shape == (1, d)
    integrand = sin(r * (θstar - θ)) / (θstar - θ)
    sum_prod_integrand = nansum(nanprod(integrand, axis=1))
    return log(sum_prod_integrand / (n * power(PI, d)))


@jit
def ln_fi_z(θstar, θ, d=1, v=V_DEFAULT, r=R_DEFAULT):
    return ln_likelihood(θstar, v=v) + ln_prior(θstar) - ln_fi_post(θstar, θ, r=r)


@jit
def get_err(θstar, θ, d=1, v=V_DEFAULT, r=R_DEFAULT):
    return ln_fi_z(θstar, θ, d=d, v=v, r=r) - ln_true_z(d=d, v=v)


@jit
def ln_true_z(d, v=V_DEFAULT):
    return log(power(2. * PI * (1 + v), -d / 2.))


@jit
def ln_true_post(θ, v=V_DEFAULT):
    return sum(norm.logpdf(θ, loc=0, scale=jnp.sqrt(v / (v + 1))))


@jit
def ln_likelihood(θ, v=V_DEFAULT):
    n, d = θ.shape
    return log(
        power(2. * PI * v, -d / 2) * nanprod(exp(-power(θ, 2) / (2 * v)))
    )


@jit
def ln_prior(θ):
    return log(
        prod(norm.pdf(θ, loc=0, scale=1))
    )


def sample_θ(d: int = 1, v: float = V_DEFAULT, nsamples: int = int(1e6)):
    return np.random.normal(loc=0, scale=np.sqrt(v / (v + 1)), size=(nsamples, d))


def error_vs_ns(d=1, v=V_DEFAULT, r=R_DEFAULT, num_runs=10, maxn=int(10**6)):
    ns = np.linspace(500, maxn, num_runs, dtype=int)
    err = np.zeros(num_runs)
    θstar = jnp.zeros((1, d))
    for i in trange(num_runs):
        err[i] = get_err(
            θstar=θstar,
            θ=sample_θ(d=d, v=v, nsamples=ns[i]),
            d=d, v=v, r=r
        )

    return ns, err


def test_lnl_lnpri_functions(d, v):
    θstar = jnp.zeros((1, d))

    lnl = ln_likelihood(θstar, v=v)
    lnpri = ln_prior(θstar)
    lnpost = ln_true_post(θstar, v=v)
    lnz = lnl + lnpri - lnpost
    truelnz = ln_true_z(d, v=v)
    print(f"lnl={lnl}")
    print(f"lnpri={lnpri}")
    print(f"lnpost={lnpost}")
    print(f'lnz={lnz}')
    assert lnz == truelnz, ('LnZ using analytical ln-posterior doenst matches the analytical ln_z!!')


def test_posterior_samples(d, v, n=int(1e5)):
    θ = sample_θ(d, v, nsamples=n)
    assert θ.shape == (n, d), f"θ.shape ={θ.shape} !={(n, d)}"
    expected_std = np.sqrt(v / (v + 1))
    obs_std = np.std(θ, axis=0)
    assert obs_std.shape == (d,), f"obs_std.shape={obs_std.shape}"
    check = jnp.all(jnp.isclose(expected_std, obs_std, atol=0.05))
    assert check, f"expected_std ! = std(θ) => {expected_std:.3f}, {obs_std}"
    θ1 = sample_θ(d=2, v=1, nsamples=4)
    θ2 = sample_θ(d=2, v=1, nsamples=4)
    assert np.sum(θ1 - θ2) != 0, 'samples not unique!'


d, v = 20, 2
test_posterior_samples(d, v)
test_lnl_lnpri_functions(d, v)


def run_sims(maxn=int(10**6), numruns=10 ,ndims=[1, 10, 20, 100], v=V_DEFAULT):
    data = {}
    for d in tqdm(ndims):
        data['ns'], data[f'd{d}'] = error_vs_ns(d=d, v=v, num_runs=numruns, maxn=maxn)
    df = pd.DataFrame(data)
    df = df.set_index('ns')
    df.to_csv('fi_errors.csv')
    plot_results(df)
    return df


def plot_results(df):
    for col in df.columns:

        plt.figure(figsize=(5, 1.5))
        plt.axhline(0, color='k')
        plt.scatter(df.index, df[col], label=col)
        plt.xlabel('N Samples')
        plt.ylabel("LnZ Error")
        plt.legend()
        plt.tight_layout()
        plt.title(f"R={R_DEFAULT}, v={V_DEFAULT}")
        plt.savefig(f"{col}_fi_errors.png")
        plt.close('all')


df = run_sims(maxn=10**7)
plot_results(df)
