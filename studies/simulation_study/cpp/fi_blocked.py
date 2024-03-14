import argparse

import numpy as np
from scipy.stats import multivariate_normal, norm
from tqdm.auto import trange
from plot_results import main_single

np.random.seed(42)

N_REPEATS = 300

def log_true_c(v, ppb, block):
    return (ppb * block / 2) * (np.log(v) - np.log(1 + v))


# unnormalised posterior density
def g(theta, v):
    like = -np.sum(theta ** 2) / (2 * v)
    prior = np.sum(norm.logpdf(theta, 0, 1))
    return like + prior


def FourierIntegral_blocked(ppb, block, R, v, n):
    # ppb = param per block
    total_dims = ppb * block
    print(f"FI estimation for d{total_dims} (in {ppb} x {block} blocks)")


    ltrue_c = log_true_c(v, ppb, block)
    ltrue_density = multivariate_normal.logpdf(np.zeros(ppb * block), cov=np.diag([v] * (ppb * block)))
    # true underlying values

    # initialisation
    density_results = np.empty(N_REPEATS)
    simulation_results = np.empty(N_REPEATS)

    # FI
    for i in trange(N_REPEATS):
        post_dens_i = np.empty(block)
        for block_i in range(block):
            # generate posterior sample for block i (independence)
            target_sample = np.random.normal(0, np.sqrt(v / (v + 1)), size=(n, ppb))

            # density estimation for block i
            as_ = np.sum(np.prod(np.sin(R * target_sample) / target_sample, axis=1))
            post_dens_i[block_i] = as_ / (n * np.pi ** ppb)

        # density estimation altogether
        post_dens = np.sum(np.log(np.abs(post_dens_i)))
        density_results[i] = post_dens

        lpriorlike = g(np.zeros(ppb), v=v)
        simulation_results[i] = lpriorlike - post_dens

    # save simulation results as a txt file
    np.savetxt(f'simulation_results_ppb{ppb}_block{block}_n{n}_r{R}_v{v}.txt', simulation_results)


if __name__ == '__main__':
    # get CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--PPB', type=int, default=1)
    parser.add_argument('--BLOCK', type=int, default=1)
    parser.add_argument('--R', type=int, default=100)
    parser.add_argument('--V', type=float, default=1)
    parser.add_argument('--N', type=int, default=int(1e6))
    args = parser.parse_args()

    # run the simulation
    FourierIntegral_blocked(args.PPB, args.BLOCK, args.R, args.V, args.N)
    main_single(f'simulation_results_ppb{args.PPB}_block{args.BLOCK}_n{args.N}_r{args.R}_v{args.V}.txt')