import argparse
import io
import logging
import os
import time
from contextlib import redirect_stdout

import bilby
import numpy as np
import pandas as pd
from tqdm.auto import trange

from funnel.fi_core import get_fi_lnz_list

logging.getLogger("bilby").setLevel(logging.ERROR)

OUTDIR = "out"


class LartillotLikelihood(bilby.Likelihood):
    def __init__(self, dim, v):
        self.dim = dim
        self.v = v
        super().__init__(parameters={f"x{i}": None for i in range(dim)})

    def log_likelihood(self):
        x = np.array(list(self.parameters.values()))
        lnl = -np.power(x, 2) / (2 * self.v)
        if self.dim == 1:
            return lnl[0]
        else:
            return np.sum(lnl)


def get_prior(n_dim):
    pri = bilby.core.prior.PriorDict()
    for i in range(n_dim):
        pri[f"x{i}"] = bilby.core.prior.TruncatedNormal(
            mu=0, sigma=1, minimum=-1000, maximum=1000
        )
    return pri


def true_lnz(v, dim):
    return (dim / 2) * (np.log(v) - np.log(1 + v))


def nested_sampling_lnz(v, dim, label="", nlive=1000, checkpoint=False):
    likelihood = LartillotLikelihood(dim, v)
    priors = get_prior(dim)
    clean = not checkpoint
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=nlive,
        label=f"d{dim}_v{v}_nlive{nlive}_{label}",
        clean=clean ,
        sample="rwalk",
        save=False,
        plot=False,
        print_method="interval-10",
        check_point=checkpoint,
        check_point_plot=False,
    )
    return (result.log_evidence, result.log_evidence_err)


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--v", type=float, default=0.01)
    parser.add_argument("-d", "--dim", type=int, default=1)
    parser.add_argument("-r", "--nrep", type=int, default=50)
    parser.add_argument("-s", "--seed", type=int, default=-1)
    parser.add_argument("-n", "--nlive", type=int, default=1000)
    parser.add_argument('-c', '--checkpoint', action='store_true', default=False)
    args = parser.parse_args()
    if args.seed < 0:
        args.seed = np.random.randint(0, 1e5)
        np.random.seed(args.seed)
    return args


def __runner(args, outfile, checkpoint_n_sec=3 * 60):
    with open(outfile, "a") as f:
        f.write(f"lnz lnz_err\n")
    t0 = time.time()
    data = []
    pbar = trange(args.nrep)
    for i in pbar:
        l = f"_seed{args.seed}"
        data.append(nested_sampling_lnz(args.v, args.dim, l, nlive=args.nlive, checkpoint=args.checkpoint))
        pbar.set_postfix_str(f"LnZ: {data[-1][0]:.2f} +/- {data[-1][1]:.2f}")
        t1 = time.time()
        if t1 - t0 > checkpoint_n_sec:
            t0 = t1
            __checkpoint_data(data, outfile)
            data = []
    __checkpoint_data(data, outfile)


def __checkpoint_data(data, outfile):
    with open(outfile, "a") as f:
        for lnz, lnz_err in data:
            f.write(f"{lnz} {lnz_err}\n")


def main():
    args = parse_cli_args()
    print(f"Lartillot LnZ (v={args.v}, dim={args.dim}, seed={args.seed})")
    print(f"True LnZ: {true_lnz(args.v, args.dim):.2f}")
    os.makedirs(OUTDIR, exist_ok=True)
    outfile = f"{OUTDIR}/lartillot_d{args.dim}_v{args.v}_seed{args.seed}.dat"
    __runner(args, outfile)


if __name__ == "__main__":
    main()
