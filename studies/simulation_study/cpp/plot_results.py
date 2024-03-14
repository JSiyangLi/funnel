"""Plot all the results from the simulation study."""
import glob
import os
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

NBIN = 30


def true_lnz(ppb, block, v):
    d = ppb * block
    return (d / 2) * (np.log(v) - np.log(1 + v))


def load_result(filename):
    """filename: simulation_results_ppb{ppb}_block{block}_n{n}_r{r}_v{v}.txt"""
    filename = os.path.basename(filename)
    data = dict(
        ppb=int(filename.split('_ppb')[1].split('_')[0]),
        block=int(filename.split('_block')[1].split('_')[0]),
        n=int(filename.split('_n')[1].split('_')[0]),
        r=int(filename.split('_r')[-1].split('_')[0]),
        v=float(filename.split('_v')[1].split('.txt')[0])
    )
    data['true_lnz'] = true_lnz(data['ppb'], data['block'], data['v'])
    data['d'] = data['ppb'] * data['block']
    data['lnz_estimates'] = np.loadtxt(filename)
    # only keep up to 300 lnz-estimates
    if len(data['lnz_estimates']) > 300:
        data['lnz_estimates'] = data['lnz_estimates'][:300]
    t = namedtuple('Result', data.keys())(*data.values())
    data['label'] = f"n={t.n}, r={t.r}, v={t.v}, d={t.d} ({t.block}bocks x {t.ppb}params/block)"
    return namedtuple('Result', data.keys())(*data.values())


def main_single(fname):
    result = load_result(fname)
    fig, ax = plt.subplots()
    ax.hist(result.lnz_estimates, bins=NBIN, density=True)
    ax.axvline(result.true_lnz, color='r', label='True lnz')
    ax.set_title(result.label)
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname.replace('.txt', '.png'))


def main():
    results = glob.glob('simulation_results_ppb*_block*_n*_r*_v*.txt')
    results = [load_result(filename) for filename in results]
    # plot histograms of lnz estimates for each result, add label with n, r, v, ppb, block, d,
    # overplot true lnz as a vertical line

    fig, axes = plt.subplots(len(results), 1, figsize=(5, 5 * len(results)))
    for i, result in enumerate(results):
        ax = axes[i]
        ax.hist(result.lnz_estimates, bins=NBIN, density=True)
        ax.axvline(result.true_lnz, color='r', label='True lnz')
        ax.set_title(result.label)
        ax.legend()
    plt.tight_layout()
    plt.savefig('simulation_study_results.png')


if __name__ == '__main__':

    if len(sys.argv) > 1:
        main_single(sys.argv[1])
    else:
        main()
