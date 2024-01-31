import h5py
import numpy as np
import matplotlib.pyplot as plt

FNAME = "gw150914_runs/results/gw150914_fi.hdf5"


class LVK_FI_data:
    def __init__(self, fname=FNAME):
        with h5py.File(fname, "r") as f:
            data = {k: v[()] for k, v in f.items()}
        self.r_vals = data.pop("r_vals")
        self.fi_lnz = data.pop("fi_lnz")
        self.ns_lnz = data.pop("ns_lnz")
        self.ss_lnz = data.pop("ss_lnz")
        self.ns_lnz_err = data.pop("ns_lnz_err")
        self.ss_lnz_err = data.pop("ss_lnz_err")


def plot_lnzs(data: LVK_FI_data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    fi_lnz = data.fi_lnz
    r_vals = data.r_vals
    ns_lnz = data.ns_lnz
    ns_lnz_err = data.ns_lnz_err
    ss_lnz = data.ss_lnz
    ss_lnz_err = data.ss_lnz_err


    ax = axes[0]
    ax.scatter(r_vals, fi_lnz, s=0.5, alpha=0.7)
    ax.set_xscale('log')
    ax.axhline(np.median(fi_lnz), ls='-', color='C0')
    ax.axhline(ns_lnz, ls='--', color='C1', label="Nested Sampling")
    ax.axhline(ss_lnz, ls='--', color='C2', label="Stepping Stone")
    ax.legend(fontsize=10)
    ax.set_xlabel("|R| Values")
    ax.set_ylabel("LnZ")
    ax.set_ylim(-6995, -6960)

    ax = axes[1]
    bins = np.linspace(-6995, -6975, 50)
    ax.hist(fi_lnz[r_vals > 10000], density=True, bins=bins, alpha=0.5)
    ax.axvline(np.median(fi_lnz[r_vals > 10000]), ls='-', color='C0')
    ax.axvline(ns_lnz, ls='--', color='C1', label="Nested Sampling")
    ax.axvspan(ns_lnz - ns_lnz_err, ns_lnz + ns_lnz_err, color="C1", alpha=0.3)
    ax.axvline(ss_lnz, ls='--', color='C2', label="Stepping Stone")
    ax.axvspan(ss_lnz - ss_lnz_err, ss_lnz + ss_lnz_err, color="C2", alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    data = LVK_FI_data()
    plot_lnzs(data)