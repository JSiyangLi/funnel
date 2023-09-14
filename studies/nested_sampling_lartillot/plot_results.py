import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
# ticklocator

# plt rc params y ticks mirrored
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["ytick.right"] = True
plt.rcParams["font.size"] = 15

DATA = "out/nested_sampling_lnzs.dat"
V = 0.01


def true_lnz(v, dim):
    return (dim / 2) * (np.log(v) - np.log(1 + v))


def read_data():
    data = np.loadtxt(DATA, skiprows=1)
    data = pd.DataFrame(data, columns=["dim", "ns_lnz", "ns_lnz_err"])
    return data


def violin_plot_of_lnzs_for_each_d():
    data = read_data()
    # different panel for each dimension
    fig, ax = plt.subplots(1, 3, figsize=(8, 5))
    for i, d in enumerate([1, 20, 100]):
        ax[i].violinplot(
            data[data["dim"] == d]["ns_lnz"],
            quantiles=[0.16, 0.5, 0.84],
            showextrema=False,
        )
        # draw ornage line at true ln(Z)
        ax[i].axhline(true_lnz(V, d), color="orange", ls="--", lw=1)
        ax[i].set_xticks([])
        # ax[i].set_xticks([1])
        if i == 0:
            ax[i].set_ylabel("ln(Z)")
        ax[i].set_title(f"d={d}")
        # only use 3 y ticks
        ax[i].yaxis.set_major_locator(MaxNLocator(4))
    fig.tight_layout()
    fig.savefig("out/nested_sampling_lnzs.png", bbox_inches="tight")


if __name__ == '__main__':
    violin_plot_of_lnzs_for_each_d()
