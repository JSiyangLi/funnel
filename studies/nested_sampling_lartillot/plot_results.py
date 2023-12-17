import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
import funnel.analytical as fa
# ticklocator

# plt rc params y ticks mirrored
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["ytick.right"] = True
plt.rcParams["font.size"] = 15

DATA = "nested_sampling_lnzs.dat"
V = fa.v#0.01


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
        dat = data[data["dim"] == d]["ns_lnz"]
        if len(dat) == 0:
            continue
        ax[i].violinplot(
            dat,
            quantiles=[0.16, 0.5, 0.84],
            showextrema=False,
        )
        # draw ornage line at true ln(Z)
        # ax[i].axhline(true_lnz(V, d), color="orange", ls="--", lw=1)
        # draw the theoratical variance bound R^p/n
        ax[i].ax.hlines(y = true_lnz(V, d) - fa.R ** fa.p / fa.n, xmin = 0.75, xmax = 1.25, color = "purple")
        ax[i].ax.hlines(y = true_lnz(V, d) + fa.R ** fa.p / fa.n, xmin = 0.75, xmax = 1.25, color = "purple")
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
