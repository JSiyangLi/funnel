import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

import colorsys
from matplotlib.colors import ColorConverter

import corner

from .fi_core import get_fi_lnz_list

rcParams.update({"xtick.major.pad": "7.0"})
rcParams.update({"xtick.major.size": "7.5"})
rcParams.update({"xtick.major.width": "1.5"})
rcParams.update({"xtick.minor.pad": "7.0"})
rcParams.update({"xtick.minor.size": "3.5"})
rcParams.update({"xtick.minor.width": "1.0"})
rcParams.update({"ytick.major.pad": "7.0"})
rcParams.update({"ytick.major.size": "7.5"})
rcParams.update({"ytick.major.width": "1.5"})
rcParams.update({"ytick.minor.pad": "7.0"})
rcParams.update({"ytick.minor.size": "3.5"})
rcParams.update({"ytick.minor.width": "1.0"})
rcParams.update({"font.size": 20})


def plot_fi_evidence_results(
    posterior_samples: pd.DataFrame = pd.DataFrame,
    sampling_lnz: np.array = [],
    r_vals: np.array = np.array([]),
    num_ref_params: int = 10,
    plot_all_lnzs: bool = False,
    plot_ci: bool = True,
    plt_kwgs: dict = {},
    lnzs=np.array([]),
):
    if len(lnzs) == 0:
        lnzs, r_vals, _ = get_fi_lnz_list(posterior_samples, r_vals, num_ref_params)
    lnz_quants = np.nanquantile(lnzs, [0.16, 0.5, 0.84], axis=0)

    frac_nans = np.sum(np.isnan(lnzs.ravel()), axis=0) / len(lnzs.ravel()) * 100

    # PLOT Nested Sampling LnZ
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if len(sampling_lnz) > 0:
        ax.fill_between(
            x=r_vals,
            y1=min(sampling_lnz),
            y2=max(sampling_lnz),
            color="tab:blue",
            interpolate=True,
            alpha=0.5,
            label="Nested Sampling",
            zorder=10,
        )
    ax.set_xlim(min(r_vals), max(r_vals))
    ax.set_xlabel(r"FI $\|R\|$")
    ax.set_ylabel(r"$\ln{\mathcal{Z}}$")
    ax.set_xscale("log")

    fi_color = "tab:green"
    if plot_all_lnzs:
        plt_kwgs["alpha"] = plt_kwgs.get("alpha", 0.05)
        for lnz in lnzs:
            ax.plot(r_vals, lnz, zorder=-1, **plt_kwgs, color=fi_color)
    if plot_ci:
        ci_color = scale_color_brightness(color_str=fi_color, scale_l=2)
        ax.fill_between(
            x=r_vals,
            y1=lnz_quants[2],
            y2=lnz_quants[0],
            color=ci_color,
            interpolate=True,
            zorder=0,
            lw=0,
            label=r"90% CI",
        )
        ax.plot(r_vals, lnz_quants[1], zorder=1, color=fi_color)
    ax.plot([], [], label="FI", color=fi_color, alpha=1, zorder=1)

    # # median along last 10% of lnzs
    # idx = int(0.9 * len(lnzs))
    # med = np.nanmedian(np.nanmedian(lnzs[idx:], axis=1), axis=0)
    # ax.axhline(med, label="FI Median", color=fi_color, alpha=0.5, zorder=100, ls="--")

    # add textbox with fraction of nans on bottom left
    if frac_nans > 0:
        ax.text(
            0.05,
            0.05,
            f"NaNs: {frac_nans:.2f}%",
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment="bottom",
        )

    ax.legend(loc=(1.1, 0.5), frameon=False)
    plt.tight_layout()
    return fig


def plot_corner_and_mark_samples(df, samps):
    # assert df.columns == samps.columns, "Columns of df and samps must match"

    fig = corner.corner(
        df,
        color="tab:gray",
        truth=None,
        label_kwargs={"fontsize": 26},
        quantiles=None,
        plot_density=False,
        plot_contours=True,
        fill_contours=True,
    )

    # overplot the FI points
    for i, s in enumerate(samps):
        # corner.overplot_lines(fig, s, color=f'C{i}')
        corner.overplot_points(
            fig,
            [[np.nan if t is None else t for t in s]],
            color=f"C{i}",
            marker="s",
            markersize=7,
        )
    return fig


def scale_color_brightness(rgb=None, color_str="", scale_l=1.0):
    if color_str != "":
        rgb = ColorConverter.to_rgb(color_str)
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)
