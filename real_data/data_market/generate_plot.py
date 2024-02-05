import sys, os
import pandas as pd
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import astropy
import shutil
import pyoperon as Operon
from iminuit import Minuit
from iminuit.cost import LeastSquares
import seaborn as sns
from os import listdir
from os.path import isfile, join
from scipy.optimize import curve_fit

sys.path.append("../")
import generate_data_market as gdm

sys.path.append("../../")
import mvsr as mvsr
import analysis as ana


def gettable():
    final_df = pd.DataFrame()
    path = "/media3/etienne/workdir/MvSR-analysis/real_data/data_market/"
    for stonks in gdm.all_runs:
        errors, parameters = gdm.fit_test(stonks[0], stonks[1], path, use_MSE=True)
        final_df[stonks[2]] = errors
    print(final_df[:-1].describe())
    final_df.to_csv("companies_MSE", index=False)


def getplot():

    final_df = pd.read_csv("companies_MSE")

    onlyfiles = [
        f for f in listdir("all_companies") if isfile(join("all_companies", f))
    ]
    onlyfiles.remove("ARM.csv")

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    filenumber = 103
    file = onlyfiles[filenumber]
    xbounds = [[-0.07, 0.07], [-0.015, 0.015]]
    titles = ["Ingersoll Rand", "S&P500"]

    for idx, asset in enumerate([file, "../spx.csv"]):
        if idx == 1:
            funarg = -1
        elif idx == 0:
            funarg = filenumber

        data = pd.read_csv(f"all_companies/{asset}")
        data["Open"] = data["Open"] / data["Open"].max()
        diffs = np.array(data["Open"][1:]) - np.array(data["Open"][:-1])
        diffs = diffs[diffs == diffs]

        hist, bins = np.histogram(diffs, bins=100)
        x, y = bins[:-1], hist
        y = 10 * y / y.max()
        mask = y > 0
        x, y = x[mask], y[mask]

        smoothX = np.linspace(x.min(), x.max(), 300)

        for info in [gdm.all_runs[0], gdm.all_runs[-1]]:
            popt, pcov = curve_fit(info[0], x, y, p0=info[1], maxfev=50000)
            ax[idx].plot(
                smoothX,
                info[0](smoothX, *popt),
                label=f"{info[2]}\nMSE={final_df.iloc[funarg][info[2]]:.3f}",
                alpha=0.65,
                linewidth=2.5,
            )

        ax[idx].scatter(x, y, c="black", marker=".")
        ax[idx].set_xlim(xbounds[idx][0], xbounds[idx][1])
        ax[idx].set_ylim(-0.1, 10.5)
        ax[idx].legend()
        ax[idx].set_title(titles[idx], fontsize=11, fontweight="bold")

        for axis in ["top", "bottom", "left", "right"]:
            for axee in ax:
                axee.spines[axis].set_linewidth(1.4)

        ax[idx].tick_params(width=1.4)

        ax[idx].tick_params(labelsize=11)
        ax[idx].legend(fontsize=11)

    fig.text(0.5, -0.02, "Asset normalized daily return", ha="center", fontsize=12)
    fig.text(
        0.08, 0.5, "Normalized count", va="center", rotation="vertical", fontsize=12
    )
    plt.show()
    fig.savefig(
        "/media3/etienne/workdir/MvSR-analysis/real_data/plots/market_plot.pdf",
        bbox_inches="tight",
    )
