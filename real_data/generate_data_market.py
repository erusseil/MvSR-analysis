from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from os import listdir
import pandas as pd
import numpy as np
import shutil
import os
from os.path import isfile, join


def fit_test(func, guess, path):
    errors = []
    parameters = []

    onlyfiles = [
        f
        for f in listdir(f"{path}/all_companies")
        if isfile(join(f"{path}/all_companies", f))
    ]
    onlyfiles.remove("ARM.csv")

    for idx, file in enumerate(onlyfiles[10:] + ["../spx.csv"]):  #
        data = pd.read_csv(f"{path}/all_companies/{file}")
        data["Open"] = data["Open"] / data["Open"].max()
        diffs = np.array(data["Open"][1:]) - np.array(data["Open"][:-1])
        diffs = diffs[diffs == diffs]
        hist, bins = np.histogram(diffs, bins=100)
        x, y = bins[:-1], hist
        y = 10 * y / y.max()
        mask = y > 0
        x, y = x[mask], y[mask]
        popt, pcov = curve_fit(func, x, y, p0=guess, maxfev=50000)
        errors.append(mean_squared_error(y, func(x, *popt)))
        parameters.append(popt)

    return errors, parameters


def cauchy(X, A, B):
    return B * (A**2) / (((X**2) + (A**2)))


cauchy_guess = [0.01, 100]


def gaussian(X, A, B):
    return A * np.exp(-(X**2) / B)


gaussian_guess = [10, 0.001]


def laplace(X, A, B):
    return A * np.exp(-B * abs(X))


laplace_guess = [10, 1]


def explaplace(X, A, B, C):
    return A * np.exp(-B * abs(X)) * np.exp(-C * X)


explaplace_guess = [10, 60, -0.1]


def linearlaplace(X, A, B, C):
    return (A - B * X) * np.exp(-C * abs(X))


linearlaplace_guess = [10, 40, 60]


def powerlaplace(X, A, B, C):
    return np.exp(A - B * abs(X) ** C)


powerlaplace_guess = [5, 20, 0.3]


all_runs = [
    [cauchy, cauchy_guess, "cauchy"],
    [gaussian, gaussian_guess, "gaussian"],
    [laplace, laplace_guess, "laplace"],
    [explaplace, explaplace_guess, "explaplace"],
    [linearlaplace, linearlaplace_guess, "linearlaplace"],
    [powerlaplace, powerlaplace_guess, "powerlaplace"],
]
