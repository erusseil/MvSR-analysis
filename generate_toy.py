import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from numpy.random import default_rng
import shutil


global_rng = default_rng(seed=0)


def func_sin(npoints, A, B):
    X = np.linspace(-0.5 * np.pi, 0.5 * np.pi, npoints)
    return X, A * np.sin(X + B)


def func_poly(npoints, A, B, C):
    X = np.linspace(-2, 2, npoints)
    return X, A + B * X + C * X**2


def func_gaussian(npoints, mu, sig):
    X = np.linspace(-1, 1, npoints)
    return X, (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((X - mu) / sig, 2.0) / 2)
    )


def gaussian_noise(y, rng, noise_ratio):
    sigma = np.std(y) * np.sqrt(noise_ratio / (1.0 - noise_ratio))
    return rng.normal(loc=0.0, scale=sigma, size=len(y))


def create_folders(name, noises):
    if not os.path.exists("toy_data"):
        os.makedirs("toy_data")

    # Delete previous data if it exists
    if os.path.isdir(f"toy_data/{name}"):
        shutil.rmtree(f"toy_data/{name}")

    if not os.path.exists(f"toy_data/{name}"):
        os.makedirs(f"toy_data/{name}")

    if not os.path.exists(f"toy_data/{name}/perfect"):
        os.makedirs(f"toy_data/{name}/perfect")

    for noise in noises:
        if not os.path.exists(f"toy_data/{name}/noisy_{noise}"):
            os.makedirs(f"toy_data/{name}/noisy_{noise}")


def generate_data(func, name, params, noises, npoints=25):
    header = ["Xaxis", "yaxis"]
    create_folders(name, noises)

    for idx, param in enumerate(params):
        x, y = func(npoints, *param)
        example = np.vstack((x, y)).T

        with open(
            f"toy_data/{name}/perfect/example{idx}.csv",
            "w",
            encoding="UTF8",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(example)

        for noise in noises:
            y_noisy = y + gaussian_noise(y, global_rng, noise)
            example = np.vstack((x, y_noisy)).T

            with open(
                f"toy_data/{name}/noisy_{noise}/example{idx}.csv",
                "w",
                encoding="UTF8",
                newline="",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(example)


if __name__ == "__main__":
    noises = [0.05]  # , .1, .5
    generate_data(func_poly, "polynomial", [[0, 1, 1], [1, 0, 1], [1, 1, 0]], noises)
    generate_data(func_sin, "sinus", [[1, 0], [1, 1], [0, 1]], noises)
    generate_data(func_gaussian, "gaussian", [[1, 0.5], [0.5, 1], [0, 2]], noises)
