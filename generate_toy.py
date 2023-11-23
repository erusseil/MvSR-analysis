import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from numpy.random import default_rng
import shutil


global_rng = default_rng(seed=0)


def func_poly(X, A, B, C):
    return A + B * X + C * X**2

def func_gaussian(X, A, mu, sig):
    return (
        A / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((X - mu) / sig, 2.0) / 2)
    )

def func_fried1(X, A, B, C, D):
    # Original functional form comes from here :
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html
    # The function was modified to incorporate 4 free parameters (and remove constants)
    return np.sin(A * X[:, 0] * X[:, 1]) + B * (X[:, 2] - C) ** 2 + D * X[:, 3] + X[:, 4]

def func_fried2(X, A, B, C, D):
    # Original functional form comes from here :
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman2.html
    # The function was modified to incorporate 4 free parameters
    return A * (B * X[:, 0] ** 2 + C + (D * X[:, 1] * X[:, 2]  - 1 / (X[:, 1] * X[:, 3])) ** 2) ** 0.5
    
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


def generate_data(func, name, Xs, params, noises):

    if len(np.shape(Xs)) == 2:
        header = ["Xaxis0", "yaxis"]
    else:
        header = []
        for i in range(np.shape(Xs)[1]):
            header.append(f"Xaxis{i}")
        header.append("yaxis")
    
    create_folders(name, noises)

    for idx, param in enumerate(params):
        x = Xs[idx]
        y = func(x.T, *param)

        if len(np.shape(x.T)) == 1:
            example = np.vstack((x.T, y)).T
        elif len(np.shape(x.T)) == 2:
            example = np.array([list(x.T[idx2])+[y[idx2]] for idx2 in range(len(x.T))])

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
    noises = [0.05, .1]  # 


    # Generate polynomial data : 
    step = 0.2
    Xs, Xs_lim = [], [-2, 2]
    
    for i in range(4):
        Xs.append(np.arange(Xs_lim[0], Xs_lim[1], step))
    
    generate_data(func_poly, "polynomial", Xs, [[2, -2, 0],[0, 2, -2],[0, 0, 2],[0, 2, 0]], noises)

    # Generate gaussian data :
    steps = 0.2
    Xs, Xs_lim = [], [[-2, 2], [-2, 2], [-2, 0], [0, 2]]
    
    for idx, lim in enumerate(Xs_lim):
        if (idx==2) | (idx==3):
            uno = np.arange(lim[0], lim[1], step)
            Xs.append(np.sort(np.concatenate([uno, uno])))
        else:
            Xs.append(np.arange(lim[0], lim[1], step))

    generate_data(func_gaussian, "gaussian", Xs, [[0, 0, 2],[2, 0, 2],[2, 0.5, .5],[2, 0.5, .5]], noises)

    # Generate friedman1 data :
    npoints = 100
    Xs = []

    #Loop through each example
    for _ in range(4):
        loop = [] 
        #Loop through each X
        for i in range(5):
            loop.append(np.random.random_sample(npoints))
        Xs.append(np.array(loop))
            
    generate_data(func_fried1, "friedman1", Xs, [[2, 2, 2, 0],
                                                  [2, 2, 0, 2],
                                                  [2, 0, 2, 2],
                                                  [0, 2, 2, 2]], noises)
    
    # Generate friedman2 data :
    npoints = 100
    Xs = []

    #Loop through each example
    for _ in range(4):
        loop = [] 
        loop.append(np.random.uniform(low=0, high=100, size=npoints))
        loop.append(np.random.uniform(low=40*np.pi, high=560*np.pi, size=npoints))
        loop.append(np.random.uniform(low=0, high=1, size=npoints))
        loop.append(np.random.uniform(low=1, high=11, size=npoints))
        Xs.append(np.array(loop))
            
    generate_data(func_fried2, "friedman2", Xs, [[2, 2, 2, 0],
                                                  [2, 2, 0, 2],
                                                  [2, 0, 2, 2],
                                                  [0, 2, 2, 2]], noises)
    




