import numpy as np
import sys, os
import csv
import mvsr as mvsr
import shutil
import json
import pandas as pd
import pyoperon as Operon
import time
import sympy as sp
from sympy import sin, exp, sqrt, log, Abs
import string
import re
from sympy import symbols, lambdify
from iminuit import Minuit
from iminuit.cost import LeastSquares
import argparse
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt


def refit_and_plot(
    folder,
    func,
    func_str,
    initial_guess,
    Xlim,
    ylim,
    labels,
    saveas,
    limits=None,
):
    """
    Refits a func to all examples of the specified folder.
    Plots the result (needs to be 2D).

    Parameters
    ---------
    folder: str
        Path to the folder of examples
    func: Python function
        Function to fit
    func_str: str
        String of the function (used as title)
    initial_guess: dict
        Initial parameter values for the minimization
    Xlim: list
        [Xmin, Xmax] bounds to display
    ylim: list
        [ymin, ymax] bounds to display
    labels: list
        [x_label, y_label] used to legend the axis
    saveas: str
        Name of the file to save
    limits: list
        List of [lower, upper] bounds to use for the minimization. 
        Needs to be in the same order as initial_guess.
        Optional

    Returns
    -------
        List of r2 error of the fit
    """

    smooth = [np.linspace(Xlim[0], Xlim[1], 500).T]
    color_palette = sns.color_palette("tab10")
    all_sets = np.sort([x for x in os.listdir(folder) if "csv" in x])
    fig, axes = plt.subplots(1, 1, figsize=(16, 8))

    all_sets = all_sets[: len(color_palette)]

    errors = []

    for idx, file in enumerate(all_sets):

        df = pd.read_csv(f"{folder}/{file}")
        X = df.iloc[:, :-1].values.T
        y = df.yaxis.values

        least_squares = LeastSquares(X, y, 1, func)
        fit = Minuit(least_squares, **initial_guess)

        if limits is not None:
            for k in range(len(limits)):
                fit.limits[list(initial_guess)[k]] = limits[k]

        fit.migrad()

        y_pred = func(X, *fit.values)
        errors.append(r2_score(y, y_pred))
        sx = np.sort(X, axis=0)
        dic = fit.values.to_dict()
        display = [f"{x}: {dic.get(x):.2f}" for x in dic]
        display = ", ".join([str(item) for item in display])

        plt.scatter(X.flatten(), y, label=display, color=color_palette[idx], s=60)
        plt.plot(
            smooth[0],
            func(smooth, *fit.values).flatten(),
            color=color_palette[idx],
            alpha=0.6,
            linewidth=3,
        )
        plt.ylim(ylim[0], ylim[1])
        plt.xlim(min(smooth[0]), max(smooth[0]))

        title = f"f(X1) = {func_str}".replace("X1", "X")
        plt.title(title, fontsize=20)
        plt.xlabel(labels[0], fontsize=18)
        plt.ylabel(labels[1], fontsize=18)

        for axis in ["top", "bottom", "left", "right"]:
            axes.spines[axis].set_linewidth(2)

        axes.tick_params(width=2, labelsize=17)
        plt.legend(fontsize=17)
        plt.savefig(f"plots/{saveas}.png", bbox_inches="tight")

    return errors

def save_2D_example(X, y, path):
    """
    Save 2D examples to the correct format to be used by MvSR 

    Parameters
    ---------
    X: array
    y: array
    path: str
        Path of the folder to stores examples
    """    
    
    header = ['Xaxis0', 'yaxis']
    example = np.vstack((X, y)).T

    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(example)

def refit_solution(func, path, initial_guess):
    """
    Fits a python function to an example and returns MSE

    Parameters
    ---------
    func: Python function
        Function to fit
    path: str
        Path of the example to fit
    initial_guess: dict
        Initial parameter values for the minimization (iminuit)

    Returns
    -------
        Mean square error of the fit
    """

    data = pd.read_csv(path)
    npoints = len(data)

    if npoints < len(initial_guess):
        return np.nan

    X, y = data.iloc[:, :-1].values, data.yaxis.values
    X = X.T

    least_squares = LeastSquares(X, y, 0.1, func)

    try:
        fit = Minuit(least_squares, **initial_guess)
        fit.migrad()
    except:
        print('Minimization error: check inputed function')

    fit = Minuit(least_squares, **initial_guess)
    fit.migrad()
    y_pred = func(X, *fit.values)
    y_pred = np.where(y_pred<1e50, y_pred, 1e50)
    MSE_mvsr = mean_squared_error(y, y_pred)
    return MSE_mvsr


def convert_string_to_func(SR_str, n_variables):
    """
    Converts a string outputed by pyOperon into a python function

    Parameters
    ---------
    SR_str: str
        Result of pyoperon
    n_variables: int
        Dimensionality of X
    """
    
    alphabet = list(string.ascii_uppercase)
    parameter_names = alphabet + [[k + i for k in alphabet for i in alphabet]]
    parameters_dict = {}

    function_str = str(sp.N(sp.sympify(SR_str), 50))

    floatzoo = 99.9
    # Zoo detection :
    while "zoo" in function_str:
        function_str = function_str.replace("zoo", str(floatzoo), 1)
        floatzoo += 1

    function_str = function_str.replace("re", "")
    function_str = function_str.replace("im", "")
    if "I" in function_str:
        function_str = function_str.replace("**I", "**1")
        function_str = function_str.replace("*I", "*1")
        function_str = function_str.replace("/I", "/1")
        function_str = function_str.replace("+ I", "+ 0")
        function_str = function_str.replace("- I", "- 0")
        function_str = str(sp.N(sp.sympify(function_str), 50))

    # Remove scientific notation
    function_str = re.sub(
        "e\d+", "", re.sub("e\+\d+", "", re.sub("e-\d+", "", function_str))
    )
    # Make sure sqrt are not mistaken for parameters up to 5 sqrt intricated
    for i, code in enumerate(["one", "two", "three", "four", "five"]):
        function_str = function_str.replace(f"**{str(0.5**(i+1))}", f"**sqrt{code}")

    all_floats = re.findall("\d+\.\d+", function_str) + ["0"]

    if len(all_floats) > len(parameter_names):
        print("WARNING WAY TOO BIG FUNCTIONS")
        return function_str, False

    n_parameters = 0
    for idx, one_float in enumerate(all_floats):
        if one_float in function_str:
            if one_float == "0":
                for zzz in [
                    i for i, letter in enumerate(function_str) if letter == "0"
                ]:
                    if not function_str[zzz - 1].isnumeric():
                        n_parameters += 1
                        function_str = function_str.replace(
                            one_float, parameter_names[idx], 1
                        )
                        parameters_dict[parameter_names[idx]] = float(one_float)
            else:
                n_parameters += 1
                function_str = function_str.replace(one_float, parameter_names[idx], 1)
                parameters_dict[parameter_names[idx]] = float(one_float)

    # Revert sqrt temporariry protection
    for i, code in enumerate(["one", "two", "three", "four", "five"]):
        function_str = function_str.replace(f"**sqrt{code}", f"**{str(0.5**(i+1))}")

    used_params = parameter_names[:n_parameters]

    X = sp.IndexedBase("X")
    param_symbols = {k: sp.Symbol(k) for k in used_params}
    param_symbols["X"] = X

    tempo_function_str = function_str
    for i in range(n_variables):
        tempo_function_str = tempo_function_str.replace(f"X{i+1}", f"X[{i}]")

    try:
        func = sp.lambdify(
            ["X"] + used_params,
            eval(tempo_function_str, globals(), param_symbols),
            modules=[
                "numpy",
                {"exp": np.exp, "log": np.log, "sin": np.sin, "abs": np.abs},
            ],
        )
    except:
        print("Original:", SR_str)
        print("After:", function_str)

    return func, function_str, parameters_dict


def create_folders(name, noises, settings):
    """
    Creates folders associated to the function

    Paramters
    ---------
    name: str
        Name of the function's folder
    noises: list (of floats or str)
        List of the noise levels to consider
    """

    if not os.path.exists("toy_results"):
        os.makedirs("toy_results")

    if not os.path.exists(f"toy_results/{name}"):
        os.makedirs(f"toy_results/{name}")

    for noise in noises:
        if not os.path.exists(f"toy_results/{name}/{noise}"):
            os.makedirs(f"toy_results/{name}/{noise}")
        for maxL in settings["maxL"]:

            # Delete previous data if it exists
            if os.path.isdir(f"toy_results/{name}/{noise}/max{maxL}"):
                shutil.rmtree(f"toy_results/{name}/{noise}/max{maxL}")

            if not os.path.exists(f"toy_results/{name}/{noise}/max{maxL}"):
                os.makedirs(f"toy_results/{name}/{noise}/max{maxL}")


def run_mvsr(name, nseeds, settings, use_single_view=None):
    """
    Run the main MvSR analysis for a given toy data at different noise levels.
    Saves results inside "toy_results" folder

    Paramters
    ---------
    name: str
        Name of the function's folder
    nseeds: int
        Number of repetition of the experiment
    settings: dict
        Parameters of the MvSR function.
        Only 4 values will be changed in the main analysis namely:
        settings = {'generations': generations,
                    'maxL': maxL, 'maxD': maxD,
                    'OperationSet': OperationSet}
    use_single_view: None or int
        If None, run MvSR normally
        If int, run normal SR using only example number "use_single_view".
        In that case the expression found is still evaluated on all examples
    """

    noises = os.listdir(f"toy_data/{name}")
    examples = sorted([x for x in os.listdir(f"toy_data/{name}/perfect") if "csv" in x])
    n_variables = np.shape(pd.read_csv(f"toy_data/{name}/perfect/{examples[0]}"))[1] - 1
    results = pd.DataFrame(
        data=np.empty(shape=(nseeds, 2)),
        columns=["expression", "losses"],
        dtype="object",
    )

    for noise in noises:
        for seed in range(nseeds):
            result = mvsr.MultiViewSR(
                f"toy_data/{name}/{noise}",
                verbose=0,
                seed=seed,
                use_single_view=use_single_view,
                **settings,
            )

            conversion = convert_string_to_func(result[0], n_variables)

            # Case where the expression was too big to be fitted realistically
            if not conversion[1]:
                results.iloc[seed] = [conversion[0], np.nan]

            else:
                func, func_str, initial_guess = conversion
                mse_refit = []
                for example in examples:
                    perfect_path = f"toy_data/{name}/perfect/{example}"
                    refit = refit_solution(
                        func, perfect_path, initial_guess
                    )
                    mse_refit.append(refit)

                results.iloc[seed] = [func_str, mse_refit]

        if use_single_view is not None:
            results.to_csv(
                f"toy_results/{name}/{noise}/max{settings['maxL']}/example{use_single_view}_results.csv",
                index=False,
            )

        else:
            results.to_csv(
                f"toy_results/{name}/{noise}/max{settings['maxL']}/MvSR_results.csv",
                index=False,
            )


def run_single_view(name, nseeds, settings):
    path = f"toy_data/{name}/perfect/"
    all_examples = sorted([x for x in os.listdir(path) if "csv" in x])

    for example in range(len(all_examples)):
        run_mvsr(name, nseeds, settings, use_single_view=example)


def run_analysis(name, nseeds, settings):
    noises = os.listdir(f"toy_data/{name}")
    create_folders(name, noises, settings)

    with open(f"toy_results/{name}/settings.txt", "w") as f:
        save_settings = settings.copy()
        save_settings["OperationSet"] = str(save_settings["OperationSet"])
        f.write(json.dumps(save_settings))

    for maxL in settings["maxL"]:
        setting = settings.copy()
        setting["maxL"] = maxL
        run_mvsr(name, nseeds, setting)
        run_single_view(name, nseeds, setting)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--maxL", nargs="*", type=int, default=[5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25], help="maxL list"
    )
    arg_parser.add_argument("--opset", default="common", type=str, help="common or sin")
    arg_parser.add_argument(
        "--function", required=True, type=str, help="Function to extract"
    )
    arg_parser.add_argument("--nseeds", default=100, type=int, help="Number of seeds")
    args = arg_parser.parse_args()

    common_operation_set = (
        Operon.NodeType.Square | Operon.NodeType.Exp | Operon.NodeType.Sqrt
    )

    if args.opset == "common":
        operation_set = common_operation_set
    elif args.opset == "sin":
        operation_set = common_operation_set | Operon.NodeType.Sin

    common_setting = {
        "generations": 1000,
        "maxL": args.maxL,
        "maxD": 5,
        "OperationSet": operation_set,
    }

    run_analysis(args.function, args.nseeds, common_setting)
