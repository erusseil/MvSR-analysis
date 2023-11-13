import numpy as np
import sys, os
import csv
import mvsr as mvsr
import shutil
import json
import pandas as pd
import pyoperon as Operon
import sympy as sp
import time
import sympy as sp
import string
import re


def refit_solution(expression, name, example):
    
    example_path = f"toy_data/{name}/perfect/{example}"
    model_path = f"toy_results/{name}/operon_{example[:-4]}.models"
    
    npoints = len(pd.read_csv(example_path))
    
    with open(model_path, 'w') as f:
        f.write(expression)

    outputs = []
    no_perfect_yet = True
    for _ in range(10):
        if no_perfect_yet:
            stream = os.popen('srtree-opt -f operon -i {0} -d {1} --hasheader --restart --niter 100 --distribution gaussian'.format(model_path, example_path))
            outputs.append(stream.read())
            no_perfect_yet = round(find_sse(outputs[-1]), 3) != 0

    output = outputs[np.argmin(np.array([find_sse(i) for i in outputs]))]
    print([find_sse(i) for i in outputs], "      :     ", find_sse(output))
    os.remove(model_path)
    return round(find_sse(output)/npoints, 3)

def convert_string_to_func(SR_str):
    alphabet = list(string.ascii_uppercase)
    parameter_names = alphabet + [[k + i for k in alphabet for i in alphabet ]]
    parameters_dict = {}
    function_str = str(sp.N(sp.sympify(SR_str), 50))
    all_floats = re.findall("\d+\.\d+", function_str)

    if len(all_floats)>len(parameter_names):
        print('WARNING WAY TOO BIG FUNCTIONS')
        return function_str, False
        
    for idx, one_float in enumerate(all_floats):
        function_str = function_str.replace(one_float, parameter_names[idx], 1)
        parameters_dict[parameter_names[idx]] = float(one_float)
        
    return function_str

def replace_wrong_symbols(expression):
    expression = expression.replace("^", "**")
    return expression
    
def find_expression(output):
    """
    Reads output of srtree-opt refiter and finds the refited expression.

    Paramters
    ---------
    output: str
        String outputed by srtree-opt refiter

    Returns
    -------
    str:
        Refited mathematical expression as outputed by srtree-opt
    """
    start1 = output.find('\n0,') + 3
    start2 = output.find(',', start1) + 1
    stop = output.find(',', start2)
    return output[start2:stop]


def find_sse(output):
    """
    Reads output of srtree-opt refiter and finds sse error

    Paramters
    ---------
    output: str
        String outputed by srtree-opt refiter

    Returns
    -------
    float:
        Sum of square error as outputed by srtree-opt
    """
    start = output.find('\n0,') + 3

    for i in range(9):
        start = output.find(',', start) + 1
    
    stop = output.find(',', start)
    return float(output[start:stop])


def create_folders(name, noises):
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

    # Delete previous data if it exists
    if os.path.isdir(f"toy_results/{name}"):
        shutil.rmtree(f"toy_results/{name}")

    if not os.path.exists(f"toy_results/{name}"):
        os.makedirs(f"toy_results/{name}")

    for noise in noises:
        if not os.path.exists(f"toy_results/{name}/{noise}"):
            os.makedirs(f"toy_results/{name}/{noise}")


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
    examples = os.listdir(f"toy_data/{name}/perfect")
    ndim = np.shape(pd.read_csv(f"toy_data/{name}/perfect/{examples[0]}"))[1]
    
    results = pd.DataFrame(
        data=np.empty(shape=(nseeds, ndim)),
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

            mse_refit = []
            for example in examples:
                before = time.time()
                refit = refit_solution(result[0], name, example)
                duration = time.time()-before
                mse_refit.append(refit)

            results.iloc[seed] = [convert_string_to_func(result[0]), mse_refit]

        if use_single_view is not None:
            results.to_csv(
                f"toy_results/{name}/{noise}/example{use_single_view}_results.csv",
                index=False,
            )

        else:
            results.to_csv(f"toy_results/{name}/{noise}/MvSR_results.csv", index=False)

        print(f"Noise {noise} done !")

def run_single_view(name, nseeds, settings):
    path = f"toy_data/{name}/perfect/"
    all_examples = [x for x in os.listdir(path) if "csv" in x]

    for example in range(len(all_examples)):
        run_mvsr(name, nseeds, settings, use_single_view=example)


def run_analysis(name, nseeds, settings):
    noises = os.listdir(f"toy_data/{name}")
    create_folders(name, noises)

    with open(f"toy_results/{name}/settings.txt", "w") as f:
        save_settings = settings.copy()
        save_settings["OperationSet"] = str(save_settings["OperationSet"])
        f.write(json.dumps(save_settings))

    run_mvsr(name, nseeds, settings)
    run_single_view(name, nseeds, settings)


if __name__ == "__main__":

    nseeds = 10
    
    polynomial_settings = {
        "generations": 1000,
        "maxL": 50,
        "maxD": 20,
        "OperationSet": None,
    }

    run_analysis("polynomial", nseeds, polynomial_settings)
