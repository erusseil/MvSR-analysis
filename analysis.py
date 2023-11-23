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
from sympy import sin, exp, sqrt
import string
import re
from sympy import symbols, lambdify
from iminuit import Minuit
from iminuit.cost import LeastSquares

def MSE(y, ypred):
    """
    Compute mean square error
    
    Parameters
    ---------
    y: array
        True answer
    ypred: array
        Predicted answers
    Returns
    -------
    float
        Mean square error metric
    
    """
    return np.square(np.subtract(y,ypred)).mean()

    
def refit_solution(func, path, initial_guess):

    data = pd.read_csv(path)
    npoints = len(data)
    
    if npoints<len(initial_guess):
        return np.nan

    X, y = data.iloc[:, :-1].values, data.yaxis.values
    X = X.T

    least_squares = LeastSquares(X, y, .1, func)
    fit = Minuit(least_squares, **initial_guess)
    fit.migrad()
    y_pred = func(X, *fit.values)
    MSE_mvsr = MSE(y, y_pred)
    return round(MSE_mvsr, 3)

def convert_string_to_func(SR_str, n_variables):
    alphabet = list(string.ascii_uppercase)
    parameter_names = alphabet + [[k + i for k in alphabet for i in alphabet ]]
    parameters_dict = {}


    function_str = str(sp.N(sp.sympify(SR_str), 50))

    floatzoo = 99.9
    # Zoo detection :
    while "zoo" in function_str:
        function_str= function_str.replace("zoo", str(floatzoo), 1)
        floatzoo += 1
        
    # Remove scientific notation
    function_str = re.sub("e\d+", '',re.sub("e\+\d+", '', re.sub("e-\d+", '', function_str)))
    # Make sure sqrt are not mistaken for parameters
    function_str = function_str.replace("**0.5", '**sqrt')

    all_floats = re.findall("\d+\.\d+", function_str) + ['0']

    if len(all_floats)>len(parameter_names):
        print('WARNING WAY TOO BIG FUNCTIONS')
        return function_str, False

    n_parameters = 0
    for idx, one_float in enumerate(all_floats):
        if one_float in function_str:
            n_parameters += 1
            function_str = function_str.replace(one_float, parameter_names[idx], 1)
            parameters_dict[parameter_names[idx]] = float(one_float)

    # Revert it once we found all floats
    function_str = function_str.replace('**sqrt', "**0.5")
    used_params = parameter_names[:n_parameters]

    X = sp.IndexedBase('X')
    param_symbols = {k:sp.Symbol(k) for k in used_params}
    param_symbols['X'] = X

    tempo_function_str = function_str
    for i in range(n_variables):
        tempo_function_str = tempo_function_str.replace(f'X{i+1}', f'X[{i}]')

    try:
        func = sp.lambdify(["X"] + used_params, eval(tempo_function_str, globals(), param_symbols), modules=["numpy", {'exp':np.exp, 'Log':np.log, 'sin':np.sin}])
    except:
        print("Original:", SR_str)
        print("After:", function_str)
    
    return func, function_str, parameters_dict


def convert_string_to_funcOLD(SR_str):
    alphabet = list(string.ascii_uppercase)
    parameter_names = alphabet + [[k + i for k in alphabet for i in alphabet ]]
    parameters_dict = {}
    function_str = str(sp.N(sp.sympify(SR_str), 50))
    # Remove scientific notation
    function_str = re.sub("e\d+", '', re.sub("e-\d+", '', function_str))
    
    all_floats = re.findall("\d+\.\d+", function_str) + ['0']

    if len(all_floats)>len(parameter_names):
        print('WARNING WAY TOO BIG FUNCTIONS')
        return function_str, False
        
    for idx, one_float in enumerate(all_floats):
        function_str = function_str.replace(one_float, parameter_names[idx], 1)
        parameters_dict[parameter_names[idx]] = float(one_float)

    xs = symbols(['X1', *parameter_names[:len(all_floats)]])
    func = lambdify(xs, function_str, ["numpy", {'exp':np.exp, 'Log':np.log}])
    return func, function_str, parameters_dict


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
        for maxL in settings['maxL']:

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
    n_variables = np.shape(pd.read_csv(f"toy_data/{name}/perfect/{examples[0]}"))[1]-1
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
                    refit = refit_solution(func, perfect_path, initial_guess)
                    mse_refit.append(refit)
    
                results.iloc[seed] = [func_str, mse_refit]


        if use_single_view is not None:
            results.to_csv(
                f"toy_results/{name}/{noise}/max{settings['maxL']}/example{use_single_view}_results.csv",
                index=False,
            )

        else:
            results.to_csv(f"toy_results/{name}/{noise}/max{settings['maxL']}/MvSR_results.csv", index=False)

def run_single_view(name, nseeds, settings):
    path = f"toy_data/{name}/perfect/"
    all_examples = sorted([x for x in os.listdir(path) if "csv" in x])

    for example in range(len(all_examples)):
        print(f"Example {example} starting :")
        run_mvsr(name, nseeds, settings, use_single_view=example)


def run_analysis(name, nseeds, settings):
    noises = os.listdir(f"toy_data/{name}")
    create_folders(name, noises, settings)

    with open(f"toy_results/{name}/settings.txt", "w") as f:
        save_settings = settings.copy()
        save_settings["OperationSet"] = str(save_settings["OperationSet"])
        f.write(json.dumps(save_settings))

    for maxL in settings['maxL']:
        setting = settings.copy()
        setting['maxL'] = maxL
        print("Multiview starting :")
        run_mvsr(name, nseeds, setting)
        run_single_view(name, nseeds, setting)


if __name__ == "__main__":

    nseeds = 100
    common_operation_set = Operon.NodeType.Square | Operon.NodeType.Exp | Operon.NodeType.Sin | Operon.NodeType.Sqrt
    common_setting = {
        "generations": 1000,
        "maxL": [20, 25], #,5, 10, 15 
        "maxD": 5,
        "OperationSet": common_operation_set, 
    }
    
    run_analysis("polynomial", nseeds, common_setting)
    #run_analysis("gaussian", nseeds, common_setting)
    #run_analysis("friedman1", nseeds, common_setting)
    #run_analysis("friedman2", nseeds, common_setting)

