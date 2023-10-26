import numpy as np
import sys, os
import csv
import mvsr as mvsr
import shutil
import json
import pandas as pd
import pyoperon as Operon

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
            results.iloc[seed] = [result[0], [round(i[0], 3) for i in result[1]]]

        if use_single_view is not None:
            results.to_csv(
                f"toy_results/{name}/{noise}/example{use_single_view}_results.csv",
                index=False,
            )

        else:
            results.to_csv(f"toy_results/{name}/{noise}/MvSR_results.csv", index=False)


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
        "maxL": 20,
        "maxD": 10,
        "OperationSet": Operon.NodeType.Square,
    }

    run_analysis("polynomial", nseeds, polynomial_settings)
