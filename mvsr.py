import numpy as np
from pyoperon.sklearn import SymbolicRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pyoperon as Operon
import random, time, sys, os, json
from scipy import stats
import sympy as sp

import warnings
warnings.filterwarnings("ignore")

def MultiViewSR(
    path,
    minL=1,
    maxL=20,
    maxD=10,
    generations=1000,
    OperationSet=None,
    seed=44,
    verbose=True,
    use_single_view=None,
):
    all_examples = [x for x in os.listdir(path) if "csv" in x]
    all_examples = sorted(all_examples)

    if use_single_view is not None:
        if use_single_view >= len(all_examples):
            raise ValueError("Example asked do not exist")

    n_sets = len(all_examples)
    problems = []
    selectors = []
    smallest_data = np.array(
        [
            len(np.loadtxt(f"{path}/{i}", delimiter=",", skiprows=1))
            for i in all_examples
        ]
    ).min()

    for i in all_examples:
        Z = np.loadtxt(f"{path}/{i}", delimiter=",", skiprows=1)
        # np.random.shuffle(Z)
        ds = Operon.Dataset(Z)

        training_range = Operon.Range(0, smallest_data)
        test_range = Operon.Range(0, smallest_data)

        # define the regression target
        target = ds.Variables[-1]  # take the last column in the dataset as the target

        # take all other variables as inputs
        inputs = [h for h in ds.VariableHashes if h != target.Hash]

        # initialize a problem object which encapsulates the data, input, target and training/test ranges
        problem = Operon.Problem(ds, training_range, test_range)
        problem.Target = target
        problem.InputHashes = inputs

        # use tournament selection with a group size of 5
        # we are doing single-objective optimization so the objective index is 0
        selector = Operon.TournamentSelector(objective_index=0)
        selector.TournamentSize = 10
        selectors.append(selector)

        # initialize the primitive set (add, sub, mul, div, exp, log, sin, cos), constants and variables are implicitly added
        if OperationSet != None:
            OperationSet = Operon.PrimitiveSet.Arithmetic | OperationSet

        else:
            OperationSet = Operon.PrimitiveSet.Arithmetic

        problem.ConfigurePrimitiveSet(OperationSet)
        pset = problem.PrimitiveSet

        problems.append(problem)

    # define a tree creator (responsible for producing trees of given lengths)
    btc = Operon.BalancedTreeCreator(pset, problems[0].InputHashes, bias=0.0)
    tree_initializer = Operon.UniformLengthTreeInitializer(btc)
    tree_initializer.ParameterizeDistribution(minL, maxL)
    tree_initializer.MaxDepth = maxD

    # define a coefficient initializer (this will initialize the coefficients in the tree)
    coeff_initializer = Operon.NormalCoefficientInitializer()
    coeff_initializer.ParameterizeDistribution(0, 1)

    # define several kinds of mutation
    mut_onepoint = Operon.NormalOnePointMutation()
    mut_changeVar = Operon.ChangeVariableMutation(inputs)
    mut_changeFunc = Operon.ChangeFunctionMutation(pset)
    mut_replace = Operon.ReplaceSubtreeMutation(btc, coeff_initializer, maxD, maxL)

    # use a multi-mutation operator to apply them at random
    mutation = Operon.MultiMutation()
    mutation.Add(mut_onepoint, 1)
    mutation.Add(mut_changeVar, 1)
    mutation.Add(mut_changeFunc, 1)
    mutation.Add(mut_replace, 1)

    # define crossover
    crossover_internal_probability = (
        0.9  # probability to pick an internal node as a cut point
    )
    crossover = Operon.SubtreeCrossover(crossover_internal_probability, maxD, maxL)

    # define fitness evaluation
    evaluator = Operon.MultiEvaluator(problems[0])
    evaluators = []
    interpreters = []
    metrics = []
    optimizers = []
    for i in range(n_sets):
        interpreter = Operon.DispatchTable()  # tree interpreter
        interpreters.append(interpreter)
        error_metric = Operon.MSE()  # use the coefficient of determination as fitness
        metrics.append(error_metric)
        evaluator_i = Operon.Evaluator(problems[i], interpreter, metrics[-1], True)
        optimizers.append(
            Operon.Optimizer(
                interpreter,
                problems[i],
                optimizer="lbfgs",
                likelihood="gaussian",
                iterations=100,
                batchsize=smallest_data,
            )
        )

        evaluator_i.Optimizer = optimizers[-1]

        #evaluator_i.LocalOptimizationIterations = 100
        evaluators.append(evaluator_i)

        if use_single_view is None:
            evaluator.Add(evaluators[-1])

        else:
            if i == use_single_view:
                evaluator.Add(evaluators[-1])

    aggregateEvaluator = Operon.AggregateEvaluator(evaluator)
    # aggregateEvaluator.AggregateType = Operon.AggregateType.HarmonicMean
    aggregateEvaluator.AggregateType = Operon.AggregateType.Max

    evaluator.Budget = 1000 * 1000  # computational budget

    # define how new offspring are created
    generator = Operon.BasicOffspringGenerator(
        aggregateEvaluator, crossover, mutation, selector, selector
    )
    # initialize an algorithm configuration
    config = Operon.GeneticAlgorithmConfig(
        generations=generations,
        max_evaluations=100000000,
        local_iterations=0,
        population_size=1000,
        pool_size=5,
        p_crossover=1.0,
        p_mutation=0.25,
        epsilon=1e-10,
        seed=seed,
        time_limit=86400,
    )
    
    # define how the offspring are merged back into the population - here we replace the worst parents with the best offspring
    reinserter = Operon.ReplaceWorstReinserter(objective_index=0)
    gp = Operon.GeneticProgrammingAlgorithm(
        problems[0], config, tree_initializer, coeff_initializer, generator, reinserter
    )
    
    # report some progress
    gen = 0
    max_ticks = 50
    interval = (
        1
        if config.Generations < max_ticks
        else int(np.round(config.Generations / max_ticks, 0))
    )
    t0 = time.time()

    # initialize a rng
    rng = Operon.RomuTrio(seed)

    # run the algorithm
    gp.Run(rng, threads=1)
    best = gp.BestModel
    agg_model_string = Operon.InfixFormatter.Format(best.Genotype, ds, 15)
    # Internally operon uses A*f(x)+B and fits A and B. We make sure that A and B appears explicity in the solutions
    # to later be replaces by parameters
    agg_model_string = "1.00001*("+ agg_model_string + ")+0.00001"


    scores = []

    if verbose:
        print("Agg. estimator: ", aggregateEvaluator(rng, best))
        print(f"{sp.sympify(agg_model_string)}\n")

    minimized = []

    for i, e in enumerate(evaluators):
        scores.append(e(rng, best))

        if verbose:
            print(e.CallCount, e.ResidualEvaluations, e.JacobianEvaluations)
            print(f"Eval. {i}: ", e(rng, best))
            print(f"{sp.N(agg_model_string, 4)}\n")


    return agg_model_string, scores
