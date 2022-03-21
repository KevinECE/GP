import math
import matplotlib.pyplot as plt

from deap import base
from deap import creator
from deap import tools
from deap import gp

from functions import *


########################################################################################################################
# INITIALIZE GP
########################################################################################################################
def initGP(toolbox, X, y, num_features, function_set="koza", selection="tournament", sample_size=None, limitDepth=None,
           limitSize=None, realData=False):
    """
    :param toolbox: toolbox used to init gp
    :param X: independent variable(s)
    :param y: dependent variables
    :param num_features: number of independent variables in X
    :param function_set: select which function set to use (Koza, Korns)
    :param selection: method of selection (Lexicase, Tournament)
    :param sample_size: number of samples to use
    :param limitDepth: optionally set a limit on the max expression tree depth
    :param limitSize: optionally set a limit on the max expression tree size
    :param realData:
    :return: none
    """

    # Initialize primitive set
    pset = gp.PrimitiveSet("MAIN", num_features)
    # args = {'ARG0': 'x0', 'ARG1': 'x1', 'ARG2': 'x2', 'ARG3': 'x3', 'ARG4': 'x4'}
    # for arg in args:
    #     pset.renameArguments(arg)

    # Define function set
    if function_set == "koza":
        koza(pset)
    elif function_set == "korns":
        korns(pset)

    # Set evaluate and select functions based on whether tournament or lexicase
    if selection == "tournament":
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox.register("evaluate", evalTournSymbReg, toolbox=toolbox, X=X, y=y)
        toolbox.register("select", tools.selTournament, tournsize=3)
    else:
        # Create individuals with a minimizing fitness (need weights to be same length as fitness data)
        creator.create("FitnessMin", base.Fitness, avalue=1, weights=(-1.0,) * int(sample_size))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox.register("evaluate", evalSymbReg, toolbox=toolbox, X=X, y=y)
        toolbox.register("select", tools.selAutomaticEpsilonLexicase)

    # Register parameters specific to the evolution process
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Optionally limit expression tree depth and size
    if limitDepth is not None:
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limitDepth))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limitDepth))
    if limitSize is not None:
        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limitSize))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limitSize))


########################################################################################################################
# EVALUATION FUNCTIONS
########################################################################################################################
def evalRealSymbReg(individual, samples, toolbox, X, y):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    sqerrors = [(func(x1, x2) - y) ** 2 for x1, x2 in zip(X[0], X[1])]
    return tuple(sqerrors)


def evalSymbReg(individual, toolbox, X, y):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    sqerrors = [(func(*x) - y(*x)) ** 2 for x in X]
    return tuple(sqerrors)


def evalTournSymbReg(individual, toolbox, X, y):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    sqerrors = [(func(*x) - y(*x)) ** 2 for x in X]

    return math.fsum(sqerrors) / len(X),


########################################################################################################################
# DISPLAY FUNCTIONS
########################################################################################################################
def dispConfig(benchmark, func_set, selection, sample_size=None, data_min=None, data_max=None, data_size=None, num_features=None):
    print("########## RUN PARAMETERS ##########")
    print("Benchmark: " + str(benchmark))
    print("Function Set: " + str(func_set))
    print("Selection: " + str(selection))
    if sample_size is not None: print("Sample size: " + str(100*(sample_size/data_size)) + "%")
    if num_features is not None: print("# Features: " + str(num_features))
    if data_min is not None: print("Data: [" + str(data_min) + ", " + str(data_max) + ", " + str(data_size) + "]")


def dispHallOfFame(hof):
    print("HALL OF FAME ")
    counter = 0
    for h in hof:
        print(str(counter) + " " + str(h))
        counter += 1


def plotData(logbook):
    """Plot data from log

    :param log:
    :return:
    """
    fit_mins = logbook.select("min")
    fit_meds = logbook.select("med")
    gen = logbook.select("gen")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Min Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax1.plot(gen, fit_meds, "g-", label="Median Fitness")
    ax2.set_ylabel("Med Fitness", color="g")
    for tl in ax2.get_yticklabels():
        tl.set_color("g")

    lns = line1 + line2
    # lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()
