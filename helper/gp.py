import logging
import operator
import math
import random

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.benchmarks import gp as benchmark

import matplotlib.pyplot as plt


# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def absLog(n):
    try:
        return math.log(math.fabs(n))
    except ValueError:
        return 1

def protectedExp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return 1000000


def initGP(toolbox, X, y, num_features, sample_size=None, limitDepth=None, limitSize=None, realData=False):
    """
    :param toolbox: toolbox used to init gp
    :param X: independent variable(s)
    :param y: dependent variables
    :param num_features: number of independent variables in X
    :param sample_size: number of samples to use
    :param limitDepth: optionally set a limit on the max expression tree depth
    :param limitSize: optionally set a limit on the max expression tree size
    :param realData:
    :return: none
    """
    # Koza Set: +, -, *, %, sin, cos, exp, ln(|x|)
    pset = gp.PrimitiveSet("MAIN", 2)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(protectedExp, 1)
    pset.addPrimitive(absLog, 1)
    pset.renameArguments(ARG0='x')
    pset.renameArguments(ARG1='y')

    # Need weights to be same length as fitness data
    if sample_size is not None:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * int(sample_size))
    else:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * len(X[0]))

    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("evaluate", evalSymbReg, toolbox=toolbox, y=y, X=X)
    toolbox.register("select", tools.selAutomaticEpsilonLexicase)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    if limitDepth is not None:
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limitDepth))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limitDepth))
    if limitSize is not None:
        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limitSize))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limitSize))


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
    # TODO: ALOW FOR N-FEATURES BY PASSING X AS AN ARG LIST OR SOMETHING
    # sqerrors = [(func(*X[i]) - y[i]) ** 2 for i in range(len(X))]
    # sqerrors = [(func(x1, x2) - y([x1, x2])) ** 2 for x1, x2 in zip(X[0], X[1])]
    sqerrors = [(func(x1, x2) - y(x1, x2)) ** 2 for x1, x2 in zip(X[0], X[1])]
    # TODO: USE TO DEMONSTRATE DOWNSAMPLING
    # print("Datapoints = " + str(X))
    # print("Errors " + str(sqerrors))
    return tuple(sqerrors)


def lexAvg(tuples):
    avgs = []
    for t in tuples:
        avgs.append(np.asarray(t).sum() / np.asarray(t).size)
    return np.mean(np.asarray(avgs))


def lexMed(tuples):
    meds = []
    for t in tuples:
        meds.append(np.asarray(t).sum() / np.asarray(t).size)
    return np.median(np.asarray(meds))


def lexMin(tuples):
    avgs = []
    for t in tuples:
        avgs.append(np.asarray(t).sum() / np.asarray(t).size)
    return np.min(np.asarray(avgs))


def dispHallOfFame(hof):
    print("HALL OF FAME ")
    counter = 0
    for h in hof:
        print(str(counter) + " " + str(h))
        counter += 1


def plotData(logbook, selection="tourn"):
    """Plot data from log

    :param log:
    :return:
    """
    gen = logbook.select("gen")
    if selection == "tourn":
        fit_mins = logbook.chapters["fitness"].select("min")
        fit = logbook.chapters["fitness"].select("avg")
        size_avgs = logbook.chapters["size"].select("size avg")

    elif selection == "elex":
        fit_mins = logbook.chapters["fitness"].select("lex min")
        fit_meds = logbook.chapters["fitness"].select("lex med")
        size_avgs = logbook.chapters["size"].select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    line2 = ax1.plot(gen, fit_meds, "g-", label="Median Fitness")

    ax3 = ax1.twinx()
    line3 = ax3.plot(gen, size_avgs, "r-", label="Average Size")
    ax3.set_ylabel("Size", color="r")
    for tl in ax3.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()
