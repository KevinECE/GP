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
import pandas as pd

DATA_MAX = 5
DATA_MIN = -5
STEP_SIZE = 0.4
DATA_SIZE = (DATA_MAX - DATA_MIN) / STEP_SIZE


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

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# Pagie 1 Function
def pagie1(x, y):
    if x == 0 and y == 0:
        return 0
    elif x == 0:
        return 1 / (1 + pow(y, -4))
    elif y == 0:
        return 1 / (1 + pow(x, -4))
    else:
        return 1 / (1 + pow(x, -4)) + 1 / (1 + pow(y, -4))


def evalSymbReg(individual, points, realFunc):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # sqerrors = ((func(x[0], x[1]) - realFunc(x)) ** 2 for x in points)
    sqerrors = [(func(x1, x2) - realFunc(x1,  x2)) ** 2 for x1, x2 in zip(points[0], points[1])]

    return math.fsum(sqerrors) / len(points[0]),


# toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
toolbox.register("evaluate", evalSymbReg, points=[np.around(np.linspace(DATA_MIN, DATA_MAX, int(DATA_SIZE)), 1).tolist() for row in range(2)],
                 realFunc=pagie1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

limitHeight = 10
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limitHeight))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=limitHeight))

limitLength = 15
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=limitLength))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=limitLength))


def plotData(logbook):
    """Plot data from log

    :param log:
    :return:
    """
    gen = logbook.select("gen")
    fit_mins = logbook.chapters["fitness"].select("min")
    fit_meds = logbook.chapters["fitness"].select("med")
    size_avgs = logbook.chapters["size"].select("avg")
    # fit_meds = list(filter(lambda x: x>4000, fit_meds))

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    line2 = ax1.plot(gen, fit_meds, "g-", label="Median Fitness")

    # ax3 = ax1.twinx()
    # line3 = ax3.plot(gen, size_avgs, "r-", label="Average Size")
    # ax3.set_ylabel("Size", color="r")
    # for tl in ax3.get_yticklabels():
    #     tl.set_color("r")

    # lns = line1 + line2 + line3
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()


def main():
    random.seed()

    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(5)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    stats_fit.register("med", np.median)
    # mstats.register("std", np.std)
    stats_fit.register("min", np.min)
    stats_size.register("avg", np.mean)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.3, 50, stats=stats_fit,
                                   halloffame=hof, verbose=True)

    print("HALL OF FAME " + str(hof[0]))

    df_log = pd.DataFrame(log)
    # df_log.to_csv('..\data.csv', index=False)

    # Plot data
    plotData(log)

    return pop, log, hof


if __name__ == "__main__":
    main()
