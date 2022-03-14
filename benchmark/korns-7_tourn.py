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
import pandas as pd

from myhelper import plotData

DATA_MAX = 50
DATA_MIN = -50
# STEP_SIZE = 0.4
DATA_SIZE = 1000
NUM_FEATURES = 5

mydata = np.transpose([np.around(np.random.uniform(DATA_MIN, DATA_MAX, int(DATA_SIZE)), 1).tolist() for row in range(NUM_FEATURES)])


# Define new functions
def protectedDiv(left, right):
    if right == 0:
        return 1
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def absLog(n):
    try:
        return math.log(math.fabs(n))
    except ValueError or OverflowError:
        return 1


def protectedExp(x):
    if math.fabs(x) > 8:
        return 1
    else:
        return math.exp(x)


def square(x):
    try:
        return x**2
    except OverflowError:
        return 1


def cube(x):
    try:
        return x**3
    except OverflowError:
        return 1


def protectedSqrt(x):
    if x < 0:
        return 1
    else:
        return math.sqrt(x)


# Koza Set: +, -, *, %, sin, cos, exp, ln(|x|)
pset = gp.PrimitiveSet("MAIN", 5)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(protectedExp, 1)
pset.addPrimitive(absLog, 1)
pset.addPrimitive(square, 1)
pset.addPrimitive(cube, 1)
pset.addPrimitive(math.tan, 1)
pset.addPrimitive(math.tanh, 1)
pset.addPrimitive(protectedSqrt, 1)
pset.addEphemeralConstant("rand64double", lambda: random.uniform(-1000, 1000))


pset.renameArguments(ARG0='x0')
pset.renameArguments(ARG1='x1')
pset.renameArguments(ARG2='x2')
pset.renameArguments(ARG3='x3')
pset.renameArguments(ARG4='x4')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


# Pagie 1 Function
def korns7(x0, x1, x2, x3, x4):
    return 213.80940889 * (1 - math.exp(-0.54723748542 * x0))


def evalSymbReg(individual, X, y):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    sqerrors = [(func(*x) - y(*x)) ** 2 for x in X]

    return math.fsum(sqerrors) / len(X),


# toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
toolbox.register("evaluate", evalSymbReg, X=mydata, y=korns7)
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



def main():
    random.seed()
    print("My data = " + str(mydata))
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(5)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    stats_fit.register("med", np.median)
    # mstats.register("std", np.std)
    stats_fit.register("min", np.min)
    stats_size.register("avg", np.mean)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.3, 100, stats=stats_fit,
                                   halloffame=hof, verbose=True)

    print("HALL OF FAME " + str(hof[0]))

    df_log = pd.DataFrame(log)
    # df_log.to_csv('..\data.csv', index=False)

    # Plot data
    plotData(log)

    return pop, log, hof


if __name__ == "__main__":
    main()
