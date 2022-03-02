#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

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


pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
pset.renameArguments(ARG0='x1')
pset.renameArguments(ARG1='x2')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual, points, realFunc):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # sqerrors = ((func(x[0], x[1]) - realFunc(x)) ** 2 for x in points)
    sqerrors = [(func(x1, x2) - realFunc([x1,  x2])) ** 2 for x1, x2 in zip(points[0], points[1])]

    return math.fsum(sqerrors) / len(points[0]),


# toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
toolbox.register("evaluate", evalSymbReg, points=[np.linspace(0, 6, 50).tolist() for row in range(2)],
                 realFunc=benchmark.sin_cos)
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
    fit_avgs = logbook.chapters["fitness"].select("med")
    size_avgs = logbook.chapters["size"].select("avg")
    # fit_avgs = list(filter(lambda x: x>4000, fit_avgs))

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, fit_avgs, "g-", label="Median Fitness")
    ax2.set_ylabel("Fitness", color="g")
    for tl in ax2.get_yticklabels():
        tl.set_color("g")

    ax3 = ax1.twinx()
    line3 = ax3.plot(gen, size_avgs, "r-", label="Average Size")
    ax3.set_ylabel("Size", color="r")
    for tl in ax3.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    # Cutoff extremely large data
    ax2.set_ylim(0, 10000)

    plt.show()


def main():
    random.seed(318)

    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(5)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("med", np.median)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.3, 50, stats=mstats,
                                   halloffame=hof, verbose=True)

    print("HALL OF FAME " + str(hof[0]))

    # Plot data
    plotData(log)

    return pop, log, hof


if __name__ == "__main__":
    main()
