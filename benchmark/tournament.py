from benchmark.myhelper import *
import pandas as pd
import random

from deap import algorithms
from benchmarks import *

DATA_MIN = -1
DATA_MAX = 1
DATA_SIZE = 20
NUM_FEATURES = 1


# Test HOF individuals
def test(toolbox, ind, X, y):
    func = toolbox.compile(expr=ind)
    sqerrors = [(func(*x) - y(*x)) ** 2 for x in X]
    df_log = pd.DataFrame(sqerrors)
    df_log.to_csv('..\hoftest.csv', index=False)
    print(np.mean(sqerrors))


# Initialize stats objectc
def stats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    stats_fit.register("med", np.median)
    # mstats.register("std", np.std)
    stats_fit.register("min", np.min)
    stats_size.register("avg", np.mean)
    return stats_fit


def main():
    # INIT
    random.seed()
    toolbox = base.Toolbox()

    # DATA
    X = np.transpose(
        [np.around(np.random.uniform(DATA_MIN, DATA_MAX, int(DATA_SIZE)), 1).tolist() for row in range(NUM_FEATURES)])
    y = koza2

    # GP
    initGP(toolbox, X, y, num_features=NUM_FEATURES, function_set="Koza", selection="Tournament",
           limitDepth=10, limitSize=15)

    # POP
    pop = toolbox.population(n=1000)

    # HOF
    hof = tools.HallOfFame(5)

    # RUN
    pop, logbook = algorithms.eaSimple(pop, toolbox, 0.9, 0.1, 100, stats=stats(),
                                   halloffame=hof, verbose=True)

    # Display hall of fame
    dispHallOfFame(hof)

    for h in hof:
        test(toolbox, h, X, y)

    # PLOT
    plotData(logbook)


if __name__ == "__main__":
    main()