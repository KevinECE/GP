from myhelper import *
import pandas as pd
import random
import time

from deap import algorithms
from benchmarks import *

from warnings import filterwarnings
filterwarnings("ignore")

DATA_MIN = -50
DATA_MAX = 50
DATA_SIZE = 100
NUM_FEATURES = 5

BENCHMARK = korns1
FUNCTION_SET = "korns"
SELECTION = "lexicase"

SAMPLE_SIZE = int(DATA_SIZE * .5)


# Test HOF individuals
def test(toolbox, ind, X, y):
    func = toolbox.compile(expr=ind)
    sqerrors = [(func(*x) - y(*x)) ** 2 for x in X]
    df_log = pd.DataFrame(sqerrors)
    df_log.to_csv('..\hoftest.csv', index=False)
    print(np.mean(sqerrors))


# Initialize stats object
def stats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.avalue)
    stats_size = tools.Statistics(len)
    stats_fit.register("med", np.median, axis=0)
    stats_fit.register("min", np.min, axis=0)
    stats_fit.register("max", np.max, axis=0)
    stats_size.register("size avg", np.mean)
    return stats_fit


def main():
    # INIT
    random.seed()
    toolbox = base.Toolbox()

    # DATA
    X = np.transpose(
        [np.around(np.random.uniform(DATA_MIN, DATA_MAX, int(DATA_SIZE)), 1).tolist() for row in range(NUM_FEATURES)])
    y = BENCHMARK

    # GP
    initGP(toolbox, X, y,
           num_features=NUM_FEATURES,
           sample_size=SAMPLE_SIZE,
           function_set=FUNCTION_SET,
           selection=SELECTION,
           limitDepth=10, limitSize=15)

    dispConfig(BENCHMARK,
               FUNCTION_SET,
               SELECTION,
               SAMPLE_SIZE,
               DATA_MIN,
               DATA_MAX,
               DATA_SIZE,
               NUM_FEATURES)


    # POP
    pop = toolbox.population(n=1000)

    # HOF
    hof = tools.HallOfFame(1)

    # START TIME
    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

    # RUN
    pop, logbook = algorithms.gpDownsample(pop, toolbox, X, y, SAMPLE_SIZE, 0.9, 0.1, 100, stats=stats(),
                                           halloffame=hof, verbose=True)

    # STOP TIME
    print("--- %s seconds ---" % (time.time() - start_time))

    # Display hall of fame
    dispHallOfFame(hof)
    for h in hof:
        test(toolbox, h, X, y)

    # Log data to csv
    # df_log = pd.DataFrame(logbook)
    # df_log.to_csv('..\data.csv', index=False)

    # PLOT
    plotData(logbook)


if __name__ == "__main__":
    main()
