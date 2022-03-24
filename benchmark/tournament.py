from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from benchmark.myhelper import *
import pandas as pd
import random
from deap import algorithms
import benchmarks
from warnings import filterwarnings

filterwarnings("ignore")

########################################################################################################################
# DATA
########################################################################################################################
# X = np.transpose(
#     [np.around(np.random.uniform(DATA_MIN, DATA_MAX, int(DATA_SIZE)), 1).tolist() for row in range(NUM_FEATURES)])
# y = BENCHMARK
X, y = fetch_data('594_fri_c2_100_5', return_X_y=True)
train_X, test_X, train_y, test_y = train_test_split(X, y)

########################################################################################################################
# DATA PARAMS
# ########################################################################################################################
# DATA_MIN = -50
# DATA_MAX = 50
DATA_SIZE = train_X
NUM_FEATURES = len(X[0])

BENCHMARK = benchmarks.koza2
FUNCTION_SET = FuncSet.PMLB
SELECTION = Select.TOURN

########################################################################################################################
# GP PARAMS
########################################################################################################################
POP_SIZE = 1000
NUM_GENS = 100
CXPB = 0.9
MUTPB = 0.1

# Test HOF individuals
def test(toolbox, ind, X, y):
    func = toolbox.compile(expr=ind)
    sqerrors = [(func(*x) - y[i]) ** 2 for i, x in enumerate(X)]
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

    # GP
    initGP(toolbox, train_X, train_y,
           num_features=NUM_FEATURES,
           function_set=FUNCTION_SET,
           selection=SELECTION,
           limitDepth=10, limitSize=15)

    # dispConfig(BENCHMARK,
    #            FUNCTION_SET,
    #            SELECTION,
    #            DATA_MIN,
    #            DATA_MAX,
    #            DATA_SIZE,
    #            NUM_FEATURES)

    # POP
    pop = toolbox.population(n=1000)

    # HOF
    hof = tools.HallOfFame(5)

    # START TIME
    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))

    # # RUN
    pop, logbook = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, NUM_GENS, stats=stats(),
                                   halloffame=hof, verbose=True)
    # STOP TIME
    print("--- %s seconds ---" % (time.time() - start_time))

    # Display hall of fame
    dispHallOfFame(hof)

    for h in hof:
        test(toolbox, h, test_X, test_y)

    # PLOT
    plotData(logbook)


if __name__ == "__main__":
    main()