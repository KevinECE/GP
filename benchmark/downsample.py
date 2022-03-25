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
datasets = ['594_fri_c2_100_5']
X, y = fetch_data('594_fri_c2_100_5', return_X_y=True)
train_X, test_X, train_y, test_y = train_test_split(X, y)

########################################################################################################################
# DATA PARAMS
########################################################################################################################
DATA_MIN = np.min(train_X)
DATA_MAX = np.max(train_X)
DATA_SIZE = len(train_X)
NUM_FEATURES = len(X[0])

BENCHMARK = benchmarks.korns1
FUNCTION_SET = FuncSet.PMLB
SELECTION = Select.LEX

SAMPLE_SIZE = int(DATA_SIZE * .15)

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

    # GP
    initGP(toolbox, train_X, train_y,
           num_features=NUM_FEATURES,
           sample_size=SAMPLE_SIZE,
           function_set=FUNCTION_SET,
           selection=SELECTION,
           limitDepth=None, limitSize=None)

    dispConfig(BENCHMARK,
               FUNCTION_SET,
               SELECTION,
               SAMPLE_SIZE,
               DATA_MIN,
               DATA_MAX,
               DATA_SIZE,
               NUM_FEATURES)

    # POP
    pop = toolbox.population(n=POP_SIZE)

    # HOF
    hof = tools.HallOfFame(1)

    # RUN
    pop, logbook = algorithms.gpDownsample(pop, toolbox, train_X, train_y, SAMPLE_SIZE, CXPB, MUTPB, NUM_GENS,
                                           stats=stats(), halloffame=hof, verbose=True)



    # Display hall of fame
    dispHallOfFame(hof)
    for h in hof:
        test(toolbox, h, test_X, test_y)
    #
    # # Log data to csv
    # df_log = pd.DataFrame(logbook)
    # df_log.to_csv('..\data\data' + str(int(SAMPLE_SIZE)) +'.csv', index=False)

    # PLOT
    plotData(logbook, SAMPLE_SIZE/DATA_SIZE)


if __name__ == "__main__":
    main()
