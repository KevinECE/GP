from benchmark.myhelper import *
import pandas as pd

DATA_MAX = 50
DATA_MIN = -50
# STEP_SIZE = 0.4
DATA_SIZE = 1000
NUM_FEATURES = 5

SAMPLE_SIZE = DATA_SIZE // 20


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

# Pagie 1 Function
def korns7(x0, x1, x2, x3, x4):
    return 213.80940889 * (1 - math.exp(-0.54723748542 * x0))

def test(toolbox, ind, X, y):
    func = toolbox.compile(expr=ind)
    sqerrors = [(func(*x) - y(*x)) ** 2 for x in X]
    df_log = pd.DataFrame(sqerrors)
    df_log.to_csv('..\hoftest.csv', index=False)
    print(np.mean(sqerrors))

def main():
    random.seed()
    toolbox = base.Toolbox()

    # DATAa
    X = np.transpose(
        [np.around(np.random.uniform(DATA_MIN, DATA_MAX, int(DATA_SIZE)), 1).tolist() for row in range(NUM_FEATURES)])
    print(X)
    # print("Sample size " + str(SAMPLE_SIZE))
    y = korns7

    # GP
    initGP(toolbox, X, y, num_features=NUM_FEATURES, sample_size=SAMPLE_SIZE, limitDepth=10, limitSize=15)

    # POP
    pop = toolbox.population(n=1000)

    # STATS
    hof = tools.HallOfFame(5)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.avalue)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    stats_fit.register("med", np.median, axis=0)
    stats_fit.register("min", np.min, axis=0)
    stats_size.register("size avg", np.mean)


    # RUN
    pop, logbook = algorithms.gpDownsample(pop, toolbox, X, y, SAMPLE_SIZE, 0.7, 0.3, 50, stats=stats_fit,
                                           halloffame=hof, verbose=True)
    # Display hall of fame
    dispHallOfFame(hof)
    for h in hof:
        test(toolbox, h, X, y)

    # Log data to csv
    df_log = pd.DataFrame(logbook)
    df_log.to_csv('..\data.csv', index=False)

    # Plot data
    plotData(logbook)


if __name__ == "__main__":
    main()