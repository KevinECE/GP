from helper.gp import *
import pandas as pd

DATA_MAX = 5
DATA_MIN = -5
STEP_SIZE = 0.4
DATA_SIZE = (DATA_MAX - DATA_MIN) / STEP_SIZE
SAMPLE_SIZE = 5


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


def main():
    random.seed()
    toolbox = base.Toolbox()

    # DATA
    X = [np.around(np.linspace(DATA_MIN, DATA_MAX, int(DATA_SIZE)), 1).tolist() for row in range(2)]
    # print(X)
    # print("Sample size " + str(SAMPLE_SIZE))
    y = pagie1

    # GP
    initGP(toolbox, X, y, num_features=2, sample_size=SAMPLE_SIZE, limitDepth=10, limitSize=15)

    # POP
    pop = toolbox.population(n=1000)

    # STATS
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    stats_fit.register("lex med", lexMed)
    stats_fit.register("lex min", lexMin)
    stats_size.register("size avg", np.mean)


    # RUN
    pop, logbook = algorithms.gpDownsample(pop, toolbox, X, y, SAMPLE_SIZE, 0.7, 0.3, 50, stats=stats_fit,
                                           halloffame=hof, verbose=True)

    # Display hall of fame
    dispHallOfFame(hof)
    # Plot data
    # plotData(logbook, selection="elex")
    df_log = pd.DataFrame(logbook)
    df_log.to_csv('..\data.csv', index=False)


if __name__ == "__main__":
    main()
