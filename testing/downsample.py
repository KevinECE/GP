from helper.gp import *

DATA_SIZE = 50
SAMPLE_SIZE = 50


def main():
    random.seed(318)
    toolbox = base.Toolbox()

    # DATA
    X = [np.linspace(-5, 5, DATA_SIZE).tolist() for row in range(2)]
    # X = [np.linspace(-5, 5, DATA_SIZE).tolist() for row in range(2)]
    # y = benchmark.sin_cos
    y = benchmark.ripple
    # from RegressionData import train_X, train_y
    # X = train_X[:10]
    # y = train_y[:10]

    # GP
    initGP(toolbox, X, y, num_features=2, sample_size=SAMPLE_SIZE, limitDepth=10, limitSize=15)

    # POP
    # TODO: USE TO DEMONSTRATE DOWNSAMPLING
    pop = toolbox.population(n=500)
    # pop = toolbox.population(n=500)

    # STATS
    hof = tools.HallOfFame(5)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    stats_fit.register("lex med", lexMed)
    stats_fit.register("lex min", lexMin)
    stats_size.register("avg", np.mean)
    # mstats.register("avg", np.mean)
    # mstats.register("std", np.std)
    # mstats.register("min", np.min)
    # mstats.register("max", np.max)

    # RUN
    pop, logbook = algorithms.gpDownsample(pop, toolbox, X, y, SAMPLE_SIZE, 0.7, 0.3, 50, stats=mstats,
                                       halloffame=hof, verbose=True)

    # Display hall of fame
    dispHallOfFame(hof)
    # Plot data
    plotData(logbook, selection="elex")


if __name__ == "__main__":
    main()
