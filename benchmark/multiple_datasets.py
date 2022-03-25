from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from myhelper import *
import random
from deap import algorithms
from warnings import filterwarnings

filterwarnings("ignore")

########################################################################################################################
# DATA
########################################################################################################################
datasets = ['594_fri_c2_100_5', '644_fri_c4_250_25', '210_cloud']
########################################################################################################################
# GP PARAMS
########################################################################################################################
POP_SIZE = 100
NUM_GENS = 5
CXPB = 0.9
MUTPB = 0.1


# Test HOF individuals
def test(toolbox, ind, X, y):
    func = toolbox.compile(expr=ind)
    sqerrors = [np.sqrt((func(*x) - y[i]) ** 2) for i, x in enumerate(X)]
    # df_log = pd.DataFrame(sqerrors)
    # df_log.to_csv('..\hoftest.csv', index=False)
    print(np.mean(sqerrors))


# Initialize stats object
def stats():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.aggregate)
    stats_size = tools.Statistics(len)
    stats_fit.register("med", np.median, axis=0)
    stats_fit.register("min", np.min, axis=0)
    stats_fit.register("max", np.max, axis=0)
    stats_size.register("size avg", np.mean)
    return stats_fit


# Display dataset and other info
def display(dataset, func_set, selection, sample_size=None, data_size=None, num_features=None):
    """ Display parameters for next run

    :param dataset:
    :param func_set:
    :param selection:
    :param sample_size:
    :param data_size:
    :param num_features:
    :return:
    """
    print("########## RUN PARAMETERS ##########")
    print("Dataset: " + str(dataset))
    print("Function Set: " + str(func_set))
    print("Selection: " + str(selection))
    print("Sample size: " + str((100 * sample_size) // data_size) + "%")
    print("Data size: " + str(data_size))
    print("# Features: " + str(num_features))
    print("####################################")


def main():
    # INIT
    random.seed()
    toolbox = base.Toolbox()

    # GP
    # Register toolbox functions that are independent of pset
    toolbox.register("evaluate", evalRealSymbReg, toolbox=toolbox)
    toolbox.register("select", tools.selAutomaticEpsilonLexicase)

    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

    # HOF
    hof = tools.HallOfFame(3)

    ####################################################################################################################
    # MAIN LOOP
    ####################################################################################################################
    # Easier to run GP with different datasets and then change GP config
    for data in datasets:

        # Get data
        X, y = fetch_data(data, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y)
        # train_X = [[1, 2 ,3 ,4], [1, 2 ,3, 4]]
        # train_y = [4, 3, 2, 1]
        num_features = len(train_X[0])
        SAMPLE_SIZE = int(len(train_X) * 1)

        # Display info
        display(data, "PMLB", Select.LEX, SAMPLE_SIZE, data_size=len(train_X), num_features=len(train_X[0]))

        # Create new fitness class for sample size
        creator.create("FitnessMin", base.Fitness, aggregate=1, weights=(-1.0,) * int(SAMPLE_SIZE))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        # Register new pset
        pset = gp.PrimitiveSet("MAIN", num_features)
        pmlb(pset, num_features)

        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=2)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        # POP
        pop = toolbox.population(n=POP_SIZE)

        # RUN
        pop, logbook = algorithms.gpDownsample(pop, toolbox, train_X, train_y, SAMPLE_SIZE, CXPB, MUTPB, NUM_GENS,
                                               stats=stats(), halloffame=hof, verbose=True)

        # HOF
        for h in hof:
            test(toolbox, h, test_X, test_y)

        # PLOT
        plotData(logbook, SAMPLE_SIZE, len(train_X))

        # Delete old Fitness class with old sample size
        del creator.FitnessMin

        # Delete old pset and unregister toolbox functions that depend on it
        del pset
        toolbox.unregister("expr")
        toolbox.unregister("individual")
        toolbox.unregister("population")
        toolbox.unregister("compile")
        toolbox.unregister("mutate")


if __name__ == "__main__":
    main()
