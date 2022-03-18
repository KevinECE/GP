import numpy as np
import operator
import random


########################################################################################################################
# GP FUNCTION SETS
########################################################################################################################
def koza(pset):
    """Koza Set: +, -, *, %, sin, cos, exp, ln(|x|)

    :param pset: deap gp problem set object used to add primitives to th emodel
    """

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(aqDiv, 2)
    pset.addPrimitive(protectedSin, 1)
    pset.addPrimitive(protectedCos, 1)
    pset.addPrimitive(protectedExp, 1)
    pset.addPrimitive(absLog, 1)

    pset.renameArguments(ARG0='x')
    pset.renameArguments(ARG1='y')


def korns(pset):
    """# Korns Set: +, -, *, %, sin, cos, exp, ln(|x|)

    :param pset: deap gp problem set object used to add primitives to th emodel
    """

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(aqDiv, 2)
    pset.addPrimitive(protectedSin, 1)
    pset.addPrimitive(protectedCos, 1)
    pset.addPrimitive(protectedExp, 1)
    pset.addPrimitive(absLog, 1)
    pset.addPrimitive(square, 1)
    pset.addPrimitive(cube, 1)
    pset.addPrimitive(protectedTan, 1)
    pset.addPrimitive(protectedTanh, 1)
    pset.addPrimitive(protectedSqrt, 1)
    pset.addEphemeralConstant("rand64double", lambda: random.uniform(-1000, 1000))

    pset.renameArguments(ARG0='x')
    pset.renameArguments(ARG1='y')
    pset.renameArguments(ARG2='z')
    pset.renameArguments(ARG3='v')
    pset.renameArguments(ARG4='w')


def keijzer(pset):
    """# Korns Set: +, *, %, sin, cos, exp, ln(|x|)

    :param pset: deap gp problem set object used to add primitives to th emodel
    """
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(operator.po, 2)
    pset.addPrimitive(protectedSqrt, 1)

########################################################################################################################
# GP FUNCTIONS
########################################################################################################################
def aqDiv(left, right):
    if left == np.inf and right == np.inf:
        return 1
    else:
        return left / np.sqrt(1 + right ** 2)


def absLog(x):
    return np.log(1 + np.fabs(x))


def protectedExp(x):
    return np.exp(x)


def square(x):
    return x ** 2


def cube(x):
    return x ** 3


def protectedSqrt(x):
    return np.sqrt(np.fabs(x))


def protectedSin(x):
    if x == float('inf'):
        return float('inf')
    else:
        return np.sin(x)


def protectedCos(x):
    if x == float('inf'):
        return float('inf')
    else:
        return np.cos(x)


def protectedTan(x):
    if x == float('inf'):
        return float('inf')
    else:
        return np.tan(x)


def protectedTanh(x):
    if x == float('inf'):
        return float('inf')
    else:
        return np.tanh(x)
