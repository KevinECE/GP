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
    pset.addPrimitive(protectedMult, 2)
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
def protectedMult(left, right):
    res = left * right
    if np.isnan(res):
        return np.inf
    return res


def protectedSub(left, right):
    res = np.subtract(left, right)
    if np.isnan(res):
        return np.inf
    return res


def aqDiv(left, right):
    res = np.divide(left, np.sqrt(1 + right ** 2))
    if np.isnan(res):
        return np.inf
    return res


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
    res = np.sin(x)
    if np.isnan(res):
        return np.inf
    return res


def protectedCos(x):
    res = np.cos(x)
    if np.isnan(res):
        return np.inf
    return res


def protectedTan(x):
    res = np.tan(x)
    if np.isnan(res):
        return np.inf
    return res


def protectedTanh(x):
    res = np.tanh(x)
    if np.isnan(res):
        return np.inf
    return res
