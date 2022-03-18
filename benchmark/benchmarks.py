########################################################################################################################
# BENCHMARK FUNCTIONS
########################################################################################################################
import numpy as np


def koza2(x):
    """Koza-2 benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\mathbf{x} \in U[-1, 1, 20]`
       * - Function
         - :math:`f(\mathbf{x}) = x^{5} - 2x^{3} + x`
    """
    return x ** 5 - 2 * x ** 3 + x


def nguyen10(x, y):
    """Nguyen-10 benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\mathbf{x} \in U[0, 1, 20]`
       * - Function
         - :math:`f(\mathbf{x}) = 2sin(x)cos(y)`
    """
    return 2 * np.sin(x) * np.cos(y)


def korns1(v):
    """Korns-1 benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\mathbf{x} \in U[-50, 50, 10000]`
       * - Function
         - :math:`f(\mathbf{x}) = 1.57 + 24.3v`
    """
    return 1.57 + 24.3 * v


def keijzer14(x, y):
    """Keijzer-14 benchmark function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Range
         - :math:`\mathbf{x} \in U[-3, 3, 20]`
       * - Function
         - :math:`f(\mathbf{x}) = 8 / (2 + x^{2} + y^{2})`
    """
    return 8 / (2 + x**2 + y**2)


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
    return 213.80940889 * (1 - np.exp(-0.54723748542 * x0))