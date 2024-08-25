import numpy as np
from numpy.linalg import det
from typing import List, Union

def kroneckerdelta(i: int, j: int) -> int:
    """
    Return the value of the Kronecker delta function δ(i, j).

    :param i: First index.
    :type i: int
    :param j: Second index.
    :type j: int
    :return: Value of the Kronecker delta function δ(i, j).
    :rtype: int
    """
    if i == j:
        return 1
    else:
        return 0

def crct(*args: Union[int, List[int]]) -> Union[int, List[int]]:
    """
    Convert indices from 1-based to 0-based.

    :param args: One or more indices.
    :type args: Union[int, List[int]]
    :return: Converted indices (0-based).
    :rtype: Union[int, List[int]]
    """
    return [ind - 1 for ind in args] if len(args) > 1 else args[0] - 1

def submatrix(M: np.ndarray, i: int) -> np.matrix:
    """
    Return the submatrix of M corresponding to the specified neutrino flavor i, obtained by deleting the i-th row and column.

    :param M: Matrix.
    :type M: np.ndarray
    :param i: Index of the row and column to delete.
    :type i: int
    :return: Submatrix obtained by deleting the i-th row and column.
    :rtype: np.matrix
    """
    i = crct(i)

    A = M[:]
    A = np.delete(A, i, axis=0)
    A = np.delete(A, i, axis=1)

    return np.matrix(A)

def adjugate(M: np.ndarray) -> np.ndarray:
    """
    Compute the adjugate of a matrix M.

    :param M: Matrix.
    :type M: np.ndarray
    :return: Adjugate of the matrix M.
    :rtype: np.ndarray
    :raises Exception: If the matrix M is singular (has determinant equal to zero).
    """
    M = M.astype(np.csingle)
    detM = det(M)
    
    if detM != 0:
        cf = np.linalg.inv(M).T * detM
        adj = cf.T
        return adj
    else:
        raise Exception("Singular matrix")