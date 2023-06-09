import numpy as np
from numpy.linalg import det

def kroneckerdelta(i, j):
    if i == j:
        return 1
    else:
        return 0

def crct(*args): # index_correction
    return [ind - 1 for ind in args] if len(args) > 1 else args[0] - 1

def submatrix(M, i):
    i = crct(i)

    A = M[:]
    A = np.delete(A, i, axis = 0)
    A = np.delete(A, i, axis = 1)

    return np.matrix(A)

def adjugate(M):
    M = M.astype(np.csingle)
    detM = det(M)
    
    if detM != 0:
        cf = np.linalg.inv(M).T * detM
        adj = cf.T
        return adj
    else:
        raise Exception("Singular matrix")
