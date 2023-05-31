import numpy as np
from numpy import sin, cos
from numpy.linalg import eigvals, eig

from .utils import kroneckerdelta, product, crct, submatrix, adjugate

class MatSolver:
    def __init__(self, baseline, energy, V, delmsq21, delmsq31, deltacp, theta12, theta13, theta23):
        self.baseline = baseline
        self.energy = energy
        self.deltacp = deltacp
        self.theta12 = theta12
        self.theta13 = theta13
        self.theta23 = theta23
        self.V = V

        self.delmsq21 = delmsq21
        self.delmsq31 = delmsq31

        alpha = delmsq21 / delmsq31
        a = (2 * energy * V)/delmsq31

        self.O12 = O12 = np.matrix([[cos(theta12), sin(theta12), 0], [-sin(theta12), cos(theta12), 0], [0, 0, 1]])
        self.O13 = O13 = np.matrix([[cos(theta13), 0, sin(theta13)], [0, 1, 0], [-sin(theta13), 0, cos(theta13)]])
        self.O23 = O23 = np.matrix([[1, 0, 0], [0, cos(theta23), sin(theta23)], [0, -sin(theta23), cos(theta23)]])
        self.Udelta = Udelta = np.matrix(np.diag([1, 1, np.exp(1j * deltacp)]))

        self.M = M = O13 @ O12 @ np.diag([0, alpha, 1]) @ O12.T @ O13.T + np.diag([a, 0, 0])
        self.H = H = O23 @ Udelta @ M @ Udelta.H @ O23.T

    def Hevals(self):
        H = self.H
        return eigvals(H)
    
    def probabilities(self):
        labelmatrix = np.matrix([["Pee", "Pemu", "Petau"], ["Pmue", "Pmumu", "Pmutau"], ["Ptaue", "Ptaumu", "Ptautau"]])

        probmatrix = np.zeros((3, 3))

        for i in range(1, 4):
            for j in range(1, 4):
                label = labelmatrix[i - 1, j - 1]

                if i == j:
                    prob = self.disappearance(i)
                else:
                    prob = self.appearance(i, j)

                # print(f"{label}: {prob}")
                
                probmatrix[i - 1, j - 1] = prob

        return probmatrix

class Eigen(MatSolver):
    def __init__(self, *args):
        super().__init__(*args)

    def Hevecs(self):
        H = self.H
        return eig(H)[1]
    
    def evolution_matrix(self):
        lda1, lda2, lda3 = self.Hevals()
        evecs = self.Hevecs()

        L = self.baseline
        En = self.energy
        delmsq31 = self.delmsq31
        delta = (1.267 * delmsq31 * L)/En

        eigmat = np.matrix(evecs)

        S = eigmat @ np.diag([np.exp(-1j * lda1 * 2 * delta), np.exp(-1j * lda2 * 2 * delta), np.exp(-1j * lda3 * 2 * delta)]) @ eigmat.H
        return S
    
    def probability(self, i, j):
        i, j = crct(i, j)

        S = self.evolution_matrix()

        return np.abs(S[j, i])**2
    
    def disappearance(self, i):
        return self.probability(i, i)

    def appearance(self, i, j):
        return self.probability(i, j)
    
class CayleyHamilton(MatSolver):
    def __init__(self, *args):
        super().__init__(*args)
    
    def evolution_matrix(self):
        H = self.H
        lda1, lda2, lda3 = self.Hevals()

        L = self.baseline
        En = self.energy
        delmsq31 = self.delmsq31
        delta = (1.267 * delmsq31 * L)/En

        S = (np.exp(-1j * lda1 * 2 * delta)/((lda1 - lda2) * (lda1 - lda3))) * (lda2 * lda3 * np.identity(3) - (lda2 + lda3) * H + H @ H) +\
        (np.exp(-1j * lda2 * 2 * delta)/((lda2 - lda1) * (lda2 - lda3))) * (lda1 * lda3 * np.identity(3) - (lda1 + lda3) * H + H @ H) +\
        (np.exp(-1j * lda3 * 2 * delta)/((lda3 - lda1) * (lda3 - lda2))) * (lda1 * lda2 * np.identity(3) - (lda1 + lda2) * H + H @ H)

        return S
    
    def probability(self, i, j):
        i, j = crct(i, j)

        S = self.evolution_matrix()

        return np.abs(S[j, i])**2
    
    def disappearance(self, i):
        return self.probability(i, i)

    def appearance(self, i, j):
        return self.probability(i, j)

class Identities(MatSolver):
    def __init__(self, *args):
        super().__init__(*args)

    def submatrix_evals(self):
        H = self.H

        He = submatrix(H, 1)
        Hmu = submatrix(H, 2)
        Htau = submatrix(H, 3)

        Xie, Chie = eigvals(He)
        Ximu, Chimu = eigvals(Hmu)
        Xitau, Chitau = eigvals(Htau)

        return [[Xie, Chie], [Ximu, Chimu], [Xitau, Chitau]]
    
    def quartic_product(self, i, alpha, beta):
        i, alpha, beta = crct(i, alpha, beta)

        H = self.H
        lda = self.Hevals()

        adj = adjugate(lda[i] * np.identity(3) - H)[alpha, beta]
        denom = product([lda[i] - lda[k] if k != i else 1 for k in range(3)])

        qprod = adj / denom

        return qprod

    def PMNSmattermodsq(self, alpha, i):
        alpha, i = crct(alpha, i)

        lda = self.Hevals()
        subeigs = self.submatrix_evals()

        sublda = subeigs[alpha]

        num = product([lda[i] - sublda[j] for j in range(len(sublda))])
        denom = product([lda[i] - lda[k] if k != i else 1 for k in range(3)])

        modsq = num / denom

        return modsq
    
    def mat_deltamsq(self, k, j):
        k, j = crct(k, j)
        lda = self.Hevals()
        delmsq31 = self.delmsq31
        
        return delmsq31 * (lda[k] - lda[j])
    
    def mat_phase(self, k, j, n=1):
        k, j = crct(k, j)

        lda = self.Hevals()

        L = self.baseline
        En = self.energy
        delmsq31 = self.delmsq31
        delta = (1.267 * delmsq31 * L)/En

        return n * (lda[k] - lda[j]) * delta
    
    def disappearance(self, i):
        sum_terms = []
        for j in range(1, 4):
            for k in range(j + 1, 4):
                term = self.PMNSmattermodsq(i, k) * self.PMNSmattermodsq(i, j) * sin(self.mat_phase(k, j))**2
                sum_terms.append(term)

        return (1 - 4 * sum(sum_terms)).real
    
    def appearance(self, a, b):
        firstsum = []
        for j in range(1, 4):
            for k in range(j + 1, 4):
                term = (self.quartic_product(k, b, a) * self.quartic_product(j, a, b)).real * sin(self.mat_phase(k, j))**2
                firstsum.append(term)
        firstsum = sum(firstsum)

        secondsum = []
        for j in range(1, 4):
            for k in range(j + 1, 4):
                term = (self.quartic_product(k, b, a) * self.quartic_product(j, a, b)).imag * sin(self.mat_phase(k, j, 2))
                secondsum.append(term)
        secondsum = sum(secondsum)

        return (kroneckerdelta(a, b) - 4 * firstsum + 2 * secondsum).real  
    
    def mat_mixing_angles(self):
        s13matsq = self.PMNSmattermodsq(1, 3).real
        c13matsq = 1 - s13matsq

        s12matsq = self.PMNSmattermodsq(1, 2).real / c13matsq
        s23matsq = self.PMNSmattermodsq(2, 3).real / c13matsq

        s12mat = np.sqrt(s12matsq)
        s13mat = np.sqrt(s13matsq)
        s23mat = np.sqrt(s23matsq)

        theta12mat = np.arcsin(s12mat)
        theta13mat = np.arcsin(s13mat)
        theta23mat = np.arcsin(s23mat)

        return theta12mat, theta13mat, theta23mat
    
    def toshev_identity(self):
        _, _, theta23mat = self.mat_mixing_angles()

        deltacp = self.deltacp
        theta23 = self.theta23

        matsindeltacp = (sin(2 * theta23)/sin(2 * theta23mat)) * sin(deltacp)
        matdeltacp = np.arcsin(matsindeltacp)

        return matdeltacp

    def mat_angles_phase(self):
        theta12mat, theta13mat, theta23mat = self.mat_mixing_angles()
        matdeltacp = self.toshev_identity()

        return matdeltacp, theta12mat, theta13mat, theta23mat
    
    def PMNSmat(self):
        deltacp, theta12, theta13, theta23 = self.mat_angles_phase()

        O12 = np.matrix([[cos(theta12), sin(theta12), 0], [-sin(theta12), cos(theta12), 0], [0, 0, 1]])
        O13 = np.matrix([[cos(theta13), 0, sin(theta13)], [0, 1, 0], [-sin(theta13), 0, cos(theta13)]])
        O23 = np.matrix([[1, 0, 0], [0, cos(theta23), sin(theta23)], [0, -sin(theta23), cos(theta23)]])
        Udelta = np.matrix(np.diag([1, 1, np.exp(1j * deltacp)]))

        PMNSmat = O23 @ Udelta @ O13 @ Udelta.H @ O12

        return PMNSmat
