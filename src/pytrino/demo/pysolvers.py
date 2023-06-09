from numpy import pi, sin, cos, array, matrix, diag, exp, identity, sqrt, arcsin, arctan2, prod
from numpy import abs as nabs
from numpy.linalg import eigvalsh, eigh

from .utils import kroneckerdelta, crct, submatrix, adjugate

class Solver:
    def __init__(self, delmsq21, delmsq31, deltacp, theta12, theta13, theta23):
        self.deltacp = deltacp
        self.theta12 = theta12
        self.theta13 = theta13
        self.theta23 = theta23

        self.delmsq21 = delmsq21
        self.delmsq31 = delmsq31

    def alpha(self):
        return self.delmsq21 / self.delmsq31

    def _PMNS_subrotations(self):
        theta12 = self.theta12
        theta13 = self.theta13
        theta23 = self.theta23
        deltacp = self.deltacp

        self._O12 = matrix([[cos(theta12), sin(theta12), 0], [-sin(theta12), cos(theta12), 0], [0, 0, 1]])
        self._O13 = matrix([[cos(theta13), 0, sin(theta13)], [0, 1, 0], [-sin(theta13), 0, cos(theta13)]])
        self._O23 = matrix([[1, 0, 0], [0, cos(theta23), sin(theta23)], [0, -sin(theta23), cos(theta23)]])
        self._Udelta = matrix(diag([1, 1, exp(1j * deltacp)]))

    def _hamiltonian(self):
        O12 = self._O12
        O13 = self._O13
        O23 = self._O23
        Udelta = self._Udelta

        alpha = self.alpha()
        delmsq31 = self.delmsq31

        energy = self.energy
        V = self.V

        a = (2 * energy * 1e+9 * V)/delmsq31

        # PMNS = O23 @ Udelta @ O13 @ Udelta.H @ O12

        M = O13 @ O12 @ diag([0, alpha, 1]) @ O12.T @ O13.T + diag([a, 0, 0])

        self._H = O23 @ Udelta @ M @ Udelta.H @ O23.T

    def _Hevals(self):
        return eigvalsh(self._H)
    
    def _computeH(self, baseline, energy, V, antineutrinos):
        if antineutrinos:
            self.deltacp = -self.deltacp
            V = -V

        self._PMNS_subrotations()

        self.baseline = baseline
        self.energy = energy
        self.V = V

        self._hamiltonian()

        if antineutrinos:
            self.deltacp = -self.deltacp
            V = -V
    
    def probability(self, i, j, baseline, energy, V, antineutrinos = False):
        i, j = crct(i, j)

        self._computeH(baseline, energy, V, antineutrinos)
        S = self._evolution_matrix()

        return nabs(S[j, i])**2
    
    def probmatrix(self, baseline, energy, V, antineutrinos = False, labels = False):
        Pemu = self.probability(1, 2, baseline, energy, V, antineutrinos)
        Pmutau = self.probability(2, 3, baseline, energy, V, antineutrinos)
        
        self.theta23 = self.theta23 + pi/2
        _Pemu = self.probability(1, 2, baseline, energy, V, antineutrinos)
        _Pmutau = self.probability(2, 3, baseline, energy, V, antineutrinos)
        self.theta23 = self.theta23 - pi/2

        Pee = 1 - (Pemu + _Pemu)
        Petau = _Pemu
        Pmue = Pemu - Pmutau + _Pmutau
        Pmumu = 1 - Pemu - _Pmutau
        Ptaue = _Pemu + Pmutau - _Pmutau
        Ptaumu = _Pmutau
        Ptautau = 1 - (_Pemu + Pmutau)

        pmat = matrix([[Pee, Pemu, Petau], [Pmue, Pmumu, Pmutau], [Ptaue, Ptaumu, Ptautau]])

        if labels:
            labelmatrix = matrix([["Pee", "Pemu", "Petau"], ["Pmue", "Pmumu", "Pmutau"], ["Ptaue", "Ptaumu", "Ptautau"]])
            if antineutrinos:
                labelmatrix = matrix([['~' + labelmatrix[i, j] for j in range(3)] for i in range(3)])
            
            labeledpmat = [[(labelmatrix[i, j], pmat[i, j]) for j in range(3)] for i in range(3)]
            pmat = array(labeledpmat)
        
        '''
        # labelmatrix = matrix([["Pee", "Pemu", "Petau"], ["Pmue", "Pmumu", "Pmutau"], ["Ptaue", "Ptaumu", "Ptautau"]])
        pmat = np.zeros((3, 3))

        for i in range(1, 4):
            for j in range(1, 4):
                # label = labelmatrix[i - 1, j - 1]
                prob = self.probability(i, j, baseline, energy, V, antineutrinos)

                # print(f"{label}: {prob}")
                
                pmat[i - 1, j - 1] = prob
        '''

        return pmat

class Eigen(Solver):
    def __init__(self, *args):
        super().__init__(*args)

    def _Hevecs(self):
        return eigh(self._H)[1]
    
    def _evolution_matrix(self):
        lda1, lda2, lda3 = self._Hevals()
        evecs = self._Hevecs()

        L = self.baseline
        En = self.energy
        delmsq31 = self.delmsq31
        delta = (1.267 * delmsq31 * L)/En

        PMNSmat = matrix(evecs)

        S = PMNSmat @ diag([exp(-1j * lda1 * 2 * delta), exp(-1j * lda2 * 2 * delta), exp(-1j * lda3 * 2 * delta)]) @ PMNSmat.H
        return S
    
class CayleyHamilton(Solver):
    def __init__(self, *args):
        super().__init__(*args)
    
    def _evolution_matrix(self):
        H = self._H
        lda1, lda2, lda3 = self._Hevals()

        L = self.baseline
        En = self.energy
        delmsq31 = self.delmsq31
        delta = (1.267 * delmsq31 * L)/En

        S = (exp(-1j * lda1 * 2 * delta)/((lda1 - lda2) * (lda1 - lda3))) * (lda2 * lda3 * identity(3) - (lda2 + lda3) * H + H @ H) +\
        (exp(-1j * lda2 * 2 * delta)/((lda2 - lda1) * (lda2 - lda3))) * (lda1 * lda3 * identity(3) - (lda1 + lda3) * H + H @ H) +\
        (exp(-1j * lda3 * 2 * delta)/((lda3 - lda1) * (lda3 - lda2))) * (lda1 * lda2 * identity(3) - (lda1 + lda2) * H + H @ H)

        return S

class Identities(Solver):
    def __init__(self, *args):
        super().__init__(*args)

    def _submatrix_evals(self):
        H = self._H

        He = submatrix(H, 1)
        Hmu = submatrix(H, 2)
        Htau = submatrix(H, 3)

        Xie, Chie = eigvalsh(He)
        Ximu, Chimu = eigvalsh(Hmu)
        Xitau, Chitau = eigvalsh(Htau)

        return [[Xie, Chie], [Ximu, Chimu], [Xitau, Chitau]]

    def _quartic_product(self, i, alpha, beta):
        i, alpha, beta = crct(i, alpha, beta)

        H = self._H
        lda = self._Hevals()

        adj = adjugate(lda[i] * identity(3) - H)[alpha, beta]
        denom = prod([lda[i] - lda[k] if k != i else 1 for k in range(3)])

        qprod = adj / denom

        return qprod
    
    def _PMNS_matter_modsq(self, alpha, i):
        alpha, i = crct(alpha, i)

        lda = self._Hevals()
        subeigs = self._submatrix_evals()

        sublda = subeigs[alpha]

        num = prod([lda[i] - sublda[j] for j in range(len(sublda))])
        denom = prod([lda[i] - lda[k] if k != i else 1 for k in range(3)])

        modsq = num / denom

        return modsq
    
    def _phase_mat(self, k, j, n=1):
        k, j = crct(k, j)

        lda = self._Hevals()

        L = self.baseline
        En = self.energy
        delmsq31 = self.delmsq31
        delta = (1.267 * delmsq31 * L)/En

        return n * (lda[k] - lda[j]) * delta
    
    def _disappearance(self, i):
        summation = 0
        for j in range(1, 4):
            for k in range(j + 1, 4):
                print(self._phase_mat(k, j))
                term = self._PMNS_matter_modsq(i, k) * self._PMNS_matter_modsq(i, j) * sin(self._phase_mat(k, j))**2                
                summation += term

        return 1 - 4 * summation
    
    def _appearance(self, a, b):
        firstsum = 0
        secondsum = 0
        for j in range(1, 4):
            for k in range(j + 1, 4):
                firstsum += (self._quartic_product(k, b, a) * self._quartic_product(j, a, b)).real * sin(self._phase_mat(k, j))**2
                secondsum += (self._quartic_product(k, b, a) * self._quartic_product(j, a, b)).imag * sin(self._phase_mat(k, j, 2))

        return kroneckerdelta(a, b) - 4 * firstsum + 2 * secondsum
    
    # @override
    def probability(self, i, j, baseline, energy, V, antineutrinos = False):
        self._computeH(baseline, energy, V, antineutrinos)
        self._disappearance(i)
        if i == j:
            return self._disappearance(i)
        else:
            return self._appearance(i, j)
    
    def _mixing_angles_mat(self):
        s13matsq = self._PMNS_matter_modsq(1, 3).real
        c13matsq = 1 - s13matsq

        s12matsq = self._PMNS_matter_modsq(1, 2).real / c13matsq
        s23matsq = self._PMNS_matter_modsq(2, 3).real / c13matsq

        s12mat = sqrt(s12matsq)
        s13mat = sqrt(s13matsq) 
        s23mat = sqrt(s23matsq)

        theta12mat = arcsin(s12mat)
        theta13mat = arcsin(s13mat)
        theta23mat = arcsin(s23mat)

        return theta12mat, theta13mat, theta23mat
    
    def _toshev_identity(self): # DOESNT WORK LOL EDIT: no it does TODO
        _, _, theta23mat = self._mixing_angles_mat()

        deltacp = self.deltacp
        theta23 = self.theta23

        sindeltacpmat = (sin(2 * theta23)/sin(2 * theta23mat)) * sin(deltacp)

        return sindeltacpmat
    
    def _cpphase_mat(self):
        lda1, lda2, lda3 = self._Hevals()
        Ximu, Chimu = self._submatrix_evals()[1]

        s12 = sin(self.theta12)
        c12 = cos(self.theta12)
        s23 = sin(self.theta23)
        c23 = cos(self.theta23)
        s13 = sin(self.theta13)

        Umu1modsq = ((lda1 - Ximu) * (lda1 - Chimu))/((lda1 - lda2) * (lda1 - lda3))

        sdcp = self._toshev_identity()
        cdcp = (Umu1modsq - s12**2 * c23**2 - c12**2 * s13**2 * s23**2)/(2 * s12 * c12 * s13 * s23 * c23)

        deltacpmat = arctan2(sdcp, cdcp)
        if deltacpmat < 0:
            deltacpmat += 2 * pi

        return deltacpmat
    
    # mass squared differences in matter
    def deltamsq_matter(self, baseline, energy, V, antineutrinos = False):
        self._computeH(baseline, energy, V, antineutrinos)

        lda = self._Hevals()
        delmsq31 = self.delmsq31
        
        pairs = [(2, 1), (3, 1), (3, 2)]
        msqsmat = []
        for k, j in pairs:
            k = k - 1
            j = j - 1

            msqsmat.append(delmsq31 * (lda[k] - lda[j]))
        return tuple(msqsmat)

    def angles_phase_matter(self, baseline, energy, V, antineutrinos = False): 
        self._computeH(baseline, energy, V, antineutrinos)

        theta12mat, theta13mat, theta23mat = self._mixing_angles_mat()
        deltacpmat = self._cpphase_mat()

        return deltacpmat, theta12mat, theta13mat, theta23mat
    
    def PMNS_matter(self, baseline, energy, V, antineutrinos = False):
        self._computeH(baseline, energy, V, antineutrinos)
    
        theta12mat, theta13mat, theta23mat = self._mixing_angles_mat()
        deltacpmat = self._cpphase_mat()
        
        O12mat = matrix([[cos(theta12mat), sin(theta12mat), 0], [-sin(theta12mat), cos(theta12mat), 0], [0, 0, 1]])
        O13mat = matrix([[cos(theta13mat), 0, sin(theta13mat)], [0, 1, 0], [-sin(theta13mat), 0, cos(theta13mat)]])
        O23mat = matrix([[1, 0, 0], [0, cos(theta23mat), sin(theta23mat)], [0, -sin(theta23mat), cos(theta23mat)]])
        Udeltamat = matrix(diag([1, 1, exp(1j * deltacpmat)]))

        PMNSmat = O23mat @ Udeltamat @ O13mat @ Udeltamat.H @ O12mat

        return PMNSmat