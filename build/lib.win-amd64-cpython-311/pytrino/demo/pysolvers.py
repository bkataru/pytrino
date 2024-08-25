from typing import List, Tuple

from numpy import pi, sin, cos, array, ndarray, matrix, diag, exp, identity, sqrt, arcsin, arctan2, prod
from numpy import abs as nabs
from numpy.linalg import eigvalsh, eigh

from .utils import kroneckerdelta, crct, submatrix, adjugate

class Solver:
    """
    Solver superclass for neutrino oscillation calculations.

    :param delmsq21: Mass squared difference between the second and first neutrino mass eigenstates in eV^2
    :type delmsq21: float

    :param delmsq31: Mass squared difference between the third and first neutrino mass eigenstates in eV^2
    :type delmsq31: float

    :param deltacp: The Dirac CP-violating phase in vacuum in radians
    :type deltacp: float

    :param theta12: Mixing angle θ12 in radians
    :type theta12: float

    :param theta13: Mixing angle θ13 in radians
    :type theta13: float

    :param theta23: Mixing angle θ23 in radians
    :type theta23: float

    :param _O12: theta12 rotation matrix.
    :type numpy.matrix:
    
    :param _O13: theta13 rotation matrix.
    :type numpy.matrix:

    :param _O23: theta23 rotation matrix
    :type numpy.matrix:

    :param _Udelta: Unitary CP phase matrix.
    :type numpy.matrix:

    :param _H: Hamiltonian in matter
    :type numpy.matrix:

    :param baseline: The baseline length in km
    :type float: 

    :param energy: The neutrino beam energy in GeV
    :type float:
    
    :param V: The effective (constant) matter potential in eV
    :type float:
    """
    def __init__(self, delmsq21: float, delmsq31: float, deltacp: float, theta12: float, theta13: float, theta23: float) -> None:
        """
        Initialize the Solver instance.

        :param delmsq21: Mass squared difference between the second and first neutrino mass eigenstates in eV^2
        :type delmsq21: float

        :param delmsq31: Mass squared difference between the third and first neutrino mass eigenstates in eV^2
        :type delmsq31: float

        :param deltacp: CP-violating phase in radians
        :type deltacp: float

        :param theta12: Mixing angle θ12 in radians
        :type theta12: float

        :param theta13: Mixing angle θ13 in radians
        :type theta13: float

        :param theta23: Mixing angle θ23 in radians
        :type theta23: float
        """
        
        self.deltacp = deltacp
        self.theta12 = theta12
        self.theta13 = theta13
        self.theta23 = theta23

        self.delmsq21 = delmsq21
        self.delmsq31 = delmsq31

    def alpha(self) -> float:
        """
        Compute the mass-hierarchy parameter α.

        :return: The value of α.
        :rtype: float
        """

        return self.delmsq21 / self.delmsq31

    def _PMNS_subrotations(self) -> None:
        """
        Compute the PMNS sub-rotation matrices.
        """

        theta12 = self.theta12
        theta13 = self.theta13
        theta23 = self.theta23
        deltacp = self.deltacp

        self._O12 = matrix([[cos(theta12), sin(theta12), 0], [-sin(theta12), cos(theta12), 0], [0, 0, 1]])
        self._O13 = matrix([[cos(theta13), 0, sin(theta13)], [0, 1, 0], [-sin(theta13), 0, cos(theta13)]])
        self._O23 = matrix([[1, 0, 0], [0, cos(theta23), sin(theta23)], [0, -sin(theta23), cos(theta23)]])
        self._Udelta = matrix(diag([1, 1, exp(1j * deltacp)]))

    def _hamiltonian(self) -> None:
        """
        Compute the Hamiltonian matrix.
        """

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

    def _Hevals(self) -> ndarray:
        """
        Compute the eigenvalues of the Hamiltonian matrix.

        :return: The eigenvalues of the Hamiltonian.
        :rtype: numpy.ndarray
        """

        return eigvalsh(self._H)
    
    def _computeH(self, baseline: float, energy: float, V: float, antineutrinos: bool) -> None:
        """
        Compute the Hamiltonian matrix.

        :param baseline: The baseline length in km
        :type baseline: float
        :param energy: The neutrino beam energy in GeV
        :type energy: float
        :param V: The effective (constant) matter potential in eV
        :type V: float
        :param antineutrinos: Flag indicating whether to compute for antineutrinos (True) or neutrinos (False).
        :type antineutrinos: bool
        """

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
    
    def probability(self, i: int, j: int, baseline: float, energy: float, V: float, antineutrinos: bool = False) -> float:
        """
        Compute the neutrino oscillation probability from flavor i to flavor j.

        :param i: Initial flavor index (e = 1, mu = 2, tau = 3).
        :type i: int
        :param j: Final flavor index (e = 1, mu = 2, tau = 3).
        :type j: int
        :param baseline: The baseline length in km
        :type baseline: float
        :param energy: The neutrino beam energy in GeV
        :type energy: float
        :param V: The effective (constant) matter potential in eV
        :type V: float
        :param antineutrinos: Flag indicating whether to compute for antineutrinos (True) or neutrinos (False).
        :type antineutrinos: bool
        :return: The oscillation probability.
        :rtype: float
        """

        i, j = crct(i, j)

        self._computeH(baseline, energy, V, antineutrinos)
        S = self._evolution_matrix()

        return nabs(S[j, i])**2
    
    def probmatrix(self, baseline: float, energy: float, V: float, antineutrinos: bool = False, labels: bool = False) -> ndarray:
        """
        Compute the neutrino oscillation probability matrix.

        :param baseline: The baseline length in km
        :type baseline: float
        :param energy: The neutrino beam energy in GeV
        :type energy: float
        :param V: The effective (constant) matter potential in eV
        :type V: float
        :param antineutrinos: Flag indicating whether to compute for antineutrinos (True) or neutrinos (False).
        :type antineutrinos: bool
        :param labels: Flag indicating whether to include labels in the output matrix.
        :type labels: bool
        :return: The neutrino/antineutrino oscillation probability matrix.
        :rtype: numpy.ndarray
        """

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

        pmat = array([[Pee, Pemu, Petau], [Pmue, Pmumu, Pmutau], [Ptaue, Ptaumu, Ptautau]])

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
    """
    Solver subclass to compute oscillation probabilities by 
        1. Calculating eigenvalues and eigenvectors of Hamiltonian in matter (thereby diagonalizing said Hamiltonian).
        2. Computing evolution matrix using said eigenvalues and eigenvectors
        3. Compute oscillation probabilites using evolution matrix.
    """
    def __init__(self, *args: float) -> None:
        """
        Initialize the Eigen solver instance.

        :param args: Arguments passed to the Solver superclass.
        :type args: float
        """

        super().__init__(*args)

    def _Hevecs(self) -> ndarray:
        """
        Compute the eigenvectors of the Hamiltonian matrix.

        :return: The eigenvectors of the Hamiltonian matrix.
        :rtype: numpy.ndarray
        """

        return eigh(self._H)[1]
    
    def _evolution_matrix(self) -> ndarray:
        """
        Compute the evolution matrix S.

        :return: The evolution matrix S.
        :rtype: numpy.ndarray
        """

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
    """
    Solver subclass to compute oscillation probabilities by 
        1. Calculating eigenvalues of Hamiltonian in matter.
        2. Computing evolution matrix using the Cayley-Hamilton formalism (See https://arxiv.org/abs/hep-ph/0402175 for more info)
        3. Compute oscillation probabilites using evolution matrix.
    """
    def __init__(self, *args: float) -> None:
        """
        Initialize the CayleyHamilton solver instance.

        :param args: Arguments passed to the Solver superclass.
        :type args: float
        """

        super().__init__(*args)
    
    def _evolution_matrix(self) -> ndarray:
        """
        Compute the evolution matrix S.

        :return: The evolution matrix S.
        :rtype: numpy.ndarray
        """

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
    """
    Solver subclass to compute matter oscillation properties by 
        1. Calculating eigenvalues of submatrices of the Hamiltonian in matter.
        2. Computing modulus squared of the elements of the PMNS matrix in matter using the Eigenvalue-Eigenvector identity (See https://arxiv.org/abs/1907.02534 for more info).
        3. Computing quartic products of the elements of the PMNS matrix in matter using the Adjugate identity (See https://arxiv.org/abs/2212.12565 for more info).
        4. Using these quantities to compute probabilities for appearance and disappearance channels.
        5. Implementing subroutines to explicitly compute mass squared differences, mixing angles + CP phase, and PMNS matrix in matter.
    """
    def __init__(self, *args: float) -> None:
        """
        Initialize the Identities solver instance.

        :param args: Arguments passed to the Solver superclass.
        :type args: float
        """

        super().__init__(*args)

    def _submatrix_evals(self) -> List[List[float]]:
        """
        Compute the eigenvalues of submatrices of the Hamiltonian.
        
        See https://arxiv.org/abs/1907.02534 for more info.

        :return: The eigenvalues of submatrices.
        :rtype: List[List[float]]
        """

        H = self._H

        He = submatrix(H, 1)
        Hmu = submatrix(H, 2)
        Htau = submatrix(H, 3)

        Xie, Chie = eigvalsh(He)
        Ximu, Chimu = eigvalsh(Hmu)
        Xitau, Chitau = eigvalsh(Htau)

        return [[Xie, Chie], [Ximu, Chimu], [Xitau, Chitau]]

    def _quartic_product(self, i: int, alpha: int, beta: int) -> complex:
        """
        Compute the quartic product using the Adjugate Identity.
        
        See https://arxiv.org/abs/2212.12565 for more info.

        :param i: Index i.
        :type i: int
        :param alpha: Flavor index alpha.
        :type alpha: int
        :param beta: Flavor index beta.
        :type beta: int
        :return: The value of the quartic product.
        :rtype: complex
        """

        i, alpha, beta = crct(i, alpha, beta)

        H = self._H
        lda = self._Hevals()

        adj = adjugate(lda[i] * identity(3) - H)[alpha, beta]
        denom = prod([lda[i] - lda[k] if k != i else 1 for k in range(3)])

        qprod = adj / denom

        return qprod
    
    def _PMNS_matter_modsq(self, alpha: int, i: int) -> float:
        """
        Compute the modulus squared of the elements of the PMNS matrix in matter using the Eigenvalue-Eigenvector Identity.

        See https://arxiv.org/abs/1907.02534 for more info.

        :param alpha: Flavor index alpha.
        :type alpha: int
        :param i: Index i.
        :type i: int
        :return: The value of the squared modulus.
        :rtype: float
        """

        alpha, i = crct(alpha, i)

        lda = self._Hevals()
        subeigs = self._submatrix_evals()

        sublda = subeigs[alpha]

        num = prod([lda[i] - sublda[j] for j in range(len(sublda))])
        denom = prod([lda[i] - lda[k] if k != i else 1 for k in range(3)])

        modsq = num / denom

        return modsq
    
    def _phase_mat(self, k: int, j: int, n: int = 1) -> float:
        """
        Compute the oscillation phase in matter.

        :param k: Index k.
        :type k: int
        :param j: Index j.
        :type j: int
        :param n: multiplier
        :type n: int
        :return: The oscillation phase in matter.
        :rtype: float
        """

        k, j = crct(k, j)

        lda = self._Hevals()

        L = self.baseline
        En = self.energy
        delmsq31 = self.delmsq31
        delta = (1.267 * delmsq31 * L)/En

        return n * (lda[k] - lda[j]) * delta
    
    def _disappearance(self, i: int) -> float:
        """
        Compute the disappearance probability using the modulus squared of the elements of the PMNS matrix in matter, obtained using the Eigenvalue-Eigenvector identity.

        See https://arxiv.org/abs/1907.02534 for more info.

        :param i: Flavor index i.
        :type i: int
        :return: The disappearance probability.
        :rtype: float
        """

        summation = 0
        for j in range(1, 4):
            for k in range(j + 1, 4):
                term = self._PMNS_matter_modsq(i, k) * self._PMNS_matter_modsq(i, j) * sin(self._phase_mat(k, j))**2                
                summation += term

        return 1 - 4 * summation
    
    def _appearance(self, a: int, b: int) -> float:
        """
        Compute the appearance probability using the adjugate of the Hamiltonian in matter, obtained using the Adjugate identity.

        See https://arxiv.org/abs/2212.12565 for more info.

        :param a: Flavor index a.
        :type a: int
        :param b: Flavor index b.
        :type b: int
        :return: The appearance probability.
        :rtype: float
        """

        firstsum = 0
        secondsum = 0
        for j in range(1, 4):
            for k in range(j + 1, 4):
                firstsum += (self._quartic_product(k, b, a) * self._quartic_product(j, a, b)).real * sin(self._phase_mat(k, j))**2
                secondsum += (self._quartic_product(k, b, a) * self._quartic_product(j, a, b)).imag * sin(self._phase_mat(k, j, 2))

        return kroneckerdelta(a, b) - 4 * firstsum + 2 * secondsum
    
    # @override
    def probability(
        self,
        i: int,
        j: int,
        baseline: float,
        energy: float,
        V: float,
        antineutrinos: bool = False,
    ) -> float:
        """
        Compute the neutrino oscillation probability for a given channel.

        :param i: Flavor index i.
        :type i: int
        :param j: Flavor index j.
        :type j: int
        :param baseline: Baseline length in km.
        :type baseline: float
        :param energy: Neutrino energy in GeV.
        :type energy: float
        :param V: Effective (constant) matter potential in eV.
        :type V: float
        :param antineutrinos: Flag indicating whether to compute for antineutrinos (default is False).
        :type antineutrinos: bool
        :return: The neutrino/antineutrino oscillation probability.
        :rtype: float
        """

        self._computeH(baseline, energy, V, antineutrinos)
        if i == j:
            return self._disappearance(i)
        else:
            return self._appearance(i, j)
    
    def _mixing_angles_mat(self) -> Tuple[float, float, float]:
        """
        Compute the mixing angles in matter.

        :return: The mixing angles in matter (theta12mat, theta13mat, theta23mat).
        :rtype: Tuple[float, float, float]
        """

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
    
    def _toshev_identity(self) -> float: # DOESNT WORK LOL EDIT: no it does TODO
        """
        Implements the Toshev identity to compute sine of the CP phase in matter.

        :return: The sine of the CP phase in matter.
        :rtype: float
        """
        
        _, _, theta23mat = self._mixing_angles_mat()

        deltacp = self.deltacp
        theta23 = self.theta23

        sindeltacpmat = (sin(2 * theta23)/sin(2 * theta23mat)) * sin(deltacp)

        return sindeltacpmat
    
    def _cpphase_mat(self) -> float:
        """
        Calculates the CP phase in matter using the Eigenvalue-Eigenvector and Toshev identities.

        Note: The CP phase being 2 pi bla bla this is outliend in eq. bla bla TODO

        See https://arxiv.org/abs/1907.02534 for more info.

        :return: The value of the CP phase in matter in radians.
        :rtype: float
        """

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
    def deltamsq_matter(self, baseline: float, energy: float, V: float, antineutrinos: bool = False) -> Tuple[float, float, float]:
        """
        Calculates the mass squared differences in matter for the given parameters.

        :param baseline: The baseline length in km.
        :type baseline: float
        :param energy: The neutrino beam energy in GeV.
        :type energy: float
        :param V: The effective (constant) matter potential in eV.
        :type V: float
        :param antineutrinos: Flag indicating whether to compute for antineutrinos. Defaults to False.
        :type antineutrinos: bool, optional
        :return: The mass squared differences in matter in eV^2.
        :rtype: Tuple[float, float, float]
        """

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

    def angles_phase_matter(self, baseline: float, energy: float, V: float, antineutrinos: bool = False) -> Tuple[float, float, float, float]:
        """
        Calculates the mixing angles and CP-violating phase in matter for the given parameters.

        See https://arxiv.org/abs/1907.02534 for more info.

        :param baseline: The baseline length in km.
        :type baseline: float
        :param energy: The neutrino beam energy in GeV.
        :type energy: float
        :param V: The effective (constant) matter potential in eV.
        :type V: float
        :param antineutrinos: Flag indicating whether to compute for antineutrinos. Defaults to False.
        :type antineutrinos: bool, optional
        :return: The CP phase and mixing angles in matter in radians.
        :rtype: Tuple[float, float, float, float]

        """

        self._computeH(baseline, energy, V, antineutrinos)

        theta12mat, theta13mat, theta23mat = self._mixing_angles_mat()
        deltacpmat = self._cpphase_mat()

        return deltacpmat, theta12mat, theta13mat, theta23mat
    
    def PMNS_matter(self, baseline: float, energy: float, V: float, antineutrinos: bool = False) -> matrix:
        """
        Calculates the PMNS matrix in matter for the given parameters.

        See https://arxiv.org/abs/1907.02534 for more info.

        :param baseline: The baseline length in km.
        :type baseline: float
        :param energy: The neutrino beam energy in GeV.
        :type energy: float
        :param V: The effective (constant) matter potential in eV.
        :type V: float
        :param antineutrinos: Flag indicating whether to compute for antineutrinos. Defaults to False.
        :type antineutrinos: bool, optional
        :return: The PMNS matrix in matter.
        :rtype: matrix
        """

        self._computeH(baseline, energy, V, antineutrinos)
    
        theta12mat, theta13mat, theta23mat = self._mixing_angles_mat()
        deltacpmat = self._cpphase_mat()
        
        O12mat = matrix([[cos(theta12mat), sin(theta12mat), 0], [-sin(theta12mat), cos(theta12mat), 0], [0, 0, 1]])
        O13mat = matrix([[cos(theta13mat), 0, sin(theta13mat)], [0, 1, 0], [-sin(theta13mat), 0, cos(theta13mat)]])
        O23mat = matrix([[1, 0, 0], [0, cos(theta23mat), sin(theta23mat)], [0, -sin(theta23mat), cos(theta23mat)]])
        Udeltamat = matrix(diag([1, 1, exp(1j * deltacpmat)]))

        PMNSmat = O23mat @ Udeltamat @ O13mat @ Udeltamat.H @ O12mat

        return PMNSmat