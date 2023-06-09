#cython: language_level=3

from libc.math cimport pi, sin, cos, sqrt, asin, acos, atan2
from cmath import exp as cexp

'''
cdef extern from "complex.h":
    double complex cexp(double complex z)

    double cabs(double complex z)
    double complex conj(double complex z)

    # Decomposing complex values
    double cimag(double complex z)
    double creal(double complex z)
'''

cdef class ThreeFlavor:
    cdef double _delmsq21, _delmsq31, _deltacp, _theta12, _theta13, _theta23
    cdef double alpha, s12, c12, s13, c13, s23, c23

    @property
    def delmsq21(self): return self._delmsq21

    @property
    def delmsq31(self): return self._delmsq31

    @property
    def deltacp(self): return self._deltacp

    @property
    def theta12(self): return self._theta12
    
    @property
    def theta13(self): return self._theta13

    @property
    def theta23(self): return self._theta23
    
    @delmsq21.setter
    def delmsq21(self, double val): 
        self._delmsq21 = val
        self.alpha = val / self._delmsq31

    @delmsq31.setter
    def delmsq31(self, double val):
        self._delmsq31 = val
        self.alpha = self._delmsq21 / val

    @deltacp.setter
    def deltacp(self, double val):
        self._deltacp = val

    @theta12.setter
    def theta12(self, double val):
        self._theta12 = val
        self.s12 = sin(val)
        self.c12 = cos(val)

    @theta13.setter
    def theta13(self, double val):
        self._theta13 = val
        self.s13 = sin(val)
        self.c13 = cos(val)

    @theta23.setter
    def theta23(self, double val):
        self._theta23 = val
        self.s23 = sin(val)
        self.c23 = cos(val)
    
    # delmsq21 and delmsq31 in eV^2
    # all angles in radians
    def __cinit__(self, double delmsq21, double delmsq31, double deltacp, double theta12, double theta13, double theta23):
        self._delmsq21 = delmsq21
        self._delmsq31 = delmsq31

        self._deltacp = deltacp
        self._theta12 = theta12
        self._theta13 = theta13
        self._theta23 = theta23

        self.alpha = delmsq21 / delmsq31

        self.s12 = sin(theta12)
        self.c12 = cos(theta12)
        self.s13 = sin(theta13)
        self.c13 = cos(theta13)
        self.s23 = sin(theta23)
        self.c23 = cos(theta23)

    cdef (double, double, double) eigenvalues(self, double E, double V):
        cdef double delmsq21 = self.delmsq21
        cdef double delmsq31 = self.delmsq31

        cdef double s12 = self.s12
        cdef double c12 = self.c12
        cdef double c13 = self.c13

        cdef double A = (2 * 1e+9 * E * V) / self.delmsq31
        
        cdef double X = delmsq21 + delmsq31 + (A * delmsq31)
        cdef double Y = delmsq21 * delmsq31 + (A * delmsq31) * (delmsq31 * c13**2 + delmsq21 * (1 - c13**2 * s12**2))
        cdef double Z = (A * delmsq31) * delmsq21 * delmsq31 * c13**2 * c12**2
        cdef double W = cos((1/3) * acos((2 * X**3 - 9 * X * Y + 27 * Z)/(2 * (X**2 - 3 * Y)**(3/2))))

        cdef double lda1 = (X/3) - (1/3) * sqrt(X**2 - 3 * Y) * (W + sqrt(3 * (1 - W**2)))
        cdef double lda2 = (X/3) - (1/3) * sqrt(X**2 - 3 * Y) * (W - sqrt(3 * (1 - W**2)))
        cdef double lda3 = (X/3) + (2/3) * sqrt(X**2 - 3 * Y) * W

        lda1 = lda1 / delmsq31  # document this TODO
        lda2 = lda2 / delmsq31
        lda3 = lda3 / delmsq31

        return lda1, lda2, lda3

    # L in km
    # E in GeV
    # V in eV
    cpdef (double, double, double) deltamsq_matter(self, double E, double V, bint antineutrinos = False):
        if antineutrinos:
            V = -V

        cdef double delmsq31 = self.delmsq31

        cdef double lda1, lda2, lda3
        lda1, lda2, lda3 = self.eigenvalues(E, V)

        cdef double delmsq21mat = delmsq31 * (lda2 - lda1)
        cdef double delmsq31mat = delmsq31 * (lda3 - lda1)
        cdef double delmsq32mat = delmsq31 * (lda3 - lda2)

        return delmsq21mat, delmsq31mat, delmsq32mat

    cpdef (double, double, double, double) angles_phase_matter(self, double L, double E, double V, bint antineutrinos = False):
        cdef double delmsq31 = self.delmsq31
        cdef double alpha = self.alpha

        cdef double deltacp
        if antineutrinos:
            V = -V
            deltacp = -self.deltacp
        else:
            deltacp = self.deltacp

        cdef double theta23 = self.theta23

        cdef double s12 = self.s12
        cdef double c12 = self.c12
        cdef double s13 = self.s13
        cdef double c13 = self.c13
        cdef double s23 = self.s23
        cdef double c23 = self.c23

        cdef double lda1, lda2, lda3
        lda1, lda2, lda3 = self.eigenvalues(E, V)

        cdef double A = (2 * 1e+9 * E * V) / self.delmsq31

        cdef double M_ee = A + s13**2 + c13**2 * s12**2 * alpha
        cdef double M_emu = c12 * c13 * s12 * alpha
        cdef double M_etau = c13 * s13 * (1 - s12**2 * alpha)

        cdef double M_mumu = c12**2 * alpha
        cdef double M_mutau = -c12 * s12 * s13 * alpha

        cdef double M_tautau = c13**2 + s12**2 * s13**2 * alpha

        cdef double sub_e_sum = M_mumu + M_tautau
        cdef double sub_e_prod = M_mumu * M_tautau - M_mutau**2

        cdef double sub_mu_sum = M_ee + c23**2 * M_tautau + M_mumu * s23**2 - 2 * c23 * M_mutau * s23 * cos(deltacp)
        cdef double sub_mu_prod = M_ee * (c23**2 * M_tautau + M_mumu * s23**2 - 2 * c23 * M_mutau * s23 * cos(deltacp)) - abs(c23 * cexp(-1j * deltacp) * M_etau - M_emu * s23)**2

        cdef double s13matsq = (lda3**2 - sub_e_sum * lda3 + sub_e_prod)/((lda3 - lda1) * (lda3 - lda2))
        cdef double c13matsq = 1 - s13matsq

        cdef double s12matsq = ((1/c13matsq) * ((lda2**2 - sub_e_sum * lda2 + sub_e_prod)/((lda2 - lda3) * (lda2 - lda1))))
        cdef double s23matsq = ((1/c13matsq) * ((lda3**2 - sub_mu_sum * lda3 + sub_mu_prod)/((lda3 - lda1) * (lda3 - lda2))))

        cdef double theta12mat = asin(sqrt(s12matsq))
        cdef double theta13mat = asin(sqrt(s13matsq))
        cdef double theta23mat = asin(sqrt(s23matsq))

        cdef double Umu1modsq = (lda1**2 - sub_mu_sum * lda1 + sub_mu_prod)/((lda1 - lda2) * (lda1 - lda3))

        cdef double sdcp = (sin(2 * theta23)/sin(2 * theta23mat)) * sin(deltacp)
        cdef double cdcp = (Umu1modsq - s12**2 * c23**2 - c12**2 * s13**2 * s23**2)/(2 * s12 * c12 * s13 * s23 * c23)

        cdef double deltacpmat = atan2(sdcp, cdcp) # sus
        if deltacpmat < 0:
            deltacpmat += 2 * pi

        return deltacpmat, theta12mat, theta13mat, theta23mat

    
    cpdef double probability(self, int alpha, int beta, double L, double E, double V, bint antineutrinos = False):
        cdef double delmsq31 = self.delmsq31
        
        cdef double deltacpmat, theta12mat, theta13mat, theta23mat
        deltacpmat, theta12mat, theta13mat, theta23mat = self.angles_phase_matter(L, E, V, antineutrinos)

        cdef double complex[3][3] Umat = [
            [
                cos(theta12mat) * cos(theta13mat), 
                sin(theta12mat) * cos(theta13mat),
                sin(theta13mat) * cexp(-1j * deltacpmat)
            ],
            [
                -sin(theta12mat) * cos(theta23mat) - cos(theta12mat) * sin(theta23mat) * sin(theta13mat) * cexp(1j * deltacpmat),
                cos(theta12mat) * cos(theta23mat) - sin(theta12mat) * sin(theta23mat) * sin(theta13mat) * cexp(1j * deltacpmat),
                sin(theta23mat) * cos(theta13mat)
            ],
            [
                sin(theta12mat) * sin(theta23mat) - cos(theta12mat) * cos(theta23mat) * sin(theta13mat) * cexp(1j * deltacpmat),
                -cos(theta12mat) * sin(theta23mat) - sin(theta12mat) * cos(theta23mat) * sin(theta13mat) * cexp(1j * deltacpmat),
                cos(theta23mat) * cos(theta13mat)
            ]
        ]

        cdef double delmsq21mat, delmsq31mat, delmsq32mat
        delmsq21mat, delmsq31mat, delmsq32mat = self.deltamsq_matter(E, V, antineutrinos)

        delmsq21mat = delmsq21mat / delmsq31
        delmsq31mat = delmsq31mat / delmsq31
        delmsq32mat = delmsq32mat / delmsq31

        cdef double delta = (1.267 * delmsq31 * L)/E

        cdef int[3][2] pairs = [[2, 1], [3, 1], [3, 2]]
        cdef double[3] msqsmat = [delmsq21mat, delmsq31mat, delmsq32mat]

        cdef int i, ind, k, j, a, b, kd
        cdef double summation, firstsum, secondsum
        if alpha == beta:
            i = alpha - 1

            ind = 0
            summation = 0

            for ind in range(3):
                k = pairs[ind][0] - 1
                j = pairs[ind][1] - 1

                summation += abs(Umat[i][k])**2 * abs(Umat[i][j])**2 * sin(msqsmat[ind] * delta)**2

            return 1 - 4 * summation
        else:
            a = alpha - 1
            b = beta - 1

            kd = 0
            if a == b:
                kd = 1

            firstsum = 0.0
            secondsum = 0.0

            ind = 0
            for ind in range(3):
                k = pairs[ind][0] - 1
                j = pairs[ind][1] - 1

                firstsum += ((Umat[a][k]).conjugate() * Umat[b][k] * Umat[a][j] * (Umat[b][j]).conjugate()).real * sin(msqsmat[ind] * delta)**2
                secondsum += ((Umat[a][k]).conjugate() * Umat[b][k] * Umat[a][j] * (Umat[b][j]).conjugate()).imag * sin(2 * msqsmat[ind] * delta)

            return kd - 4 * firstsum + 2 * secondsum

    def probmatrix(self, double L, double E, double V, bint antineutrinos = False):
        cdef double[3][3] probarray

        # TODO compute matrix more efficiently

        cdef int i, j
        for i in range(0, 3):
            for j in range(0, 3):
                probarray[i][j] = self.probability(i + 1, j + 1, L, E, V, antineutrinos)
        
        return probarray


