#cython: language_level=3

cdef extern from "math.h":
    double sin(double x)
    double cos(double x)
    double sqrt(double x)

cdef class TwoFlavor:
    cdef double _delmsq, _theta

    @property
    def delmsq(self): return self._delmsq
    @property
    def theta(self): return self._theta

    @delmsq.setter
    def delmsq(self, double val): self._delmsq = val
    @theta.setter
    def theta(self, double val): self._theta = val

    # delmsq in eV^2
    # theta in radians
    def __cinit__(self, double delmsq, double theta):
        self._delmsq = delmsq
        self._theta = theta

    # E in GeV
    # L in km
    # V in eV
    def probability(self, double L, double E, double V = 0):
        cdef double delmsq = self.delmsq
        cdef double theta = self.theta

        cdef double sin2theta = sin(2 * theta)
        cdef double cos2theta = cos(2 * theta)
        cdef double A = (2 * 1e+9 * E * V) / delmsq
        cdef double sin2theta_sq = sin2theta ** 2
        cdef double cos2theta_term_sq = (cos2theta - A) ** 2

        cdef double deltamat = ((1.267 * delmsq * L) / E) * sqrt(sin2theta_sq + cos2theta_term_sq)
        cdef double sin2thetam = sin2theta / sqrt(sin2theta_sq + cos2theta_term_sq)

        cdef double Pemu = sin2thetam ** 2 * sin(deltamat) ** 2
        cdef double Pee = 1 - Pemu

        return Pee, Pemu
        
        