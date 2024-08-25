#cython: language_level=3

cdef extern from "math.h":
    double sin(double x)
    double cos(double x)
    double sqrt(double x)

cdef class TwoFlavor:
    """
    Cython solver class to compute properties of a two-flavor neutrino oscillation model.
    """

    cdef double _delmsq, _theta

    @property
    def delmsq(self): 
        """
        Get the value of delmsq in eV^2.

        :return: The value of delmsq.
        :rtype: float
        """

        return self._delmsq
    
    @property
    def theta(self):
        """
        Get the value of theta in radians.

        :return: The value of theta.
        :rtype: float
        """
        return self._theta

    @delmsq.setter
    def delmsq(self, double val):
        """
        Set the value of delmsq.

        :param val: The value of delmsq in eV^2.
        :type val: float
        """
        self._delmsq = val

    @theta.setter
    def theta(self, double val):
        """
        Set the value of theta.

        :param val: The value of theta in radians.
        :type val: float
        """
        self._theta = val

    # delmsq in eV^2
    # theta in radians
    def __cinit__(self, double delmsq, double theta):
        """
        Initialize the TwoFlavor solver.

        :param delmsq: The value of delmsq in eV^2.
        :type delmsq: float
        :param theta: The value of theta in radians.
        :type theta: float
        """
        self._delmsq = delmsq
        self._theta = theta

    # E in GeV
    # L in km
    # V in eV
    def probability(self, double L, double E, double V = 0, bint antineutrinos = False):
        """
        Calculate the two flavor neutrino oscillation probability in vacuum/matter.

        :param L: The distance traveled by the neutrino in km.
        :type L: float
        :param E: The neutrino beam energy in GeV.
        :type E: float
        :param V: The effective (constant) matter potential in eV. Default is 0.
        :type V: float
        :param antineutrinos: Whether to compute for antineutrinos (True) or neutrinos (False). Default is False.
        :type antineutrinos: bool
        :return: The probabilities (Pee, Pemu) of oscillations. Note that the other two probabilities can be computed using these as Pmue = Pemu and Pmumu = Pee
        :rtype: Tuple[float, float]
        """
        if antineutrinos:
            V = -V
        
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
        
        