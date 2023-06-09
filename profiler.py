import pstats, cProfile

import pyximport
pyximport.install()

import three_flavor_matter
import numpy as np

delmsq31 = 2.5e-3
delmsq21 = 7.55e-5 
deltacp = 1.32 * np.pi
theta13 = np.radians(4.4)
theta12 = np.radians(33.2)
theta23 = np.radians(46.1)

constants = [delmsq21, delmsq31, deltacp, theta12, theta13, theta23]

baseline = 1000
energy = 0.001
V = 0 #  1e-14

solver = three_flavor_matter.ThreeFlavor(*constants)

func = lambda: solver.probmatrix(baseline, energy, V)

cProfile.runctx("func()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()